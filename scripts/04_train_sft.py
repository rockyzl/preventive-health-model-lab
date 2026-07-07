#!/usr/bin/env python3
"""QLoRA supervised fine-tuning (SFT), config-driven by ``sft_lora_medical.yaml``.

Loads a base model in 4-bit (nf4, double-quant, bf16 compute — the exact path
proven by ``scripts/00b_smoke_test_4bit.py`` on this Blackwell GPU), prepares it
for k-bit training, attaches a LoRA adapter (peft) with the config's
r/alpha/target_modules, and trains with TRL's ``SFTTrainer`` on
``data/processed/{train,val}.jsonl``.

Each ``{instruction, input, output}`` record is rendered into ONE training
string by ``training.formatting`` — the same module the baseline runner uses to
build its prompt, so training targets and inference prompts cannot drift apart.

Guard rails (by design):
  * NO silent gated download. Without ``--smoke``, if the config model isn't
    already cached locally, the script prints the exact license-acceptance +
    ``hf login`` steps and exits non-zero (never triggers a multi-GB pull).
  * ``--smoke`` overrides the model to an UNGATED tiny model
    (``Qwen/Qwen2.5-0.5B-Instruct``), runs a handful of steps on a tiny subset,
    and saves the adapter to ``adapters/smoke/`` — this proves the whole path
    (4-bit base -> LoRA -> SFTTrainer -> saved adapter) on THIS GPU without any
    license gate. The real config model (MedGemma) is untouched until the user
    accepts its HF license and runs without ``--smoke``.

Usage:
    python scripts/04_train_sft.py --smoke            # tiny ungated proof run
    python scripts/04_train_sft.py                    # real run (needs weights)
    python scripts/04_train_sft.py --config configs/sft_lora_medical.yaml
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make the src-layout package importable when run as a script.
sys.path.insert(0, str(REPO_ROOT / "src"))

# stdlib-only at module import time so ``--help`` never pays for torch/transformers.
from preventive_health_model_lab.utils.config import load_yaml  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "sft_lora_medical.yaml"

# Ungated, Apache-2.0, ~1 GB. Exercises the identical 4-bit + LoRA kernel path
# as a real 4B run, with no HF license gate. NOT a project model.
SMOKE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SMOKE_OUTPUT_DIR = "adapters/smoke"
SMOKE_TRAIN_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "tiny_train.jsonl"
SMOKE_VAL_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "tiny_val.jsonl"


def weights_available_locally(model_id: str) -> bool:
    """Best-effort check for a locally-cached model, WITHOUT importing torch.

    Mirrors ``scripts/03_run_baseline.py`` so both scripts refuse a silent pull
    the same way. Returns False when we can't confirm a local snapshot.
    """
    if not model_id:
        return False
    if Path(model_id).expanduser().exists():
        return True
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    slug = "models--" + model_id.replace("/", "--")
    return (cache / slug).exists()


def _gated_model_message(model_id: str) -> str:
    return (
        "\nModel weights are not available locally, and this script will NOT\n"
        "download them automatically (the configured model is gated on HF).\n\n"
        f"  model_id: {model_id}\n\n"
        "To proceed:\n"
        "  1. Accept the model license on Hugging Face (MedGemma is gated).\n"
        "  2. Authenticate:      hf auth login\n"
        f"  3. Pre-download once: hf download {model_id}\n"
        "  4. Re-run this script.\n\n"
        "Or prove the pipeline now on an ungated tiny model:\n"
        "  python scripts/04_train_sft.py --smoke\n"
    )


def resolve_data_paths(cfg: dict, args: argparse.Namespace) -> tuple[Path, Path | None]:
    """Resolve train/eval JSONL paths.

    Precedence: explicit ``--train-file/--val-file`` > config paths (if they
    exist) > (smoke only) the bundled tiny fixtures. Returns absolute paths;
    the eval path may be None if no eval data is available.
    """
    data_cfg = cfg.get("data", {})

    def _resolve(explicit: str | None, cfg_key: str, fixture: Path) -> Path | None:
        if explicit:
            return Path(explicit).resolve()
        cfg_path = data_cfg.get(cfg_key)
        if cfg_path:
            p = (REPO_ROOT / cfg_path).resolve()
            if p.exists():
                return p
        if args.smoke and fixture.exists():
            return fixture.resolve()
        return None

    train = _resolve(args.train_file, "train_path", SMOKE_TRAIN_FIXTURE)
    val = _resolve(args.val_file, "eval_path", SMOKE_VAL_FIXTURE)
    if train is None:
        src = args.train_file or data_cfg.get("train_path", "(unset)")
        raise FileNotFoundError(
            f"Training data not found: {src}\n"
            "  The data track writes data/processed/train.jsonl. For a smoke\n"
            "  proof run without it, use --smoke (falls back to tests/fixtures)."
        )
    return train, val


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to the SFT config YAML")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="ungated tiny-model proof run: overrides model_id, runs a few steps "
        "on a tiny subset, saves adapter to adapters/smoke/",
    )
    p.add_argument("--train-file", default=None, help="override path to train JSONL")
    p.add_argument("--val-file", default=None, help="override path to eval/val JSONL")
    p.add_argument("--model", default=None, help="override base model_id (non-smoke), e.g. the control model")
    p.add_argument("--output-dir", default=None, help="override adapter output dir (non-smoke)")
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="cap optimizer steps (default: 5 in --smoke, else config epochs)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="use only the first N train rows (default: 8 in --smoke, else all)",
    )
    return p


def _dtype_from_name(name: str, torch_mod):
    return {
        "bfloat16": torch_mod.bfloat16,
        "float16": torch_mod.float16,
        "float32": torch_mod.float32,
    }.get(str(name).lower(), torch_mod.bfloat16)


def setup_tracking(cfg: dict, args: argparse.Namespace) -> list[str]:
    """Wire mlflow if present; degrade gracefully to no tracking if not.

    Returns the ``report_to`` list for SFTConfig. Smoke runs never track.
    """
    if args.smoke:
        return []
    tracking = cfg.get("tracking", {})
    if tracking.get("backend") != "mlflow":
        return []
    if importlib.util.find_spec("mlflow") is None:
        print(
            "[tracking] mlflow is not installed — continuing WITHOUT experiment "
            "tracking. `pip install mlflow` to enable local ./mlruns logging.",
            file=sys.stderr,
        )
        return []
    import os

    os.environ.setdefault("MLFLOW_TRACKING_URI", f"file:{REPO_ROOT / 'mlruns'}")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", tracking.get("experiment_name", "preventive-health-sft"))
    # mlflow 3.x rejects the local file store unless opted in; keep the simple
    # ./mlruns workflow (readable by `mlflow ui`) instead of forcing a DB backend.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    print(f"[tracking] mlflow -> {os.environ['MLFLOW_TRACKING_URI']}")
    return ["mlflow"]


def train(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config)

    model_cfg = cfg.get("model", {})
    quant_cfg = cfg.get("quantization", {})
    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg.get("training", {})
    tracking_cfg = cfg.get("tracking", {})

    # --- resolve model + run shape (smoke overrides) ---
    if args.smoke:
        model_id = SMOKE_MODEL_ID
        output_dir = REPO_ROOT / SMOKE_OUTPUT_DIR
        max_steps = args.max_steps if args.max_steps is not None else 5
        limit = args.limit if args.limit is not None else 8
        max_length = min(int(train_cfg.get("max_seq_length", 2048)), 512)
        print(f"[smoke] overriding model_id -> {model_id} (ungated), max_steps={max_steps}")
    else:
        model_id = args.model or model_cfg.get("model_id", "")
        _out = args.output_dir or train_cfg.get("output_dir", "adapters/sft")
        output_dir = (REPO_ROOT / _out).resolve()
        max_steps = args.max_steps if args.max_steps is not None else -1
        limit = args.limit
        max_length = int(train_cfg.get("max_seq_length", 2048))
        if not weights_available_locally(model_id):
            print(_gated_model_message(model_id), file=sys.stderr)
            return 2

    train_path, eval_path = resolve_data_paths(cfg, args)
    report_to = setup_tracking(cfg, args)

    # --- heavy imports (kept out of module scope so --help stays fast/GPU-free) ---
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # noqa: BLE001
        print(
            f"Training stack import failed: {exc}\n"
            "  Install it first: pip install -r requirements.txt (torch per pytorch.org).",
            file=sys.stderr,
        )
        return 2

    if not torch.cuda.is_available():
        print(
            "CUDA is not available — QLoRA training needs a GPU.\n"
            "  This project targets the local Blackwell GPU; see reports/environment_verified.md.",
            file=sys.stderr,
        )
        return 2

    compute_dtype = _dtype_from_name(quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"), torch)

    # --- 4-bit quantization config (identical path to the passing smoke test) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bool(quant_cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(quant_cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print(f"\nloading base model {model_id} in 4-bit (nf4) on cuda …")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=bool(model_cfg.get("trust_remote_code", False))
    )
    if tokenizer.pad_token is None:
        # LoRA SFT needs a pad token; fall back to EOS (standard for decoder-only LMs).
        tokenizer.pad_token = tokenizer.eos_token

    # Gemma 3 / MedGemma ship as multimodal Gemma3ForConditionalGeneration. Load
    # with that NATIVE class (Gemma3ForCausalLM leaves 445 text keys randomly
    # initialised — a silent-garbage trap, verified). We feed text only, so the
    # vision tower is loaded (small in 4-bit, ~+0.3 GiB) but never in the forward
    # path. Other bases (e.g. the Qwen smoke model) use Auto* unchanged.
    _cfg = AutoConfig.from_pretrained(
        model_id, trust_remote_code=bool(model_cfg.get("trust_remote_code", False))
    )
    is_gemma3 = getattr(_cfg, "model_type", "") == "gemma3"
    _load_kwargs = dict(
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=compute_dtype,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        attn_implementation=model_cfg.get("attn_implementation", "eager"),
    )
    if is_gemma3:
        from transformers import Gemma3ForConditionalGeneration
        print("  (gemma3 detected -> Gemma3ForConditionalGeneration, text-only)")
        model = Gemma3ForConditionalGeneration.from_pretrained(model_id, **_load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **_load_kwargs)
    model.config.use_cache = False  # required with gradient checkpointing

    # --- prepare for k-bit training + attach LoRA ---
    use_gc = bool(train_cfg.get("gradient_checkpointing", True))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gc)

    _target_names = list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]))
    if is_gemma3:
        # The same proj names also exist in the vision tower; scope LoRA to the
        # language model only (a str target_modules is treated as a full regex).
        target_modules = r".*language_model.*(" + "|".join(_target_names) + r")"
    else:
        target_modules = _target_names
    lora_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- datasets ---
    from preventive_health_model_lab.training.formatting import make_formatting_func

    train_ds = load_dataset("json", data_files=str(train_path), split="train")
    if limit:
        train_ds = train_ds.select(range(min(limit, len(train_ds))))
    eval_ds = None
    if eval_path is not None:
        eval_ds = load_dataset("json", data_files=str(eval_path), split="train")
        if args.smoke:
            eval_ds = eval_ds.select(range(min(4, len(eval_ds))))
    print(f"train rows: {len(train_ds)}  |  eval rows: {len(eval_ds) if eval_ds is not None else 0}")

    formatting_func = make_formatting_func(eos_token=tokenizer.eos_token or "")

    # --- trainer config ---
    do_eval = eval_ds is not None and not args.smoke
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        max_length=max_length,
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=1 if args.smoke else int(train_cfg.get("gradient_accumulation_steps", 16)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        max_steps=max_steps,
        gradient_checkpointing=use_gc,
        bf16=bool(train_cfg.get("bf16", True)),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        logging_steps=1 if args.smoke else int(train_cfg.get("logging_steps", 10)),
        save_strategy="no" if args.smoke else train_cfg.get("save_strategy", "epoch"),
        eval_strategy="epoch" if do_eval else "no",
        seed=int(train_cfg.get("seed", 42)),
        report_to=report_to,
        run_name=tracking_cfg.get("run_name"),
        # formatting_func writes to the default "text" column; leave loss on the
        # full sequence (language-modeling style) — appropriate for these short,
        # structured targets.
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds if do_eval else None,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    print("\n=== starting training ===")
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))       # saves the LoRA adapter for a PeftModel
    tokenizer.save_pretrained(str(output_dir))

    peak = torch.cuda.max_memory_allocated(0) / 1024**3
    print(f"\nadapter saved to: {output_dir}")
    print(f"peak VRAM        : {peak:.2f} GiB")
    saved = sorted(p.name for p in output_dir.iterdir())
    print(f"adapter dir contents: {saved}")
    print("\nRESULT: training completed and adapter written.")
    return 0


def main() -> int:
    args = build_argparser().parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
