#!/usr/bin/env python3
"""Run a baseline inference pass over synthetic patient timelines.

SKELETON (Phase 1), config-driven by ``configs/eval.yaml``.

Guard rails, by design:
  - NO silent network downloads. If the model weights are not already
    present locally, the script prints a clear, actionable message and
    exits non-zero (it never triggers a multi-GB pull on its own).
  - ``--dry-run`` resolves and prints the full plan (model, generation
    params, prompt files, safety flags) and exits 0 without loading any
    model. This is what the Phase 1 tests exercise.

Usage:
    python scripts/03_run_baseline.py --dry-run
    python scripts/03_run_baseline.py --config configs/eval.yaml --dry-run
    python scripts/03_run_baseline.py            # real run (Phase 2+, needs weights)
    python scripts/03_run_baseline.py --adapter adapters/smoke --dry-run  # eval a LoRA adapter

The optional ``--adapter <path>`` loads a trained LoRA adapter (peft) on top of
the base model, so the baseline can compare base vs fine-tuned. It overrides
``model.adapter_path`` from the config. ``--dry-run`` validates the adapter path
without loading anything or touching the network.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from preventive_health_model_lab.utils.config import load_yaml  # noqa: E402
from preventive_health_model_lab.data.schema import OUTPUT_SECTIONS  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "eval.yaml"


def resolve_prompt_files(cfg: dict) -> list[Path]:
    prompts = cfg.get("prompts", {})
    base = (REPO_ROOT / prompts.get("path", "examples/")).resolve()
    glob = prompts.get("glob", "*.json")
    return sorted(base.glob(glob))


def resolve_adapter_path(cfg: dict, cli_adapter: str | None) -> str | None:
    """CLI ``--adapter`` overrides ``model.adapter_path``; None means base-only."""
    if cli_adapter:
        return cli_adapter
    return cfg.get("model", {}).get("adapter_path")


def validate_adapter_dir(adapter_path: str | None) -> str:
    """Return a human-readable validation status for an adapter path.

    A valid LoRA adapter directory contains ``adapter_config.json``. Pure
    filesystem check — no torch/peft import, no network.
    """
    if not adapter_path:
        return "none (base model only)"
    p = Path(adapter_path).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    if not p.exists():
        return f"MISSING — path does not exist: {p}"
    if not (p / "adapter_config.json").exists():
        return f"INVALID — no adapter_config.json in: {p}"
    return f"OK — {p}"


def print_plan(cfg: dict, prompt_files: list[Path], adapter_path: str | None) -> None:
    model = cfg.get("model", {})
    gen = cfg.get("generation", {})
    safety = cfg.get("safety", {})
    print("=== Baseline run plan (resolved) ===")
    print(f"  model_id        : {model.get('model_id')}")
    print(f"  adapter_path    : {adapter_path}")
    print(f"  adapter status  : {validate_adapter_dir(adapter_path)}")
    print(f"  max_new_tokens  : {gen.get('max_new_tokens')}")
    print(f"  temperature     : {gen.get('temperature')}")
    print(f"  top_p           : {gen.get('top_p')}")
    print(f"  prompt files    : {len(prompt_files)} found")
    for f in prompt_files:
        print(f"      - {f.relative_to(REPO_ROOT)}")
    print(f"  require_disclaimer   : {safety.get('require_disclaimer')}")
    print(f"  forbid_diagnostic    : {safety.get('forbid_diagnostic_language')}")
    print(f"  target output schema : {', '.join(OUTPUT_SECTIONS)}")


def weights_available_locally(model_id: str) -> bool:
    """Best-effort check for a locally-cached model, WITHOUT importing torch.

    Looks in the HF cache dir for a matching snapshot. Returns False if we
    can't confirm — the caller then refuses to proceed (no silent pull).
    """
    if not model_id:
        return False
    # A local path that exists is trivially available.
    if Path(model_id).expanduser().exists():
        return True
    # HF hub cache: ~/.cache/huggingface/hub/models--<org>--<name>
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    slug = "models--" + model_id.replace("/", "--")
    return (cache / slug).exists()


def real_run(cfg: dict, prompt_files: list[Path], adapter_path: str | None) -> int:
    model_id = cfg.get("model", {}).get("model_id", "")

    if not weights_available_locally(model_id):
        print(
            "\nModel weights are not available locally, and this script will NOT\n"
            "download them automatically.\n\n"
            f"  model_id: {model_id}\n\n"
            "To proceed (Phase 2+):\n"
            "  1. Accept the model license on Hugging Face (MedGemma is gated).\n"
            "  2. Authenticate:      hf auth login\n"
            "  3. Pre-download once: hf download "
            f"{model_id}\n"
            "  4. Re-run this script (or use --dry-run to preview the plan).\n",
            file=sys.stderr,
        )
        return 2

    # If an adapter was requested, it must be a real LoRA dir before we bother
    # loading a multi-GB base model. Refuse loudly on a bad path.
    if adapter_path:
        status = validate_adapter_dir(adapter_path)
        if not status.startswith("OK"):
            print(f"\nRequested adapter is not usable: {status}\n", file=sys.stderr)
            return 2

    if importlib.util.find_spec("transformers") is None:
        print(
            "transformers is not installed. Install the training stack first:\n"
            "  pip install -r requirements.txt   (and torch per pytorch.org)",
            file=sys.stderr,
        )
        return 2

    # Load base 4-bit, then (optionally) stack the LoRA adapter with peft. This
    # is the base-vs-fine-tuned comparison path.
    model, tokenizer = load_base_and_adapter(cfg, adapter_path)  # noqa: F841

    # The generation loop turns each patient TIMELINE into an instruction prompt
    # (via training.formatting.build_prompt) and calls model.generate. Building
    # that prompt from a raw timeline needs the Phase-2 serializer
    # (scripts/02 timeline_to_instruction_record), so the loop lands with it.
    raise NotImplementedError(
        "Base model + adapter loaded successfully, but the timeline->prompt "
        "generation loop depends on the Phase-2 dataset serializer "
        "(scripts/02_build_instruction_dataset.timeline_to_instruction_record). "
        "Use --dry-run to preview the plan, or train/evaluate on already-built "
        "instruction records."
    )


def load_base_and_adapter(cfg: dict, adapter_path: str | None):
    """Load the 4-bit base model and, if given, stack a LoRA adapter (peft).

    Uses the exact 4-bit (nf4/double-quant/bf16) path proven by the smoke test.
    Returns ``(model, tokenizer)``. Imports torch/transformers/peft lazily so
    ``--dry-run`` never pays for them.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg = cfg.get("model", {})
    model_id = model_cfg.get("model_id", "")
    trust = bool(model_cfg.get("trust_remote_code", False))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
        trust_remote_code=trust,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to eval config YAML")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved plan and exit 0")
    parser.add_argument(
        "--adapter",
        default=None,
        help="path to a trained LoRA adapter dir to stack on the base (overrides model.adapter_path)",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    prompt_files = resolve_prompt_files(cfg)
    adapter_path = resolve_adapter_path(cfg, args.adapter)

    if args.dry_run:
        print_plan(cfg, prompt_files, adapter_path)
        print("\n[dry-run] No model loaded, no network access. Exiting 0.")
        return 0

    return real_run(cfg, prompt_files, adapter_path)


if __name__ == "__main__":
    raise SystemExit(main())
