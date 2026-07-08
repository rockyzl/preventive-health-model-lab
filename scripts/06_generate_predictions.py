#!/usr/bin/env python3
"""Generate model predictions on a test set, for scoring by scripts/05_evaluate.py.

Loads a base model (optionally with a trained LoRA adapter on top), runs it over
each record of a JSONL test file using the SAME prompt the trainer used
(``training.formatting.build_prompt``), and writes prediction records:

    {"input": <same timeline text>, "prediction": <generated text>, "patient_id": ...}

which feed straight into ``scripts/05_evaluate.py --predictions`` / ``--compare``.

Usage (base vs fine-tuned for one model):
    python scripts/06_generate_predictions.py --out preds/medgemma_base.jsonl
    python scripts/06_generate_predictions.py --adapter adapters/medgemma-... \
        --out preds/medgemma_tuned.jsonl
    python scripts/06_generate_predictions.py --model google/gemma-3-4b-it \
        --adapter adapters/gemma3-... --out preds/gemma3_tuned.jsonl

No silent gated download: if the base weights aren't cached, it exits with
guidance (same guard as scripts/04).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from preventive_health_model_lab.utils.config import load_yaml  # noqa: E402
from preventive_health_model_lab.training.formatting import build_prompt  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "eval.yaml"
DEFAULT_TEST = REPO_ROOT / "data" / "processed" / "test.jsonl"


def _hf_cached(model_id: str) -> bool:
    """True if the model looks present in the local HF cache (no network)."""
    from huggingface_hub.constants import HF_HUB_CACHE

    slug = "models--" + model_id.replace("/", "--")
    return (Path(HF_HUB_CACHE) / slug).exists()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--model", default=None, help="base model id (default: config model_id)")
    ap.add_argument("--adapter", default=None, help="path to a trained LoRA adapter (optional)")
    ap.add_argument("--test-file", default=str(DEFAULT_TEST))
    ap.add_argument("--out", required=True, help="output predictions JSONL")
    ap.add_argument("--max-new-tokens", type=int, default=1100)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    model_id = args.model or cfg.get("model", {}).get("model_id", "")
    if not model_id:
        print("No model_id (pass --model or set model.model_id in the config).", file=sys.stderr)
        return 2
    if not _hf_cached(model_id):
        print(f"Base weights for '{model_id}' are not in the local HF cache.\n"
              "  Accept the license on huggingface.co, `hf auth login`, then let it\n"
              "  download — this script never pulls a gated model silently.", file=sys.stderr)
        return 2
    if args.adapter and not (Path(args.adapter) / "adapter_config.json").exists():
        print(f"--adapter '{args.adapter}' has no adapter_config.json", file=sys.stderr)
        return 2

    records = []
    with open(args.test_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit:
        records = records[: args.limit]
    if not records:
        print(f"No records in {args.test_file}", file=sys.stderr)
        return 2

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"loading {model_id} in 4-bit"
          + (f" + adapter {args.adapter}" if args.adapter else " (base only)") + " …")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    _cfg = AutoConfig.from_pretrained(model_id)
    load_kwargs = dict(quantization_config=bnb, device_map={"": 0},
                       dtype=torch.bfloat16, attn_implementation="eager")
    if getattr(_cfg, "model_type", "") == "gemma3":
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as out_fh:
        for rec in records:
            prompt = build_prompt(rec)
            enc = tok(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tok.pad_token_id,
                )
            text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out_fh.write(json.dumps({
                "input": rec.get("input", ""),
                "prediction": text.strip(),
                "patient_id": rec.get("patient_id", ""),
            }, ensure_ascii=False) + "\n")
            n += 1
            print(f"  [{n}/{len(records)}] {rec.get('patient_id','?')}  ({len(text)} chars)")

    print(f"\nwrote {n} predictions -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
