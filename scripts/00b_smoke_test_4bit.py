#!/usr/bin/env python3
"""4-bit (QLoRA-path) smoke test for Blackwell / sm_120.

This is the single most important de-risking step before any training:
it proves that ``bitsandbytes`` can build a working 4-bit (nf4) kernel on
this specific GPU and run a real forward + generate pass. If this FAILS
with a "no kernel image is available for execution on the device" (the
classic sm_120 symptom), the local-QLoRA plan needs a bitsandbytes source
build or the Unsloth prebuilt stack (see requirements.txt / project_design.md).

By design it loads a SMALL, UNGATED, Apache-2.0 model (Qwen2.5-0.5B-Instruct)
— NOT MedGemma — so it needs no license acceptance and downloads ~1 GB only.
The kernel path it exercises is identical to the one a 4B QLoRA run uses.

Usage:
    python scripts/00b_smoke_test_4bit.py               # default tiny model
    python scripts/00b_smoke_test_4bit.py --model <hf_id>
    python scripts/00b_smoke_test_4bit.py --no-generate # load-only (faster)

Exit codes: 0 = PASS, 2 = a dependency/kernel problem (actionable), 1 = other.
"""
from __future__ import annotations

import argparse
import sys
import textwrap

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # ungated, Apache-2.0, ~1 GB


def _fail(msg: str, code: int = 2) -> None:
    print("\nRESULT: FAIL")
    print(textwrap.indent(msg.strip(), "  "))
    sys.exit(code)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"HF model id (default: {DEFAULT_MODEL}; must be ungated)")
    ap.add_argument("--no-generate", action="store_true",
                    help="only load in 4-bit, skip the generate pass")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    args = ap.parse_args()

    # --- torch / device diagnostics ---
    try:
        import torch
    except ImportError:
        _fail("torch is not installed. Install the GPU stack first:\n"
              "  pip install torch --index-url https://download.pytorch.org/whl/cu128")

    print(f"torch                 : {torch.__version__}")
    print(f"CUDA available        : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        _fail("torch.cuda.is_available() is False — no usable CUDA device.\n"
              "Check the driver and that the cu128 wheel is installed.")
    cap = torch.cuda.get_device_capability(0)
    print(f"device                : {torch.cuda.get_device_name(0)}")
    print(f"compute capability    : sm_{cap[0]}{cap[1]}")
    print(f"bf16 supported        : {torch.cuda.is_bf16_supported()}")

    # --- bitsandbytes presence ---
    try:
        import bitsandbytes as bnb  # noqa: F401
        print(f"bitsandbytes          : {bnb.__version__}")
    except ImportError:
        _fail("bitsandbytes is not installed.\n"
              "  pip install 'bitsandbytes>=0.48'")

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"\nloading {args.model} in 4-bit (nf4) on cuda …")
    try:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant, device_map="cuda",
            dtype=torch.bfloat16,
        )
    except Exception as exc:  # noqa: BLE001
        text = f"{type(exc).__name__}: {exc}"
        if "no kernel image" in text.lower() or "sm_120" in text.lower():
            _fail("4-bit kernel is NOT available for this GPU (sm_120 / Blackwell).\n"
                  f"Underlying error: {text}\n\n"
                  "Fix options:\n"
                  "  1. Install a cu128 bitsandbytes build: pip install -U 'bitsandbytes>=0.48'\n"
                  "  2. Or use the Unsloth prebuilt stack (bundles a matching bnb+torch)\n"
                  "  3. Or build bitsandbytes from source with CUDA 12.8 for sm_120")
        _fail(f"4-bit load failed:\n{text}")

    used = torch.cuda.max_memory_allocated(0) / 1024**3
    print(f"4-bit load OK         : peak VRAM {used:.2f} GiB")

    if not args.no_generate:
        prompt = "In one sentence, what is preventive medicine?"
        msgs = [{"role": "user", "content": prompt}]
        try:
            # transformers 5.x: apply_chat_template(return_tensors=...) yields a
            # BatchEncoding, so render to text then tokenize for a stable path.
            text = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
            enc = tok(text, return_tensors="pt").to("cuda")
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
            gen = tok.decode(out[0][enc["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            print(f"generate OK           : {gen.strip()[:120]!r}")
        except Exception as exc:  # noqa: BLE001
            _fail(f"4-bit load succeeded but generate failed:\n{type(exc).__name__}: {exc}")
        peak = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"peak VRAM (with gen)  : {peak:.2f} GiB")

    print("\nRESULT: PASS — bitsandbytes 4-bit works on this Blackwell GPU.")
    print("The QLoRA path is viable locally. Safe to proceed to a real 4B run.")
    sys.exit(0)


if __name__ == "__main__":
    main()
