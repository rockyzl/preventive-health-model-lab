# Environment verified — GPU stack + 4-bit smoke test

> Date: 2026-07-07. This records the actual working software stack and the
> result of the Blackwell/sm_120 4-bit smoke test — the #1 de-risking step
> before any QLoRA training. Reproduce with `scripts/00b_smoke_test_4bit.py`.

## Verified working stack (installed in `.venv`)

| Package | Version |
|---|---|
| torch | **2.11.0+cu128** |
| transformers | 5.13.0 |
| bitsandbytes | **0.49.2** |
| peft | 0.19.1 |
| trl | 1.7.1 |
| accelerate | 1.14.0 |
| datasets | 5.0.0 |
| sentencepiece | 0.2.1 |

Install path (Blackwell/sm_120): `torch` from the cu128 index
(`--index-url https://download.pytorch.org/whl/cu128`), the rest from PyPI.

## Smoke test result — PASS

```
torch                 : 2.11.0+cu128
CUDA available        : True
device                : NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU
compute capability    : sm_120
bf16 supported        : True
bitsandbytes          : 0.49.2
loading Qwen/Qwen2.5-0.5B-Instruct in 4-bit (nf4) on cuda …
4-bit load OK         : peak VRAM 0.43 GiB
generate OK           : 'Preventive medicine aims to prevent disease and health problems before they occur ...'
peak VRAM (with gen)  : 0.44 GiB
RESULT: PASS — bitsandbytes 4-bit works on this Blackwell GPU.
```

**Interpretation:** bitsandbytes builds a working nf4 kernel on this sm_120
laptop GPU under WSL2 — no "no kernel image for sm_120" failure, no source
build or Unsloth fallback needed. The local QLoRA path for a 4B model is
viable. The 0.5B smoke model used 0.44 GiB; a 4-bit 4B base is estimated at
~2.5–3 GB weights + LoRA/activations/optimizer, well within the measured
8151 MiB (≈8 GB) with grad-checkpointing + batch 1.

Model used is `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0, ungated) purely to
exercise the kernel path with no license gate — NOT a project model.

## Two non-blocking warnings observed (cosmetic)

- `transformers`: `torch_dtype` deprecated → use `dtype` (fixed in the script).
- `bitsandbytes`: a `FutureWarning` about `_check_is_size` from a newer torch
  internal — harmless, upstream will update.
