# Phase 2 status — pipeline built & proven end-to-end (synthetic)

> Date: 2026-07-07. Everything below runs and is verified on THIS machine.
> The one remaining gate is external: the real MedGemma/Gemma runs need the
> user to accept those models' licenses on Hugging Face (gated). No real
> patient data is used anywhere; all training data is clearly-labeled synthetic.

## What is done in Phase 2

1. **GPU stack installed & the #1 risk cleared.** torch 2.11.0+cu128,
   transformers 5.13, trl 1.7.1, peft 0.19.1, accelerate 1.14, bitsandbytes
   0.49.2, datasets 5.0, mlflow 3.14. The 4-bit (nf4) Blackwell/sm_120 smoke
   test PASSES — `scripts/00b_smoke_test_4bit.py`, evidence in
   `reports/environment_verified.md`. bitsandbytes on this laptop GPU is no
   longer a risk.

2. **v1 synthetic dataset — correct-by-construction, non-diagnostic, leak-free.**
   `src/.../data/synthetic_generator.py` + `scripts/01 --generate` +
   `scripts/02 --build`. 60 synthetic patients across 5 trajectory archetypes
   (cardiometabolic-drift, stable-healthy, improving-after-intervention,
   isolated-hypertension, lipid-plus-family-risk), biomarkers clamped to the
   preventive/borderline zone so the correct answer is always "monitor + refer",
   never a diagnosis. Each patient's gold 7-part output is a **pure function of
   the emitted timeline** (no train/serve skew); the archetype label is stripped
   from model input. Patient-level split **48/6/6 with zero overlap**.
   Interface: JSONL `{instruction, input, output, patient_id, synthetic:true}`.

3. **QLoRA training pipeline.** `scripts/04_train_sft.py` (config-driven, nf4 +
   LoRA r16/alpha32/7 target-modules via `configs/sft_lora_medical.yaml`),
   `src/.../training/formatting.py` (single source of prompt formatting shared
   by training and baseline inference), `scripts/03 --adapter` (load a trained
   adapter for base-vs-fine-tuned comparison). mlflow tracking wired (local
   ./mlruns) for the real run.

4. **End-to-end proof on REAL synthetic data.** `04 --smoke` on the actual
   `data/processed/{train,val}.jsonl` (Qwen2.5-0.5B, ungated): loss fell
   **2.10 → 1.03** over 8 steps, mean-token-accuracy 0.62 → 0.77, adapter
   written, peak VRAM 1.47 GiB. The chain works and the data is learnable.

5. **QA gate passed** — an independent reviewer re-ran everything: 26 tests
   green; 0 leakage (archetype/label never in model input); every numeric value
   cited in a gold output verified present in that record's timeline (0 invented
   numbers); 0 diagnostic red-flags; disclaimer verbatim in every output; split
   integrity 0 overlap; git hygiene clean (bulk JSONL gitignored).

## The one gate before real training (needs the user)

The chosen models are gated on Hugging Face. To run the real experiment:

```bash
# 1. On huggingface.co, accept the license for:
#    google/medgemma-1.5-4b-it   (primary)   and   google/gemma-3-4b-it (control)
# 2. Authenticate locally:
hf auth login
# 3. Real QLoRA run (NO --smoke → uses the config model on the full dataset):
.venv/bin/python scripts/04_train_sft.py                     # MedGemma primary
#    (the script refuses to silently download a gated model; it prints these
#     exact steps if the weights aren't cached)
```

Everything up to this line was done without the user's credentials and is fully
reversible. Training the actual 4B models is the next step, on the user's go.

## Known minor / deferred (non-blocking)

- `synthetic_generator.py` has a small dead `_REF` dict (band language is
  hardcoded in headlines) — cosmetic cleanup.
- `configs/sft_lora_medical.yaml` uses `warmup_ratio`; transformers suggests
  `warmup_steps` — still works, deprecation only.
- Wording diversity of gold outputs is template-ish (acceptable tradeoff for a
  correct-by-construction v1; add paraphrase variety in v2).
- Dataset covers the preventive/borderline zone only — no frank-disease values
  (HbA1c ≥ 6.5, stage-2 BP) by design; note this when interpreting eval.
- Synthea (real-ish longitudinal generator) deferred to v2; needs Java, not
  installed. Pure-Python generator chosen for v1 to keep timeline↔gold coupling.
