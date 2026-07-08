# Preventive Health Model Lab

A small, reproducible **fine-tuning experiment**: does medical continued-pretraining
help a 4B LLM reason about a patient's health **over time** — surfacing preventive
risk signals (a slowly rising HbA1c, a blood-pressure creep against family history)
in a structured, **non-diagnostic** form — after identical QLoRA fine-tuning? Runs
end-to-end on one **8 GB laptop GPU**.

> **Research / education only. Synthetic data only. Not a diagnostic product, not
> clinical decision support, not for real patient data or PHI.**

## The research question

MedGemma 1.5 4B is Gemma 3 4B **plus** medical continued-pretraining. So fine-tune
**both** with the *same* QLoRA recipe on the *same* synthetic data and ask: does the
medical base actually reason better about preventive longitudinal signals? Same base
family ⇒ the only variable is the medical pretraining. That is what makes this a
controlled experiment rather than "make a model."

## Headline result (honest)

On a held-out **synthetic** test set (n=6), scored by **automatic** metrics:

| condition | overall | safety hard-fail | hallucinated #s |
|---|---:|---:|---:|
| Gemma 3 base | 0.612 | 100 % | 2 |
| **Gemma 3 QLoRA** | **1.000** | 0 % | 0 |
| MedGemma base | 0.614 | 100 % | 7 |
| MedGemma QLoRA | 0.997 | 0 % | 2 |

- **QLoRA worked, dramatically, for both** — the biggest, most consistent effect was
  turning models that routinely broke the safety contract (no disclaimer, diagnostic
  language) into ones that reliably follow the 7-part, non-diagnostic, faithful
  format.
- **Medical pretraining showed *no measurable advantage*** here: the non-medical
  control matched (marginally beat) MedGemma. With n=6, automatic metrics, and
  synthetic data, the honest conclusion is "no evidence of an advantage on this
  benchmark," **not** "the medical model is better." Full analysis + caveats:
  [`reports/final_experiment_report.md`](reports/final_experiment_report.md).

The value of this project is that it reads like a **real experiment with an honest,
slightly inconvenient result** — not an over-packaged demo.

## Why synthetic data

No public dataset is simultaneously longitudinal + preventive + instruction-formatted
+ license-clean, and real EHR data (MIMIC, EHRSHOT) is credential-gated and legally
can't be sent through third-party services. So v1 uses a seeded Python generator: 60
synthetic patients across 5 trajectory archetypes, with each patient's **gold answer
derived from the same emitted numbers** (correct-by-construction, no hallucinated
values) and all biomarkers clamped to the **preventive/borderline** range so the
right answer is always "monitor + refer," never a diagnosis. Real-data validation is
explicitly future work. See [`hf_release/dataset_card.md`](hf_release/dataset_card.md).

## Method

- **QLoRA**: base loaded in 4-bit (nf4); train a ~30M-param LoRA adapter (<1% of 4B,
  ~57 MB) on the language-model projections; base weights frozen. This is what makes
  a 4B fine-tune fit in 8 GB.
- **Stack**: PyTorch 2.11 (cu128, Blackwell/sm_120), Transformers 5, PEFT, TRL,
  bitsandbytes 0.49. Config: [`configs/sft_lora_medical.yaml`](configs/sft_lora_medical.yaml).
- **Recipe (identical for both models)**: r=16/α=32, 5 epochs, effective batch 8,
  lr 2e-4 cosine, seq≤2048.

## Evaluation

Automatic, per-output metrics (no clinician in the loop): **schema conformance**
(all 7 sections), **safety disclaimer present**, **non-diagnostic** (a red-flag
scanner; diagnostic phrasing is a hard fail), and **numeric grounding** (every
clinical number cited must appear in the input — the anti-hallucination check). Gold
answers score a perfect 1.000 as the ceiling. These measure **form + faithfulness,
not clinical correctness.**

## Pipeline (reproduce)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # + torch per pytorch.org (cu128 for Blackwell)

python scripts/00_check_environment.py                 # env + GPU
python scripts/00b_smoke_test_4bit.py                  # prove 4-bit works on this GPU
python scripts/01_download_or_prepare_data.py --generate --n 60 --seed 42 --out data/synthetic/timelines.jsonl
python scripts/02_build_instruction_dataset.py --build # -> data/processed/{train,val,test}.jsonl (patient split)
# accept the gated MedGemma + Gemma-3 licenses on huggingface.co, then `hf auth login`
python scripts/04_train_sft.py --no-track --no-eval                                   # MedGemma
python scripts/04_train_sft.py --model google/gemma-3-4b-it --output-dir adapters/gemma3-4b-preventive-sft-v0 --no-track --no-eval  # control
python scripts/06_generate_predictions.py ...          # 4 conditions -> outputs/predictions/
python scripts/07_build_comparison.py                  # scores + comparison + demo artifacts
python -m pytest -q                                    # 47 tests
```

(`--no-track --no-eval` + `TOKENIZERS_PARALLELISM=false` avoid a WSL2 trainer-start
deadlock; see the report/commits.)

## Repo layout

```
configs/    sft_lora_medical.yaml, eval.yaml
scripts/    00 env · 00b 4-bit smoke · 01 generate · 02 build · 03 baseline
            04 train · 05 evaluate · 06 predictions · 07 comparison
src/preventive_health_model_lab/  data/ (schema, generator, rendering)
            safety/ training/ evaluation/ inference/ utils/
outputs/    predictions/  evaluation/   (scores, comparison CSVs, failure cases)
demo_artifacts/  precomputed, synthetic-only, public-demo-safe bundle + safety banner
hf_release/ draft adapter model cards, dataset card, Space README (adapter-only)
reports/    final_experiment_report.md, final_handoff_summary.md, phase2_status.md, ...
tests/      47 tests (data, safety gate, training smoke, evaluation)
```

## The 7-part output schema

1. Longitudinal summary · 2. Risk signals · 3. Evidence (exact data points) ·
4. Missing information · 5. Clinician questions · 6. Safety disclaimer ·
7. What NOT to conclude. Canonical list in `src/.../data/schema.py::OUTPUT_SECTIONS`;
hand-written gold example in `examples/sample_model_output.md`.

## Safety boundaries (structural, not optional)

No real PHI ever (the schema hard-fails any record not marked `synthetic: true`); the
model must not diagnose; every output carries the disclaimer + "what not to conclude";
not a medical device. Any public demo is **precomputed and read-only** — it must
never accept user health data or run live inference on user text (see
[`demo_artifacts/safety_disclaimer.md`](demo_artifacts/safety_disclaimer.md)).

## Limitations (read before believing the numbers)

Tiny test set (n=6 ⇒ rank gaps are noise); automatic metrics measure form +
faithfulness, **not** clinical correctness; synthetic + template-derived gold ⇒ high
scores partly reflect format-matching; base MedGemma outputs partly degenerated and
hit the token cap; test split lacks one archetype; **no clinical validation**. Real
clinical data + clinician review are required to say anything about real-world use.

## What's demo-safe vs not

- **Safe to show publicly:** everything in `demo_artifacts/` (synthetic timelines,
  the 4 model outputs, scores, honest failures), the report, the code.
- **Not for release without a separate license/legal review:** merged base-model
  weights (ship **adapter-only**); any real patient data (there is none here).

## License

Code: MIT (`LICENSE`). Adapters are derivative of their gated base models
(MedGemma / Gemma terms) — accept those separately; confirm redistribution terms
before publishing adapters.
