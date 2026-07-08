# Model card (draft) — Gemma 3 4B · Preventive-Health QLoRA adapter (v1, control)

**Draft — do not upload until explicitly asked. Adapter-only release; never ship
merged base weights without a separate license/legal review.**

## What this is

A **LoRA/QLoRA adapter** (not a standalone model) fine-tuned on top of
`google/gemma-3-4b-it`. It is the **experimental control**: the same base family as
MedGemma but **without** medical continued-pretraining, fine-tuned with an identical
recipe on identical synthetic data, to isolate the effect of medical pretraining.
~30M trainable params (<1% of the 4B base); ~57 MB adapter.

Requires separate access to the gated base `google/gemma-3-4b-it`; load the adapter
on top with PEFT.

## Intended use

Research and education only (the control arm of a fine-tuning experiment). **Not**
for diagnosis, treatment, triage, prognosis, or clinical decision support; **not**
for real patient data / PHI.

## Training

- Data: 60 **synthetic** patients, patient-level split 48/6/6, zero overlap
  (dataset card). No real data, no PHI.
- QLoRA: 4-bit nf4 base, LoRA r=16/α=32 on language-model projections, 5 epochs,
  effective batch 8, lr 2e-4 cosine, seq≤2048, single 8 GB GPU.
- Final training loss ≈ 0.157, token accuracy ≈ 94 %.

## Evaluation (held-out synthetic test, n=6, automatic metrics)

| metric | base | this adapter |
|---|---:|---:|
| overall (mean of 4) | 0.612 | **1.000** |
| safety disclaimer present | 0.000 | 1.000 |
| non-diagnostic | 0.500 | 1.000 |
| numeric grounding | 0.950 | 1.000 |
| 7-section schema | 1.000 | 1.000 |
| hard-fail rate | 100 % | 0 % |
| hallucinated numbers (total) | 2 | 0 |

**Honest headline:** on this synthetic benchmark, this non-medical control
**matched or marginally beat** the medical MedGemma adapter (0 hallucinated numbers
here vs 2 there). Interpret with care — n=6, automatic metrics, synthetic data. See
`reports/final_experiment_report.md`.

## Limitations

Same as the MedGemma adapter: synthetic-data bias, template-derived gold, automatic
metrics only, no clinical validation, tiny test set, narrow lab/vital scope, possible
hallucination.

## License

Adapter weights are derivative of `google/gemma-3-4b-it` and bound by the Gemma
terms. Confirm the exact license and redistribution allowance before any release.
