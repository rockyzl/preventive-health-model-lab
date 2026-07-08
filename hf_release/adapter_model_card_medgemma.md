# Model card (draft) — MedGemma 1.5 4B · Preventive-Health QLoRA adapter (v1)

**Draft — do not upload until explicitly asked. Adapter-only release; never ship
merged base weights without a separate license/legal review.**

## What this is

A **LoRA/QLoRA adapter** (not a standalone model) fine-tuned on top of
`google/medgemma-1.5-4b-it` for structured, **non-diagnostic** preventive-health
reasoning over a synthetic longitudinal patient record. ~30M trainable params
(<1% of the 4B base); ~57 MB adapter.

To use it you must **separately obtain access to the gated base model**
`google/medgemma-1.5-4b-it` (accept its Health AI Developer Foundations terms on
Hugging Face) and load the adapter on top with PEFT.

## Intended use

Research and education only: studying whether medical continued-pretraining helps
preventive longitudinal reasoning, and demonstrating safety-aware fine-tuning.
**Not** for diagnosis, treatment, triage, prognosis, or clinical decision support,
and **not** for real patient data / PHI.

## Training

- Data: 60 **synthetic** patients (no real data, no PHI), patient-level split
  train 48 / val 6 / test 6, zero overlap. See the dataset card.
- QLoRA: 4-bit nf4 base, LoRA r=16/α=32 on language-model projections, 5 epochs,
  effective batch 8, lr 2e-4 cosine, seq≤2048, on a single 8 GB GPU.
- Final training loss ≈ 0.155, token accuracy ≈ 94 %.

## Evaluation (held-out synthetic test, n=6, automatic metrics)

| metric | base | this adapter |
|---|---:|---:|
| overall (mean of 4) | 0.614 | **0.997** |
| safety disclaimer present | 0.000 | 1.000 |
| non-diagnostic | 0.667 | 1.000 |
| numeric grounding | 0.954 | 0.987 |
| 7-section schema | 0.833 | 1.000 |
| hard-fail rate | 100 % | 0 % |
| hallucinated numbers (total) | 7 | 2 |

Fine-tuning greatly improved safety-framing compliance and reduced hallucinated
numbers. **Finding:** after identical fine-tuning, this medical adapter and the
non-medical control (Gemma 3 4B) were indistinguishable (both near-ceiling; gap within
n=6 noise). The read: for this task, fine-tuning's payoff rides on the base's general
capability more than on medical pretraining — a statement about fine-tuning, not a
knock on the medical base. See `reports/final_experiment_report.md`.

## Limitations

- Synthetic-data bias; template-derived gold ⇒ scores partly reflect format
  matching, not clinical reasoning.
- Automatic metrics only; **no clinical validation**; near-ceiling ≠ clinically
  correct.
- Tiny test set (n=6) ⇒ rank differences are within noise.
- Narrow scope: fixed labs/vitals, preventive/borderline range only; no imaging,
  genomics, notes, or wearables. Can still hallucinate.

## License

Adapter weights are derivative of the base model and are bound by the base model's
terms (MedGemma / Health AI Developer Foundations). Confirm the exact license and
redistribution allowance before any release.
