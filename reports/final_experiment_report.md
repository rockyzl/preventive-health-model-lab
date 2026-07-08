# Final experiment report — does medical pretraining help preventive-health reasoning?

> Research/education only. **Synthetic data, automatic metrics, tiny test set.**
> Not a diagnostic product, not clinical decision support. Date: 2026-07-08.

## Question

After **identical** QLoRA fine-tuning on the **same** synthetic data and split, does
the medically continued-pretrained **MedGemma 1.5 4B** reason better about
longitudinal preventive-health signals than its non-medical same-family base
**Gemma 3 4B**?

## Setup (identical across the two models)

- Base models loaded in 4-bit (nf4); LoRA r=16/α=32 on the language-model
  projections; 5 epochs, effective batch 8, lr 2e-4 cosine, seq≤2048. Same recipe,
  same code, same seed.
- Data: 60 synthetic patients, patient-level split **train 48 / val 6 / test 6**,
  **zero patient overlap** (verified). Test covers 4 of 5 archetypes
  (isolated-hypertension ×2, stable-healthy ×2, lipids+family-risk ×1,
  cardiometabolic-drift ×1; **improving-after-intervention is absent from the test
  split** — a coverage gap).
- Four conditions scored on the held-out test set: each model **before** and
  **after** fine-tuning.
- Metrics are **automatic**: schema conformance (all 7 sections present), safety
  disclaimer present (verbatim), non-diagnostic (a red-flag scanner; any diagnostic
  phrase is a hard fail), numeric grounding (fraction of cited clinical numbers that
  actually appear in the input — the anti-hallucination metric). "overall" is the
  mean of these four. Gold answers score a perfect 1.000 as the ceiling.

## Results (held-out test, n=6)

| condition       | disclaimer | non-diagnostic | numeric grounding | schema | overall | hard-fail rate | hallucinated #s |
|-----------------|-----------:|---------------:|------------------:|-------:|--------:|---------------:|----------------:|
| Gemma 3 base    |      0.000 |          0.500 |             0.950 |  1.000 |   0.612 |          100 % |               2 |
| **Gemma 3 QLoRA** |    1.000 |          1.000 |         **1.000** |  1.000 | **1.000** |          0 % |           **0** |
| MedGemma base   |      0.000 |          0.667 |             0.954 |  0.833 |   0.614 |          100 % |               7 |
| MedGemma QLoRA  |      1.000 |          1.000 |             0.987 |  1.000 |   0.997 |          0 % |               2 |
| gold (ceiling)  |      1.000 |          1.000 |             1.000 |  1.000 |   1.000 |          0 % |               0 |

Full numbers: `outputs/evaluation/comparison_summary.csv`, per-archetype in
`comparison_by_archetype.csv`, every failure in `failure_cases.jsonl`.

## Answers

1. **Did QLoRA improve Gemma 3?** Yes, dramatically — overall 0.612 → 1.000,
   hard-fail 100 % → 0 %, hallucinated numbers 2 → 0.
2. **Did QLoRA improve MedGemma?** Yes, dramatically — overall 0.614 → 0.997,
   hard-fail 100 % → 0 %, hallucinated numbers 7 → 2.
3. **After fine-tuning, does MedGemma outperform Gemma 3?** Not measurably — they
   land effectively tied at the ceiling (1.000 vs 0.997; the 2-number gap is within
   n=6 noise). The point isn't that one model "wins"; it's that **medical pretraining
   was not the deciding factor once both capable bases were fine-tuned on the task**.
4. **Where does MedGemma help most?** Nowhere clearly, on the held-out test. Its
   only visible edge is a lower *training* loss (avg 0.547 vs 0.629 — it fit the
   format slightly faster), which did not translate into better held-out scores.
5. **Where does Gemma 3 match or beat MedGemma?** Across the board here. Gemma 3
   QLoRA had 0 hallucinated numbers (MedGemma QLoRA: 2). Even *before* fine-tuning,
   base Gemma 3 was cleaner than base MedGemma (2 vs 7 hallucinated numbers; schema
   1.000 vs 0.833).
6. **Did either model get less safe / more overconfident / more diagnostic after
   fine-tuning?** No — the opposite. Both base models almost always hard-failed on
   safety (no verbatim disclaimer; diagnostic words like "prescribe"). Fine-tuning
   made both reliably include the disclaimer and drop diagnostic language. **The
   single biggest, most consistent effect of fine-tuning was safety-framing
   compliance**, for both models.
7. **Does medical pretraining appear to improve preventive-signal reasoning on this
   synthetic benchmark?** Not on top of what fine-tuning already delivers here — both
   converge to near-ceiling and the automatic metrics can't separate them. What the
   platform did surface is that **a capable base plus task fine-tuning is what carried
   the result**; the domain of the base's pretraining was secondary for this task.
8. **What remains unproven because the data is synthetic?** A lot — see below.

## Honest caveats (do not overclaim)

- **Tiny test set (n=6).** A single patient's two hallucinated numbers is the entire
  Gemma-3-vs-MedGemma gap. Treat rank differences as noise.
- **Automatic metrics measure form + faithfulness, not clinical correctness.** A
  1.000 means "well-structured, cites only numbers that are in the record, carries
  the disclaimer, avoids diagnostic phrasing" — **not** "clinically right." No
  clinician reviewed any output.
- **Synthetic + template-derived gold.** Fine-tuning largely learns the target
  *format*; high scores partly reflect format-matching, which is easier than genuine
  reasoning. The result likely overstates how well this would work on real data.
- **Base MedGemma outputs partly degenerated** (repeating the system prompt) and hit
  the 1100-token generation cap, which contributes to its low base schema score —
  i.e. its base numbers reflect "not instruction-tuned for this format" + truncation,
  not a fair test of medical knowledge.
- **Coverage gap:** the test split lacks the improving-after-intervention archetype.
- **What's unproven:** whether medical pretraining helps on real, messy,
  distribution-shifted EHR data; whether the reasoning is clinically sound;
  generalization beyond 5 stylized archetypes and a narrow lab/vital panel.

## Bottom line

Two clean findings, both useful. **First, fine-tuning worked and mattered** — it
turned two models that routinely broke the safety contract into two that reliably
follow the 7-part, non-diagnostic, disclaimer-bearing format with faithful numbers.
**Second — the part we didn't anticipate — the medical base did not separate from the
general one after fine-tuning; both reached the ceiling.** The useful reading is not
"one model lost" but a statement about *what fine-tuning depends on*: for this task,
the payoff rides on the base model's general capability more than on domain (medical)
pretraining — a capable base adapts well regardless of what it was pretrained on. That
takes nothing away from medical models in general; it's a finding about fine-tuning,
and exactly the kind of thing this controlled platform was built to surface. What it
means for real, messy clinical data is still open — and the obvious next experiment.
