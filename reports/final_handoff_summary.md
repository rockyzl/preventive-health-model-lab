# Final handoff summary

> Date: 2026-07-08. Research/education only; synthetic data; not clinical.

## What was completed

An end-to-end, reproducible QLoRA fine-tuning experiment on one 8 GB GPU, plus a
public-demo package:

- Synthetic longitudinal preventive-health dataset (60 patients, 5 archetypes,
  patient-level 48/6/6 split, zero overlap, correct-by-construction gold).
- Both 4B models fine-tuned with an identical recipe: **MedGemma 1.5 4B** (primary)
  and **Gemma 3 4B** (control).
- Predictions for all 4 conditions on the held-out test set, scored by the validated
  automatic evaluator; head-to-head comparison + honest report.
- Precomputed, synthetic-only demo artifacts + draft HF adapter/dataset cards + Space
  plan. 47 automated tests pass.

## Where things are

| artifact | path |
|---|---|
| Trained adapters (gitignored, ~57 MB each) | `adapters/medgemma-1.5-4b-preventive-sft-v0/`, `adapters/gemma3-4b-preventive-sft-v0/` |
| Predictions (4 conditions) | `outputs/predictions/*_test_predictions.jsonl` |
| Evaluation scores + comparison | `outputs/evaluation/` (`comparison_summary.csv`, `comparison_by_archetype.csv`, `failure_cases.jsonl`, `<cond>_scores.json`) |
| Final report | `reports/final_experiment_report.md` |
| Public demo package | `demo_artifacts/` |
| HF release drafts | `hf_release/` |

## Headline result

QLoRA fine-tuning reliably instilled the safe, non-diagnostic, 7-part format in both
models (safety hard-fail 100 % → 0 %; overall 0.61 → ~1.0). **The unexpected, useful
finding: medical pretraining was not the deciding factor** — after fine-tuning the
medical and general bases were indistinguishable (Gemma 3 QLoRA 1.000; MedGemma QLoRA
0.997; the gap is within n=6 noise). Read: for this task, fine-tuning's payoff rides
on the base's general capability more than on domain pretraining — a finding about
fine-tuning, not a knock on medical models. n=6 + automatic metrics + synthetic data
⇒ do not overclaim.

## Top 3 strengths

1. **Clean controlled design** (same base family, identical recipe/data/split) with
   an honest, non-cherry-picked result.
2. **Real engineering on constrained hardware** — 4B QLoRA in 8 GB, incl. solving a
   multimodal weight-loading trap, vision-tower LoRA scoping, a context-window
   truncation bug, and a WSL2 trainer deadlock.
3. **Safety made structural** — synthetic-only hard gate, non-diagnostic scanner,
   numeric-grounding faithfulness metric, precomputed demo that refuses user PHI.

## Top 3 limitations

1. Tiny synthetic test set (n=6) + automatic metrics ⇒ measures form/faithfulness,
   not clinical correctness; rank differences are within noise.
2. Template-derived gold ⇒ fine-tuning largely learns the target format; likely
   overstates real-world performance.
3. No real clinical data and no clinician review; one archetype missing from the
   test split.

## Recommended next steps

1. Scale the synthetic test set (100+ patients, all archetypes) so differences are
   meaningful; add paraphrase diversity to gold answers.
2. Add an LLM/clinician judge for the non-automatic dimensions (clinical usefulness,
   hedging quality) — currently intentionally left unscored.
3. Pursue credentialed real data (MIMIC-IV-Ext-Instr / EHRSHOT) as a v2 track and
   re-run the same comparison to test whether the medical base helps on real data.
4. Ship the static demo (Space/site) from `demo_artifacts/`; keep it read-only.
