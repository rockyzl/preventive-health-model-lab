# demo_artifacts/ — precomputed public demo package

Read-only, **synthetic-only** data for a static demo (website or Hugging Face
Space). **No live inference. No user input. No PHI.** Every file here is derived
from the held-out synthetic test set and the automatic evaluation.

| file | contents |
|---|---|
| `safety_disclaimer.md` | the banner that must appear on every page |
| `space_plan.md` | how to build the static viewer |
| `synthetic_cases.json` | the 24 held-out synthetic patients: timeline + gold answer |
| `model_outputs.json` | for each patient, the 4 model outputs (base/QLoRA × Gemma3/MedGemma) |
| `evaluation_summary.json` | per-case scores + aggregates for all conditions |
| `comparison_summary.csv` | aggregate head-to-head table |
| `selected_failure_cases.json` | honest failure highlights (hard-fails + hallucinated numbers) |

Regenerate everything with `python scripts/07_build_comparison.py --suffix _holdout24` after
`scripts/06_generate_predictions.py` has produced `outputs/predictions/`.

**Headline:** QLoRA made both 4B models reliably safe and well-formed (100 %→0 %
safety hard-fails). The unexpected, useful finding: after fine-tuning the medical and
general bases were **indistinguishable** — for this task, the payoff rode on the
base's general capability more than on medical pretraining. Automatic metrics,
synthetic data, n=24 — not clinical evidence. See `reports/final_experiment_report.md`.
