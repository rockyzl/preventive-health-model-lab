# Public demo plan — static, precomputed (no live GPU inference)

Goal: a free Hugging Face Space (or a page on the personal site) that lets anyone
inspect the experiment's results, without any model running and without accepting
any user health data.

## Version 1 (this plan): precomputed viewer

- **Framework:** Gradio or Streamlit, CPU-only, no GPU, no model weights loaded.
- **Data source:** loads the JSON/CSV files in `demo_artifacts/` at startup. No
  network calls, no inference.
- **Interaction:** the user picks one of 5–8 synthetic patients from a dropdown.
- **What it shows for the selected case:**
  1. the compact synthetic timeline (clearly labeled synthetic),
  2. the four model outputs side by side — Gemma 3 base, Gemma 3 QLoRA,
     MedGemma base, MedGemma QLoRA,
  3. the gold reference answer,
  4. automatic evaluation scores per output (schema, disclaimer, non-diagnostic,
     numeric grounding), with any hallucinated numbers highlighted,
  5. a short, behavior-focused note on what changed after fine-tuning.
- **Always visible:** the safety banner from `safety_disclaimer.md`, on every page.
- **A "Results & limitations" tab:** the headline finding, the by-archetype
  breakdown, and an explicit "what remains unproven because the data is synthetic"
  section. Failure cases are shown, not hidden.

## Explicitly out of scope for v1

- No text box for user input; no file/image upload; no "analyze my labs".
- No live model inference; no base-model weights shipped in the Space.
- No claim of clinical validity anywhere.

## Files the Space reads (all in `demo_artifacts/`)

- `synthetic_cases.json` — the selected synthetic timelines
- `model_outputs.json` — the four outputs per case
- `evaluation_summary.json` — per-case, per-condition scores
- `comparison_summary.csv` — aggregate head-to-head
- `selected_failure_cases.json` — honest failure highlights
- `safety_disclaimer.md` — the banner text

## Version 2 (future, only after clinical/legal review)

Optional live inference on a fixed menu of synthetic cases behind a GPU, still with
no user PHI input. Not built in v1.
