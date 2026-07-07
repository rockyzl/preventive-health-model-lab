# Data — rules, provenance, and current state

> **Current state: EMPTY by design.** No datasets are committed. `raw/`,
> `processed/`, and `eval/` are gitignored. Only small, clearly-synthetic
> examples live under `examples/` (tracked) and may be mirrored into
> `synthetic/`.

This directory is governed by hard rules. They exist because this is a
*medical* project and the failure mode (leaking real patient data, or
training a model that quietly diagnoses) is serious.

## The rules

1. **No real PHI. Ever.**
   Only two kinds of data are allowed here:
   - **Synthetic** data, programmatically or hand-generated, every record
     explicitly marked `synthetic: true`.
   - **Properly licensed, de-identified** public/research corpora, with the
     de-identification and license **verified and recorded** before use.

   The timeline validator (`src/.../data/schema.py`) hard-fails any record
   that is not explicitly `synthetic: true`, so unlabeled data cannot enter
   the pipeline silently.

2. **Provenance is mandatory.**
   Every source that lands in `raw/` must have an entry in the source
   registry (`scripts/01_download_or_prepare_data.py::SOURCE_REGISTRY`)
   recording: `id`, `kind`, `license`, `phi` flag, and a note. No provenance,
   no ingestion.

3. **Licenses are respected.**
   Base models (MedGemma, etc.) and any external corpora carry their own
   licenses. Record them. Do not commit weights or licensed corpora to git.

4. **Split discipline: patient-level, no leakage.**
   Train / val / eval splits are made **by `patient_id`**. A single (synthetic)
   patient must never appear in more than one split. This is enforced in
   `build_splits()` (Phase 2).

5. **Nothing large in git.**
   `raw/`, `processed/`, model weights, and adapters are gitignored. Keep the
   repo small; data is reproduced from the documented sources/generators.

## Layout

```
data/
  raw/         staged source material (gitignored; empty now)
  processed/   built instruction dataset: train/val JSONL (gitignored; empty)
  eval/        held-out eval set (gitignored; empty)
  synthetic/   small synthetic samples; large binaries gitignored
```

## Where the shape is defined

- Patient timeline shape + validator: `src/preventive_health_model_lab/data/schema.py`
- A worked synthetic example: `examples/sample_patient_timeline.json`
- The target model output schema: `examples/sample_model_output.md`
