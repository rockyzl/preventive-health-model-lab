# Experiment log

Append one row per run. Keep it honest: log failures and dead-ends too — a
"what didn't work" line is worth more than a silent gap. Link the MLflow run
where relevant.

| Date | Run ID | Phase | Base model | Change / hypothesis | Dataset (n, split) | Key config | Result / metric | Safety notes | Next |
|------|--------|-------|-----------|---------------------|--------------------|-----------|-----------------|--------------|------|
| _pending_ | — | 1 | — | Scaffold only; no training yet | — | — | env-check + dry-run + schema tests green | Framing enforced via safety/ + schema | Phase 2: synthetic data generator |

<!--
Row-filling notes:
  - Phase:        1..5 per README roadmap.
  - Dataset:      record count + how the split was made (must be patient-level).
  - Key config:   the few settings that actually differ from last run.
  - Result:       schema-conformance %, safety-flag count, rubric score, loss.
  - Safety notes: any diagnostic-language flags, missing disclaimers, etc.
-->
