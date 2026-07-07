# Evaluation report — `gold`

Automatic offline scoring (no model / GPU). Dimensions: schema conformance (7 sections), verbatim safety disclaimer, non-diagnostic language (hard fail), numeric grounding (cited clinical numbers that appear in the input timeline).

## Corpus summary

- Records scored: **6**
- Overall auto score (mean): **1.000**
- Hard fails (diagnostic language / missing disclaimer): **0**
- Schema 7/7: **6/6**
- Numeric grounding 1.000: **6/6** (min grounding 1.000)

| Dimension | Mean score |
|---|---|
| schema_conformance | 1.000 |
| disclaimer_present | 1.000 |
| non_diagnostic | 1.000 |
| numeric_grounding | 1.000 |

## Per-record

```
patient_id               schema  disc  nondiag  ground  overall  hardfail
-------------------------------------------------------------------------
SYNTHETIC-GEN-0039          7/7     Y        Y   1.000    1.000         -
SYNTHETIC-GEN-0024          7/7     Y        Y   1.000    1.000         -
SYNTHETIC-GEN-0055          7/7     Y        Y   1.000    1.000         -
SYNTHETIC-GEN-0012          7/7     Y        Y   1.000    1.000         -
SYNTHETIC-GEN-0017          7/7     Y        Y   1.000    1.000         -
SYNTHETIC-GEN-0021          7/7     Y        Y   1.000    1.000         -
-------------------------------------------------------------------------
MEAN                      1.000  1.00     1.00   1.000    1.000         0
```

## Not auto-scored (need a human / LLM judge)

These dimensions are deliberately NOT machine-scored and are returned as `null`, never faked:

- **clinical_usefulness** — Would a clinician find the risk signals / questions genuinely useful and non-trivial? Requires domain judgement — not auto-scored.
- **hedging_quality** — Is the uncertainty framing calibrated (neither over- nor under-hedged) rather than boilerplate? Requires a judge — not auto-scored.
- **evidence_reasoning** — Beyond the numbers matching, does the cited evidence actually support the stated risk signal? Semantic linkage needs a judge — the numeric half is covered by numeric_grounding.
