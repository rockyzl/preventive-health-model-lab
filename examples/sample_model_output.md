# Gold-standard target output — style example

> This is a **hand-written** reference answer for record `SYNTHETIC-001`
> (`examples/sample_patient_timeline.json`). It defines the target *style*
> and the **7-part output schema** the fine-tuned model should learn to
> produce. It is an illustration of desired form and tone, not clinical
> guidance. The section headings below are the canonical schema — see
> `OUTPUT_SECTIONS` in `src/preventive_health_model_lab/data/schema.py`.

---

## 1. Longitudinal summary

Over roughly three years (2023-03 to 2025-06), this synthetic 47-year-old
record shows a gradual cardiometabolic drift rather than any acute event:

- **HbA1c** rose 5.5% → 5.9% → 6.3%, moving from normal into the
  prediabetes range (5.7–6.4%), then edged back to 6.1% at the last visit.
- **Fasting glucose** climbed 96 → 104 → 108 → 112 mg/dL.
- **LDL cholesterol** trended up 128 → 149 mg/dL before improving to
  118 mg/dL after a statin was started (2025-02).
- **Blood pressure** crept from 122/78 to 134/86 mmHg (single readings).
- **Weight/BMI** rose then partially reversed (BMI 26.5 → 27.5 → 26.8).

The most recent visit suggests an early, partial response to lifestyle
changes plus a statin.

## 2. Risk signals

- **Prediabetes trajectory:** three consecutive HbA1c values in/near the
  prediabetes band, with rising fasting glucose — a signal worth close
  follow-up.
- **Lipid + strong family CAD history:** rising LDL against a father with
  CAD and a grandfather with early MI.
- **Blood-pressure creep** into the elevated / stage-1 range on the
  available single readings.
- **Clustering:** these signals co-occur (glucose, lipids, BP, weight),
  which is the pattern that matters more than any single value.

## 3. Evidence (what in the record supports each signal)

| Signal | Supporting data points |
|---|---|
| Prediabetes | HbA1c 5.9 (2024-03), 6.3 (2025-02); fasting glucose 104→112 |
| Lipids | LDL 128→141→149; then 118 on atorvastatin |
| BP creep | 122/78 (2023) → 134/86 (2025-02) |
| Family risk | father T2DM @52 + CAD @61; grandfather MI @64 |

## 4. Missing information (what would change the picture)

- BP is from **single office readings** — no home/ambulatory series, so
  "elevated" is provisional.
- No **kidney markers** (eGFR, urine albumin/creatinine) in the record.
- No **ASCVD 10-year risk** inputs fully captured (e.g., no documented
  smoking status change over time beyond "never").
- No **medication adherence** or dosage-response detail after the statin
  start.
- Lab timing/fasting state not always stated.

## 5. Clinician questions (for the next visit)

- Should HbA1c be rechecked on the planned ~6-month interval, or sooner
  given the earlier upward trend?
- Would repeat/home BP monitoring change whether BP is treated?
- Are kidney function and a fuller lipid/risk panel warranted given the
  family history?
- Is a structured lifestyle / prevention program indicated and available?

## 6. Safety disclaimer

This is research/education output from an experimental language model. It
is **not** medical advice, **not** a diagnosis, and **not** a substitute
for a licensed clinician. Every observation above is provisional and may
be wrong. This record is **synthetic** — no real patient data was used.
Any health decision must be made with a qualified healthcare professional.

## 7. What NOT to conclude

- Do **not** conclude the person "has diabetes" — the values are in the
  prediabetes range and a diagnosis requires clinician assessment and
  confirmatory testing.
- Do **not** conclude the statin "fixed" anything — one improved LDL value
  is not a durable trend.
- Do **not** treat single-visit BP readings as a hypertension diagnosis.
- Do **not** infer causation from these co-occurring trends.
- Do **not** use this output to start, stop, or change any medication.
