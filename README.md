# Preventive Health Model Lab

Fine-tuning open **medical LLMs** (MedGemma-class 4B, via QLoRA) for
**longitudinal preventive-health reasoning support** — reading a patient's
history over time and surfacing *trends, risk signals, and good questions for
a clinician*, never a diagnosis.

> **Status: Phase 0/1 — design under review.** This repo is a scaffold. No
> model has been trained or downloaded. What runs today is the environment
> check, the baseline **dry-run**, and schema validation of the example data.

---

## ⚠️ Safety boundaries (read first)

This project is **research and education only**. It is built to make the
safety framing structural, not optional:

- **No real PHI, ever.** Only synthetic or properly-licensed, de-identified
  data enters the pipeline. The timeline validator hard-fails any record not
  explicitly marked `synthetic: true`.
- **The model must not diagnose.** The target output is *reasoning support*:
  trends, risk signals, missing information, and questions for a clinician.
- **Every output carries uncertainty + a clinician-review disclaimer.** This
  is enforced through `src/preventive_health_model_lab/safety/` and is part of
  the 7-part output schema, not an afterthought.
- **Not a medical device.** Nothing here is validated for clinical use, and it
  must never be used to make an actual health decision.

If you are looking for medical advice, talk to a licensed clinician.

---

## Why this project

**Why preventive health?** The high-value, lower-acute-risk reasoning task is
noticing *drift over time* — an HbA1c creeping from 5.5 → 6.3, an LDL trend, a
blood-pressure creep against a strong family history — and turning that into
the right questions before anything becomes acute. That is a longitudinal,
multi-signal reasoning problem, which is exactly where a fine-tuned model can
add structure.

**Why fine-tuning, not RAG alone?** Retrieval is great for *facts* ("what is
the prediabetes range?"). It does not teach a model to:

- read a **multi-year, multi-signal timeline** and weigh co-occurring trends,
- consistently emit a **structured 7-part answer** with calibrated hedging,
- reliably **stay inside the safety envelope** (no diagnosis, always disclaim).

Those are *behaviors/format/tone*, which supervised fine-tuning shapes far
more reliably than prompting + RAG. RAG stays useful and complementary for
grounding factual claims — the two are not mutually exclusive.

---

## Hardware

Developed for a modest local box:

- GPU: **NVIDIA RTX PRO 2000 (Blackwell), 8 GB VRAM** → 4-bit QLoRA only
- CUDA 13.1, 31 GB system RAM, WSL2, Python 3.12
- Everything is sized for a single small GPU: 4B base, `r=16` LoRA, batch 1 +
  grad accumulation, gradient checkpointing. See `configs/sft_lora_medical.yaml`.

`bitsandbytes` Blackwell/`sm_120` support must be **verified** before relying
on it — see `requirements.txt`.

---

## What works TODAY

```bash
# 0. (optional) create a light venv for the non-GPU tooling
python3 -m venv .venv && source .venv/bin/activate
pip install pyyaml pandas pytest        # light tools only; NOT the GPU stack

# 1. Check the environment (safe to run anywhere; PASS/WARN table)
python scripts/00_check_environment.py

# 2. Validate the example synthetic timeline against the schema
python scripts/02_build_instruction_dataset.py --validate-examples

# 3. Dry-run the baseline: resolve config + prompts, print the plan, exit 0
#    (loads NO model, no network access)
python scripts/03_run_baseline.py --dry-run

# 4. List intended data sources (nothing is fetched)
python scripts/01_download_or_prepare_data.py --list-sources

# 5. Run the test suite
python -m pytest -q
```

Everything else (data build, training, real inference, eval scoring) is a
**skeleton** and will raise a clear `NotImplementedError` or exit with guidance.

---

## Repository layout

```
configs/     sft_lora_medical.yaml (training design contract), eval.yaml
data/        raw/ processed/ eval/ synthetic/  (+ README with data rules)
scripts/     00_check_environment  01_download_or_prepare_data
             02_build_instruction_dataset  03_run_baseline
src/preventive_health_model_lab/
             data/ (schema + validators)  safety/ (disclaimer + guardrails)
             training/ evaluation/ inference/ utils/
examples/    sample_patient_timeline.json (SYNTHETIC-001)
             sample_model_output.md (gold-standard 7-part output)
tests/       schema, env-check, and baseline dry-run tests
reports/     experiment_log.md
notebooks/ app/
```

## The 7-part output schema

Every model answer must contain these sections (canonical list in
`src/preventive_health_model_lab/data/schema.py::OUTPUT_SECTIONS`):

1. **Longitudinal summary** — the trend story over time
2. **Risk signals** — what stands out, and why it clusters
3. **Evidence** — the specific data points behind each signal
4. **Missing information** — what would change the picture
5. **Clinician questions** — what to ask / check next
6. **Safety disclaimer** — uncertainty + not-a-diagnosis
7. **What NOT to conclude** — explicit anti-overreach guardrails

See `examples/sample_model_output.md` for a hand-written gold example.

---

## Phase roadmap

- **Phase 0/1 — design + scaffold (this repo):** structure, safety framing,
  schema, runnable env-check + dry-run + validation, tests. *Under review.*
- **Phase 2 — data:** synthetic timeline generator, instruction-dataset build,
  patient-level splits, dedup. Real generation still not wired.
- **Phase 3 — baseline:** pull a gated MedGemma-class model (deliberately),
  run the baseline over synthetic timelines, score schema conformance + safety.
- **Phase 4 — training:** QLoRA SFT per `configs/sft_lora_medical.yaml`, track
  in MLflow, evaluate the adapter vs. baseline.
- **Phase 5 — app + eval harness:** Streamlit demo (clearly labeled research
  tool), richer rubric scoring, safety red-team pass.

## License

MIT — see `LICENSE`. Note: the *base models* (e.g. MedGemma) carry their own,
stricter licenses you must accept separately.
