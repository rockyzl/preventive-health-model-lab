# Project Design — Preventive Health Model Lab (Phase 0 feasibility)

> Status: **design under review** (end of Phase 0/1 — no training started).
> Date: 2026-07-07. All model/dataset facts verified against primary sources
> (HF model pages, Google HAI-DEF docs, PhysioNet, MITRE, PyTorch/bitsandbytes
> releases) as of this date. Hardware facts measured locally.

One-liner: *fine-tuning open medical LLMs for longitudinal preventive-health
reasoning using public or credentialed clinical instruction data, with
safety-aware evaluation and clinician-review-oriented outputs.*

---

## 1. Hardware (measured, not assumed)

| Item | Measured value |
|---|---|
| GPU | NVIDIA RTX PRO 2000 Blackwell **Laptop** GPU (sm_120) |
| VRAM | **8151 MiB (~8 GB)** — laptop variant, NOT the 16 GB desktop card |
| CUDA driver | 13.1 (driver 591.64), WSL2 Ubuntu |
| RAM / CPU | 31 GB (~16 GB free) / 16 cores |
| Disk | 742 GB free on repo drive |

**Consequence:** the 8 GB ceiling locks v1 to the **~4B parameter class with
QLoRA (4-bit)**. Full fine-tuning and 8B-class models (e.g. Meditron3-8B) are
out of scope for local training.

## 2. Model shortlist

| # | Model (HF id) | Params | License · fine-tune terms | Gated | Context | QLoRA on 8 GB | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | `google/medgemma-1.5-4b-it` | 4B | HAI-DEF (custom, non-OSI) · fine-tune allowed; **adapter redistribution not explicit** | Yes | ≥128K | ~5–6.5 GB with grad-ckpt + bs1 → **fits** | **Primary.** Current small-size SOTA open clinical model; Gemma 3 base |
| 2 | `google/gemma-3-4b-it` | 4B | Gemma Terms (custom) · fine-tune + derivative distribution allowed | Yes | 128K | fits | **Control baseline** — same base as MedGemma, isolates the "does medical pretraining help" variable |
| 3 | `Qwen/Qwen3.5-4B`-class | ~4B | **Apache 2.0** (clean) | No | 256K-class | fits | License-clean alternate baseline (different base → weaker control) |
| 4 | `epfl-llm/meditron3-8b`-class | 8B | Llama 3.1 Community (AUP) | Partly | 128K | ~7–8 GB+ → **too tight** | Stretch goal for a 16 GB machine only |
| 5 | `stanford-crfm/BioMedLM` | 2.7B | GPT-2-era, no instruct, ctx ~1024 | — | ~1024 | fits but pointless | **Rejected** — obsolete, context too short for clinical timelines |

MedGemma 27B exists but cannot be trained on 8 GB; noted for reference only.
Sources: [medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it),
[MedGemma model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card),
[NVIDIA RTX PRO 2000](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-2000/).

### Recommended pair

**`medgemma-1.5-4b-it` (primary) + `gemma-3-4b-it` (control).** MedGemma 1.5
is Gemma 3 4B plus continued medical training, so the same-base control
isolates exactly one variable — the scientific comparison this project exists
to make. If license cleanliness ever outranks scientific control (adapter
redistribution, commercial use), swap the control to a Qwen3.5-4B-class
Apache-2.0 model and accept the weaker comparison.

## 3. Dataset shortlist

| Dataset | Size / access | License · LLM-training terms | Longitudinal? | Preventive fit | Verdict |
|---|---|---|---|---|---|
| [MIMIC-IV-Ext-Instr v1.0.0](https://physionet.org/content/mimic-iv-ext-instr/1.0.0/) | 450K+ instruction pairs; **credentialed** (PhysioNet + CITI + DUA + reference), realistic lead time **2–6+ weeks** | PhysioNet Credentialed License 1.5.0; training is intended use, but see LLM rules below | Partial (ICU/ED-centric, not lifetime) | Weak–moderate; GPT-3.5-generated, no expert validation | v2 material — start credentialing now, don't block v1 on it |
| [MIMIC-IV demo v2.2](https://physionet.org/content/mimic-iv-demo/2.2/) | 100 patients, 15.5 MB, open **today** | ODC (permissive) | within-stay only | Low | Schema-realistic sandbox only |
| [Synthea](https://github.com/synthetichealth/synthea) (MITRE) | generator — unlimited synthetic patients; FHIR/CSV; open **today** | **Apache 2.0** — only fully commercial-clean option | **Yes — lifetime timelines** | **Best** — screening/immunization/wellness modules built in | **v1 backbone** (needs our own event→instruction templating layer) |
| [Asclepius synthetic notes](https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes) | 157K notes + QA, open today | CC-BY-NC-SA (non-commercial) | No (single notes) | Moderate | Auxiliary note-reasoning seed only |
| [EHRSHOT](https://ehrshot.stanford.edu/) (Stanford) | 6,739 patients / 41.6M events; Stanford DUA (days–weeks) | non-commercial research | **Yes — strongest real longitudinal** | Good structure | v2 material; benchmark not instruction data — needs templating |
| [MedAlpaca / Medical Meadow](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) | open today | CC (varies by subset) | No | Low | Auxiliary general-medical mix-in only |

**Key structural finding:** no existing dataset is simultaneously
longitudinal + preventive + instruction-formatted + license-clean. That
combination must be synthesized — which is why the v1 backbone is Synthea plus
our own templating layer, not a download.

### PhysioNet LLM rules we must respect (v2 track)

Per PhysioNet's [Responsible Use of MIMIC with LLMs](https://physionet.org/news/post/llm-responsible-use/):
MIMIC data must **not** be sent through third-party APIs (OpenAI, ChatGPT, …)
or retained by third-party LLM services; locally deployed models are the
recommended path. All MIMIC-based fine-tuning or augmentation in this project
therefore runs on local/self-hosted models only — which matches our local-GPU
training plan anyway.

## 4. Final v1 decision — simplest credible path

**Two-track plan; v1 trains entirely on Track 1.**

- **Track 1 (starts now, zero gating):** generate longitudinal preventive-care
  patients with **Synthea** → build an **event→instruction templating layer**
  (`scripts/02_build_instruction_dataset.py`) producing
  `{instruction, input, output}` pairs targeting the 7-part clinician-review
  output schema → **QLoRA fine-tune `medgemma-1.5-4b-it`** on 8 GB →
  same-recipe control run on `gemma-3-4b-it` → safety-aware eval (0–2 scoring,
  7 dimensions) + error analysis. Optionally mix in a small share of
  Asclepius/MedAlpaca (non-commercial license noted in the data card).
- **Track 2 (starts now in parallel, lands in weeks):** file PhysioNet
  credentialing + CITI immediately; sign the EHRSHOT DUA. These unlock a v2 on
  real EHR-grounded data — evaluated but never blocking v1.

Why this is the "simplest credible" path: everything in Track 1 is available
today, license-clean at the core (Apache 2.0), PHI-free by construction, fits
the measured 8 GB budget, and still yields a real scientific claim (medical vs
general base, same data, same recipe).

## 5. Compute & software estimate

| Item | Estimate |
|---|---|
| QLoRA memory (4B, 4-bit, seq 1–2k, bs1 + grad-accum 16, grad-ckpt, paged_adamw_8bit) | ~5–6.5 GB of 8 GB — fits with margin (config contract: `configs/sft_lora_medical.yaml`) |
| Training time (ballpark, 5–10K examples × 1–2 epochs) | hours-class on this laptop GPU, not days — acceptable for iteration |
| torch | **≥2.7.0 with cu128 wheel** (`--index-url https://download.pytorch.org/whl/cu128`) — first stable release with Blackwell/sm_120 support ([PyTorch 2.7 release](https://pytorch.org/blog/pytorch-2-7/)) |
| transformers / peft / trl / accelerate | current versions (Gemma 3 support landed in transformers ≥4.50) |
| bitsandbytes | **≥0.48, cu128 build** — smoke test **PASSED 2026-07-07** on this GPU: torch 2.11.0+cu128 + bnb 0.49.2, 4-bit load+generate OK at 0.44 GiB (`reports/environment_verified.md`). Unsloth fallback no longer needed. |

## 6. Risks & fallbacks

| Risk | Level | Fallback |
|---|---|---|
| MedGemma HAI-DEF license: adapter redistribution not explicit; model card forbids direct clinical use | **Project-level** | Keep repo framing strictly research/education (already structural); if redistribution blocked, publish evals + code only, or switch primary to Apache-2.0 Qwen-class |
| ~~bitsandbytes kernel on sm_120~~ **CLEARED** | ~~Medium~~ | Smoke test passed 2026-07-07 (torch 2.11 cu128 + bnb 0.49.2, 4-bit OK). No longer a risk. |
| Synthetic-only training data (Synthea realism ceiling) | Medium | Honest framing as pipeline-validation v1; Track 2 real data lands in v2; document the gap in the model card |
| PhysioNet credentialing delay / solo-researcher reference hurdle | Low (v1 unblocked) | v1 never depends on it; EHRSHOT DUA as second real-data path |
| 8 GB OOM during training | Low | seq→1k, LoRA r 16→8, accum ↑; last resort: rent a cloud GPU for the final run, keep local for dev |

## 7. Safety constraints (restated, binding)

Research/education only — not a diagnostic product. No real PHI ever; every
training record must carry `synthetic: true` (schema-enforced hard fail).
Every model output targets the 7-part schema ending in uncertainty +
clinician-review framing; the model reasons over preventive signals and never
diagnoses. Failures are documented, not hidden (`reports/experiment_log.md`).

## 8. Concrete next steps (Phase 2+, pending review)

1. Install GPU stack (torch 2.7 cu128 + bnb ≥0.48) → run 4-bit load smoke test
   on `gemma-3-4b-it` (ungated proxy for kernel check is fine).
2. Accept HAI-DEF terms on HF → cache `medgemma-1.5-4b-it` → run
   `scripts/03_run_baseline.py` on the SYNTHETIC-001 example → commit
   `reports/baseline_outputs.md`.
3. Synthea generation run + design the event→instruction templating layer
   (the real intellectual work of Track 1).
4. File PhysioNet credentialing + CITI; sign EHRSHOT DUA (Track 2 clock).
5. First QLoRA run per `configs/sft_lora_medical.yaml`; log to
   `reports/experiment_log.md`.
