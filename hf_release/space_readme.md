# Space README (draft) — Preventive Health Model Lab · results viewer

**Draft — do not upload until explicitly asked.**

```
---
title: Preventive Health Model Lab (research demo)
emoji: 🩺
colorFrom: indigo
colorTo: gray
sdk: streamlit
pinned: false
license: other
---
```

## ⚠️ Safety

This research demo uses **synthetic patient timelines only**. It studies model
behavior in preventive-signal reasoning and is **not intended to diagnose, treat,
predict, or manage any medical condition**. **Do not enter personal health
information.** Outputs may be incomplete or incorrect and must not be used for
health decisions. Always consult a qualified clinician. (Full text:
`demo_artifacts/safety_disclaimer.md`.)

## What this Space shows

A **precomputed, read-only** comparison — **no live model runs**, no user input. It
loads fixed JSON/CSV from `demo_artifacts/` and lets you inspect, for a handful of
synthetic patients:

- the synthetic timeline and the gold reference answer,
- four model outputs side by side — Gemma 3 and MedGemma, each **before and after**
  QLoRA fine-tuning,
- automatic evaluation scores, with hallucinated numbers highlighted,
- a "results & limitations" view with the honest head-to-head and its caveats.

## The honest result (headline)

QLoRA fine-tuning reliably instilled the safe, non-diagnostic, 7-part format in both
4B models (both went from 100 % → 0 % safety hard-fails). On this synthetic benchmark
the medical model (MedGemma) showed **no measurable advantage** over the non-medical
control (Gemma 3). Synthetic data + automatic metrics + n=6 ⇒ do not overclaim.

## What it must never do

No uploads, no user PHI, no live inference on user text, no diagnosis/treatment
claims. See the safety file. Adapter-only; the Space ships **no base-model weights**.
