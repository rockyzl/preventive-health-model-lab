# Safety notice (must appear on every page of any demo)

> **This research demo uses synthetic patient timelines only.** It is designed to
> study model behavior in preventive-signal reasoning and is **not intended to
> diagnose, treat, predict, or manage any medical condition**. Do **not** enter
> personal health information. Outputs may be incomplete or incorrect and should
> not be used for health decisions. Always consult a qualified clinician for
> medical concerns.

## What this demo is

A read-only, **precomputed** comparison of how four model conditions
(Gemma 3 and MedGemma, each before and after QLoRA fine-tuning) respond to a set
of **synthetic** longitudinal patient records. It exists to make a research
result inspectable, not to provide any health service.

## What this demo must never do

- Never accept real patient data of any kind: no uploads of lab reports, MyChart
  / EHR records, PDFs, screenshots, images, names, birthdays, addresses, phone
  numbers, medication lists, or any personal health information.
- Never present model output as a diagnosis, prognosis, triage, or treatment
  recommendation.
- Never run live inference on user-supplied text in this first version — the demo
  serves fixed, precomputed outputs for fixed synthetic cases only.

## Framing rules for any commentary shown

Describe **model behavior**, not clinical correctness (no clinical review has been
performed). Prefer wording like: "the fine-tuned model followed the required
schema more reliably", "cited fewer unsupported numbers", "better preserved the
non-diagnostic framing", "surfaced the longitudinal trend more clearly". Failures
and limitations are shown, not hidden.
