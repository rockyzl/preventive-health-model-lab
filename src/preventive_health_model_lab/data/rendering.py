"""Compact, human-readable rendering of a patient timeline.

The model's ``input`` is this compact clinical flowsheet, NOT pretty-printed
JSON. Rationale: the JSON form of a 3-year timeline runs ~1400 tokens (repeated
keys + indentation), which pushes ``input + gold output`` past the training
window and silently truncates the target answer. This rendering keeps every
clinical number but drops JSON scaffolding and boilerplate prose, cutting the
input ~70% (to ~450 tokens) so the full example fits with headroom. It also
reads the way a clinician scans a flowsheet, which is a more realistic input.

The ``synthetic_meta`` block (the archetype latent = answer key) must already be
stripped by the caller; this renderer never emits it.
"""
from __future__ import annotations

from typing import Any

_SEX = {"female": "F", "male": "M"}
# (timeline-key, label, unit-suffix) for labs, in display order.
_LAB_FIELDS = [
    ("hba1c_pct", "HbA1c", "%"),
    ("fasting_glucose_mgdl", "fasting", ""),
    ("ldl_mgdl", "LDL", ""),
    ("hdl_mgdl", "HDL", ""),
    ("triglycerides_mgdl", "trig", ""),
    ("total_chol_mgdl", "totChol", ""),
]


def _encounter_line(e: dict[str, Any]) -> str:
    vitals = e.get("vitals") or {}
    labs = e.get("labs") or {}
    parts: list[str] = []
    if vitals.get("bp_systolic") and vitals.get("bp_diastolic"):
        parts.append(f"BP {vitals['bp_systolic']}/{vitals['bp_diastolic']}")
    if vitals.get("bmi") is not None:
        parts.append(f"BMI {vitals['bmi']}")
    if vitals.get("weight_kg") is not None:
        parts.append(f"wt {vitals['weight_kg']}kg")
    parts += [
        f"{label} {labs[key]}{unit}"
        for key, label, unit in _LAB_FIELDS
        if labs.get(key) is not None
    ]
    seg = ", ".join(parts)
    return f"{e.get('date', '?')} {e.get('type', 'encounter')}: {seg}."


def render_timeline(timeline: dict[str, Any]) -> str:
    """Render a (synthetic_meta-free) patient timeline as a compact flowsheet."""
    lines: list[str] = []
    d = timeline.get("demographics", {}) or {}
    sex = _SEX.get(d.get("sex_at_birth", ""), d.get("sex_at_birth", ""))
    lines.append(
        f"Synthetic patient (training only). "
        f"{d.get('age', '?')}{sex}, height {d.get('height_cm', '?')}cm."
    )

    fh = timeline.get("family_history") or []
    if fh:
        items = [
            f"{x.get('relation', '?')} {x.get('condition', '?')}"
            + (f" (onset {x['onset_age']})" if x.get("onset_age") else "")
            for x in fh
        ]
        lines.append("Family history: " + "; ".join(items) + ".")

    meds = timeline.get("medications") or []
    active = [
        m.get("name") + (f" from {m['start']}" if m.get("start") else "")
        for m in meds
        if m.get("name") and m.get("name") != "none"
    ]
    lines.append("Medications: " + ("; ".join(active) if active else "none") + ".")

    imm = timeline.get("immunizations") or []
    if imm:
        lines.append(
            "Immunizations: "
            + ", ".join(f"{x.get('vaccine')} {x.get('date')}" for x in imm)
            + "."
        )

    ls = timeline.get("lifestyle") or {}
    if ls:
        lines.append("Lifestyle: " + "; ".join(f"{k} {v}" for k, v in ls.items()) + ".")

    lines.append("Encounters:")
    lines.extend(_encounter_line(e) for e in timeline.get("timeline", []))
    return "\n".join(lines)
