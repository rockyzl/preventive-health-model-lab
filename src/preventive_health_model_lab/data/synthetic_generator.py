"""Controlled synthetic longitudinal preventive-health record generator (v1).

From ONE set of latent parameters per patient this module co-generates:

  1. a multi-year :class:`PatientTimeline` (labs / vitals / history), and
  2. the gold **7-part, NON-diagnostic** reasoning output over that timeline.

Because both are produced from the same latent trajectory, the target output
is *correct-by-construction*: every risk signal and evidence citation refers to
a data point the generator actually emitted, the framing is preventive (never
diagnostic), and no real PHI is involved. No Synthea, no Java — that realism
upgrade is deferred to v2.

Design choices that keep the gold output honest and safe:

  * **Preventive zone only.** Every biomarker is clamped to the borderline /
    "worth-watching" band (prediabetes *not* diabetes, elevated / stage-1 BP
    *not* a crisis, borderline-high LDL). The correct answer is therefore
    always "monitor + clinician follow-up", never a diagnosis. This is what
    lets the derived output pass the diagnostic red-flag scanner in
    :mod:`preventive_health_model_lab.safety.disclaimer`.
  * **The gold output is a pure function of the emitted timeline.** The deriver
    reads the timeline the model will see (plus the archetype latent carried in
    ``synthetic_meta``) — so there is zero train/serve skew and the evidence
    always cites real emitted values.

Public API
----------
``ARCHETYPES``                              the trajectory archetypes
``generate_patient(index, seed)``           -> :class:`GeneratedPatient`
``generate_dataset(n, seed)``               -> list[:class:`GeneratedPatient`]
``build_timeline(params)``                  -> timeline record (dict)
``derive_output_sections(timeline)``        -> {section_name: markdown_body}
``render_output_markdown(sections)``        -> gold 7-part text
"""
from __future__ import annotations

import datetime as _dt
import random
from dataclasses import dataclass, field
from typing import Any

from ..safety.disclaimer import SAFETY_DISCLAIMER
from .schema import OUTPUT_SECTIONS

GENERATOR_VERSION = "synthetic_generator/v1"

# Trajectory archetypes. Order is stable: the dataset assigns them round-robin
# by patient index so every archetype is represented and balanced.
ARCHETYPES: tuple[str, ...] = (
    "cardiometabolic-drift",
    "stable-healthy",
    "improving-after-intervention",
    "isolated-hypertension",
    "lipid-plus-family-risk",
)

# Fictional pseudonym pool — deliberately generic, always suffixed "(fictional)".
_FIRST_NAMES = (
    "Alex", "Casey", "Jordan", "Riley", "Sam", "Taylor", "Jamie", "Morgan",
    "Avery", "Quinn", "Devon", "Reese", "Emerson", "Rowan", "Sage",
)
_LAST_NAMES = (
    "Rivers", "Lake", "Stone", "Fields", "Vale", "Brooks", "Marsh", "Glen",
    "Ford", "Reed", "Hale", "Nash", "Wren", "Frost", "Pike",
)

_SMOKING = ("never", "never", "former (quit >5y)", "never")
_ALCOHOL = ("none", "1-2 drinks/week", "2-4 drinks/week", "occasional social")
_SLEEP = ("7-8h/night", "6h/night average, reports work stress", "7h/night", "6-7h/night")


# ---------------------------------------------------------------------------
# Reference bands (for DESCRIPTIVE, non-diagnostic language only).
# ---------------------------------------------------------------------------
_REF = {
    "hba1c_pct": "prediabetes reference range 5.7-6.4%",
    "fasting_glucose_mgdl": "impaired fasting glucose range 100-125 mg/dL",
    "ldl_mgdl": "borderline-high LDL at/above 130 mg/dL",
    "hdl_mgdl": "low HDL below 40 mg/dL",
    "triglycerides_mgdl": "borderline-high triglycerides at/above 150 mg/dL",
    "bp": "elevated / stage-1 range (single office readings)",
    "bmi": "overweight range at/above 25",
}


@dataclass(frozen=True)
class _Marker:
    """Distribution for one biomarker within an archetype.

    ``response`` is applied only if the patient has a documented intervention
    (statin / lifestyle), ramping in over ``response_ramp`` years after it.
    """

    baseline_mean: float
    baseline_sd: float
    slope_mean: float          # per-year drift
    slope_sd: float
    noise_sd: float            # per-observation noise
    lo: float                  # hard clamp (keeps values in the preventive zone)
    hi: float
    round_to: int              # decimals; 0 -> int
    response: float = 0.0      # signed delta at full ramp (negative == improves)
    response_ramp: float = 1.5


# A profile: how each archetype's story is parameterised.
@dataclass(frozen=True)
class _Profile:
    descriptor: str
    age_lo: int
    age_hi: int
    intervention_year: float | None   # when a statin / lifestyle change starts
    statin: bool
    family_history: tuple[tuple[str, str, int, int], ...]  # (relation, condition, onset_lo, onset_hi)
    markers: dict[str, _Marker]


# vitals shared shape helpers — hr / height are non-reasoning, kept simple.
def _hr(rng: random.Random) -> int:
    return rng.randint(60, 82)


_PROFILES: dict[str, _Profile] = {
    "cardiometabolic-drift": _Profile(
        descriptor="a gradual cardiometabolic drift rather than any acute event",
        age_lo=42, age_hi=60, intervention_year=2.0, statin=True,
        family_history=(
            ("father", "type 2 diabetes", 48, 56),
            ("father", "coronary artery disease", 58, 64),
            ("mother", "hypertension", 50, 60),
            ("paternal_grandfather", "myocardial infarction", 60, 68),
        ),
        markers={
            "hba1c_pct": _Marker(5.5, 0.15, 0.28, 0.05, 0.05, 5.2, 6.4, 1, response=-0.25, response_ramp=1.5),
            "fasting_glucose_mgdl": _Marker(96, 3, 6.0, 1.0, 2.5, 88, 124, 0, response=-4, response_ramp=1.5),
            "ldl_mgdl": _Marker(128, 6, 8.0, 1.5, 3.0, 95, 175, 0, response=-34, response_ramp=1.0),
            "hdl_mgdl": _Marker(44, 3, -0.5, 0.3, 1.5, 34, 60, 0),
            "triglycerides_mgdl": _Marker(155, 12, 8.0, 2.0, 6.0, 90, 240, 0, response=-25, response_ramp=1.0),
            "bp_systolic": _Marker(122, 3, 4.0, 0.8, 2.0, 112, 139, 0),
            "bp_diastolic": _Marker(78, 2, 2.5, 0.6, 1.5, 70, 89, 0),
            "bmi": _Marker(26.5, 0.6, 0.45, 0.15, 0.15, 22.0, 31.5, 1, response=-0.9, response_ramp=2.0),
        },
    ),
    "stable-healthy": _Profile(
        descriptor="a stable pattern with values staying within typical reference ranges",
        age_lo=35, age_hi=58, intervention_year=None, statin=False,
        family_history=(
            ("mother", "hypertension", 70, 78),
        ),
        markers={
            "hba1c_pct": _Marker(5.2, 0.12, 0.02, 0.03, 0.05, 4.9, 5.6, 1),
            "fasting_glucose_mgdl": _Marker(88, 3, 0.5, 0.5, 2.0, 78, 98, 0),
            "ldl_mgdl": _Marker(100, 6, 1.0, 1.0, 3.0, 70, 125, 0),
            "hdl_mgdl": _Marker(58, 4, 0.2, 0.3, 1.5, 45, 75, 0),
            "triglycerides_mgdl": _Marker(100, 10, 1.0, 1.5, 5.0, 60, 145, 0),
            "bp_systolic": _Marker(116, 3, 0.6, 0.6, 2.0, 106, 126, 0),
            "bp_diastolic": _Marker(74, 2, 0.3, 0.4, 1.5, 64, 79, 0),
            "bmi": _Marker(23.0, 0.6, 0.05, 0.1, 0.15, 20.0, 24.8, 1),
        },
    ),
    "improving-after-intervention": _Profile(
        descriptor="elevated baseline values that partially improve after a documented intervention",
        age_lo=40, age_hi=62, intervention_year=0.5, statin=True,
        family_history=(
            ("father", "type 2 diabetes", 50, 58),
            ("mother", "hypertension", 52, 62),
        ),
        markers={
            "hba1c_pct": _Marker(6.2, 0.12, 0.05, 0.04, 0.05, 5.6, 6.4, 1, response=-0.4, response_ramp=1.5),
            "fasting_glucose_mgdl": _Marker(114, 3, 1.0, 0.8, 2.5, 96, 124, 0, response=-10, response_ramp=1.5),
            "ldl_mgdl": _Marker(158, 7, 2.0, 1.5, 3.0, 100, 185, 0, response=-42, response_ramp=1.0),
            "hdl_mgdl": _Marker(40, 3, 0.5, 0.3, 1.5, 33, 58, 0, response=3, response_ramp=2.0),
            "triglycerides_mgdl": _Marker(190, 14, 2.0, 2.0, 6.0, 100, 245, 0, response=-40, response_ramp=1.5),
            "bp_systolic": _Marker(136, 3, 1.0, 0.8, 2.0, 118, 139, 0, response=-9, response_ramp=1.5),
            "bp_diastolic": _Marker(87, 2, 0.5, 0.5, 1.5, 76, 89, 0, response=-6, response_ramp=1.5),
            "bmi": _Marker(30.5, 0.6, 0.1, 0.15, 0.15, 25.0, 32.0, 1, response=-1.8, response_ramp=2.0),
        },
    ),
    "isolated-hypertension": _Profile(
        descriptor="an isolated upward drift in blood pressure with other markers near typical ranges",
        age_lo=38, age_hi=60, intervention_year=None, statin=False,
        family_history=(
            ("mother", "hypertension", 48, 56),
            ("father", "hypertension", 52, 60),
        ),
        markers={
            "hba1c_pct": _Marker(5.3, 0.12, 0.03, 0.03, 0.05, 5.0, 5.6, 1),
            "fasting_glucose_mgdl": _Marker(90, 3, 0.8, 0.6, 2.0, 80, 99, 0),
            "ldl_mgdl": _Marker(108, 6, 1.5, 1.0, 3.0, 80, 128, 0),
            "hdl_mgdl": _Marker(52, 4, 0.2, 0.3, 1.5, 42, 68, 0),
            "triglycerides_mgdl": _Marker(120, 10, 1.5, 1.5, 5.0, 70, 148, 0),
            "bp_systolic": _Marker(132, 3, 3.5, 0.8, 2.0, 122, 139, 0),
            "bp_diastolic": _Marker(85, 2, 2.0, 0.6, 1.5, 76, 89, 0),
            "bmi": _Marker(25.5, 0.6, 0.15, 0.12, 0.15, 22.0, 28.5, 1),
        },
    ),
    "lipid-plus-family-risk": _Profile(
        descriptor="a lipid-focused pattern against a strong family cardiovascular history",
        age_lo=40, age_hi=60, intervention_year=2.5, statin=True,
        family_history=(
            ("father", "coronary artery disease", 50, 58),
            ("paternal_grandfather", "myocardial infarction", 56, 66),
            ("mother", "high cholesterol", 50, 60),
        ),
        markers={
            "hba1c_pct": _Marker(5.4, 0.12, 0.04, 0.03, 0.05, 5.0, 5.6, 1),
            "fasting_glucose_mgdl": _Marker(92, 3, 1.0, 0.6, 2.0, 82, 99, 0),
            "ldl_mgdl": _Marker(152, 7, 6.0, 1.5, 3.0, 100, 185, 0, response=-38, response_ramp=1.0),
            "hdl_mgdl": _Marker(38, 3, -0.3, 0.3, 1.5, 32, 52, 0),
            "triglycerides_mgdl": _Marker(175, 12, 6.0, 2.0, 6.0, 100, 245, 0, response=-30, response_ramp=1.0),
            "bp_systolic": _Marker(120, 3, 1.5, 0.7, 2.0, 110, 132, 0),
            "bp_diastolic": _Marker(76, 2, 1.0, 0.5, 1.5, 68, 84, 0),
            "bmi": _Marker(26.0, 0.6, 0.15, 0.12, 0.15, 22.0, 29.0, 1),
        },
    ),
}


# ---------------------------------------------------------------------------
# Latent parameters
# ---------------------------------------------------------------------------
@dataclass
class LatentParams:
    """Everything needed to render one patient, realized from the archetype."""

    index: int
    seed: int
    archetype: str
    pseudonym: str
    age: int
    sex_at_birth: str
    height_cm: int
    n_encounters: int
    start_date: _dt.date
    intervention_year: float | None
    statin: bool
    descriptor: str
    family_history: list[dict[str, Any]]
    lifestyle: dict[str, str]
    # realized per-marker numbers: field -> (baseline, slope, response, ramp, noise, lo, hi, round_to)
    markers: dict[str, tuple[float, float, float, float, float, float, float, int]] = field(default_factory=dict)


def _rng_for(seed: int, index: int) -> random.Random:
    """Deterministic, order-independent per-patient RNG.

    ``random.Random`` seeded with a string uses a stable sha512-based scheme
    that is unaffected by PYTHONHASHSEED, so runs reproduce across processes.
    """
    return random.Random(f"phml-v1:{seed}:{index}")


def sample_latent(index: int, seed: int) -> LatentParams:
    """Realize one patient's latent parameters (deterministic in seed+index)."""
    rng = _rng_for(seed, index)
    archetype = ARCHETYPES[index % len(ARCHETYPES)]
    profile = _PROFILES[archetype]

    age = rng.randint(profile.age_lo, profile.age_hi)
    sex = rng.choice(("male", "female"))
    height = rng.randint(150, 165) if sex == "female" else rng.randint(165, 188)
    n_enc = rng.randint(4, 7)

    start_year = rng.choice((2021, 2022, 2023))
    start_date = _dt.date(start_year, rng.randint(1, 4), rng.randint(1, 28))

    family = [
        {
            "relation": rel,
            "condition": cond,
            "onset_age": rng.randint(lo, hi),
        }
        for (rel, cond, lo, hi) in profile.family_history
    ]

    lifestyle = {
        "smoking": rng.choice(_SMOKING),
        "alcohol": rng.choice(_ALCOHOL),
        "activity": rng.choice((
            "sedentary desk job",
            "light activity, walks a few times a week",
            "moderately active",
        )),
        "diet": rng.choice((
            "mixed; frequent takeout",
            "home-cooked most days",
            "self-reports high sugar intake, cutting back",
        )),
        "sleep": rng.choice(_SLEEP),
    }

    markers: dict[str, tuple[float, float, float, float, float, float, float, int]] = {}
    for name, m in profile.markers.items():
        baseline = rng.gauss(m.baseline_mean, m.baseline_sd)
        slope = rng.gauss(m.slope_mean, m.slope_sd)
        markers[name] = (
            baseline, slope, m.response, m.response_ramp, m.noise_sd, m.lo, m.hi, m.round_to,
        )

    return LatentParams(
        index=index,
        seed=seed,
        archetype=archetype,
        pseudonym=f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)} (fictional)",
        age=age,
        sex_at_birth=sex,
        height_cm=height,
        n_encounters=n_enc,
        start_date=start_date,
        intervention_year=profile.intervention_year,
        statin=profile.statin,
        descriptor=profile.descriptor,
        family_history=family,
        lifestyle=lifestyle,
        markers=markers,
    )


# ---------------------------------------------------------------------------
# Timeline construction
# ---------------------------------------------------------------------------
def _marker_value(
    rng: random.Random,
    spec: tuple[float, float, float, float, float, float, float, int],
    years: float,
    intervention_year: float | None,
) -> float | int:
    baseline, slope, response, ramp, noise_sd, lo, hi, round_to = spec
    val = baseline + slope * years
    if response and intervention_year is not None and years >= intervention_year:
        frac = min(1.0, (years - intervention_year) / ramp) if ramp > 0 else 1.0
        val += response * frac
    val += rng.gauss(0.0, noise_sd)
    val = max(lo, min(hi, val))
    return round(val, round_to) if round_to > 0 else int(round(val))


def build_timeline(params: LatentParams) -> dict[str, Any]:
    """Emit the full synthetic :class:`PatientTimeline` record for one patient."""
    rng = _rng_for(params.seed, params.index * 7 + 3)  # separate stream from sampling
    record_id = f"SYNTHETIC-GEN-{params.index + 1:04d}"

    # Encounter schedule: ~6-month spacing with small jitter.
    offsets = [i * 0.5 + rng.uniform(-0.04, 0.04) for i in range(params.n_encounters)]
    dates = [
        params.start_date + _dt.timedelta(days=round(y * 365.25))
        for y in offsets
    ]

    # Which encounter first documents the intervention (closest visit at/after it).
    statin_idx: int | None = None
    if params.statin and params.intervention_year is not None:
        candidates = [i for i, y in enumerate(offsets) if y >= params.intervention_year]
        statin_idx = candidates[0] if candidates else None

    events: list[dict[str, Any]] = []
    for i, (years, date) in enumerate(zip(offsets, dates)):
        def mk(name: str) -> float | int:
            return _marker_value(rng, params.markers[name], years, params.intervention_year)

        # Assign encounter type: first + roughly-annual are physicals; the
        # statin-start visit is a follow-up; the rest are lab-only.
        if i == 0 or i % 2 == 0:
            etype = "annual_physical"
        else:
            etype = "lab_only"
        if statin_idx is not None and i == statin_idx:
            etype = "follow_up_visit"

        event: dict[str, Any] = {"date": date.isoformat(), "type": etype}

        if etype == "annual_physical":
            bmi = mk("bmi")
            weight = round(bmi * (params.height_cm / 100.0) ** 2, 1)
            ldl, hdl, trig = mk("ldl_mgdl"), mk("hdl_mgdl"), mk("triglycerides_mgdl")
            total_chol = int(round(ldl + hdl + trig / 5.0))
            event["vitals"] = {
                "bp_systolic": mk("bp_systolic"),
                "bp_diastolic": mk("bp_diastolic"),
                "hr": _hr(rng),
                "weight_kg": weight,
                "bmi": bmi,
            }
            event["labs"] = {
                "hba1c_pct": mk("hba1c_pct"),
                "fasting_glucose_mgdl": mk("fasting_glucose_mgdl"),
                "ldl_mgdl": ldl,
                "hdl_mgdl": hdl,
                "triglycerides_mgdl": trig,
                "total_chol_mgdl": total_chol,
            }
            event["note"] = "Routine physical; full panel." if i else "Baseline visit; full panel."
        elif etype == "follow_up_visit":
            bmi = mk("bmi")
            weight = round(bmi * (params.height_cm / 100.0) ** 2, 1)
            event["vitals"] = {
                "bp_systolic": mk("bp_systolic"),
                "bp_diastolic": mk("bp_diastolic"),
                "hr": _hr(rng),
                "weight_kg": weight,
                "bmi": bmi,
            }
            event["labs"] = {"hba1c_pct": mk("hba1c_pct"), "ldl_mgdl": mk("ldl_mgdl")}
            note = "Follow-up visit."
            if statin_idx is not None and i == statin_idx:
                note = "Follow-up; atorvastatin 10mg started given LDL trend and family history; SYNTHETIC."
            event["note"] = note
        else:  # lab_only
            if i % 4 == 1:
                event["labs"] = {
                    "ldl_mgdl": mk("ldl_mgdl"),
                    "hdl_mgdl": mk("hdl_mgdl"),
                    "triglycerides_mgdl": mk("triglycerides_mgdl"),
                }
                event["note"] = "Interval lipid panel."
            else:
                event["labs"] = {
                    "fasting_glucose_mgdl": mk("fasting_glucose_mgdl"),
                    "ldl_mgdl": mk("ldl_mgdl"),
                }
                event["note"] = "Interval labs."

        events.append(event)

    # Medications.
    if params.statin and statin_idx is not None:
        statin_month = dates[statin_idx].strftime("%Y-%m")
        medications = [
            {"name": "none (baseline)", "start": dates[0].strftime("%Y-%m"),
             "stop": statin_month, "note": "no regular medications; SYNTHETIC"},
            {"name": "atorvastatin 10mg", "start": statin_month, "stop": None,
             "note": "started given LDL trend and family history; SYNTHETIC"},
        ]
    else:
        medications = [
            {"name": "none", "start": dates[0].strftime("%Y-%m"), "stop": None,
             "note": "no regular medications; SYNTHETIC"},
        ]

    # Immunizations: an influenza shot each autumn in range, plus one Tdap.
    immunizations: list[dict[str, str]] = []
    for yr in range(dates[0].year, dates[-1].year + 1):
        immunizations.append({"vaccine": "influenza", "date": f"{yr}-10-{rng.randint(5, 25):02d}"})
    immunizations.append({"vaccine": "Tdap booster", "date": f"{dates[0].year}-{rng.randint(2, 6):02d}-15"})

    record: dict[str, Any] = {
        "record_id": record_id,
        "synthetic": True,
        # Human-facing provenance is deliberately generic: the archetype / seed
        # (the answer key) live ONLY in synthetic_meta, which is stripped from
        # the model-facing input so the label cannot leak into training.
        "provenance": (
            f"Programmatically generated synthetic record ({GENERATOR_VERSION}). "
            "NOT a real person. No PHI."
        ),
        "synthetic_meta": {
            "archetype": params.archetype,
            "generator": GENERATOR_VERSION,
            "seed": params.seed,
            "index": params.index,
        },
        "demographics": {
            "pseudonym": params.pseudonym,
            "age": params.age,
            "sex_at_birth": params.sex_at_birth,
            "height_cm": params.height_cm,
            "notes": "Fictional record; values invented for illustration. No PHI.",
        },
        "family_history": params.family_history,
        "medications": medications,
        "immunizations": immunizations,
        "lifestyle": params.lifestyle,
        "timeline": events,
        "notes": (
            "SYNTHETIC record for pipeline / training use only. No claim of "
            "clinical accuracy for any real individual."
        ),
    }
    return record


# ---------------------------------------------------------------------------
# Gold 7-part output derivation (pure function of the emitted timeline)
# ---------------------------------------------------------------------------
_LAB_FIELDS = (
    "hba1c_pct", "fasting_glucose_mgdl", "ldl_mgdl", "hdl_mgdl",
    "triglycerides_mgdl", "total_chol_mgdl",
)
_VITAL_FIELDS = ("bp_systolic", "bp_diastolic", "weight_kg", "bmi", "hr")

_UNIT = {
    "hba1c_pct": "%",
    "fasting_glucose_mgdl": " mg/dL",
    "ldl_mgdl": " mg/dL",
    "hdl_mgdl": " mg/dL",
    "triglycerides_mgdl": " mg/dL",
    "total_chol_mgdl": " mg/dL",
    "bmi": "",
}


def _series(timeline: dict[str, Any], field_name: str) -> list[tuple[str, float]]:
    """Ordered (date, value) pairs for a lab or vital field across encounters."""
    out: list[tuple[str, float]] = []
    for ev in timeline.get("timeline", []):
        for bucket in ("labs", "vitals"):
            block = ev.get(bucket)
            if isinstance(block, dict) and field_name in block:
                out.append((ev.get("date", "?"), block[field_name]))
    return out


# Fields that read better with a fixed single decimal (e.g. HbA1c 6.0%, BMI 27.0).
_ONE_DECIMAL = {"hba1c_pct", "bmi"}


def _fmt_num(field_name: str, value: float) -> str:
    unit = _UNIT.get(field_name, "")
    if field_name in _ONE_DECIMAL:
        return f"{float(value):.1f}{unit}"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value}{unit}"


def _arrow(field_name: str, series: list[tuple[str, float]]) -> str:
    """Compact value chain, collapsing consecutive duplicates: 5.5 -> 5.9 -> 6.1."""
    vals: list[str] = []
    last: float | None = None
    for _, v in series:
        if v != last:
            vals.append(_fmt_num(field_name, v))
            last = v
    return " -> ".join(vals)


@dataclass
class _Signal:
    key: str
    headline: str
    evidence: list[str]


def _detect_signals(timeline: dict[str, Any]) -> list[_Signal]:
    """Read the emitted timeline and return the risk signals it supports.

    Purely descriptive / range-based — never asserts a diagnosis.
    """
    signals: list[_Signal] = []

    hba1c = _series(timeline, "hba1c_pct")
    glucose = _series(timeline, "fasting_glucose_mgdl")
    ldl = _series(timeline, "ldl_mgdl")
    hdl = _series(timeline, "hdl_mgdl")
    trig = _series(timeline, "triglycerides_mgdl")
    sys = _series(timeline, "bp_systolic")
    dia = _series(timeline, "bp_diastolic")
    bmi = _series(timeline, "bmi")

    # --- glycemic ---
    hi_a1c = [(d, v) for d, v in hba1c if v >= 5.7]
    hi_glu = [(d, v) for d, v in glucose if v >= 100]
    if hi_a1c or hi_glu:
        rising = len(hba1c) >= 2 and hba1c[-1][1] - hba1c[0][1] >= 0.2
        head = "Glycemic drift: HbA1c / fasting glucose sit in the prediabetes reference band"
        head += " on a rising trajectory." if rising else "."
        ev = [f"HbA1c {_fmt_num('hba1c_pct', v)} ({d})" for d, v in hi_a1c]
        ev += [f"fasting glucose {_fmt_num('fasting_glucose_mgdl', v)} ({d})" for d, v in hi_glu]
        signals.append(_Signal("glycemic", head, ev[:5]))

    # --- lipids ---
    hi_ldl = [(d, v) for d, v in ldl if v >= 130]
    lo_hdl = [(d, v) for d, v in hdl if v < 40]
    hi_trig = [(d, v) for d, v in trig if v >= 150]
    if hi_ldl or lo_hdl or hi_trig:
        statin = any("atorvastatin" in (m.get("name") or "").lower()
                     for m in timeline.get("medications", []))
        improved = ""
        if statin and len(ldl) >= 2:
            peak = max(v for _, v in ldl)
            later = ldl[-1][1]
            if peak - later >= 15:
                improved = (f" LDL later reads {_fmt_num('ldl_mgdl', later)} after a statin was "
                            "started (a single improved value, not a durable trend).")
        head = "Lipid signal: LDL / lipid panel in the borderline-high band against family history." + improved
        ev = [f"LDL {_fmt_num('ldl_mgdl', v)} ({d})" for d, v in hi_ldl[:3]]
        ev += [f"HDL {_fmt_num('hdl_mgdl', v)} ({d})" for d, v in lo_hdl[:2]]
        ev += [f"triglycerides {_fmt_num('triglycerides_mgdl', v)} ({d})" for d, v in hi_trig[:2]]
        signals.append(_Signal("lipid", head, ev[:5]))

    # --- blood pressure ---
    hi_sys = [(d, v) for d, v in sys if v >= 130]
    hi_dia = [(d, v) for d, v in dia if v >= 80]
    if hi_sys or hi_dia:
        sys_map = dict(sys)
        dia_map = dict(dia)
        ev = []
        for d, v in (hi_sys or hi_dia):
            ev.append(f"{int(sys_map.get(d, 0))}/{int(dia_map.get(d, 0))} mmHg ({d})")
        signals.append(_Signal(
            "bp",
            "Blood-pressure creep into the elevated / stage-1 range on single office readings.",
            ev[:4],
        ))

    # --- weight (only flagged when in the obese band) ---
    hi_bmi = [(d, v) for d, v in bmi if v >= 30]
    if hi_bmi:
        signals.append(_Signal(
            "weight",
            "BMI in the obese range across the record.",
            [f"BMI {_fmt_num('bmi', v)} ({d})" for d, v in hi_bmi[:3]],
        ))

    # --- family history context ---
    fam_terms = ("diabetes", "coronary", "myocardial", "infarction", "cholesterol", "hypertension")
    fam = [f for f in timeline.get("family_history", [])
           if any(t in (f.get("condition") or "").lower() for t in fam_terms)]
    strong_cardiac = [f for f in fam if any(t in (f.get("condition") or "").lower()
                                            for t in ("coronary", "myocardial", "infarction"))]
    if strong_cardiac:
        ev = [f"{f['relation']} {f['condition']}" + (f" @{f['onset_age']}" if f.get("onset_age") else "")
              for f in fam[:4]]
        signals.append(_Signal(
            "family",
            "Non-modifiable family cardiovascular risk (context, not a current finding).",
            ev,
        ))

    # --- clustering ---
    metabolic = {s.key for s in signals} & {"glycemic", "lipid", "bp"}
    if len(metabolic) >= 2:
        signals.append(_Signal(
            "clustering",
            "These signals co-occur (glucose / lipids / blood pressure), which is the "
            "pattern that matters more than any single value.",
            [],
        ))

    return signals


def _span_text(timeline: dict[str, Any]) -> tuple[str, str, str]:
    dates = [ev.get("date") for ev in timeline.get("timeline", []) if ev.get("date")]
    if not dates:
        return ("?", "?", "an unknown period")
    start, end = dates[0], dates[-1]
    try:
        d0 = _dt.date.fromisoformat(start)
        d1 = _dt.date.fromisoformat(end)
        years = round((d1 - d0).days / 365.25, 1)
        span = f"roughly {years} years"
    except ValueError:
        span = "the recorded period"
    return (start[:7], end[:7], span)


def derive_output_sections(timeline: dict[str, Any]) -> dict[str, str]:
    """Derive the gold 7-part output (markdown bodies keyed by section name).

    The output is a pure function of the emitted timeline plus the archetype
    latent carried in ``synthetic_meta`` — so it always cites real data points
    and never drifts from what the model will actually see.
    """
    meta = timeline.get("synthetic_meta", {}) or {}
    archetype = meta.get("archetype", "")
    descriptor = _PROFILES[archetype].descriptor if archetype in _PROFILES else "a longitudinal preventive-health picture"

    demo = timeline.get("demographics", {}) or {}
    age = demo.get("age", "unknown-age")
    start, end, span = _span_text(timeline)
    signals = _detect_signals(timeline)
    keys = {s.key for s in signals}

    # -- 1. longitudinal_summary --
    lines = [
        f"Over {span} ({start} to {end}), this synthetic {age}-year-old record shows "
        f"{descriptor}:",
        "",
    ]
    summary_fields = [
        ("hba1c_pct", "HbA1c"),
        ("fasting_glucose_mgdl", "Fasting glucose"),
        ("ldl_mgdl", "LDL cholesterol"),
        ("bmi", "BMI"),
    ]
    for fname, label in summary_fields:
        s = _series(timeline, fname)
        if len(s) >= 2:
            lines.append(f"- **{label}:** {_arrow(fname, s)}")
    sys = _series(timeline, "bp_systolic")
    dia = _series(timeline, "bp_diastolic")
    if len(sys) >= 2 and len(dia) >= 2:
        first = f"{int(sys[0][1])}/{int(dia[0][1])}"
        last = f"{int(sys[-1][1])}/{int(dia[-1][1])}"
        lines.append(f"- **Blood pressure:** {first} -> {last} mmHg (single readings)")
    longitudinal_summary = "\n".join(lines)

    # -- 2. risk_signals --
    if signals:
        risk_signals = "\n".join(f"- {s.headline}" for s in signals)
    else:
        risk_signals = (
            "- No concerning trajectory: the emitted values stay within typical "
            "reference ranges, so nothing here calls for more than routine "
            "age-appropriate preventive follow-up."
        )

    # -- 3. evidence --
    ev_signals = [s for s in signals if s.evidence]
    if ev_signals:
        rows = ["| Signal | Supporting data points |", "|---|---|"]
        for s in ev_signals:
            rows.append(f"| {s.key} | {'; '.join(s.evidence)} |")
        evidence = "\n".join(rows)
    else:
        evidence = (
            "No risk signals were detected, so there is nothing to substantiate: "
            "the labs and vitals across the record stay within typical reference ranges."
        )

    # -- 4. missing_information --
    missing = []
    if _series(timeline, "bp_systolic"):
        missing.append(
            "Blood pressure comes from single office readings — no home / ambulatory "
            "series, so any 'elevated' label is provisional."
        )
    missing.append("No kidney markers (eGFR, urine albumin/creatinine) are recorded.")
    if "lipid" in keys or "glycemic" in keys:
        missing.append("No formal 10-year cardiovascular-risk inputs are fully captured.")
    if any("atorvastatin" in (m.get("name") or "").lower() for m in timeline.get("medications", [])):
        missing.append("No medication-adherence or dose-response detail after the statin start.")
    missing.append("Fasting state and lab timing are not stated for every draw.")
    missing_information = "\n".join(f"- {m}" for m in missing)

    # -- 5. clinician_questions --
    questions = []
    if "glycemic" in keys:
        questions.append("Should HbA1c be rechecked sooner than the routine interval given the trend?")
    if "bp" in keys:
        questions.append("Would repeat or home blood-pressure monitoring change whether BP is acted on?")
    if "lipid" in keys or "family" in keys:
        questions.append("Are a fuller lipid / risk panel and kidney function warranted given the history?")
    questions.append("Is a structured lifestyle / prevention program indicated and available?")
    if not keys:
        questions = [
            "Is anything beyond routine age-appropriate preventive screening indicated at this time?",
            "Is a structured lifestyle / prevention program indicated and available?",
        ]
    clinician_questions = "\n".join(f"- {q}" for q in questions)

    # -- 6. safety_disclaimer (single source of truth) --
    safety_disclaimer = SAFETY_DISCLAIMER

    # -- 7. what_not_to_conclude --
    nots = []
    if "glycemic" in keys:
        nots.append(
            "Do not conclude the person has diabetes — the values sit in the prediabetes "
            "range and any diagnosis requires clinician assessment and confirmatory testing."
        )
    if "bp" in keys:
        nots.append("Do not treat single-visit blood-pressure readings as hypertension.")
    if any("atorvastatin" in (m.get("name") or "").lower() for m in timeline.get("medications", [])) \
            and ("lipid" in keys):
        nots.append("Do not conclude the statin 'fixed' anything — one improved value is not a durable trend.")
    if not keys:
        nots.append(
            "Do not read the absence of flags as a clean bill of health — this is a "
            "limited synthetic record, not a full evaluation."
        )
    nots.append("Do not infer causation from co-occurring trends.")
    nots.append("Do not use this output to start, stop, or change any medication.")
    what_not_to_conclude = "\n".join(f"- {n}" for n in nots)

    return {
        "longitudinal_summary": longitudinal_summary,
        "risk_signals": risk_signals,
        "evidence": evidence,
        "missing_information": missing_information,
        "clinician_questions": clinician_questions,
        "safety_disclaimer": safety_disclaimer,
        "what_not_to_conclude": what_not_to_conclude,
    }


_SECTION_HEADINGS = (
    ("longitudinal_summary", "1. Longitudinal summary"),
    ("risk_signals", "2. Risk signals"),
    ("evidence", "3. Evidence (what in the record supports each signal)"),
    ("missing_information", "4. Missing information (what would change the picture)"),
    ("clinician_questions", "5. Clinician questions (for the next visit)"),
    ("safety_disclaimer", "6. Safety disclaimer"),
    ("what_not_to_conclude", "7. What NOT to conclude"),
)


def render_output_markdown(sections: dict[str, str]) -> str:
    """Render the 7-part sections dict into gold-style markdown text."""
    # Fail loud if a section is missing — keeps train targets schema-complete.
    missing = [k for k in OUTPUT_SECTIONS if k not in sections]
    if missing:
        raise ValueError(f"cannot render, missing sections: {missing}")
    parts: list[str] = []
    for key, heading in _SECTION_HEADINGS:
        parts.append(f"## {heading}\n\n{sections[key].strip()}")
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Top-level generation
# ---------------------------------------------------------------------------
@dataclass
class GeneratedPatient:
    patient_id: str
    archetype: str
    timeline: dict[str, Any]
    output_sections: dict[str, str]
    output_text: str


def generate_patient(index: int, seed: int) -> GeneratedPatient:
    """Co-generate one patient's timeline and its gold 7-part reasoning."""
    params = sample_latent(index, seed)
    timeline = build_timeline(params)
    sections = derive_output_sections(timeline)
    text = render_output_markdown(sections)
    return GeneratedPatient(
        patient_id=timeline["record_id"],
        archetype=params.archetype,
        timeline=timeline,
        output_sections=sections,
        output_text=text,
    )


def generate_dataset(n: int, seed: int) -> list[GeneratedPatient]:
    """Generate ``n`` patients (balanced across archetypes by index)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return [generate_patient(i, seed) for i in range(n)]
