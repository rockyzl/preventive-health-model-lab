"""Schemas + validators for this project's two core data shapes.

1. ``PatientTimeline`` — a synthetic longitudinal record (see
   ``examples/sample_patient_timeline.json``). This is the *input* the
   model reasons over.

2. ``InstructionRecord`` — one ``{instruction, input, output}`` training
   example (see :mod:`...data` and ``scripts/02_build_instruction_dataset``).

The validators are pure-Python (no jsonschema dependency) so they run in
the light-tools environment and in CI without the GPU stack. They return
a list of human-readable error strings; an empty list means "valid".

SAFETY: every record MUST be flagged synthetic. ``validate_patient_timeline``
hard-fails if the ``synthetic`` marker is missing or false, so real PHI can
never quietly enter the pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# The 7-part output schema every model answer must contain.
# Kept here so training-target construction and eval scoring agree.
# ---------------------------------------------------------------------------
OUTPUT_SECTIONS: tuple[str, ...] = (
    "longitudinal_summary",
    "risk_signals",
    "evidence",
    "missing_information",
    "clinician_questions",
    "safety_disclaimer",
    "what_not_to_conclude",
)

# Top-level keys expected on a patient timeline.
_TIMELINE_REQUIRED: tuple[str, ...] = (
    "record_id",
    "synthetic",
    "demographics",
    "timeline",
)
_TIMELINE_OPTIONAL: tuple[str, ...] = (
    "family_history",
    "medications",
    "immunizations",
    "lifestyle",
    "notes",
)

# Keys expected on an instruction-tuning record.
_INSTRUCTION_REQUIRED: tuple[str, ...] = ("instruction", "input", "output", "patient_id")


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def validate_patient_timeline(record: dict[str, Any]) -> list[str]:
    """Validate one synthetic patient-timeline record.

    Returns a list of error strings (empty == valid).
    """
    errors: list[str] = []

    if not isinstance(record, dict):
        return [f"record must be a JSON object, got {type(record).__name__}"]

    for key in _TIMELINE_REQUIRED:
        if key not in record:
            errors.append(f"missing required key: '{key}'")

    # --- hard safety gate: must be explicitly synthetic ---
    if record.get("synthetic") is not True:
        errors.append(
            "SAFETY: 'synthetic' must be present and exactly true. "
            "Real or unlabeled data is not allowed in this pipeline."
        )

    if "record_id" in record and not _is_nonempty_str(record["record_id"]):
        errors.append("'record_id' must be a non-empty string")

    demo = record.get("demographics")
    if demo is not None and not isinstance(demo, dict):
        errors.append("'demographics' must be an object")
    elif isinstance(demo, dict):
        if "age" in demo and not isinstance(demo["age"], (int, float)):
            errors.append("'demographics.age' must be a number")

    timeline = record.get("timeline")
    if timeline is not None:
        if not isinstance(timeline, list) or not timeline:
            errors.append("'timeline' must be a non-empty list of events")
        else:
            for i, event in enumerate(timeline):
                if not isinstance(event, dict):
                    errors.append(f"timeline[{i}] must be an object")
                    continue
                if not _is_nonempty_str(event.get("date")):
                    errors.append(f"timeline[{i}] missing non-empty 'date'")
                if "type" not in event:
                    errors.append(f"timeline[{i}] missing 'type'")

    return errors


def validate_instruction_record(record: dict[str, Any]) -> list[str]:
    """Validate one instruction-tuning record ({instruction,input,output,...})."""
    errors: list[str] = []
    if not isinstance(record, dict):
        return [f"record must be a JSON object, got {type(record).__name__}"]

    for key in _INSTRUCTION_REQUIRED:
        if key not in record:
            errors.append(f"missing required key: '{key}'")
        elif not _is_nonempty_str(record[key]):
            errors.append(f"'{key}' must be a non-empty string")
    return errors


def validate_output_sections(output_text: str) -> list[str]:
    """Check that a model output string contains all 7 required sections.

    Matching is heading-based and lenient: we look for each section token
    (with '_' or ' ' interchangeable) as a heading anywhere in the text.
    Returns the list of MISSING section names.
    """
    lowered = output_text.lower()
    missing: list[str] = []
    for section in OUTPUT_SECTIONS:
        needle_underscore = section.lower()
        needle_spaced = section.replace("_", " ").lower()
        if needle_underscore not in lowered and needle_spaced not in lowered:
            missing.append(section)
    return missing


def load_and_validate_timeline(path: str | Path) -> tuple[dict[str, Any], list[str]]:
    """Load a timeline JSON file and validate it.

    Returns ``(record, errors)``. Raises only on unreadable / non-JSON files.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        record = json.load(fh)
    return record, validate_patient_timeline(record)
