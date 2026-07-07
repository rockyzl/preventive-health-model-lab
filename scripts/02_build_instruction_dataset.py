#!/usr/bin/env python3
"""Build the instruction-tuning dataset from staged patient timelines.

SKELETON (Phase 1). The transform logic (timeline -> instruction record)
is NOT implemented yet. What IS runnable now is the schema-validation path:

    python scripts/02_build_instruction_dataset.py --validate-examples

validates ``examples/sample_patient_timeline.json`` against the timeline
schema and exits non-zero if it fails. This is the piece the tests exercise.

Target record shape (one training example):
    {
      "patient_id": "SYNTHETIC-001",
      "instruction": "<task framing: longitudinal preventive-health reasoning>",
      "input":       "<serialized patient timeline>",
      "output":      "<7-part answer: see examples/sample_model_output.md>"
    }

Split discipline (see build_splits): patient-level, never leak a patient
across train/val/eval.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make the src-layout package importable when run as a script.
sys.path.insert(0, str(REPO_ROOT / "src"))

from preventive_health_model_lab.data.schema import (  # noqa: E402
    load_and_validate_timeline,
    validate_instruction_record,
)

EXAMPLES_DIR = REPO_ROOT / "examples"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


# --------------------------------------------------------------------------
# Skeleton function signatures — docstrings define the contract; bodies land
# in a later phase. Kept as NotImplementedError so accidental calls are loud.
# --------------------------------------------------------------------------
def timeline_to_instruction_record(timeline: dict[str, Any]) -> dict[str, Any]:
    """Turn one validated patient timeline into one instruction record.

    Serializes the timeline into ``input``, attaches the standard
    ``instruction`` framing, and (for supervised examples) the gold ``output``
    built to the 7-part schema. Not implemented in Phase 1.
    """
    raise NotImplementedError("timeline_to_instruction_record: Phase 2")


def dedup_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove near-duplicate records (e.g. by hashed input). Phase 2."""
    raise NotImplementedError("dedup_records: Phase 2")


def build_splits(
    records: list[dict[str, Any]],
    *,
    val_frac: float = 0.1,
    eval_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split records into train/val/eval **by patient_id** (no leakage).

    Guarantees no ``patient_id`` appears in more than one split. Returns a
    dict with keys ``train``, ``val``, ``eval``. Phase 2.
    """
    raise NotImplementedError("build_splits: Phase 2")


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records as JSONL. Phase 2 (used by the build pipeline)."""
    raise NotImplementedError("write_jsonl: Phase 2")


# --------------------------------------------------------------------------
# Runnable now: validate the shipped examples.
# --------------------------------------------------------------------------
def validate_examples() -> int:
    """Validate every example timeline; return process exit code."""
    files = sorted(EXAMPLES_DIR.glob("sample_patient_timeline*.json"))
    if not files:
        print(f"No example timelines found in {EXAMPLES_DIR}", file=sys.stderr)
        return 1

    any_fail = False
    for f in files:
        record, errors = load_and_validate_timeline(f)
        if errors:
            any_fail = True
            print(f"FAIL  {f.name}")
            for e in errors:
                print(f"        - {e}")
        else:
            n_events = len(record.get("timeline", []))
            print(f"OK    {f.name}  (record_id={record.get('record_id')}, {n_events} events)")

    print()
    print("VALID" if not any_fail else "INVALID — fix the errors above.")
    return 1 if any_fail else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-examples",
        action="store_true",
        help="validate examples/*.json against the timeline schema and exit",
    )
    args = parser.parse_args()

    if args.validate_examples:
        return validate_examples()

    print(
        "Phase 1 skeleton. The build pipeline is not implemented yet.\n"
        "Runnable today:  python scripts/02_build_instruction_dataset.py --validate-examples"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
