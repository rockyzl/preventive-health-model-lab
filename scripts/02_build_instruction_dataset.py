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
import hashlib
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
    validate_output_sections,
)
from preventive_health_model_lab.data.synthetic_generator import (  # noqa: E402
    derive_output_sections,
    render_output_markdown,
)
from preventive_health_model_lab.safety.disclaimer import (  # noqa: E402
    contains_diagnostic_language,
)

EXAMPLES_DIR = REPO_ROOT / "examples"
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_TIMELINES = SYNTHETIC_DIR / "timelines.jsonl"

# The task framing every training example shares. Locked with the ML engineer:
# dataset shape = {"instruction", "input", "output", "patient_id", "synthetic": true}.
INSTRUCTION = (
    "You are a preventive-health reasoning assistant. You are given a SYNTHETIC "
    "longitudinal patient record (labs, vitals, history across multiple visits). "
    "Produce a structured, NON-diagnostic preventive-health review. Do not diagnose, "
    "prescribe, or tell the patient what to do. Organize your answer into exactly "
    "these seven sections: Longitudinal summary; Risk signals; Evidence; Missing "
    "information; Clinician questions; Safety disclaimer; What NOT to conclude."
)


# --------------------------------------------------------------------------
# Skeleton function signatures — docstrings define the contract; bodies land
# in a later phase. Kept as NotImplementedError so accidental calls are loud.
# --------------------------------------------------------------------------
def timeline_to_instruction_record(timeline: dict[str, Any]) -> dict[str, Any]:
    """Turn one validated patient timeline into one instruction record.

    The gold ``output`` is DERIVED from the emitted timeline (7-part schema,
    correct-by-construction, non-diagnostic). The ``input`` is the timeline
    JSON with the ``synthetic_meta`` block dropped, so the archetype latent
    (the answer key) never leaks to the model — but ``synthetic: true`` stays.
    """
    sections = derive_output_sections(timeline)
    output_text = render_output_markdown(sections)

    input_view = {k: v for k, v in timeline.items() if k != "synthetic_meta"}
    input_str = json.dumps(input_view, ensure_ascii=False, indent=2)

    return {
        "instruction": INSTRUCTION,
        "input": input_str,
        "output": output_text,
        "patient_id": timeline["record_id"],
        "synthetic": True,
    }


def dedup_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop exact-duplicate records, keyed by a hash of (input, output)."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for r in records:
        key = hashlib.sha256(
            (r.get("input", "") + "\x00" + r.get("output", "")).encode("utf-8")
        ).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def build_splits(
    records: list[dict[str, Any]],
    *,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split records into train/val/test **by patient_id** (no leakage).

    Patients (not rows) are partitioned, so no ``patient_id`` can appear in
    more than one split. Returns a dict with keys ``train``, ``val``, ``test``.
    """
    import random

    by_patient: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_patient.setdefault(r["patient_id"], []).append(r)

    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)

    n = len(patients)
    n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
    n_test = max(1, int(round(n * test_frac))) if n >= 3 else 0
    # Never let val+test swallow train.
    n_val = min(n_val, max(0, n - 1))
    n_test = min(n_test, max(0, n - 1 - n_val))

    test_ids = set(patients[:n_test])
    val_ids = set(patients[n_test:n_test + n_val])

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for pid in patients:
        if pid in test_ids:
            bucket = "test"
        elif pid in val_ids:
            bucket = "val"
        else:
            bucket = "train"
        splits[bucket].extend(by_patient[pid])
    return splits


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records as JSONL (one compact object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_timelines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it first:\n"
            "  python scripts/01_download_or_prepare_data.py --generate --n 60 --seed 42"
        )
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i} is not valid JSON: {exc}") from exc
    return out


def build(timelines_path: Path, seed: int) -> int:
    """Read timelines -> instruction records -> validate -> split -> write."""
    from preventive_health_model_lab.data.schema import validate_patient_timeline

    timelines = _read_timelines(timelines_path)
    if not timelines:
        print(f"No timelines found in {timelines_path}", file=sys.stderr)
        return 1

    records: list[dict[str, Any]] = []
    for tl in timelines:
        # Defence in depth: re-validate the timeline (synthetic gate) here too.
        tl_errors = validate_patient_timeline(tl)
        if tl_errors:
            print(f"FAIL  timeline {tl.get('record_id')}: {tl_errors}", file=sys.stderr)
            return 1

        record = timeline_to_instruction_record(tl)

        rec_errors = validate_instruction_record(record)
        if rec_errors:
            print(f"FAIL  record {record.get('patient_id')}: {rec_errors}", file=sys.stderr)
            return 1

        missing = validate_output_sections(record["output"])
        if missing:
            print(f"FAIL  record {record['patient_id']}: missing sections {missing}", file=sys.stderr)
            return 1

        flags = contains_diagnostic_language(record["output"])
        if flags:
            print(f"FAIL  record {record['patient_id']}: diagnostic red flags {flags}", file=sys.stderr)
            return 1

        records.append(record)

    before = len(records)
    records = dedup_records(records)
    deduped = before - len(records)

    splits = build_splits(records, seed=seed)

    # Hard invariant: zero patient overlap across splits.
    id_sets = {name: {r["patient_id"] for r in rows} for name, rows in splits.items()}
    overlap = (
        (id_sets["train"] & id_sets["val"])
        | (id_sets["train"] & id_sets["test"])
        | (id_sets["val"] & id_sets["test"])
    )
    if overlap:
        print(f"FAIL  patient leakage across splits: {sorted(overlap)}", file=sys.stderr)
        return 1

    for name in ("train", "val", "test"):
        write_jsonl(splits[name], PROCESSED_DIR / f"{name}.jsonl")

    print(f"Built instruction dataset from {timelines_path}")
    print(f"  input timelines : {len(timelines)}")
    print(f"  records (post-dedup) : {len(records)}  (dropped {deduped} duplicate(s))")
    print("  patient-level split (no leakage):")
    for name in ("train", "val", "test"):
        rows = splits[name]
        print(f"    - {name:<5} {len(rows):>4} records  /  {len(id_sets[name]):>3} patients "
              f"-> {(PROCESSED_DIR / f'{name}.jsonl').relative_to(REPO_ROOT)}")
    print("  every output: 7 sections present + diagnostic-red-flag scan clean.")
    return 0


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
    parser.add_argument(
        "--build",
        action="store_true",
        help="build train/val/test.jsonl from a timelines JSONL",
    )
    parser.add_argument(
        "--timelines",
        type=Path,
        default=DEFAULT_TIMELINES,
        help="input timelines JSONL (with --build)",
    )
    parser.add_argument("--seed", type=int, default=42, help="split RNG seed (with --build)")
    args = parser.parse_args()

    if args.validate_examples:
        return validate_examples()
    if args.build:
        return build(timelines_path=args.timelines, seed=args.seed)

    print(
        "Nothing to do. Choose a mode:\n"
        "  --validate-examples   validate examples/*.json against the timeline schema\n"
        "  --build               build train/val/test.jsonl from data/synthetic/timelines.jsonl"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
