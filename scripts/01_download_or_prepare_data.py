#!/usr/bin/env python3
"""Prepare / stage source data for the instruction-dataset build.

This script performs NO network download. Its job is to stage *approved,
licensed, de-identified or synthetic* source material with a recorded
provenance. In Phase 2 the runnable path is the **synthetic generator**:

    python scripts/01_download_or_prepare_data.py \
        --generate --n 60 --seed 42 --out data/synthetic/timelines.jsonl

which writes N controlled synthetic patient timelines, each hard-validated
against ``validate_patient_timeline`` before it is written (any error aborts
the whole run). Non-synthetic corpora still require a verified license +
de-identification review and are not staged here.

SAFETY / DATA RULES (see data/README.md):
  - No real PHI. Ever.
  - Every source must have a recorded license + provenance.
  - Synthetic generators must clearly mark output as synthetic.

Usage:
    python scripts/01_download_or_prepare_data.py --list-sources
    python scripts/01_download_or_prepare_data.py --dry-run
    python scripts/01_download_or_prepare_data.py --generate --n 60 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make the src-layout package importable when run as a script.
sys.path.insert(0, str(REPO_ROOT / "src"))

RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_OUT = REPO_ROOT / "data" / "synthetic" / "timelines.jsonl"

# Registry of intended sources. Nothing is fetched in Phase 1 — this is the
# design contract for what MAY be staged, and under what terms.
SOURCE_REGISTRY: list[dict] = [
    {
        "id": "synthetic_timelines_v1",
        "kind": "synthetic",
        "license": "project-internal (generated)",
        "phi": False,
        "status": "available (--generate)",
        "note": "Controlled synthetic longitudinal records from synthetic_generator (v1). See examples/ for the shape.",
    },
    # Additional entries (public, licensed, de-identified corpora) go here
    # only AFTER their license + de-identification is verified and recorded.
]


def list_sources() -> None:
    print("Registered data sources (Phase 1 — none fetched yet):")
    for s in SOURCE_REGISTRY:
        print(f"  - {s['id']:<28} kind={s['kind']:<10} phi={s['phi']} status={s['status']}")
        print(f"      license: {s['license']}")
        print(f"      note:    {s['note']}")


def generate_synthetic(n: int, seed: int, out: Path, start: int = 0) -> int:
    """Generate ``n`` synthetic timelines, hard-validate each, write JSONL.

    Any validation error on any record aborts the whole run (non-zero exit)
    so malformed or non-synthetic data can never reach the dataset build.
    ``start`` offsets the patient index range (for disjoint held-out sets).
    """
    from preventive_health_model_lab.data.synthetic_generator import generate_dataset
    from preventive_health_model_lab.data.schema import validate_patient_timeline

    if n < 1:
        print("--n must be >= 1", file=sys.stderr)
        return 2

    patients = generate_dataset(n, seed, start=start)

    # Validate everything BEFORE writing a single line.
    archetypes: dict[str, int] = {}
    for p in patients:
        errors = validate_patient_timeline(p.timeline)
        if errors:
            print(f"FAIL  {p.patient_id}: timeline failed validation:", file=sys.stderr)
            for e in errors:
                print(f"        - {e}", file=sys.stderr)
            return 1
        archetypes[p.archetype] = archetypes.get(p.archetype, 0) + 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for p in patients:
            fh.write(json.dumps(p.timeline, ensure_ascii=False) + "\n")

    print(f"Generated {len(patients)} synthetic timelines (seed={seed}) -> {out}")
    print("Archetype balance:")
    for name, count in sorted(archetypes.items()):
        print(f"  - {name:<32} {count}")
    print("All records passed validate_patient_timeline (synthetic gate included).")
    return 0


def prepare(dry_run: bool) -> int:
    """Stage non-synthetic sources into data/raw/. Not implemented."""
    if dry_run:
        print("[dry-run] Would stage the following into", RAW_DIR)
        for s in SOURCE_REGISTRY:
            print(f"  - {s['id']} ({s['kind']})")
        print("[dry-run] No files written, no network access.")
        print("[dry-run] Synthetic data: use --generate --n <N> --seed <S>.")
        return 0
    raise NotImplementedError(
        "Non-synthetic data staging is not implemented. Run with --list-sources, "
        "--dry-run, or --generate (synthetic). Staging a real corpus requires a "
        "verified license + de-identification review for every non-synthetic source."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-sources", action="store_true", help="print the source registry and exit")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without writing anything")
    parser.add_argument("--generate", action="store_true", help="generate synthetic timelines and write JSONL")
    parser.add_argument("--n", type=int, default=60, help="number of synthetic patients (with --generate)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility (with --generate)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output JSONL path (with --generate)")
    parser.add_argument("--start-index", type=int, default=0,
                        help="patient index offset (with --generate); use to make a disjoint held-out set")
    args = parser.parse_args()

    if args.list_sources:
        list_sources()
        return 0
    if args.generate:
        return generate_synthetic(n=args.n, seed=args.seed, out=args.out, start=args.start_index)
    return prepare(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
