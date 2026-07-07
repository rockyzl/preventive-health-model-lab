#!/usr/bin/env python3
"""Prepare / stage source data for the instruction-dataset build.

SKELETON (Phase 1). This script deliberately performs NO network download
in Phase 1. Its job, once implemented, is to stage *approved, licensed,
de-identified or synthetic* source material into ``data/raw/`` with a
provenance manifest.

SAFETY / DATA RULES (see data/README.md):
  - No real PHI. Ever.
  - Every source must have a recorded license + provenance.
  - Synthetic generators must clearly mark output as synthetic.

Usage (Phase 1):
    python scripts/01_download_or_prepare_data.py --list-sources
    python scripts/01_download_or_prepare_data.py --dry-run
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

# Registry of intended sources. Nothing is fetched in Phase 1 — this is the
# design contract for what MAY be staged, and under what terms.
SOURCE_REGISTRY: list[dict] = [
    {
        "id": "synthetic_timelines_v0",
        "kind": "synthetic",
        "license": "project-internal (generated)",
        "phi": False,
        "status": "planned",
        "note": "Programmatically generated longitudinal records. See examples/ for the shape.",
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


def prepare(dry_run: bool) -> int:
    """Stage sources into data/raw/. Not implemented in Phase 1."""
    if dry_run:
        print("[dry-run] Would stage the following into", RAW_DIR)
        for s in SOURCE_REGISTRY:
            print(f"  - {s['id']} ({s['kind']})")
        print("[dry-run] No files written, no network access.")
        return 0
    raise NotImplementedError(
        "Data staging is not implemented in Phase 1. "
        "Run with --list-sources or --dry-run. Implementing this requires a "
        "verified license + de-identification review for every non-synthetic source."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-sources", action="store_true", help="print the source registry and exit")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without writing anything")
    args = parser.parse_args()

    if args.list_sources:
        list_sources()
        return 0
    return prepare(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
