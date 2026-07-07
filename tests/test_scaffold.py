"""Phase 1 scaffold tests.

These exercise exactly the paths that are supposed to run TODAY:
  - the environment-check script exits 0,
  - the shipped synthetic timeline validates,
  - the safety gate rejects non-synthetic data,
  - the baseline --dry-run exits 0 and prints a plan,
  - the 7-part output validator recognizes the gold example.

No GPU stack, no network, no model.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
EXAMPLES = REPO_ROOT / "examples"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_env_check_runs_and_exits_zero():
    proc = _run(str(SCRIPTS / "00_check_environment.py"))
    assert proc.returncode == 0, proc.stderr
    assert "CHECK" in proc.stdout
    assert "RESULT:" in proc.stdout


def test_sample_timeline_validates():
    from preventive_health_model_lab.data.schema import load_and_validate_timeline

    record, errors = load_and_validate_timeline(EXAMPLES / "sample_patient_timeline.json")
    assert errors == [], f"unexpected validation errors: {errors}"
    assert record["record_id"] == "SYNTHETIC-001"
    assert record["synthetic"] is True
    assert len(record["timeline"]) >= 3


def test_safety_gate_rejects_non_synthetic():
    from preventive_health_model_lab.data.schema import validate_patient_timeline

    bad = {
        "record_id": "X",
        # missing/false synthetic flag on purpose
        "synthetic": False,
        "demographics": {"age": 40},
        "timeline": [{"date": "2024-01-01", "type": "lab_only"}],
    }
    errors = validate_patient_timeline(bad)
    assert any("synthetic" in e.lower() for e in errors), errors


def test_baseline_dry_run_exits_zero_with_plan():
    proc = _run(str(SCRIPTS / "03_run_baseline.py"), "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert "Baseline run plan" in proc.stdout
    assert "target output schema" in proc.stdout
    # dry-run must not have loaded a model / hit the network
    assert "No model loaded" in proc.stdout


def test_build_dataset_validate_examples_exits_zero():
    proc = _run(str(SCRIPTS / "02_build_instruction_dataset.py"), "--validate-examples")
    assert proc.returncode == 0, proc.stderr
    assert "SYNTHETIC-001" in proc.stdout
    assert "VALID" in proc.stdout


def test_gold_output_has_all_seven_sections():
    from preventive_health_model_lab.data.schema import validate_output_sections

    text = (EXAMPLES / "sample_model_output.md").read_text(encoding="utf-8")
    missing = validate_output_sections(text)
    assert missing == [], f"gold example is missing sections: {missing}"
