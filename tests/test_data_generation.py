"""Tests for the v1 synthetic data generator and the instruction-build path.

These assert the properties the dataset MUST hold before any training run:
  - the generator is deterministic under a fixed seed,
  - every generated timeline passes ``validate_patient_timeline``,
  - every instruction record passes ``validate_instruction_record`` and
    ``validate_output_sections`` (all 7 sections present),
  - every gold output is NON-diagnostic (diagnostic red-flag scanner is clean),
  - the patient-level split leaks zero patients across train/val/test.

No GPU stack, no network, no model.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT / "src"))

from preventive_health_model_lab.data import synthetic_generator as gen  # noqa: E402
from preventive_health_model_lab.data.schema import (  # noqa: E402
    validate_instruction_record,
    validate_output_sections,
    validate_patient_timeline,
)
from preventive_health_model_lab.safety.disclaimer import (  # noqa: E402
    contains_diagnostic_language,
)


def _load_build_module():
    """Import scripts/02_build_instruction_dataset.py (name starts with a digit)."""
    path = SCRIPTS / "02_build_instruction_dataset.py"
    spec = importlib.util.spec_from_file_location("build_dataset_mod", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SEED = 42
N = 40


@pytest.fixture(scope="module")
def patients():
    return gen.generate_dataset(N, SEED)


def test_generator_is_deterministic():
    a = gen.generate_patient(7, SEED)
    b = gen.generate_patient(7, SEED)
    assert json.dumps(a.timeline, sort_keys=True) == json.dumps(b.timeline, sort_keys=True)
    assert a.output_text == b.output_text


def test_generator_differs_across_seeds():
    a = gen.generate_patient(3, SEED)
    b = gen.generate_patient(3, SEED + 1)
    assert a.timeline != b.timeline


def test_all_archetypes_are_represented(patients):
    seen = {p.archetype for p in patients}
    assert seen == set(gen.ARCHETYPES), seen


def test_every_timeline_validates(patients):
    for p in patients:
        errors = validate_patient_timeline(p.timeline)
        assert errors == [], (p.patient_id, errors)
        assert p.timeline["synthetic"] is True
        assert p.timeline["record_id"].startswith("SYNTHETIC-")
        assert len(p.timeline["timeline"]) >= 3


def test_every_output_has_seven_sections(patients):
    for p in patients:
        missing = validate_output_sections(p.output_text)
        assert missing == [], (p.patient_id, missing)


def test_every_output_is_non_diagnostic(patients):
    for p in patients:
        flags = contains_diagnostic_language(p.output_text)
        assert flags == [], (p.patient_id, p.archetype, flags)


def test_disclaimer_is_reused_verbatim(patients):
    from preventive_health_model_lab.safety.disclaimer import SAFETY_DISCLAIMER

    for p in patients:
        assert SAFETY_DISCLAIMER in p.output_text


def test_stable_archetype_does_not_over_flag():
    # A stable-healthy patient must not invent risk signals.
    idx = gen.ARCHETYPES.index("stable-healthy")
    p = gen.generate_patient(idx, SEED)
    signals = gen._detect_signals(p.timeline)
    assert signals == [], [s.headline for s in signals]
    assert "No concerning trajectory" in p.output_sections["risk_signals"]


def test_synthetic_meta_is_stripped_from_model_input(patients):
    mod = _load_build_module()
    for p in patients:
        record = mod.timeline_to_instruction_record(p.timeline)
        # archetype label must not leak into what the model sees
        assert "synthetic_meta" not in record["input"]
        assert p.archetype not in record["input"]
        # but the synthetic flag must survive
        assert '"synthetic": true' in record["input"]


def test_instruction_records_validate(patients):
    mod = _load_build_module()
    for p in patients:
        record = mod.timeline_to_instruction_record(p.timeline)
        assert validate_instruction_record(record) == []
        assert validate_output_sections(record["output"]) == []
        assert record["synthetic"] is True
        assert record["patient_id"] == p.timeline["record_id"]


def test_patient_level_split_has_zero_overlap(patients):
    mod = _load_build_module()
    records = [mod.timeline_to_instruction_record(p.timeline) for p in patients]
    splits = mod.build_splits(records, seed=SEED)

    assert set(splits) == {"train", "val", "test"}
    id_sets = {name: {r["patient_id"] for r in rows} for name, rows in splits.items()}

    assert id_sets["train"] & id_sets["val"] == set()
    assert id_sets["train"] & id_sets["test"] == set()
    assert id_sets["val"] & id_sets["test"] == set()

    # every record accounted for, no patient dropped
    total = sum(len(rows) for rows in splits.values())
    assert total == len(records)
    assert id_sets["train"] and id_sets["val"] and id_sets["test"]


def test_dedup_removes_exact_duplicates(patients):
    mod = _load_build_module()
    records = [mod.timeline_to_instruction_record(p.timeline) for p in patients]
    doubled = records + records[:5]  # 5 exact dupes
    unique = mod.dedup_records(doubled)
    assert len(unique) == len(records)
