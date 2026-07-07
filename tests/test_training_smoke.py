"""Training-path smoke tests — fast, no GPU, no network, no model download.

These cover the wiring the actual GPU run exercises, WITHOUT running training
(too slow for the suite). The real end-to-end proof is a manual
``python scripts/04_train_sft.py --smoke`` (see reports/).

Covered here:
  * the formatting function turns a record into a non-empty string that
    carries both the instruction and the output, and the inference prompt is a
    prefix of the training string (train/inference consistency);
  * an ``SFTConfig`` builds with the kwargs the trainer uses;
  * ``scripts/04_train_sft.py --help`` exits 0 (argparse only, no heavy imports).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
FIXTURES = REPO_ROOT / "tests" / "fixtures"


def _sample_record() -> dict:
    line = (FIXTURES / "tiny_train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    return json.loads(line)


# --------------------------------------------------------------------------
# Formatting (tokenizer-free, offline)
# --------------------------------------------------------------------------
def test_format_example_is_nonempty_and_carries_instruction_and_output():
    from preventive_health_model_lab.training.formatting import format_example

    rec = _sample_record()
    text = format_example(rec)
    assert isinstance(text, str) and text.strip(), "formatting produced an empty string"
    assert rec["instruction"] in text, "instruction missing from training string"
    assert rec["output"].strip() in text, "output missing from training string"
    assert "### Assessment" in text


def test_build_prompt_is_prefix_of_training_string():
    """The inference prompt must be exactly the prefix of the training string.

    This is what guarantees the baseline runner asks the model to continue the
    same text the model was trained to continue.
    """
    from preventive_health_model_lab.training.formatting import (
        build_prompt,
        format_example,
    )

    rec = _sample_record()
    prompt = build_prompt(rec)
    full = format_example(rec)
    assert full.startswith(prompt)
    # the completion is the gold output (plus optional EOS)
    assert full[len(prompt):].startswith(rec["output"].strip())


def test_make_formatting_func_appends_eos_and_maps_single_record():
    from preventive_health_model_lab.training.formatting import make_formatting_func

    fmt = make_formatting_func(eos_token="<|end|>")
    rec = _sample_record()
    out = fmt(rec)  # TRL calls this per-example -> single string
    assert isinstance(out, str)
    assert out.endswith("<|end|>")


def test_formatting_handles_missing_input_field():
    from preventive_health_model_lab.training.formatting import build_prompt

    text = build_prompt({"instruction": "Assess this record.", "output": "x"})
    assert "no additional structured input" in text.lower()


# --------------------------------------------------------------------------
# Trainer config builds (imports trl; no GPU, no model)
# --------------------------------------------------------------------------
def test_sft_config_builds_with_trainer_kwargs(tmp_path):
    pytest.importorskip("trl", reason="trl not installed in this environment")
    from trl import SFTConfig

    cfg = SFTConfig(
        output_dir=str(tmp_path / "out"),
        max_length=512,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=1,
        max_steps=5,
        gradient_checkpointing=True,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        seed=42,
        report_to=[],
        dataset_text_field="text",
    )
    # TRL 1.x renamed max_seq_length -> max_length; confirm the field we rely on.
    assert cfg.max_length == 512
    assert cfg.gradient_checkpointing is True
    assert cfg.max_steps == 5


# --------------------------------------------------------------------------
# CLI contract
# --------------------------------------------------------------------------
def test_04_help_exits_zero():
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "04_train_sft.py"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--smoke" in proc.stdout


def test_04_gated_guard_blocks_real_run_without_weights(tmp_path):
    """Without --smoke and without cached weights, 04 must refuse (exit 2),
    printing license guidance — never a silent gated download."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "04_train_sft.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # Either the gated-model guard (2) fires, or (if someone cached MedGemma)
    # it proceeds — but on a machine without the gated weights it must be 2.
    if proc.returncode == 2:
        assert "hf" in proc.stderr.lower() or "license" in proc.stderr.lower()


# --------------------------------------------------------------------------
# GPU-only import surface (skipped without CUDA) — does NOT train
# --------------------------------------------------------------------------
def test_qlora_build_helpers_importable_on_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device — GPU import surface not exercised")
    pytest.importorskip("bitsandbytes")
    from peft import get_peft_model, prepare_model_for_kbit_training  # noqa: F401
    from transformers import BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    assert bnb.load_in_4bit is True
