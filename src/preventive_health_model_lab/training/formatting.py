"""Instruction-record formatting — the single source of truth for prompts.

One record ``{instruction, input, output, ...}`` becomes model-facing text
here, and NOWHERE else, so the SFT training targets and the baseline
inference prompt can never drift apart:

  * ``build_prompt(record)``  -> the text fed to ``model.generate`` (ends at
    the "### Assessment" header; the model continues from there).
  * ``format_example(record)`` -> the full supervised string, i.e. exactly
    ``build_prompt(record)`` + the gold ``output`` + EOS. Because the training
    string is literally the inference prompt plus the target, what the model
    learns to continue is what the baseline runner will ask it to continue.

Deliberately tokenizer-free (a plain-text Alpaca-style template rather than a
model-specific chat template) so it is unit-testable offline with no model
download and behaves identically across base models.
"""
from __future__ import annotations

from typing import Any, Callable

from preventive_health_model_lab.data.schema import OUTPUT_SECTIONS

# Task framing shown to the model. Kept terse on purpose: the per-example
# specifics live in each record's ``instruction`` field (built by the data
# track). Naming the 7-part schema here nudges the model toward the structure
# the evaluation harness scores against (see schema.validate_output_sections).
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful preventive-health reasoning assistant. You read a "
    "SYNTHETIC longitudinal patient record and produce a structured, "
    "non-diagnostic assessment. Be conservative, surface uncertainty, and "
    "never state a diagnosis or prescribe treatment. Structure every answer "
    "with these sections: " + ", ".join(OUTPUT_SECTIONS) + "."
)

_PROMPT_TEMPLATE = (
    "{system}\n\n"
    "### Task\n{instruction}\n\n"
    "### Patient record\n{input}\n\n"
    "### Assessment\n"
)

# Shown when a record carries no separate structured ``input`` (the timeline is
# sometimes folded into ``instruction``). Keeps the template well-formed.
_MISSING_INPUT = "(no additional structured input provided)"


def build_prompt(
    record: dict[str, Any], *, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    """Render the INFERENCE prompt for one instruction record.

    Returns everything up to and including the ``### Assessment`` header — what
    the baseline runner feeds to ``model.generate``. ``format_example`` appends
    the gold output to this exact prefix; that shared prefix is what keeps
    training and inference consistent.
    """
    instruction = str(record.get("instruction", "")).strip()
    raw_input = record.get("input")
    body = str(raw_input).strip() if raw_input not in (None, "") else _MISSING_INPUT
    return _PROMPT_TEMPLATE.format(
        system=system_prompt, instruction=instruction, input=body
    )


def format_example(
    record: dict[str, Any],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    eos_token: str = "",
) -> str:
    """Render the full TRAINING string = prompt prefix + gold output + EOS.

    ``eos_token`` is appended verbatim so each example terminates cleanly; pass
    ``tokenizer.eos_token`` at train time (defaults to "" so the function stays
    tokenizer-free and testable offline).
    """
    output = str(record.get("output", "")).strip()
    return build_prompt(record, system_prompt=system_prompt) + output + eos_token


def make_formatting_func(
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    eos_token: str = "",
) -> Callable[[dict[str, Any]], str]:
    """Build a TRL ``formatting_func``: one record dict -> one training string.

    TRL 1.x applies this per example (``dataset.map(..., batched=False)``) and
    tokenizes the returned string into a ``text`` column. The tokenizer's
    ``eos_token`` is closed over here so the trainer never has to know about
    our template.
    """

    def _fmt(example: dict[str, Any]) -> str:
        return format_example(
            example, system_prompt=system_prompt, eos_token=eos_token
        )

    return _fmt
