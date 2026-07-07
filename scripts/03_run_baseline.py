#!/usr/bin/env python3
"""Run a baseline inference pass over synthetic patient timelines.

SKELETON (Phase 1), config-driven by ``configs/eval.yaml``.

Guard rails, by design:
  - NO silent network downloads. If the model weights are not already
    present locally, the script prints a clear, actionable message and
    exits non-zero (it never triggers a multi-GB pull on its own).
  - ``--dry-run`` resolves and prints the full plan (model, generation
    params, prompt files, safety flags) and exits 0 without loading any
    model. This is what the Phase 1 tests exercise.

Usage:
    python scripts/03_run_baseline.py --dry-run
    python scripts/03_run_baseline.py --config configs/eval.yaml --dry-run
    python scripts/03_run_baseline.py            # real run (Phase 2+, needs weights)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from preventive_health_model_lab.utils.config import load_yaml  # noqa: E402
from preventive_health_model_lab.data.schema import OUTPUT_SECTIONS  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "eval.yaml"


def resolve_prompt_files(cfg: dict) -> list[Path]:
    prompts = cfg.get("prompts", {})
    base = (REPO_ROOT / prompts.get("path", "examples/")).resolve()
    glob = prompts.get("glob", "*.json")
    return sorted(base.glob(glob))


def print_plan(cfg: dict, prompt_files: list[Path]) -> None:
    model = cfg.get("model", {})
    gen = cfg.get("generation", {})
    safety = cfg.get("safety", {})
    print("=== Baseline run plan (resolved) ===")
    print(f"  model_id        : {model.get('model_id')}")
    print(f"  adapter_path    : {model.get('adapter_path')}")
    print(f"  max_new_tokens  : {gen.get('max_new_tokens')}")
    print(f"  temperature     : {gen.get('temperature')}")
    print(f"  top_p           : {gen.get('top_p')}")
    print(f"  prompt files    : {len(prompt_files)} found")
    for f in prompt_files:
        print(f"      - {f.relative_to(REPO_ROOT)}")
    print(f"  require_disclaimer   : {safety.get('require_disclaimer')}")
    print(f"  forbid_diagnostic    : {safety.get('forbid_diagnostic_language')}")
    print(f"  target output schema : {', '.join(OUTPUT_SECTIONS)}")


def weights_available_locally(model_id: str) -> bool:
    """Best-effort check for a locally-cached model, WITHOUT importing torch.

    Looks in the HF cache dir for a matching snapshot. Returns False if we
    can't confirm — the caller then refuses to proceed (no silent pull).
    """
    if not model_id:
        return False
    # A local path that exists is trivially available.
    if Path(model_id).expanduser().exists():
        return True
    # HF hub cache: ~/.cache/huggingface/hub/models--<org>--<name>
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    slug = "models--" + model_id.replace("/", "--")
    return (cache / slug).exists()


def real_run(cfg: dict, prompt_files: list[Path]) -> int:
    model_id = cfg.get("model", {}).get("model_id", "")

    if not weights_available_locally(model_id):
        print(
            "\nModel weights are not available locally, and this script will NOT\n"
            "download them automatically.\n\n"
            f"  model_id: {model_id}\n\n"
            "To proceed (Phase 2+):\n"
            "  1. Accept the model license on Hugging Face (MedGemma is gated).\n"
            "  2. Authenticate:      huggingface-cli login\n"
            "  3. Pre-download once: huggingface-cli download "
            f"{model_id}\n"
            "  4. Re-run this script (or use --dry-run to preview the plan).\n",
            file=sys.stderr,
        )
        return 2

    if importlib.util.find_spec("transformers") is None:
        print(
            "transformers is not installed. Install the training stack first:\n"
            "  pip install -r requirements.txt   (and torch per pytorch.org)",
            file=sys.stderr,
        )
        return 2

    # Real generation is Phase 2+. Fail loud rather than pretend.
    raise NotImplementedError(
        "Baseline generation is not implemented in Phase 1. "
        "Weights and transformers are present — implement the HF pipeline call here."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to eval config YAML")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved plan and exit 0")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    prompt_files = resolve_prompt_files(cfg)

    if args.dry_run:
        print_plan(cfg, prompt_files)
        print("\n[dry-run] No model loaded, no network access. Exiting 0.")
        return 0

    return real_run(cfg, prompt_files)


if __name__ == "__main__":
    raise SystemExit(main())
