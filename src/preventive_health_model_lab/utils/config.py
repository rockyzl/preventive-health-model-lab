"""Tiny YAML config loader — the one piece of util the Phase 1 scripts share.

Kept dependency-light on purpose: only PyYAML, which is in the light-tools
block of requirements.txt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict.

    Raises FileNotFoundError with an actionable message if the config is
    missing, rather than a bare traceback.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Config not found: {p}\n"
            f"  cwd={Path.cwd()}\n"
            f"  Pass a path relative to the repo root, e.g. configs/eval.yaml"
        )
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config {p} did not parse to a mapping (got {type(data)}).")
    return data
