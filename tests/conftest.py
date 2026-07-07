"""Shared pytest fixtures / path setup for the scaffold tests."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Make the src-layout package importable without an install step.
sys.path.insert(0, str(REPO_ROOT / "src"))
