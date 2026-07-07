#!/usr/bin/env python3
"""Environment check for the Preventive Health Model Lab.

Runnable RIGHT NOW, with or without the GPU stack installed. Reports:
  - Python version
  - torch present? / CUDA available? (graceful if torch absent)
  - GPU name + VRAM (torch first, nvidia-smi fallback)
  - free disk space on the repo's drive
  - key training-stack packages present?

Prints a PASS / WARN / FAIL table and a "what to install next" hint.
Exit code is 0 unless a HARD requirement is missing (Python too old).

Usage:
    python scripts/00_check_environment.py
    python scripts/00_check_environment.py --json    # machine-readable
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MIN_PYTHON = (3, 10)
# VRAM (GB) we consider comfortable for 4B QLoRA. Below this -> WARN.
MIN_VRAM_GB = 7.0
MIN_FREE_DISK_GB = 40.0

# Packages the training stack needs. Presence-only check (no version gate here).
TRAINING_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "peft",
    "trl",
    "accelerate",
    "bitsandbytes",
    "mlflow",
)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def _pkg_installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def check_python() -> dict:
    ok = sys.version_info[:2] >= MIN_PYTHON
    return {
        "name": "Python version",
        "status": PASS if ok else FAIL,
        "detail": platform.python_version(),
        "hint": "" if ok else f"Need Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
    }


def check_torch_and_cuda() -> list[dict]:
    rows: list[dict] = []
    if not _pkg_installed("torch"):
        rows.append({
            "name": "PyTorch",
            "status": WARN,
            "detail": "not installed",
            "hint": "Install per pytorch.org for CUDA 13.x / Blackwell (see requirements.txt).",
        })
        rows.append({
            "name": "CUDA (via torch)",
            "status": WARN,
            "detail": "unknown (torch absent)",
            "hint": "Re-run this check after installing torch.",
        })
        return rows

    try:
        import torch  # type: ignore

        rows.append({
            "name": "PyTorch",
            "status": PASS,
            "detail": f"{torch.__version__}",
            "hint": "",
        })
        cuda_ok = bool(torch.cuda.is_available())
        rows.append({
            "name": "CUDA (via torch)",
            "status": PASS if cuda_ok else WARN,
            "detail": (
                f"available; torch cuda {getattr(torch.version, 'cuda', '?')}"
                if cuda_ok else "torch present but CUDA not available"
            ),
            "hint": "" if cuda_ok else "Check driver / CUDA build; Blackwell needs a recent wheel.",
        })
        if cuda_ok:
            try:
                props = torch.cuda.get_device_properties(0)
                vram_gb = props.total_memory / (1024**3)
                rows.append({
                    "name": "GPU (via torch)",
                    "status": PASS if vram_gb >= MIN_VRAM_GB else WARN,
                    "detail": f"{props.name} — {vram_gb:.1f} GB VRAM",
                    "hint": "" if vram_gb >= MIN_VRAM_GB else
                            f"< {MIN_VRAM_GB} GB: use 4-bit QLoRA, small batch, grad-checkpointing.",
                })
            except Exception as exc:  # pragma: no cover - hardware dependent
                rows.append({
                    "name": "GPU (via torch)",
                    "status": WARN,
                    "detail": f"query failed: {exc}",
                    "hint": "Falling back to nvidia-smi.",
                })
    except Exception as exc:  # pragma: no cover
        rows.append({
            "name": "PyTorch",
            "status": WARN,
            "detail": f"import failed: {exc}",
            "hint": "torch is installed but not importable in this interpreter.",
        })
    return rows


def check_gpu_via_smi() -> dict:
    """nvidia-smi fallback so we can report the GPU even without torch."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return {
            "name": "GPU (via nvidia-smi)",
            "status": WARN,
            "detail": "nvidia-smi not found",
            "hint": "No NVIDIA tooling on PATH; GPU status unknown.",
        }
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip()
        first = out.splitlines()[0] if out else "(no output)"
        # Parse "<name>, <NNNN> MiB" -> GB for a WARN threshold.
        status = PASS
        hint = ""
        if "MiB" in first:
            try:
                mib = float(first.split(",")[-1].strip().split()[0])
                if (mib / 1024.0) < MIN_VRAM_GB:
                    status, hint = WARN, "Low VRAM: rely on 4-bit QLoRA + small batch."
            except (ValueError, IndexError):
                pass
        return {"name": "GPU (via nvidia-smi)", "status": status, "detail": first, "hint": hint}
    except (subprocess.SubprocessError, OSError) as exc:
        return {
            "name": "GPU (via nvidia-smi)",
            "status": WARN,
            "detail": f"nvidia-smi failed: {exc}",
            "hint": "GPU present but query failed.",
        }


def check_disk() -> dict:
    usage = shutil.disk_usage(REPO_ROOT)
    free_gb = usage.free / (1024**3)
    ok = free_gb >= MIN_FREE_DISK_GB
    return {
        "name": "Free disk (repo drive)",
        "status": PASS if ok else WARN,
        "detail": f"{free_gb:.0f} GB free",
        "hint": "" if ok else f"Model weights + adapters want >= {MIN_FREE_DISK_GB:.0f} GB.",
    }


def check_training_packages() -> list[dict]:
    rows: list[dict] = []
    missing: list[str] = []
    for pkg in TRAINING_PACKAGES:
        present = _pkg_installed(pkg)
        if not present:
            missing.append(pkg)
        rows.append({
            "name": f"pkg: {pkg}",
            "status": PASS if present else WARN,
            "detail": "installed" if present else "missing",
            "hint": "",
        })
    if missing:
        rows.append({
            "name": "Training stack",
            "status": WARN,
            "detail": f"{len(missing)} package(s) missing",
            "hint": "pip install -r requirements.txt (and torch per pytorch.org).",
        })
    return rows


def gather() -> list[dict]:
    rows: list[dict] = [check_python()]
    rows.extend(check_torch_and_cuda())
    rows.append(check_gpu_via_smi())
    rows.append(check_disk())
    rows.extend(check_training_packages())
    return rows


def render_table(rows: list[dict]) -> str:
    name_w = max(len(r["name"]) for r in rows)
    lines = []
    header = f"{'CHECK'.ljust(name_w)}  STATUS  DETAIL"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(f"{r['name'].ljust(name_w)}  {r['status']:<6}  {r['detail']}")
    hints = [f"  - [{r['name']}] {r['hint']}" for r in rows if r["hint"]]
    if hints:
        lines.append("")
        lines.append("What to do next:")
        lines.extend(hints)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args()

    rows = gather()
    has_fail = any(r["status"] == FAIL for r in rows)
    has_warn = any(r["status"] == WARN for r in rows)

    if args.json:
        print(json.dumps({"rows": rows, "fail": has_fail, "warn": has_warn}, indent=2))
    else:
        print(render_table(rows))
        print()
        if has_fail:
            print("RESULT: FAIL — a hard requirement is unmet (see above).")
        elif has_warn:
            print("RESULT: WARN — runnable for scaffold/design work; GPU stack not fully ready.")
        else:
            print("RESULT: PASS — environment looks ready.")

    # Only a hard FAIL (e.g. Python too old) is a non-zero exit. WARN is fine:
    # Phase 1 scaffold work runs without the GPU stack by design.
    return 1 if has_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
