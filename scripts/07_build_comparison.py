#!/usr/bin/env python3
"""Assemble the head-to-head comparison + public-demo artifacts from predictions.

Reads the four prediction files in ``outputs/predictions/`` (base/QLoRA x
Gemma3/MedGemma), scores every record with the validated evaluation metrics, and
writes:

  outputs/evaluation/<condition>_scores.json   per-record + aggregate
  outputs/evaluation/comparison_summary.csv    aggregate head-to-head
  outputs/evaluation/comparison_by_archetype.csv
  outputs/evaluation/failure_cases.jsonl       every hard-fail / hallucination
  demo_artifacts/synthetic_cases.json          timelines + gold (for the viewer)
  demo_artifacts/model_outputs.json            4 outputs per case
  demo_artifacts/evaluation_summary.json       scores per case + aggregate
  demo_artifacts/comparison_summary.csv        (copy)
  demo_artifacts/selected_failure_cases.json   honest failure highlights

No model, no GPU, no network — it scores already-generated text.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from preventive_health_model_lab.evaluation.metrics import score_record  # noqa: E402

PRED_DIR = REPO_ROOT / "outputs" / "predictions"
EVAL_DIR = REPO_ROOT / "outputs" / "evaluation"
DEMO_DIR = REPO_ROOT / "demo_artifacts"
CONDITIONS = ["gemma3_base", "gemma3_qlora", "medgemma_base", "medgemma_qlora"]


def _load(cond: str) -> list[dict]:
    p = PRED_DIR / f"{cond}_test_predictions.jsonl"
    if not p.exists():
        raise SystemExit(f"missing predictions: {p}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _score_one(rec: dict) -> dict:
    rs = score_record(rec.get("input", ""), rec.get("prediction", ""), rec.get("patient_id", ""))
    auto = {n: d for n, d in rs.dimensions.items() if d.auto}
    grounding = rs.dimensions.get("numeric_grounding")
    ungrounded = (grounding.detail.get("ungrounded", []) if grounding else [])
    return {
        "patient_id": rec.get("patient_id", ""),
        "archetype": rec.get("archetype", "unknown"),
        "dims": {n: d.score for n, d in auto.items()},
        "overall": rs.overall_auto_score,
        "hard_fail": rs.hard_fail,
        "hard_fail_reasons": rs.hard_fail_reasons,
        "ungrounded_numbers": ungrounded,
    }


def _agg(scores: list[dict], dim_names: list[str]) -> dict:
    n = len(scores)
    return {
        "n": n,
        "dims": {d: round(mean(s["dims"].get(d, 0.0) for s in scores), 4) for d in dim_names},
        "overall_auto_score": round(mean(s["overall"] for s in scores), 4),
        "hard_fail_rate": round(sum(1 for s in scores if s["hard_fail"]) / n, 4),
        "n_hallucinated_numbers": sum(len(s["ungrounded_numbers"]) for s in scores),
    }


def main() -> int:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    preds = {c: _load(c) for c in CONDITIONS}
    scored = {c: [_score_one(r) for r in preds[c]] for c in CONDITIONS}
    dim_names = sorted(next(iter(scored.values()))[0]["dims"].keys())

    # gold ceiling (score the gold answers themselves)
    gold_recs = [{"input": r["input"], "prediction": r.get("gold", ""),
                  "patient_id": r["patient_id"], "archetype": r.get("archetype", "unknown")}
                 for r in preds[CONDITIONS[0]]]
    gold_scored = [_score_one(r) for r in gold_recs]

    # --- per-condition score files + aggregates ---
    aggregates = {}
    for c in CONDITIONS:
        aggregates[c] = _agg(scored[c], dim_names)
        (EVAL_DIR / f"{c}_scores.json").write_text(
            json.dumps({"condition": c, "aggregate": aggregates[c], "per_record": scored[c]},
                       indent=2, ensure_ascii=False), encoding="utf-8")
    aggregates["gold_ceiling"] = _agg(gold_scored, dim_names)

    # --- comparison_summary.csv ---
    cols = ["condition"] + dim_names + ["overall_auto_score", "hard_fail_rate", "n_hallucinated_numbers"]
    summary_rows = []
    for c in list(CONDITIONS) + ["gold_ceiling"]:
        a = aggregates[c]
        row = {"condition": c, **a["dims"], "overall_auto_score": a["overall_auto_score"],
               "hard_fail_rate": a["hard_fail_rate"], "n_hallucinated_numbers": a["n_hallucinated_numbers"]}
        summary_rows.append(row)
    for dest in (EVAL_DIR / "comparison_summary.csv", DEMO_DIR / "comparison_summary.csv"):
        with open(dest, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(summary_rows)

    # --- comparison_by_archetype.csv ---
    archetypes = sorted({s["archetype"] for c in CONDITIONS for s in scored[c]})
    with open(EVAL_DIR / "comparison_by_archetype.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["condition", "archetype", "n", *dim_names,
                                           "overall_auto_score", "hard_fail_rate"])
        w.writeheader()
        for c in CONDITIONS:
            for arch in archetypes:
                sub = [s for s in scored[c] if s["archetype"] == arch]
                if not sub:
                    continue
                a = _agg(sub, dim_names)
                w.writerow({"condition": c, "archetype": arch, "n": a["n"], **a["dims"],
                            "overall_auto_score": a["overall_auto_score"],
                            "hard_fail_rate": a["hard_fail_rate"]})

    # --- failure_cases.jsonl (hard-fail OR any hallucinated number OR overall<1) ---
    n_fail = 0
    with open(EVAL_DIR / "failure_cases.jsonl", "w", encoding="utf-8") as fh:
        for c in CONDITIONS:
            for s in scored[c]:
                if s["hard_fail"] or s["ungrounded_numbers"] or s["overall"] < 1.0:
                    fh.write(json.dumps({"condition": c, **s}, ensure_ascii=False) + "\n")
                    n_fail += 1

    # --- demo artifacts ---
    base = preds[CONDITIONS[0]]
    (DEMO_DIR / "synthetic_cases.json").write_text(json.dumps(
        [{"patient_id": r["patient_id"], "archetype": r.get("archetype", "unknown"),
          "timeline": r["input"], "gold": r.get("gold", "")} for r in base],
        indent=2, ensure_ascii=False), encoding="utf-8")

    by_pid_cond = {c: {r["patient_id"]: r["prediction"] for r in preds[c]} for c in CONDITIONS}
    (DEMO_DIR / "model_outputs.json").write_text(json.dumps(
        {r["patient_id"]: {c: by_pid_cond[c].get(r["patient_id"], "") for c in CONDITIONS}
         for r in base}, indent=2, ensure_ascii=False), encoding="utf-8")

    score_by_pid = {c: {s["patient_id"]: s for s in scored[c]} for c in CONDITIONS}
    (DEMO_DIR / "evaluation_summary.json").write_text(json.dumps(
        {"aggregate": aggregates,
         "per_case": {r["patient_id"]: {c: {
             "dims": score_by_pid[c][r["patient_id"]]["dims"],
             "overall": score_by_pid[c][r["patient_id"]]["overall"],
             "hard_fail": score_by_pid[c][r["patient_id"]]["hard_fail"],
             "ungrounded_numbers": score_by_pid[c][r["patient_id"]]["ungrounded_numbers"],
         } for c in CONDITIONS} for r in base}},
        indent=2, ensure_ascii=False), encoding="utf-8")

    # curated honest failure highlights for the demo (cap at 12)
    fails = []
    for c in CONDITIONS:
        for s in scored[c]:
            if s["hard_fail"] or s["ungrounded_numbers"]:
                fails.append({"condition": c, "patient_id": s["patient_id"],
                              "archetype": s["archetype"], "hard_fail": s["hard_fail"],
                              "hard_fail_reasons": s["hard_fail_reasons"],
                              "hallucinated_numbers": s["ungrounded_numbers"]})
    (DEMO_DIR / "selected_failure_cases.json").write_text(
        json.dumps(fails[:12], indent=2, ensure_ascii=False), encoding="utf-8")

    # --- console summary ---
    print(f"{'condition':16} " + " ".join(f"{d[:10]:>10}" for d in dim_names)
          + f" {'overall':>8} {'hardfail':>8} {'halluc#':>7}")
    for c in list(CONDITIONS) + ["gold_ceiling"]:
        a = aggregates[c]
        print(f"{c:16} " + " ".join(f"{a['dims'].get(d,0):>10.3f}" for d in dim_names)
              + f" {a['overall_auto_score']:>8.3f} {a['hard_fail_rate']:>8.3f}"
              + f" {a['n_hallucinated_numbers']:>7d}")
    print(f"\nfailure records written: {n_fail}")
    print("artifacts -> outputs/evaluation/ and demo_artifacts/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
