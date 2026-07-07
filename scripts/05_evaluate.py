#!/usr/bin/env python3
"""Score already-generated model outputs against the held-out test set.

This is the offline evaluator. It loads NO model, touches NO GPU, downloads
nothing — it scores TEXT that some model (base or fine-tuned) already produced,
on the automatic dimensions in
``preventive_health_model_lab.evaluation.metrics`` (schema conformance, verbatim
safety disclaimer, non-diagnostic language, and numeric grounding — the
faithfulness check that every cited lab/vital number is actually in the input).

Prediction file format (JSONL, one record per line):
    {"input": "<timeline JSON string, same as training input>",
     "prediction": "<model output text>",     # "output" also accepted
     "patient_id": "SYNTHETIC-GEN-0001"}

Modes:
    # sanity ceiling: score test.jsonl's own gold outputs (should be ~perfect)
    python scripts/05_evaluate.py --gold

    # score one prediction file
    python scripts/05_evaluate.py --predictions preds_base.jsonl --tag base

    # compare two prediction files side by side (e.g. base vs adapter)
    python scripts/05_evaluate.py --compare preds_base.jsonl preds_adapter.jsonl

Every mode prints a per-dimension table and writes both a Markdown report and a
machine-readable JSON to ``reports/`` (override with --out-dir).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from preventive_health_model_lab.evaluation.metrics import (  # noqa: E402
    JUDGE_REQUIRED_DIMENSIONS,
    RecordScore,
    record_score_to_dict,
    score_record,
)

DEFAULT_TEST_SET = REPO_ROOT / "data" / "processed" / "test.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "reports"

# The auto dimensions, in the order they appear in every table.
_AUTO_DIMS: tuple[str, ...] = (
    "schema_conformance",
    "disclaimer_present",
    "non_diagnostic",
    "numeric_grounding",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{i}: invalid JSON ({exc})") from exc
    return records


def _prediction_text(record: dict[str, Any], *, gold: bool) -> str:
    """Pull the text to score. Gold mode scores the record's own 'output'."""
    if gold:
        return str(record.get("output", ""))
    for key in ("prediction", "output", "generated", "text"):
        if record.get(key):
            return str(record[key])
    return ""


def score_file(path: Path, *, gold: bool) -> list[RecordScore]:
    """Score every record in a JSONL file, returning one RecordScore each."""
    scores: list[RecordScore] = []
    for i, rec in enumerate(_load_jsonl(path)):
        timeline = str(rec.get("input", ""))
        prediction = _prediction_text(rec, gold=gold)
        pid = str(rec.get("patient_id", f"record_{i}"))
        scores.append(score_record(timeline, prediction, pid))
    return scores


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(scores: list[RecordScore]) -> dict[str, Any]:
    """Corpus-level means / counts across a list of RecordScores."""
    n = len(scores)
    if n == 0:
        return {"n_records": 0}
    dim_means = {
        name: mean(s.dimensions[name].score for s in scores) for name in _AUTO_DIMS
    }
    return {
        "n_records": n,
        "overall_auto_score": mean(s.overall_auto_score for s in scores),
        "dimension_means": dim_means,
        "n_hard_fail": sum(1 for s in scores if s.hard_fail),
        "n_schema_perfect": sum(
            1 for s in scores if s.dimensions["schema_conformance"].passed
        ),
        "n_grounding_perfect": sum(
            1 for s in scores if s.dimensions["numeric_grounding"].passed
        ),
        "min_grounding": min(
            s.dimensions["numeric_grounding"].score for s in scores
        ),
        "records_with_ungrounded": [
            {
                "patient_id": s.patient_id,
                "ungrounded": s.dimensions["numeric_grounding"].detail["ungrounded"],
            }
            for s in scores
            if not s.dimensions["numeric_grounding"].passed
        ],
        "hard_fail_records": [
            {"patient_id": s.patient_id, "reasons": s.hard_fail_reasons}
            for s in scores
            if s.hard_fail
        ],
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_table(scores: list[RecordScore], agg: dict[str, Any]) -> str:
    """A fixed-width per-record table with a corpus-mean footer."""
    header = (
        f"{'patient_id':<22} {'schema':>8} {'disc':>5} {'nondiag':>8} "
        f"{'ground':>7} {'overall':>8} {'hardfail':>9}"
    )
    lines = [header, "-" * len(header)]
    for s in scores:
        d = s.dimensions
        schema = f"{d['schema_conformance'].detail['present']}/7"
        lines.append(
            f"{s.patient_id:<22} {schema:>8} "
            f"{('Y' if d['disclaimer_present'].passed else 'N'):>5} "
            f"{('Y' if d['non_diagnostic'].passed else 'N'):>8} "
            f"{d['numeric_grounding'].score:>7.3f} "
            f"{s.overall_auto_score:>8.3f} "
            f"{('YES' if s.hard_fail else '-'):>9}"
        )
    lines.append("-" * len(header))
    if agg.get("n_records"):
        dm = agg["dimension_means"]
        lines.append(
            f"{'MEAN':<22} {dm['schema_conformance']:>8.3f} "
            f"{dm['disclaimer_present']:>5.2f} {dm['non_diagnostic']:>8.2f} "
            f"{dm['numeric_grounding']:>7.3f} {agg['overall_auto_score']:>8.3f} "
            f"{agg['n_hard_fail']:>9}"
        )
    return "\n".join(lines)


def render_markdown(tag: str, scores: list[RecordScore], agg: dict[str, Any]) -> str:
    """The full Markdown report body."""
    out: list[str] = [
        f"# Evaluation report — `{tag}`",
        "",
        "Automatic offline scoring (no model / GPU). Dimensions: schema "
        "conformance (7 sections), verbatim safety disclaimer, non-diagnostic "
        "language (hard fail), numeric grounding (cited clinical numbers that "
        "appear in the input timeline).",
        "",
        "## Corpus summary",
        "",
        f"- Records scored: **{agg['n_records']}**",
    ]
    if agg.get("n_records"):
        dm = agg["dimension_means"]
        out += [
            f"- Overall auto score (mean): **{agg['overall_auto_score']:.3f}**",
            f"- Hard fails (diagnostic language / missing disclaimer): "
            f"**{agg['n_hard_fail']}**",
            f"- Schema 7/7: **{agg['n_schema_perfect']}/{agg['n_records']}**",
            f"- Numeric grounding 1.000: "
            f"**{agg['n_grounding_perfect']}/{agg['n_records']}** "
            f"(min grounding {agg['min_grounding']:.3f})",
            "",
            "| Dimension | Mean score |",
            "|---|---|",
            f"| schema_conformance | {dm['schema_conformance']:.3f} |",
            f"| disclaimer_present | {dm['disclaimer_present']:.3f} |",
            f"| non_diagnostic | {dm['non_diagnostic']:.3f} |",
            f"| numeric_grounding | {dm['numeric_grounding']:.3f} |",
        ]
    out += ["", "## Per-record", "", "```", render_table(scores, agg), "```", ""]

    if agg.get("records_with_ungrounded"):
        out += ["## Ungrounded (hallucinated) numbers", ""]
        for rec in agg["records_with_ungrounded"]:
            out.append(f"- **{rec['patient_id']}**")
            for u in rec["ungrounded"]:
                out.append(
                    f"  - `{u['raw']}` ({u['kind']}={u['value']}) — “{u['context']}”"
                )
        out.append("")
    if agg.get("hard_fail_records"):
        out += ["## Hard fails", ""]
        for rec in agg["hard_fail_records"]:
            out.append(f"- **{rec['patient_id']}**: {'; '.join(rec['reasons'])}")
        out.append("")

    out += [
        "## Not auto-scored (need a human / LLM judge)",
        "",
        "These dimensions are deliberately NOT machine-scored and are returned "
        "as `null`, never faked:",
        "",
    ]
    out += [f"- **{name}** — {desc}" for name, desc in JUDGE_REQUIRED_DIMENSIONS]
    out.append("")
    return "\n".join(out)


def _write_reports(
    tag: str, scores: list[RecordScore], agg: dict[str, Any], out_dir: Path
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"eval_{tag}.md"
    json_path = out_dir / f"eval_{tag}.json"
    md_path.write_text(render_markdown(tag, scores, agg), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "tag": tag,
                "summary": agg,
                "records": [record_score_to_dict(s) for s in scores],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return md_path, json_path


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------
def render_comparison(
    tag_a: str,
    scores_a: list[RecordScore],
    tag_b: str,
    scores_b: list[RecordScore],
) -> str:
    """Align two runs by patient_id and show overall + grounding deltas."""
    by_a = {s.patient_id: s for s in scores_a}
    by_b = {s.patient_id: s for s in scores_b}
    shared = [pid for pid in by_a if pid in by_b]

    header = (
        f"{'patient_id':<22} {tag_a[:10]+' ovr':>14} {tag_b[:10]+' ovr':>14} "
        f"{'Δoverall':>9} {'Δground':>8}"
    )
    lines = [header, "-" * len(header)]
    for pid in shared:
        a, b = by_a[pid], by_b[pid]
        d_ovr = b.overall_auto_score - a.overall_auto_score
        d_grd = (
            b.dimensions["numeric_grounding"].score
            - a.dimensions["numeric_grounding"].score
        )
        lines.append(
            f"{pid:<22} {a.overall_auto_score:>14.3f} {b.overall_auto_score:>14.3f} "
            f"{d_ovr:>+9.3f} {d_grd:>+8.3f}"
        )
    if shared:
        ma = mean(by_a[p].overall_auto_score for p in shared)
        mb = mean(by_b[p].overall_auto_score for p in shared)
        lines.append("-" * len(header))
        lines.append(
            f"{'MEAN':<22} {ma:>14.3f} {mb:>14.3f} {mb - ma:>+9.3f} {'':>8}"
        )
    only_a = [p for p in by_a if p not in by_b]
    only_b = [p for p in by_b if p not in by_a]
    if only_a or only_b:
        lines.append("")
        lines.append(f"(unmatched: {tag_a}-only={only_a}  {tag_b}-only={only_b})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _tag_from_path(path: Path) -> str:
    return path.stem


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--gold",
        action="store_true",
        help="score the test set's own gold outputs (sanity ceiling)",
    )
    mode.add_argument(
        "--predictions",
        type=Path,
        help="a JSONL prediction file to score",
    )
    mode.add_argument(
        "--compare",
        type=Path,
        nargs=2,
        metavar=("A", "B"),
        help="two prediction files to score side by side",
    )
    p.add_argument(
        "--test-set",
        type=Path,
        default=DEFAULT_TEST_SET,
        help=f"held-out test set for --gold (default {DEFAULT_TEST_SET})",
    )
    p.add_argument("--tag", default=None, help="label for report filenames")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args(argv)

    if args.compare:
        path_a, path_b = args.compare
        scores_a = score_file(path_a, gold=False)
        scores_b = score_file(path_b, gold=False)
        agg_a, agg_b = aggregate(scores_a), aggregate(scores_b)
        tag_a, tag_b = _tag_from_path(path_a), _tag_from_path(path_b)

        print(f"\n=== {tag_a} ===")
        print(render_table(scores_a, agg_a))
        print(f"\n=== {tag_b} ===")
        print(render_table(scores_b, agg_b))
        print(f"\n=== comparison ({tag_a} vs {tag_b}) ===")
        print(render_comparison(tag_a, scores_a, tag_b, scores_b))

        md_a, json_a = _write_reports(tag_a, scores_a, agg_a, args.out_dir)
        md_b, json_b = _write_reports(tag_b, scores_b, agg_b, args.out_dir)
        cmp_tag = args.tag or f"compare_{tag_a}_vs_{tag_b}"
        cmp_md = args.out_dir / f"eval_{cmp_tag}.md"
        cmp_md.write_text(
            "# Comparison — "
            f"`{tag_a}` vs `{tag_b}`\n\n```\n"
            + render_comparison(tag_a, scores_a, tag_b, scores_b)
            + "\n```\n",
            encoding="utf-8",
        )
        print(f"\nWrote {md_a}, {md_b}, {cmp_md}")
        return 0

    if args.gold:
        if not args.test_set.exists():
            raise SystemExit(f"test set not found: {args.test_set}")
        scores = score_file(args.test_set, gold=True)
        tag = args.tag or "gold"
    else:
        if not args.predictions.exists():
            raise SystemExit(f"prediction file not found: {args.predictions}")
        scores = score_file(args.predictions, gold=False)
        tag = args.tag or _tag_from_path(args.predictions)

    agg = aggregate(scores)
    print(render_table(scores, agg))
    md_path, json_path = _write_reports(tag, scores, agg, args.out_dir)
    print(f"\nWrote {md_path}")
    print(f"Wrote {json_path}")
    if agg.get("records_with_ungrounded"):
        print(
            f"\nWARNING: {len(agg['records_with_ungrounded'])} record(s) cite "
            "ungrounded (hallucinated) numbers — see report."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
