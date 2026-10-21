from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_summaries(root: str | Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in Path(root).rglob("summary.json")]


def rank_completed(summaries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank completed runs per task with feasibility before objective value."""
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        if summary.get("status") != "completed":
            continue
        best = summary.get("result", {}).get("best")
        if not best:
            continue
        lower_valid = best.get("lower_validation", {}).get("valid", True)
        rows.append(
            {
                "problem_id": summary["problem_id"],
                "optimizer": summary["optimizer"],
                "seed": summary["seed"],
                "run_id": summary["run_id"],
                "objective": float(best["objective"]),
                "feasible": bool(best.get("feasible", True) and lower_valid),
                "constraint_violation": float(best.get("constraint_violation", 0.0)),
            }
        )
    rows.sort(
        key=lambda row: (
            row["problem_id"],
            not row["feasible"],
            row["constraint_violation"],
            row["objective"],
        )
    )
    counters: dict[str, int] = {}
    for row in rows:
        counters[row["problem_id"]] = counters.get(row["problem_id"], 0) + 1
        row["rank"] = counters[row["problem_id"]]
    return rows
