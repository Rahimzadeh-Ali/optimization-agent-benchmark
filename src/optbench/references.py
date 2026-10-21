from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from optbench.base import BenchmarkProblem, BilevelProblem


@dataclass(frozen=True)
class ObjectiveReference:
    objective: float | None
    kind: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def objective_reference(problem: BenchmarkProblem | BilevelProblem) -> ObjectiveReference:
    """Return evaluator-only regret reference data; never include it in agent_spec."""
    problem_id = problem.problem_id
    if problem_id.startswith("D") and problem_id in {"D04", "D05"}:
        return ObjectiveReference(float(problem.bias), "defined_global", "CEC task bias at the hidden shift point")
    if problem_id in {"S04", "S05"}:
        return ObjectiveReference(float(problem.bias), "defined_global", "CEC task bias at the hidden shift point")
    if problem_id in {"N04", "N05"}:
        return ObjectiveReference(
            None,
            "noisy_validation_required",
            "A noiseless optimum is not a valid reference for individual noisy observations",
        )
    if problem_id in {"C04", "C05"}:
        front = np.asarray(problem._problem.pareto_front(), dtype=float).reshape(-1)
        return ObjectiveReference(float(front[0]), "published_global", "pymoo reference optimum")
    if problem_id in {"B01", "B02", "B03"}:
        return ObjectiveReference(0.0, "published_global", "Official SMD analytical optimum")
    if problem_id in {"B04", "B05"}:
        return ObjectiveReference(float(problem.known_solution[2]), "published_global", "BOLIB published global solution")
    if problem.family.startswith("bbob"):
        target = getattr(problem._problem, "final_target_fvalue1", None)
        if target is not None and np.isfinite(float(target)):
            return ObjectiveReference(
                float(target) - 1.0e-8,
                "suite_target",
                "COCO final target minus the standard 1e-8 target offset",
            )
    if problem.family == "yahpo":
        try:
            stats = problem._bench.target_stats
            # target_stats is a pandas DataFrame in YAHPO 1.0.2.
            selected = stats[
                (stats["metric"] == "logloss")
                & (stats["statistic"] == "min")
                & (stats["instance"].astype(str) == str(problem.instance))
            ]
            if len(selected):
                return ObjectiveReference(
                    float(selected.iloc[0]["value"]),
                    "empirical_500k",
                    "YAHPO empirical minimum from 500,000 random configurations",
                )
        except Exception:
            pass
        return ObjectiveReference(None, "run_best", "YAHPO exact optimum is unavailable; use final campaign incumbent")
    return ObjectiveReference(None, "run_best", "No external exact reference is available")
