from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

from optbench.base import BilevelEvaluation, BilevelProblem, normalize_constraints, require_vector


def _quadratic(x: np.ndarray, y: np.ndarray, q: dict[str, Any]) -> float:
    return float(
        0.5 * x @ q["P"] @ x
        + 0.5 * y @ q["Q"] @ y
        + x @ q["R"] @ y
        + q["c"] @ x
        + q["d"] @ y
        + q["s"]
    )


class BolibProblem(BilevelProblem):
    """Two source-traced BOLIB problems with BOLIB's g>=0 converted to g<=0."""

    def __init__(self, *, problem_id: str, dataset: str):
        self.dataset = dataset
        if dataset == "calamai_vicente_1994b":
            self._init_calamai(problem_id)
        elif dataset == "allende_still_2013":
            self._init_allende(problem_id)
        else:
            raise ValueError(f"Unsupported BOLIB dataset: {dataset}")

    def _init_calamai(self, problem_id: str) -> None:
        self.Fq = {
            "P": np.eye(4), "Q": np.eye(2), "R": np.zeros((4, 2)),
            "c": -np.ones(4), "d": np.zeros(2), "s": 2.0,
        }
        self.fq = {
            "P": np.zeros((4, 4)), "Q": np.eye(2),
            "R": np.array([[-1.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.0, 0.0]]),
            "c": np.zeros(4), "d": np.zeros(2), "s": 0.0,
        }
        self.A_lower = np.array([
            [-1, 0, 0, 0], [0, -1, 0, 0], [-1, 0, 0, 0],
            [0, -1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0],
        ], dtype=float)
        self.B_lower = np.array([
            [1, 0], [0, 1], [-1, 0], [0, -1], [1, 0], [0, 1],
        ], dtype=float)
        self.rhs_lower = np.array([-1, -1, -1.5, -3, 1, 1], dtype=float)
        self.known_solution = (
            np.array([1.25, 0.5, 1.0, 1.0]), np.array([0.25, 0.5]), 0.3125
        )
        super().__init__(
            problem_id=problem_id,
            name="BOLIB Calamai-Vicente 1994b",
            family="BOLIB",
            upper_bounds=[(-5.0, 5.0)] * 4,
            lower_bounds=[(-5.0, 5.0)] * 2,
            upper_budget=40,
            lower_budget_per_upper_candidate=100,
            lower_constraint_count=6,
            metadata={
                "dataset": self.dataset,
                "search_box_source": "benchmark-added; BOLIB dataset is unbounded",
            },
        )

    def _init_allende(self, problem_id: str) -> None:
        self.Fq = {
            "P": np.array([[-2.0, 0.0], [0.0, 2.0]]), "Q": 2.0 * np.eye(2),
            "R": np.zeros((2, 2)), "c": np.array([-2.0, -2.0]),
            "d": np.zeros(2), "s": 0.0,
        }
        self.fq = {
            "P": np.zeros((2, 2)), "Q": 2.0 * np.eye(2),
            "R": -2.0 * np.eye(2), "c": np.zeros(2), "d": np.zeros(2), "s": 0.0,
        }
        self.known_solution = (
            np.array([0.5, 0.5]), np.array([0.5, 0.5]), -1.5
        )
        super().__init__(
            problem_id=problem_id,
            name="BOLIB Allende-Still 2013",
            family="BOLIB",
            upper_bounds=[(0.0, 2.0)] * 2,
            lower_bounds=[(0.0, 2.0)] * 2,
            upper_budget=40,
            lower_budget_per_upper_candidate=100,
            upper_constraint_count=5,
            lower_constraint_count=2,
            metadata={
                "dataset": self.dataset,
                "search_box_source": "benchmark-added; published x lower bound retained as constraint",
            },
        )

    def _upper_constraints(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.dataset == "calamai_vicente_1994b":
            return np.empty(0)
        # BOLIB g>=0: 2-x1, y1, y2, x1, x2. Return the negation for g<=0.
        return -np.array([2.0 - x[0], y[0], y[1], x[0], x[1]])

    def _lower_constraints(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.dataset == "calamai_vicente_1994b":
            return -(self.A_lower @ x + self.B_lower @ y - self.rhs_lower)
        # 0.25 - (yi - 1)^2 >= 0 in BOLIB.
        return -np.array([0.25 - (y[0] - 1.0) ** 2, 0.25 - (y[1] - 1.0) ** 2])

    def _evaluate_upper(self, xu: np.ndarray, xl: np.ndarray) -> BilevelEvaluation:
        g, violation, feasible = normalize_constraints(self._upper_constraints(xu, xl))
        return BilevelEvaluation(
            objective=_quadratic(xu, xl, self.Fq), feasible=feasible,
            constraint_violation=violation, constraints=g,
        )

    def _evaluate_lower(self, xu: np.ndarray, xl: np.ndarray) -> BilevelEvaluation:
        g, violation, feasible = normalize_constraints(self._lower_constraints(xu, xl))
        return BilevelEvaluation(
            objective=_quadratic(xu, xl, self.fq), feasible=feasible,
            constraint_violation=violation, constraints=g,
        )

    def _solve_lower_reference(self, xu: np.ndarray) -> tuple[np.ndarray, float, bool]:
        if self.dataset == "allende_still_2013":
            y = np.clip(xu, 0.5, 1.5)
            return y, _quadratic(xu, y, self.fq), True

        constraints = {
            "type": "ineq",
            "fun": lambda y: -self._lower_constraints(xu, np.asarray(y, dtype=float)),
        }
        starts = [
            np.zeros(self.lower_dimension),
            np.mean(self.lower_bounds, axis=1),
            np.array([xu[0], xu[1]]),
        ]
        best = None
        for start in starts:
            result = minimize(
                lambda y: _quadratic(xu, np.asarray(y), self.fq),
                np.clip(start, self.lower_bounds[:, 0], self.lower_bounds[:, 1]),
                method="SLSQP",
                bounds=self.lower_bounds,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 500},
            )
            violation = np.maximum(self._lower_constraints(xu, result.x), 0.0).sum()
            if violation <= 1e-7 and (best is None or result.fun < best.fun):
                best = result
        if best is None:
            return np.full(self.lower_dimension, np.nan), np.inf, False
        return np.asarray(best.x), float(best.fun), bool(best.success)

    def validate_lower_solution(self, xu: Any, xl: Any, tolerance: float = 1e-6) -> dict[str, Any]:
        xu = require_vector(xu, self.upper_dimension, self.upper_bounds)
        xl = require_vector(xl, self.lower_dimension, self.lower_bounds)
        candidate = self._evaluate_lower(xu, xl)
        reference_xl, reference_value, solver_success = self._solve_lower_reference(xu)
        gap = max(0.0, float(candidate.objective - reference_value))
        valid = candidate.feasible and solver_success and gap <= tolerance
        return {
            "valid": bool(valid),
            "candidate_feasible": candidate.feasible,
            "lower_objective": candidate.objective,
            "reference_lower_objective": reference_value,
            "optimality_gap": gap,
            "reference_xl": reference_xl.tolist(),
            "tolerance": tolerance,
        }

