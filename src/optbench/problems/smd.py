from __future__ import annotations

from typing import Any

import numpy as np

from optbench.base import BilevelEvaluation, BilevelProblem, require_vector


class SMDProblem(BilevelProblem):
    """Direct minimization-form implementation of SMD2, SMD3, and SMD6."""

    def __init__(self, *, problem_id: str, variant: int):
        if variant not in {2, 3, 6}:
            raise ValueError("Implemented SMD variants are 2, 3, and 6.")
        self.variant = variant
        if variant == 2:
            upper_bounds = [(-5.0, 10.0), (-5.0, 1.0)]
            lower_bounds = [(-5.0, 10.0), (np.finfo(float).eps, np.e)]
            property_name = "upper/lower conflict"
        elif variant == 3:
            eps = 1e-10
            upper_bounds = [(-5.0, 10.0), (-5.0, 10.0)]
            lower_bounds = [(-5.0, 10.0), (-np.pi / 2 + eps, np.pi / 2 - eps)]
            property_name = "multimodal lower level"
        else:
            upper_bounds = [(-5.0, 10.0), (-5.0, 10.0)]
            # p=q=r=1, s=2 => lower vector [l1_q, l1_s0, l1_s1, l2].
            lower_bounds = [(-5.0, 10.0)] * 4
            property_name = "set-valued lower optimum"

        super().__init__(
            problem_id=problem_id,
            name=f"SMD{variant} (p=1, q=1, r=1" + (", s=2)" if variant == 6 else ")"),
            family="SMD",
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            upper_budget=40,
            lower_budget_per_upper_candidate=100,
            metadata={"variant": variant, "property": property_name},
        )

    def _objectives(self, xu: np.ndarray, xl: np.ndarray) -> tuple[float, float]:
        u1, u2 = float(xu[0]), float(xu[1])
        if self.variant == 2:
            l1, l2 = float(xl[0]), float(xl[1])
            coupling = (u2 - np.log(l2)) ** 2
            upper = u1**2 - l1**2 + u2**2 - coupling
            lower = u1**2 + l1**2 + coupling
        elif self.variant == 3:
            l1, l2 = float(xl[0]), float(xl[1])
            coupling = (u2**2 - np.tan(l2)) ** 2
            # Official SMD3 equation (10): F3 = u2^2 + (u2^2 - tan(l2))^2.
            # The previous `(u2 - 1)^2` term moved the global optimum from
            # xu=(0, 0) to xu=(0, 1), defining a different problem.
            upper = u1**2 + l1**2 + u2**2 + coupling
            lower = u1**2 + 1.0 + l1**2 - np.cos(2.0 * np.pi * l1) + coupling
        else:
            q_value = float(xl[0])
            s_values = xl[1:3]
            l2 = float(xl[3])
            coupling = (u2 - l2) ** 2
            upper = u1**2 - q_value**2 + float(np.sum(s_values**2)) + u2**2 - coupling
            lower = u1**2 + q_value**2 + float((s_values[1] - s_values[0]) ** 2) + coupling
        return float(upper), float(lower)

    def _evaluate_upper(self, xu: np.ndarray, xl: np.ndarray) -> BilevelEvaluation:
        upper, _ = self._objectives(xu, xl)
        return BilevelEvaluation(objective=upper)

    def _evaluate_lower(self, xu: np.ndarray, xl: np.ndarray) -> BilevelEvaluation:
        _, lower = self._objectives(xu, xl)
        return BilevelEvaluation(objective=lower)

    def analytic_lower_response(self, xu: np.ndarray) -> np.ndarray:
        if self.variant == 2:
            return np.array([0.0, np.exp(xu[1])])
        if self.variant == 3:
            return np.array([0.0, np.arctan(xu[1] ** 2)])
        return np.array([0.0, 0.0, 0.0, xu[1]])

    def validate_lower_solution(self, xu: Any, xl: Any, tolerance: float = 1e-6) -> dict[str, Any]:
        xu = require_vector(xu, self.upper_dimension, self.upper_bounds)
        xl = require_vector(xl, self.lower_dimension, self.lower_bounds)
        candidate = self._objectives(xu, xl)[1]
        reference_xl = self.analytic_lower_response(xu)
        optimum = self._objectives(xu, reference_xl)[1]
        gap = max(0.0, float(candidate - optimum))
        return {
            "valid": bool(gap <= tolerance),
            "lower_objective": float(candidate),
            "reference_lower_objective": float(optimum),
            "optimality_gap": gap,
            "tolerance": tolerance,
        }

