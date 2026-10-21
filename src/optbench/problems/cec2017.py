from __future__ import annotations

from typing import Any

import numpy as np

from optbench.base import BackendUnavailable, BenchmarkProblem, EvaluationResult, require_vector


_NAMES = {
    5: "Shifted and Rotated Rastrigin",
    6: "Shifted and Rotated Expanded Schaffer F6",
    8: "Shifted and Rotated Non-Continuous Rastrigin",
}


class CEC2017Problem(BenchmarkProblem):
    """CEC-2017 F5/F6/F8 using official transform data and direct equations.

    `opfunu` supplies its packaged copies of the official shift vectors and
    rotation matrices. Its evaluator classes are deliberately not called.
    """

    def __init__(self, *, problem_id: str, function_id: int, dimension: int, budget: int, direction: str):
        if function_id not in _NAMES:
            raise ValueError(f"Only CEC-2017 F5, F6, and F8 are implemented, got F{function_id}.")
        try:
            from opfunu.cec_based import cec2017
        except ImportError as exc:
            raise BackendUnavailable(
                "CEC-2017 tasks require `python -m pip install -e \".[cec]\"`."
            ) from exc

        data_holder = getattr(cec2017, f"F{function_id}2017")(ndim=dimension)
        self.function_id = function_id
        self.shift = np.asarray(data_holder.f_shift, dtype=float).reshape(-1)[:dimension]
        matrix = np.asarray(data_holder.f_matrix, dtype=float)
        self.rotation = matrix[:dimension, :dimension]
        self.bias = float(function_id * 100)
        if self.shift.shape != (dimension,) or self.rotation.shape != (dimension, dimension):
            raise ValueError("CEC transform data have unexpected dimensions.")

        super().__init__(
            problem_id=problem_id,
            name=f"CEC-2017 F{function_id}: {_NAMES[function_id]}",
            direction=direction,
            family="cec2017",
            dimension=dimension,
            variable_types=["continuous"] * dimension,
            bounds=[(-100.0, 100.0)] * dimension,
            budget=budget,
            metadata={
                "function_id": function_id,
                "global_minimum_value": self.bias,
                "definition_policy": "named equation takes priority over conflicting CEC-2017 ID",
            },
        )

    @staticmethod
    def _rastrigin(z: np.ndarray) -> float:
        return float(np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0))

    @staticmethod
    def _expanded_schaffer_f6(z: np.ndarray) -> float:
        z_next = np.roll(z, -1)
        radius2 = z * z + z_next * z_next
        numerator = np.sin(np.sqrt(radius2)) ** 2 - 0.5
        return float(np.sum(0.5 + numerator / (1.0 + 0.001 * radius2) ** 2))

    def _evaluate(self, x: Any) -> EvaluationResult:
        x = require_vector(x, self.dimension, self.bounds)
        delta = x - self.shift
        if self.function_id == 8:
            delta = np.where(
                np.abs(delta) > 0.5,
                np.floor(2.0 * delta + 0.5) / 2.0,
                delta,
            )

        scale = 0.0512 if self.function_id in {5, 8} else 1.0
        z = self.rotation @ (scale * delta)
        if self.function_id in {5, 8}:
            value = self._rastrigin(z)
        else:
            value = self._expanded_schaffer_f6(z)
        return EvaluationResult(objective=value + self.bias)

