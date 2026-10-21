from __future__ import annotations

from typing import Any

import numpy as np

from optbench.base import BackendUnavailable, BenchmarkProblem, EvaluationResult, require_vector


class NevergradNoisyProblem(BenchmarkProblem):
    def __init__(self, *, problem_id: str, function_name: str, seed: int = 0):
        try:
            from nevergrad.functions import ArtificialFunction
        except ImportError as exc:
            raise BackendUnavailable(
                "Nevergrad tasks require `python -m pip install -e \".[nevergrad]\"`."
            ) from exc
        self._ArtificialFunction = ArtificialFunction
        self.function_name = function_name.lower()
        self.seed = seed
        self._function = None
        super().__init__(
            problem_id=problem_id,
            name=f"Nevergrad noisy {function_name}",
            direction="noise",
            family="nevergrad",
            dimension=20,
            variable_types=["continuous"] * 20,
            bounds=[(-5.0, 5.0)] * 20,
            budget=500,
            noisy=True,
            metadata={"noise_level": 10.0, "rotation": False, "benchmark_box_assumption": [-5.0, 5.0]},
        )
        self._reset_backend(seed)

    def _reset_backend(self, seed: int | None) -> None:
        if seed is not None:
            self.seed = int(seed)
        # Nevergrad's ArtificialFunction currently uses NumPy's legacy global
        # RNG for its stochastic transform/noise. Seed immediately before
        # construction to make independent campaign creation reproducible.
        np.random.seed(self.seed)
        self._function = self._ArtificialFunction(
            name=self.function_name,
            block_dimension=self.dimension,
            noise_level=10.0,
            rotation=False,
        )

    def _evaluate(self, x: Any) -> EvaluationResult:
        x = require_vector(x, self.dimension, self.bounds)
        return EvaluationResult(objective=float(self._function(x)))

