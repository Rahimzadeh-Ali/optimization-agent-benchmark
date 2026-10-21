from __future__ import annotations

from typing import Any

import numpy as np

from optbench.base import BackendUnavailable, BenchmarkProblem, EvaluationResult, normalize_constraints, require_vector


class PymooGProblem(BenchmarkProblem):
    def __init__(self, *, problem_id: str, problem_name: str, budget: int = 500):
        try:
            from pymoo.problems import get_problem
        except ImportError as exc:
            raise BackendUnavailable(
                "pymoo tasks require `python -m pip install -e \".[constraints]\"`."
            ) from exc

        self._problem = get_problem(problem_name.lower())
        dimension = int(self._problem.n_var)
        constraints = int(self._problem.n_ieq_constr)
        bounds = np.column_stack([self._problem.xl, self._problem.xu])
        super().__init__(
            problem_id=problem_id,
            name=f"pymoo {problem_name.upper()}",
            direction="constraints",
            family="pymoo_g",
            dimension=dimension,
            variable_types=["continuous"] * dimension,
            bounds=bounds,
            budget=budget,
            number_of_constraints=constraints,
            metadata={"problem": problem_name.lower()},
        )

    def _evaluate(self, x: Any) -> EvaluationResult:
        x = require_vector(x, self.dimension, self.bounds)
        output = self._problem.evaluate(x, return_values_of=["F", "G"], return_as_dictionary=True)
        objective = float(np.asarray(output["F"]).reshape(-1)[0])
        constraints, violation, feasible = normalize_constraints(output["G"])
        return EvaluationResult(
            objective=objective,
            feasible=feasible,
            constraint_violation=violation,
            constraints=constraints,
        )

