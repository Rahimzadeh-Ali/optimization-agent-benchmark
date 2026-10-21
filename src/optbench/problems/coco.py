from __future__ import annotations

from typing import Any

import numpy as np

from optbench.base import BackendUnavailable, BenchmarkProblem, EvaluationResult, normalize_constraints, require_vector


class CocoProblem(BenchmarkProblem):
    """Thin adapter over a single problem from a COCO suite."""

    def __init__(
        self,
        *,
        problem_id: str,
        name: str,
        direction: str,
        suite_name: str,
        function_id: int,
        dimension: int,
        budget: int,
        instance: int = 1,
        noisy: bool = False,
        expected_constraints: int = 0,
    ) -> None:
        try:
            import cocoex
        except ImportError as exc:
            raise BackendUnavailable(
                "COCO tasks require `python -m pip install -e \".[coco]\"`."
            ) from exc

        self.suite_name = suite_name
        self.function_id = function_id
        self.instance = instance
        self._coco_dimension = dimension
        self._cocoex = cocoex
        self._suite = cocoex.Suite(suite_name, "", "")
        self._problem = self._suite.get_problem_by_function_dimension_instance(
            function_id, dimension, instance
        )
        if self._problem is None:
            raise ValueError(
                f"COCO has no {suite_name} function={function_id}, "
                f"dimension={dimension}, instance={instance}."
            )

        bounds = np.column_stack(
            [np.asarray(self._problem.lower_bounds), np.asarray(self._problem.upper_bounds)]
        )
        integer_count = int(getattr(self._problem, "number_of_integer_variables", 0))
        if suite_name == "bbob-mixint":
            # COCO orders the 80% discretized variables before the 20% continuous block.
            variable_types = ["integer"] * integer_count + ["continuous"] * (
                dimension - integer_count
            )
        else:
            variable_types = ["continuous"] * dimension

        actual_constraints = int(getattr(self._problem, "number_of_constraints", 0))
        if expected_constraints and actual_constraints != expected_constraints:
            raise ValueError(
                f"{problem_id} expected {expected_constraints} COCO constraints, "
                f"but backend reports {actual_constraints}."
            )

        super().__init__(
            problem_id=problem_id,
            name=name,
            direction=direction,
            family=suite_name,
            dimension=dimension,
            variable_types=variable_types,
            bounds=bounds,
            budget=budget,
            noisy=noisy,
            number_of_constraints=actual_constraints,
            metadata={"function_id": function_id, "instance": instance},
        )

    def _evaluate(self, x: Any) -> EvaluationResult:
        x = require_vector(x, self.dimension, self.bounds)
        objective = float(np.asarray(self._problem(x)).reshape(-1)[0])
        if self.number_of_constraints:
            constraints, violation, feasible = normalize_constraints(self._problem.constraint(x))
        else:
            constraints, violation, feasible = (), 0.0, True
        return EvaluationResult(
            objective=objective,
            feasible=feasible,
            constraint_violation=violation,
            constraints=constraints,
        )

    def _reset_backend(self, seed: int | None) -> None:
        del seed
        reset = getattr(self._problem, "reset", None)
        if callable(reset):
            reset()
            return
        # coco-experiment 2.8.2 does not expose Problem.reset().  Recreating
        # the suite/problem is the supported way to obtain fresh counters and
        # a fresh noisy-function state when a wrapper is reused.
        self._suite = self._cocoex.Suite(self.suite_name, "", "")
        self._problem = self._suite.get_problem_by_function_dimension_instance(
            self.function_id, self._coco_dimension, self.instance
        )
        if self._problem is None:
            raise RuntimeError("COCO could not recreate the benchmark problem.")


def make_bbob(problem_id: str, function_id: int, dimension: int, budget: int, direction: str, name: str) -> CocoProblem:
    return CocoProblem(
        problem_id=problem_id,
        name=name,
        direction=direction,
        suite_name="bbob",
        function_id=function_id,
        dimension=dimension,
        budget=budget,
    )


def make_mixint(problem_id: str, function_id: int, name: str) -> CocoProblem:
    return CocoProblem(
        problem_id=problem_id,
        name=name,
        direction="variable_type",
        suite_name="bbob-mixint",
        function_id=function_id,
        dimension=10,
        budget=500,
    )


def make_noisy(problem_id: str, function_id: int, name: str) -> CocoProblem:
    return CocoProblem(
        problem_id=problem_id,
        name=name,
        direction="noise",
        suite_name="bbob-noisy",
        function_id=function_id,
        dimension=5,
        budget=500,
        noisy=True,
    )


def make_constrained(problem_id: str, function_id: int, name: str) -> CocoProblem:
    return CocoProblem(
        problem_id=problem_id,
        name=name,
        direction="constraints",
        suite_name="bbob-constrained",
        function_id=function_id,
        dimension=10,
        budget=500,
        expected_constraints=3,
    )

