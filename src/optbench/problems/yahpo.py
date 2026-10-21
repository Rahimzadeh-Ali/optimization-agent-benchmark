from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from optbench.base import BackendUnavailable, BenchmarkProblem, EvaluationResult


class YahpoProblem(BenchmarkProblem):
    """YAHPO single-objective view with fixed instance and fidelity."""

    def __init__(self, *, problem_id: str, scenario: str, instance: str, dimension: int, seed: int = 0):
        try:
            from yahpo_gym import benchmark_set
        except ImportError as exc:
            raise BackendUnavailable(
                "YAHPO tasks require `python -m pip install -e \".[yahpo]\"` and the "
                "v1.0.2 yahpo_data bundle described in README.md."
            ) from exc

        self.scenario = scenario
        self.instance = str(instance)
        self.seed = int(seed)
        try:
            self._bench = benchmark_set.BenchmarkSet(scenario=scenario, check=False)
            self._bench.set_instance(self.instance)
            self._bench.set_constant("trainsize", 1.0)
            self._bench.set_constant("repl", 1)
            self._space = self._bench.get_opt_space(drop_fidelity_params=True, seed=self.seed)
            # YAHPO 1.0.2 calls a private ConfigSpace 0.6 sorting hook which
            # was removed in ConfigSpace 1.x.  Modern SMAC needs ConfigSpace
            # 1.x; its order is already stable, so the old hook is a no-op.
            space_type = type(self._bench.config_space)
            if not hasattr(space_type, "_sort_hyperparameters"):
                setattr(space_type, "_sort_hyperparameters", lambda space: None)
        except Exception as exc:
            raise BackendUnavailable(
                "YAHPO is installed but its v1.0.2 surrogate data are unavailable or misconfigured. "
                "Follow the YAHPO setup commands in README.md."
            ) from exc

        # YAHPO keeps the selected task_id as a ConfigSpace Constant. It is
        # part of the surrogate input, but is not an optimization variable.
        hyperparameters = [
            hp
            for hp in self._space.get_hyperparameters()
            if "constant" not in type(hp).__name__.lower()
        ]
        self._variable_names = {hp.name for hp in hyperparameters}
        if len(hyperparameters) != dimension:
            raise ValueError(
                f"{problem_id} expected a {dimension}D optimization space, got {len(hyperparameters)}D."
            )
        variable_types = []
        for hp in hyperparameters:
            class_name = type(hp).__name__.lower()
            if "categorical" in class_name:
                variable_types.append("categorical")
            elif "integer" in class_name:
                variable_types.append("integer")
            else:
                variable_types.append("continuous")

        super().__init__(
            problem_id=problem_id,
            name=f"YAHPO {scenario} on OpenML dataset {instance}",
            direction="variable_type",
            family="yahpo",
            dimension=dimension,
            variable_types=variable_types,
            bounds=None,
            budget=150,
            metadata={
                "scenario": scenario,
                "instance": self.instance,
                "target": "logloss",
                "fixed_constants": {"trainsize": 1.0, "repl": 1},
                "hierarchical": True,
                "parameter_names": sorted(self._variable_names),
            },
        )

    def _reset_backend(self, seed: int | None) -> None:
        if seed is not None:
            self.seed = int(seed)
            self._space.seed(self.seed)

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        self._space.seed(int(rng.integers(0, 2**31 - 1)))
        config = self._space.sample_configuration()
        return {
            key: value
            for key, value in config.get_dictionary().items()
            if key in self._variable_names
        }

    def _evaluate(self, x: Any) -> EvaluationResult:
        if not isinstance(x, Mapping):
            raise TypeError(f"{self.problem_id} expects a configuration dictionary.")
        configuration = dict(x)
        unknown = set(configuration) - self._variable_names
        if unknown:
            raise ValueError(f"Unknown YAHPO parameter(s): {sorted(unknown)}")
        configuration.update({"task_id": self.instance, "trainsize": 1.0, "repl": 1})
        prediction = self._bench.objective_function(configuration, seed=self.seed)[0]
        metrics = {key: float(value) for key, value in prediction.items()}
        return EvaluationResult(objective=metrics["logloss"], metadata={"metrics": metrics})

