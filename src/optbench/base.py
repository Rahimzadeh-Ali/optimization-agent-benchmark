from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


class BudgetExceeded(RuntimeError):
    """Raised before an evaluation that would exceed a fixed budget."""


class BackendUnavailable(ImportError):
    """Raised with an actionable install message for an optional backend."""


@dataclass(frozen=True)
class EvaluationResult:
    objective: float
    feasible: bool = True
    constraint_violation: float = 0.0
    cost: float = 1.0
    constraints: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BilevelEvaluation(EvaluationResult):
    pass


def normalize_constraints(values: Sequence[float] | np.ndarray) -> tuple[tuple[float, ...], float, bool]:
    """Normalize a g<=0 vector into values, L1 violation, and feasibility."""
    g = np.asarray(values, dtype=float).reshape(-1)
    violation = float(np.maximum(g, 0.0).sum())
    return tuple(float(v) for v in g), violation, bool(violation <= 1e-10)


def require_vector(x: Any, dimension: int, bounds: np.ndarray | None = None) -> np.ndarray:
    value = np.asarray(x, dtype=float)
    if value.shape != (dimension,):
        raise ValueError(f"Expected a vector with shape ({dimension},), got {value.shape}.")
    if not np.all(np.isfinite(value)):
        raise ValueError("Candidate contains NaN or infinity.")
    if bounds is not None:
        if np.any(value < bounds[:, 0]) or np.any(value > bounds[:, 1]):
            raise ValueError("Candidate is outside the declared search domain.")
    return value


class BenchmarkProblem(ABC):
    objective_sense = "minimize"
    kind = "single_level"

    def __init__(
        self,
        *,
        problem_id: str,
        name: str,
        direction: str,
        family: str,
        dimension: int,
        variable_types: Sequence[str],
        bounds: Sequence[Sequence[float]] | None,
        budget: int,
        noisy: bool = False,
        number_of_constraints: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.problem_id = problem_id
        self.name = name
        self.direction = direction
        self.family = family
        self.dimension = int(dimension)
        self.variable_types = tuple(variable_types)
        self.bounds = None if bounds is None else np.asarray(bounds, dtype=float)
        self.budget = int(budget)
        self.noisy = bool(noisy)
        self.number_of_constraints = int(number_of_constraints)
        self.metadata = dict(metadata or {})
        self.evaluations = 0
        self._budget_limit: int | None = None

        if len(self.variable_types) != self.dimension:
            raise ValueError("variable_types length must equal dimension")
        if self.bounds is not None and self.bounds.shape != (self.dimension, 2):
            raise ValueError("bounds must have shape (dimension, 2)")

    def set_budget_limit(self, limit: int | None) -> None:
        self._budget_limit = limit

    def reset(self, seed: int | None = None) -> None:
        self.evaluations = 0
        self._reset_backend(seed)

    def _reset_backend(self, seed: int | None) -> None:
        del seed

    def evaluate(self, x: Any) -> EvaluationResult:
        if self._budget_limit is not None and self.evaluations >= self._budget_limit:
            raise BudgetExceeded(
                f"{self.problem_id} exhausted its budget of {self._budget_limit} evaluations."
            )
        result = self._evaluate(x)
        if not isinstance(result, EvaluationResult):
            raise TypeError("Problem implementations must return EvaluationResult.")
        self.evaluations += 1
        return result

    @abstractmethod
    def _evaluate(self, x: Any) -> EvaluationResult:
        raise NotImplementedError

    def sample(self, rng: np.random.Generator) -> Any:
        if self.bounds is None:
            raise TypeError(f"{self.problem_id} has no numeric box sampler.")
        x = rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
        for i, variable_type in enumerate(self.variable_types):
            if variable_type in {"integer", "binary", "discrete"}:
                x[i] = np.rint(x[i])
        return x

    @property
    def is_numeric_box(self) -> bool:
        return self.bounds is not None and all(
            t in {"continuous", "integer", "binary", "discrete"} for t in self.variable_types
        )

    def agent_spec(self) -> dict[str, Any]:
        """Public task description. Hidden optimum/transform data is omitted."""
        return {
            "problem_id": self.problem_id,
            "name": self.name,
            "direction": self.direction,
            "family": self.family,
            "objective": self.objective_sense,
            "dimension": self.dimension,
            "variable_types": list(self.variable_types),
            "bounds": None if self.bounds is None else self.bounds.tolist(),
            "budget": self.budget,
            "noisy": self.noisy,
            "number_of_inequality_constraints": self.number_of_constraints,
            "constraint_convention": "g(x) <= 0",
            "metadata": dict(self.metadata),
        }


class BilevelProblem(ABC):
    objective_sense = "minimize"
    kind = "bilevel"

    def __init__(
        self,
        *,
        problem_id: str,
        name: str,
        family: str,
        upper_bounds: Sequence[Sequence[float]],
        lower_bounds: Sequence[Sequence[float]],
        upper_budget: int,
        lower_budget_per_upper_candidate: int,
        upper_constraint_count: int = 0,
        lower_constraint_count: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.problem_id = problem_id
        self.name = name
        self.family = family
        self.direction = "true_bilevel"
        self.upper_bounds = np.asarray(upper_bounds, dtype=float)
        self.lower_bounds = np.asarray(lower_bounds, dtype=float)
        self.upper_dimension = self.upper_bounds.shape[0]
        self.lower_dimension = self.lower_bounds.shape[0]
        self.upper_budget = int(upper_budget)
        self.lower_budget_per_upper_candidate = int(lower_budget_per_upper_candidate)
        self.upper_constraint_count = int(upper_constraint_count)
        self.lower_constraint_count = int(lower_constraint_count)
        self.metadata = dict(metadata or {})
        self.upper_evaluations = 0
        self.lower_evaluations = 0

    def reset(self, seed: int | None = None) -> None:
        del seed
        self.upper_evaluations = 0
        self.lower_evaluations = 0

    def evaluate_upper(self, xu: Any, xl: Any) -> BilevelEvaluation:
        xu = require_vector(xu, self.upper_dimension, self.upper_bounds)
        xl = require_vector(xl, self.lower_dimension, self.lower_bounds)
        result = self._evaluate_upper(xu, xl)
        self.upper_evaluations += 1
        return result

    def evaluate_lower(self, xu: Any, xl: Any) -> BilevelEvaluation:
        xu = require_vector(xu, self.upper_dimension, self.upper_bounds)
        xl = require_vector(xl, self.lower_dimension, self.lower_bounds)
        result = self._evaluate_lower(xu, xl)
        self.lower_evaluations += 1
        return result

    def upper_constraints(self, xu: Any, xl: Any) -> tuple[float, ...]:
        """Return public upper constraints without consuming an objective budget."""
        xu = require_vector(xu, self.upper_dimension, self.upper_bounds)
        xl = require_vector(xl, self.lower_dimension, self.lower_bounds)
        return self._evaluate_upper(xu, xl).constraints

    def lower_constraints(self, xu: Any, xl: Any) -> tuple[float, ...]:
        """Return public lower constraints without consuming an objective budget."""
        xu = require_vector(xu, self.upper_dimension, self.upper_bounds)
        xl = require_vector(xl, self.lower_dimension, self.lower_bounds)
        return self._evaluate_lower(xu, xl).constraints

    @abstractmethod
    def _evaluate_upper(self, xu: np.ndarray, xl: np.ndarray) -> BilevelEvaluation:
        raise NotImplementedError

    @abstractmethod
    def _evaluate_lower(self, xu: np.ndarray, xl: np.ndarray) -> BilevelEvaluation:
        raise NotImplementedError

    @abstractmethod
    def validate_lower_solution(self, xu: Any, xl: Any, tolerance: float = 1e-6) -> dict[str, Any]:
        raise NotImplementedError

    def agent_spec(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "name": self.name,
            "direction": self.direction,
            "family": self.family,
            "objective": self.objective_sense,
            "upper_dimension": self.upper_dimension,
            "lower_dimension": self.lower_dimension,
            "upper_bounds": self.upper_bounds.tolist(),
            "lower_bounds": self.lower_bounds.tolist(),
            "upper_budget": self.upper_budget,
            "lower_budget_per_upper_candidate": self.lower_budget_per_upper_candidate,
            "upper_inequality_constraints": self.upper_constraint_count,
            "lower_inequality_constraints": self.lower_constraint_count,
            "constraint_convention": "g(xu, xl) <= 0",
            "metadata": dict(self.metadata),
        }
