"""Function-visible optimization-agent benchmark suite."""

from .base import (
    BenchmarkProblem,
    BilevelEvaluation,
    BilevelProblem,
    BudgetExceeded,
    EvaluationResult,
)
from .registry import PROBLEMS, create_problem, list_problems

__all__ = [
    "BenchmarkProblem",
    "BilevelEvaluation",
    "BilevelProblem",
    "BudgetExceeded",
    "EvaluationResult",
    "PROBLEMS",
    "create_problem",
    "list_problems",
]

