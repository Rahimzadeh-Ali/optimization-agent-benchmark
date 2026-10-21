import numpy as np
import pytest

from optbench.base import BudgetExceeded, normalize_constraints
from optbench.registry import create_problem


def test_constraint_convention_is_nonpositive() -> None:
    values, violation, feasible = normalize_constraints([-2.0, 0.0, 0.25])
    assert values == (-2.0, 0.0, 0.25)
    assert violation == pytest.approx(0.25)
    assert not feasible


def test_single_level_budget_is_enforced_before_extra_call() -> None:
    problem = create_problem("C04")
    problem.reset(0)
    problem.set_budget_limit(2)
    midpoint = np.mean(problem.bounds, axis=1)
    problem.evaluate(midpoint)
    problem.evaluate(midpoint)
    with pytest.raises(BudgetExceeded):
        problem.evaluate(midpoint)
    assert problem.evaluations == 2


@pytest.mark.parametrize("problem_id", ["C04", "C05"])
def test_pymoo_adapter_shapes(problem_id: str) -> None:
    problem = create_problem(problem_id)
    result = problem.evaluate(np.mean(problem.bounds, axis=1))
    assert np.isfinite(result.objective)
    assert len(result.constraints) == problem.number_of_constraints
