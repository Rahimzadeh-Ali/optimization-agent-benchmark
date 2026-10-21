import numpy as np
import pytest

from optbench.registry import create_problem


@pytest.mark.parametrize(
    ("problem_id", "xu", "expected"),
    [
        ("B01", np.array([0.0, 0.0]), 0.0),
        ("B02", np.array([0.0, 1.0]), 0.0),
        ("B03", np.array([0.0, 0.0]), 0.0),
    ],
)
def test_smd_analytic_optimum(problem_id: str, xu: np.ndarray, expected: float) -> None:
    problem = create_problem(problem_id)
    xl = problem.analytic_lower_response(xu)
    upper = problem.evaluate_upper(xu, xl)
    assert upper.objective == pytest.approx(expected, abs=1e-12)
    assert problem.validate_lower_solution(xu, xl)["valid"]


@pytest.mark.parametrize("problem_id", ["B04", "B05"])
def test_bolib_published_solution(problem_id: str) -> None:
    problem = create_problem(problem_id)
    xu, xl, expected = problem.known_solution
    upper = problem.evaluate_upper(xu, xl)
    lower = problem.evaluate_lower(xu, xl)
    assert upper.objective == pytest.approx(expected, abs=1e-10)
    assert upper.feasible and lower.feasible
    assert problem.validate_lower_solution(xu, xl)["valid"]
    assert len(problem.upper_constraints(xu, xl)) == problem.upper_constraint_count
    assert len(problem.lower_constraints(xu, xl)) == problem.lower_constraint_count
    assert problem.upper_evaluations == 1
    assert problem.lower_evaluations == 1
