import numpy as np
import pytest

from optbench.registry import create_problem


@pytest.mark.parametrize("problem_id", ["D04", "D05", "S04", "S05"])
def test_cec_shift_point_has_declared_bias(problem_id: str) -> None:
    pytest.importorskip("opfunu")
    problem = create_problem(problem_id)
    result = problem.evaluate(problem.shift)
    assert result.objective == pytest.approx(problem.bias, abs=1e-12)


@pytest.mark.parametrize("problem_id", ["N04", "N05"])
def test_nevergrad_noisy_backend_is_seed_reproducible(problem_id: str) -> None:
    pytest.importorskip("nevergrad")
    problem = create_problem(problem_id, seed=13)
    x = np.zeros(problem.dimension)
    first = problem.evaluate(x).objective
    problem.reset(13)
    second = problem.evaluate(x).objective
    assert first == pytest.approx(second)
