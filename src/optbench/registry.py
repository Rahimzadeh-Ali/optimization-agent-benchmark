from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from optbench.base import BenchmarkProblem, BilevelProblem


Problem = BenchmarkProblem | BilevelProblem


@dataclass(frozen=True)
class ProblemSpec:
    problem_id: str
    direction: str
    family: str
    name: str
    budget: int
    factory: Callable[..., Problem]

    def public_record(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("factory")
        return data


def _bbob(pid: str, fid: int, dim: int, budget: int, direction: str, name: str) -> Problem:
    from optbench.problems.coco import make_bbob
    return make_bbob(pid, fid, dim, budget, direction, name)


def _mixint(pid: str, fid: int, name: str) -> Problem:
    from optbench.problems.coco import make_mixint
    return make_mixint(pid, fid, name)


def _noisy(pid: str, fid: int, name: str) -> Problem:
    from optbench.problems.coco import make_noisy
    return make_noisy(pid, fid, name)


def _constrained(pid: str, fid: int, name: str) -> Problem:
    from optbench.problems.coco import make_constrained
    return make_constrained(pid, fid, name)


def _cec(pid: str, fid: int, dim: int, budget: int, direction: str) -> Problem:
    from optbench.problems.cec2017 import CEC2017Problem
    return CEC2017Problem(problem_id=pid, function_id=fid, dimension=dim, budget=budget, direction=direction)


def _yahpo(pid: str, scenario: str, dim: int, seed: int = 0) -> Problem:
    from optbench.problems.yahpo import YahpoProblem
    return YahpoProblem(problem_id=pid, scenario=scenario, instance="40981", dimension=dim, seed=seed)


def _nevergrad(pid: str, name: str, seed: int = 0) -> Problem:
    from optbench.problems.nevergrad_noise import NevergradNoisyProblem
    return NevergradNoisyProblem(problem_id=pid, function_name=name, seed=seed)


def _pymoo(pid: str, name: str) -> Problem:
    from optbench.problems.pymoo_g import PymooGProblem
    return PymooGProblem(problem_id=pid, problem_name=name)


def _smd(pid: str, variant: int) -> Problem:
    from optbench.problems.smd import SMDProblem
    return SMDProblem(problem_id=pid, variant=variant)


def _bolib(pid: str, dataset: str) -> Problem:
    from optbench.problems.bolib import BolibProblem
    return BolibProblem(problem_id=pid, dataset=dataset)


PROBLEMS: dict[str, ProblemSpec] = {
    "D01": ProblemSpec("D01", "dimensionality", "coco_bbob", "Rosenbrock f8 D=10", 250, lambda **_: _bbob("D01", 8, 10, 250, "dimensionality", "Rosenbrock Original")),
    "D02": ProblemSpec("D02", "dimensionality", "coco_bbob", "Rosenbrock f8 D=20", 500, lambda **_: _bbob("D02", 8, 20, 500, "dimensionality", "Rosenbrock Original")),
    "D03": ProblemSpec("D03", "dimensionality", "coco_bbob", "Rosenbrock f8 D=40", 1000, lambda **_: _bbob("D03", 8, 40, 1000, "dimensionality", "Rosenbrock Original")),
    "D04": ProblemSpec("D04", "dimensionality", "cec2017", "Expanded Schaffer F6 D=30", 750, lambda **_: _cec("D04", 6, 30, 750, "dimensionality")),
    "D05": ProblemSpec("D05", "dimensionality", "cec2017", "Expanded Schaffer F6 D=50", 1250, lambda **_: _cec("D05", 6, 50, 1250, "dimensionality")),
    "V01": ProblemSpec("V01", "variable_type", "bbob_mixint", "Mixed Rosenbrock f8", 500, lambda **_: _mixint("V01", 8, "Mixed-integer Rosenbrock")),
    "V02": ProblemSpec("V02", "variable_type", "bbob_mixint", "Mixed Rastrigin f15", 500, lambda **_: _mixint("V02", 15, "Mixed-integer Rastrigin")),
    "V03": ProblemSpec("V03", "variable_type", "bbob_mixint", "Mixed Schaffer F7 f17", 500, lambda **_: _mixint("V03", 17, "Mixed-integer Schaffer F7")),
    "V04": ProblemSpec("V04", "variable_type", "yahpo", "rbv2_svm dataset 40981", 150, lambda seed=0, **_: _yahpo("V04", "rbv2_svm", 6, seed)),
    "V05": ProblemSpec("V05", "variable_type", "yahpo", "rbv2_xgboost dataset 40981", 150, lambda seed=0, **_: _yahpo("V05", "rbv2_xgboost", 14, seed)),
    "N01": ProblemSpec("N01", "noise", "bbob_noisy", "Rosenbrock Gaussian f110", 500, lambda **_: _noisy("N01", 110, "Rosenbrock with Gaussian noise")),
    "N02": ProblemSpec("N02", "noise", "bbob_noisy", "Rosenbrock uniform f111", 500, lambda **_: _noisy("N02", 111, "Rosenbrock with uniform noise")),
    "N03": ProblemSpec("N03", "noise", "bbob_noisy", "Rosenbrock seldom-Cauchy f112", 500, lambda **_: _noisy("N03", 112, "Rosenbrock with seldom-Cauchy noise")),
    "N04": ProblemSpec("N04", "noise", "nevergrad", "Noisy Rosenbrock D=20", 500, lambda seed=0, **_: _nevergrad("N04", "rosenbrock", seed)),
    "N05": ProblemSpec("N05", "noise", "nevergrad", "Noisy Cigar D=20", 500, lambda seed=0, **_: _nevergrad("N05", "cigar", seed)),
    "S01": ProblemSpec("S01", "smoothness", "coco_bbob", "Step Ellipsoid f7 D=10", 500, lambda **_: _bbob("S01", 7, 10, 500, "smoothness", "Step Ellipsoid")),
    "S02": ProblemSpec("S02", "smoothness", "coco_bbob", "Sharp Ridge f13 D=10", 500, lambda **_: _bbob("S02", 13, 10, 500, "smoothness", "Sharp Ridge")),
    "S03": ProblemSpec("S03", "smoothness", "coco_bbob", "Weierstrass f16 D=10", 500, lambda **_: _bbob("S03", 16, 10, 500, "smoothness", "Weierstrass")),
    "S04": ProblemSpec("S04", "smoothness", "cec2017", "Rastrigin F5 D=30", 750, lambda **_: _cec("S04", 5, 30, 750, "smoothness")),
    "S05": ProblemSpec("S05", "smoothness", "cec2017", "Non-Continuous Rastrigin F8 D=30", 750, lambda **_: _cec("S05", 8, 30, 750, "smoothness")),
    "C01": ProblemSpec("C01", "constraints", "bbob_constrained", "Rotated Ellipsoid ID 20", 500, lambda **_: _constrained("C01", 20, "Constrained Rotated Ellipsoid")),
    "C02": ProblemSpec("C02", "constraints", "bbob_constrained", "Different Powers ID 38", 500, lambda **_: _constrained("C02", 38, "Constrained Different Powers")),
    "C03": ProblemSpec("C03", "constraints", "bbob_constrained", "Rastrigin ID 44", 500, lambda **_: _constrained("C03", 44, "Constrained Rastrigin")),
    "C04": ProblemSpec("C04", "constraints", "pymoo_g", "G04", 500, lambda **_: _pymoo("C04", "g4")),
    "C05": ProblemSpec("C05", "constraints", "pymoo_g", "G07", 500, lambda **_: _pymoo("C05", "g7")),
    "B01": ProblemSpec("B01", "true_bilevel", "SMD", "SMD2", 40, lambda **_: _smd("B01", 2)),
    "B02": ProblemSpec("B02", "true_bilevel", "SMD", "SMD3", 40, lambda **_: _smd("B02", 3)),
    "B03": ProblemSpec("B03", "true_bilevel", "SMD", "SMD6", 40, lambda **_: _smd("B03", 6)),
    "B04": ProblemSpec("B04", "true_bilevel", "BOLIB", "Calamai-Vicente 1994b", 40, lambda **_: _bolib("B04", "calamai_vicente_1994b")),
    "B05": ProblemSpec("B05", "true_bilevel", "BOLIB", "Allende-Still 2013", 40, lambda **_: _bolib("B05", "allende_still_2013")),
}


def list_problems() -> list[dict[str, object]]:
    return [PROBLEMS[key].public_record() for key in sorted(PROBLEMS)]


def create_problem(problem_id: str, *, seed: int = 0) -> Problem:
    try:
        spec = PROBLEMS[problem_id.upper()]
    except KeyError as exc:
        raise KeyError(f"Unknown problem ID {problem_id!r}. Choose one of {', '.join(sorted(PROBLEMS))}.") from exc
    return spec.factory(seed=seed)

