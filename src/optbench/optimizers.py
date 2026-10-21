from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc

from optbench.base import BackendUnavailable, BenchmarkProblem, BilevelProblem, BudgetExceeded


Objective = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    description: str
    runner: Callable[[Objective, np.ndarray, int, int], None] | None


def _random_search(objective: Objective, bounds: np.ndarray, budget: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(budget):
        objective(rng.uniform(bounds[:, 0], bounds[:, 1]))


def _sobol(objective: Objective, bounds: np.ndarray, budget: int, seed: int) -> None:
    sampler = qmc.Sobol(d=len(bounds), scramble=True, seed=seed)
    exponent = int(np.ceil(np.log2(max(1, budget))))
    points = qmc.scale(
        sampler.random_base2(exponent)[:budget], bounds[:, 0], bounds[:, 1]
    )
    for x in points:
        objective(x)


def _scipy_de(objective: Objective, bounds: np.ndarray, budget: int, seed: int) -> None:
    # BudgetExceeded is deliberately allowed to terminate SciPy exactly at the
    # external evaluation budget; the campaign runner retains all observations.
    differential_evolution(
        objective,
        [tuple(b) for b in bounds],
        seed=seed,
        popsize=10,
        mutation=0.8,
        recombination=0.9,
        polish=False,
        workers=1,
        updating="immediate",
        maxiter=max(1, budget),
    )


def _local(method: str) -> Callable[[Objective, np.ndarray, int, int], None]:
    def run(objective: Objective, bounds: np.ndarray, budget: int, seed: int) -> None:
        del seed
        x0 = np.mean(bounds, axis=1)
        options = {"maxiter": max(1, budget)}
        if method == "Nelder-Mead":
            options.update({"adaptive": True, "maxfev": max(1, budget)})
        else:
            options["maxfun"] = max(1, budget)
        minimize(objective, x0, method=method, bounds=[tuple(b) for b in bounds], options=options)
    return run


def _cma_es(objective: Objective, bounds: np.ndarray, budget: int, seed: int) -> None:
    try:
        import cma
    except ImportError as exc:
        raise BackendUnavailable("CMA-ES requires `python -m pip install -e \".[optimizers]\"`.") from exc
    lower, upper = bounds[:, 0], bounds[:, 1]
    x0 = np.mean(bounds, axis=1)
    sigma0 = float(np.mean(upper - lower) / 4.0)
    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {"bounds": [lower.tolist(), upper.tolist()], "seed": seed, "verbose": -9},
    )
    used = 0
    while used < budget and not es.stop():
        population = es.ask()
        remaining = budget - used
        if len(population) > remaining:
            for x in population[:remaining]:
                objective(np.asarray(x))
            return
        values = [objective(np.asarray(x)) for x in population]
        used += len(population)
        es.tell(population, values)


def _bobyqa(objective: Objective, bounds: np.ndarray, budget: int, seed: int) -> None:
    del seed
    try:
        import pybobyqa
    except ImportError as exc:
        raise BackendUnavailable("BOBYQA requires `python -m pip install -e \".[optimizers]\"`.") from exc
    pybobyqa.solve(
        objective,
        np.mean(bounds, axis=1),
        bounds=(bounds[:, 0], bounds[:, 1]),
        scaling_within_bounds=True,
        maxfun=budget,
        do_logging=False,
    )


def _gp_bo(objective: Objective, bounds: np.ndarray, budget: int, seed: int) -> None:
    try:
        from skopt import Optimizer
        from skopt.space import Real
    except ImportError as exc:
        raise BackendUnavailable("GP-BO requires `python -m pip install -e \".[optimizers]\"`.") from exc
    optimizer = Optimizer(
        [Real(float(lo), float(hi)) for lo, hi in bounds],
        base_estimator="GP",
        acq_func="EI",
        n_initial_points=min(10, budget),
        random_state=seed,
    )
    for _ in range(budget):
        x = optimizer.ask()
        y = objective(np.asarray(x, dtype=float))
        optimizer.tell(x, y)


def _turbo(objective: Objective, bounds: np.ndarray, budget: int, seed: int) -> None:
    """Compact sequential TuRBO-1 implementation following the BoTorch recipe."""
    try:
        import torch
        from botorch.acquisition import LogExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as exc:
        raise BackendUnavailable("TuRBO requires `python -m pip install -e \".[optimizers]\"`.") from exc

    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lower, upper = bounds[:, 0], bounds[:, 1]
    x_unit: list[np.ndarray] = []
    values: list[float] = []
    initial = min(2 * dim, budget)
    sobol = qmc.Sobol(dim, scramble=True, seed=seed)
    for u in sobol.random(initial):
        values.append(objective(lower + u * (upper - lower)))
        x_unit.append(u)
    length, successes, failures = 0.8, 0, 0
    best = min(values)
    while len(values) < budget:
        train_x = torch.tensor(np.asarray(x_unit), dtype=torch.double)
        train_y = torch.tensor(np.asarray(values)[:, None], dtype=torch.double)
        model = SingleTaskGP(
            train_x,
            train_y,
            input_transform=Normalize(d=dim),
            outcome_transform=Standardize(m=1),
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        center = train_x[int(np.argmin(values))]
        tr_lower = torch.clamp(center - length / 2.0, 0.0, 1.0)
        tr_upper = torch.clamp(center + length / 2.0, 0.0, 1.0)
        acquisition = LogExpectedImprovement(model, best_f=float(best), maximize=False)
        candidate, _ = optimize_acqf(
            acquisition,
            bounds=torch.stack([tr_lower, tr_upper]),
            q=1,
            num_restarts=min(10, max(2, dim)),
            raw_samples=min(512, max(64, 16 * dim)),
        )
        u = candidate.detach().cpu().numpy().reshape(-1)
        value = objective(lower + u * (upper - lower))
        x_unit.append(u)
        values.append(value)
        if value < best - 1e-3 * max(1.0, abs(best)):
            best, successes, failures = value, successes + 1, 0
        else:
            failures, successes = failures + 1, 0
        if successes >= 3:
            length, successes = min(1.6, 2.0 * length), 0
        if failures >= max(4, dim):
            length, failures = length / 2.0, 0
        if length < 0.5**7 and len(values) < budget:
            # Fresh random center; retain global observations for scoring.
            u = rng.random(dim)
            value = objective(lower + u * (upper - lower))
            x_unit.append(u)
            values.append(value)
            length, successes, failures = 0.8, 0, 0


OPTIMIZERS: dict[str, OptimizerSpec] = {
    "random_search": OptimizerSpec("random_search", "Uniform random search", _random_search),
    "sobol": OptimizerSpec("sobol", "Scrambled Sobol search", _sobol),
    "cma_es": OptimizerSpec("cma_es", "CMA-ES", _cma_es),
    "de": OptimizerSpec("de", "Differential evolution", _scipy_de),
    "gp_bo": OptimizerSpec("gp_bo", "Gaussian-process Bayesian optimization", _gp_bo),
    "turbo": OptimizerSpec("turbo", "Trust-region Bayesian optimization", _turbo),
    "bobyqa": OptimizerSpec("bobyqa", "Bounded quadratic approximation", _bobyqa),
    "nelder_mead": OptimizerSpec("nelder_mead", "Adaptive Nelder-Mead", _local("Nelder-Mead")),
    "lbfgsb": OptimizerSpec("lbfgsb", "L-BFGS-B", _local("L-BFGS-B")),
    "nested_random": OptimizerSpec("nested_random", "Nested random search for bilevel tasks", None),
}


def applicability(problem: BenchmarkProblem | BilevelProblem, optimizer_name: str) -> tuple[bool, str]:
    if optimizer_name not in OPTIMIZERS:
        return False, "unknown optimizer"
    if isinstance(problem, BilevelProblem):
        return (optimizer_name == "nested_random", "true bilevel tasks require a nested method")
    if optimizer_name == "nested_random":
        return False, "nested_random is only for true bilevel tasks"
    if not problem.is_numeric_box:
        return (optimizer_name == "random_search", "categorical/hierarchical tasks require configuration sampling")
    if any(t != "continuous" for t in problem.variable_types):
        allowed = optimizer_name in {"random_search", "de"}
        return allowed, "method must tolerate repaired integer/discrete coordinates"
    if problem.noisy and optimizer_name in {"bobyqa", "nelder_mead", "lbfgsb", "gp_bo", "turbo"}:
        return False, "this implementation assumes deterministic observations"
    return True, "applicable"


def run_numeric_optimizer(
    optimizer_name: str,
    objective: Objective,
    bounds: np.ndarray,
    budget: int,
    seed: int,
) -> None:
    spec = OPTIMIZERS[optimizer_name]
    if spec.runner is None:
        raise ValueError(f"{optimizer_name} does not use the numeric single-level runner.")
    try:
        spec.runner(objective, bounds, budget, seed)
    except BudgetExceeded:
        pass

