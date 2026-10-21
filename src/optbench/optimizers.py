from __future__ import annotations

from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Any, Callable

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc

from optbench.base import (
    BackendUnavailable,
    BenchmarkProblem,
    BilevelProblem,
    BudgetExceeded,
    EvaluationResult,
)


Objective = Callable[[np.ndarray], float]
ProblemEvaluator = Callable[[Any], EvaluationResult]
ProblemRunner = Callable[[BenchmarkProblem, ProblemEvaluator, int, int], Any | None]


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    description: str
    runner: Callable[[Objective, np.ndarray, int, int], None] | None
    problem_runner: ProblemRunner | None = None


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


def _nevergrad_continuous(name: str) -> ProblemRunner:
    def run(
        problem: BenchmarkProblem,
        evaluate: ProblemEvaluator,
        budget: int,
        seed: int,
    ) -> Any:
        try:
            import nevergrad as ng
        except ImportError as exc:
            raise BackendUnavailable(
                "Nevergrad SOTA methods require `python -m pip install -e \".[nevergrad]\"`."
            ) from exc
        assert problem.bounds is not None
        parametrization = ng.p.Array(
            init=np.mean(problem.bounds, axis=1)
        ).set_bounds(problem.bounds[:, 0], problem.bounds[:, 1])
        parametrization.random_state.seed(seed)
        optimizer_class = getattr(ng.optimizers, name)
        optimizer = optimizer_class(parametrization=parametrization, budget=budget, num_workers=1)
        for _ in range(budget):
            candidate = optimizer.ask()
            result = evaluate(np.asarray(candidate.value, dtype=float))
            value = result.objective if result.feasible else (
                1.0e12 + 1.0e6 * result.constraint_violation
            )
            optimizer.tell(candidate, float(value))
        return np.asarray(optimizer.provide_recommendation().value, dtype=float)

    return run


def _nevergrad_mixed(
    problem: BenchmarkProblem,
    evaluate: ProblemEvaluator,
    budget: int,
    seed: int,
) -> Any:
    try:
        import nevergrad as ng
    except ImportError as exc:
        raise BackendUnavailable(
            "Nevergrad mixed optimization requires `python -m pip install -e \".[nevergrad]\"`."
        ) from exc
    assert problem.bounds is not None
    parameters = []
    for variable_type, (lower, upper) in zip(problem.variable_types, problem.bounds):
        scalar = ng.p.Scalar(lower=float(lower), upper=float(upper))
        if variable_type in {"integer", "binary", "discrete"}:
            scalar.set_integer_casting()
        parameters.append(scalar)
    parametrization = ng.p.Tuple(*parameters)
    parametrization.random_state.seed(seed)
    optimizer = ng.optimizers.PortfolioDiscreteOnePlusOne(
        parametrization=parametrization,
        budget=budget,
        num_workers=1,
    )
    for _ in range(budget):
        candidate = optimizer.ask()
        x = np.asarray(candidate.value, dtype=float)
        result = evaluate(x)
        optimizer.tell(candidate, float(result.objective))
    return np.asarray(optimizer.provide_recommendation().value, dtype=float)


def _smac_hpo(
    problem: BenchmarkProblem,
    evaluate: ProblemEvaluator,
    budget: int,
    seed: int,
) -> Any:
    try:
        from smac import HyperparameterOptimizationFacade, Scenario
    except ImportError as exc:
        raise BackendUnavailable(
            "SMAC HPO requires `python -m pip install -e \".[hpo]\"`."
        ) from exc

    def target(configuration: Any, seed: int = seed) -> float:
        del seed
        values = dict(configuration)
        values = {key: value for key, value in values.items() if key in problem._variable_names}
        return float(evaluate(values).objective)

    with TemporaryDirectory(prefix="optbench-smac-") as directory:
        scenario = Scenario(
            problem._space,
            deterministic=True,
            n_trials=budget,
            seed=seed,
            output_directory=directory,
        )
        # Keep the official SMAC HPO pipeline while scaling its overhead to a
        # cheap 2,000-query surrogate benchmark.  The defaults search 10,000
        # acquisition challengers and refit every eight suggestions, which can
        # cost far more than the YAHPO objective itself.
        acquisition_maximizer = HyperparameterOptimizationFacade.get_acquisition_maximizer(
            scenario,
            challengers=1024,
            local_search_iterations=5,
        )
        config_selector = HyperparameterOptimizationFacade.get_config_selector(
            scenario,
            retrain_after=32,
        )
        optimizer = HyperparameterOptimizationFacade(
            scenario,
            target,
            acquisition_maximizer=acquisition_maximizer,
            config_selector=config_selector,
            overwrite=True,
            logging_level=False,
        )
        incumbent = optimizer.optimize()
    values = dict(incumbent)
    return {key: value for key, value in values.items() if key in problem._variable_names}


def _pymoo_isres(
    problem: BenchmarkProblem,
    evaluate: ProblemEvaluator,
    budget: int,
    seed: int,
) -> Any:
    try:
        import pymoo.algorithms.soo.nonconvex.sres as sres_module
        from pymoo.algorithms.soo.nonconvex.isres import ISRES
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.functions import FunctionLoader
        from pymoo.optimize import minimize as pymoo_minimize
    except ImportError as exc:
        raise BackendUnavailable(
            "ISRES requires `python -m pip install -e \".[constraints]\"`."
        ) from exc
    assert problem.bounds is not None

    class WrappedProblem(ElementwiseProblem):
        def __init__(self) -> None:
            super().__init__(
                n_var=problem.dimension,
                n_obj=1,
                n_ieq_constr=problem.number_of_constraints,
                xl=problem.bounds[:, 0],
                xu=problem.bounds[:, 1],
            )

        def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            result = evaluate(np.asarray(x, dtype=float))
            out["F"] = result.objective
            out["G"] = np.asarray(result.constraints, dtype=float)

    # ISRES requires at least two offspring per parent.  A compact population
    # leaves enough stochastic-ranking generations inside the fixed 2,000-call
    # benchmark budget (large default populations can spend the whole budget
    # before reaching the narrow feasible region of G7).  Complete generations
    # still fit exactly inside the external accounting limit.
    if budget < 6:
        rng = np.random.default_rng(seed)
        recommendation = None
        for _ in range(budget):
            recommendation = rng.uniform(problem.bounds[:, 0], problem.bounds[:, 1])
            evaluate(recommendation)
        return recommendation
    population = min(20, budget // 3)
    generations = max(1, (budget // population - 1) // 2)
    isres_budget = population * (1 + 2 * generations)
    # pymoo 0.6.1.6's Windows Cython extension is not NumPy-2 compatible
    # (C long versus long-long index buffers).  The official pure-Python
    # implementation is algorithmically identical and portable.
    loader = FunctionLoader.get_instance()
    previous_mode = loader.mode
    previous_sres_loader = sres_module.load_function
    loader.mode = "python"
    sres_module.load_function = lambda name: loader.load(name, mode="python")
    try:
        result = pymoo_minimize(
            WrappedProblem(),
            ISRES(pop_size=population, n_offsprings=2 * population),
            termination=("n_eval", isres_budget),
            # pymoo's setup path treats integer zero as an unset/falsy seed.
            # Offset the public benchmark seed so seed=0 stays reproducible.
            seed=seed + 1,
            verbose=False,
        )
    finally:
        sres_module.load_function = previous_sres_loader
        loader.mode = previous_mode
    recommendation = None if result.X is None else np.asarray(result.X, dtype=float)
    rng = np.random.default_rng(seed + 1)
    for _ in range(budget - isres_budget):
        candidate = rng.uniform(problem.bounds[:, 0], problem.bounds[:, 1])
        evaluate(candidate)
        if recommendation is None:
            recommendation = candidate
    return recommendation


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
    "ngopt": OptimizerSpec(
        "ngopt", "Nevergrad adaptive NGOpt portfolio", None,
        _nevergrad_continuous("NGOpt"),
    ),
    "tbpsa": OptimizerSpec(
        "tbpsa", "Nevergrad noise-specialist TBPSA", None,
        _nevergrad_continuous("TBPSA"),
    ),
    "ng_mixed_portfolio": OptimizerSpec(
        "ng_mixed_portfolio", "Nevergrad mixed/discrete portfolio", None,
        _nevergrad_mixed,
    ),
    "smac_hpo": OptimizerSpec(
        "smac_hpo", "SMAC model-based hyperparameter optimization", None,
        _smac_hpo,
    ),
    "pymoo_isres": OptimizerSpec(
        "pymoo_isres", "pymoo improved stochastic ranking", None,
        _pymoo_isres,
    ),
    "source_bilevel": OptimizerSpec(
        "source_bilevel", "Source-structured bilevel optimization", None,
    ),
}


def applicability(problem: BenchmarkProblem | BilevelProblem, optimizer_name: str) -> tuple[bool, str]:
    if optimizer_name not in OPTIMIZERS:
        return False, "unknown optimizer"
    if isinstance(problem, BilevelProblem):
        return (
            optimizer_name in {"nested_random", "source_bilevel"},
            "true bilevel tasks require a bilevel method",
        )
    if optimizer_name in {"nested_random", "source_bilevel"}:
        return False, "bilevel optimizer is only for true bilevel tasks"
    if optimizer_name == "smac_hpo":
        return (problem.family == "yahpo", "SMAC HPO is reserved for YAHPO ConfigSpace tasks")
    if optimizer_name == "ng_mixed_portfolio":
        return (
            problem.is_numeric_box and any(t != "continuous" for t in problem.variable_types),
            "mixed portfolio requires a numeric mixed-variable box",
        )
    if optimizer_name == "tbpsa":
        return (problem.is_numeric_box and problem.noisy, "TBPSA is selected for noisy numeric tasks")
    if optimizer_name == "pymoo_isres":
        return (
            problem.is_numeric_box and problem.number_of_constraints > 0,
            "ISRES requires explicit numeric inequality constraints",
        )
    if optimizer_name == "ngopt":
        return (problem.is_numeric_box, "NGOpt requires a numeric box")
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


def run_problem_optimizer(
    optimizer_name: str,
    problem: BenchmarkProblem,
    evaluate: ProblemEvaluator,
    budget: int,
    seed: int,
) -> Any | None:
    runner = OPTIMIZERS[optimizer_name].problem_runner
    if runner is None:
        raise ValueError(f"{optimizer_name} does not use the problem-aware runner.")
    return runner(problem, evaluate, budget, seed)

