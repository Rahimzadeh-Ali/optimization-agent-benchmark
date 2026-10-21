from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from optbench.base import BenchmarkProblem, BilevelProblem, EvaluationResult
from optbench.optimizers import (
    OPTIMIZERS,
    applicability,
    run_numeric_optimizer,
    run_problem_optimizer,
)
from optbench.references import objective_reference
from optbench.registry import PROBLEMS, create_problem
from optbench.sota import sota_method


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(value), sort_keys=True) + "\n")


def _quality(result: EvaluationResult) -> tuple[int, float, float]:
    """Lexicographic feasibility-first ordering used only to select incumbents."""
    return (0 if result.feasible else 1, result.constraint_violation, result.objective)


def _merit(result: EvaluationResult) -> float:
    """Finite scalar exposed to optimizers that only accept one objective."""
    if result.feasible:
        return result.objective
    return 1.0e12 + 1.0e6 * result.constraint_violation + float(np.tanh(result.objective))


def _versions() -> dict[str, str]:
    names = ["numpy", "scipy", "coco-experiment", "opfunu", "pymoo", "nevergrad", "yahpo-gym"]
    found: dict[str, str] = {"python": platform.python_version()}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return found


def _make_run_dir(root: str | Path, problem_id: str, optimizer: str, seed: int) -> tuple[str, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_id = f"{problem_id}_{optimizer}_seed{seed}_{stamp}"
    path = Path(root).resolve() / run_id
    path.mkdir(parents=True, exist_ok=False)
    return run_id, path


def _repair_numeric(problem: BenchmarkProblem, x: np.ndarray) -> np.ndarray:
    assert problem.bounds is not None
    candidate = np.clip(np.asarray(x, dtype=float), problem.bounds[:, 0], problem.bounds[:, 1])
    for index, variable_type in enumerate(problem.variable_types):
        if variable_type in {"integer", "binary", "discrete"}:
            candidate[index] = np.rint(candidate[index])
    return candidate


def _run_single(
    problem: BenchmarkProblem,
    optimizer: str,
    seed: int,
    budget: int,
    trace_path: Path,
) -> dict[str, Any]:
    problem.reset(seed)
    problem.set_budget_limit(budget)
    started = time.perf_counter()
    best: tuple[Any, EvaluationResult] | None = None

    def record(candidate: Any) -> EvaluationResult:
        nonlocal best
        if isinstance(candidate, np.ndarray) and problem.bounds is not None:
            candidate = _repair_numeric(problem, candidate)
        result = problem.evaluate(candidate)
        if best is None or _quality(result) < _quality(best[1]):
            best = (candidate, result)
        _append_jsonl(
            trace_path,
            {
                "phase": "optimization",
                "evaluation": problem.evaluations,
                "elapsed_seconds": time.perf_counter() - started,
                "candidate": candidate,
                **result.to_dict(),
            },
        )
        return result

    def evaluate(candidate: Any) -> float:
        return _merit(record(candidate))

    recommendation: Any | None = None
    if optimizer == "random_search":
        rng = np.random.default_rng(seed)
        for _ in range(budget):
            evaluate(problem.sample(rng))
    else:
        spec = OPTIMIZERS[optimizer]
        if spec.problem_runner is not None:
            recommendation = run_problem_optimizer(optimizer, problem, record, budget, seed)
        else:
            if problem.bounds is None:
                raise TypeError(f"{optimizer} requires a numeric box.")
            run_numeric_optimizer(optimizer, evaluate, problem.bounds, budget, seed)

    optimization_evaluations = problem.evaluations
    validation: dict[str, Any] | None = None
    if problem.noisy and best is not None:
        # Validation calls are deliberately outside the optimization budget and
        # are marked separately so they cannot be mistaken for agent queries.
        problem.set_budget_limit(None)
        repeats = 20
        values: list[float] = []
        validation_candidate = best[0] if recommendation is None else recommendation
        for repeat in range(repeats):
            result = problem.evaluate(validation_candidate)
            values.append(result.objective)
            _append_jsonl(
                trace_path,
                {
                    "phase": "validation",
                    "repeat": repeat + 1,
                    "candidate": validation_candidate,
                    **result.to_dict(),
                },
            )
        validation = {
            "repeats": repeats,
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)),
            "candidate": validation_candidate,
        }

    return {
        "optimization_evaluations": optimization_evaluations,
        "best": None if best is None else {"candidate": best[0], **best[1].to_dict()},
        "noisy_validation": validation,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _run_bilevel(
    problem: BilevelProblem,
    optimizer: str,
    seed: int,
    upper_budget: int,
    inner_budget: int,
    trace_path: Path,
) -> dict[str, Any]:
    if optimizer == "source_bilevel":
        return _run_source_bilevel(problem, seed, upper_budget, inner_budget, trace_path)

    problem.reset(seed)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    best: tuple[np.ndarray, np.ndarray, EvaluationResult, dict[str, Any]] | None = None

    for outer_index in range(upper_budget):
        xu = rng.uniform(problem.upper_bounds[:, 0], problem.upper_bounds[:, 1])
        lower_best: tuple[np.ndarray, EvaluationResult] | None = None
        for inner_index in range(inner_budget):
            xl = rng.uniform(problem.lower_bounds[:, 0], problem.lower_bounds[:, 1])
            lower_result = problem.evaluate_lower(xu, xl)
            if lower_best is None or _quality(lower_result) < _quality(lower_best[1]):
                lower_best = (xl, lower_result)
            _append_jsonl(
                trace_path,
                {
                    "phase": "lower_optimization",
                    "outer_evaluation": outer_index + 1,
                    "inner_evaluation": inner_index + 1,
                    "xu": xu,
                    "xl": xl,
                    **lower_result.to_dict(),
                },
            )

        assert lower_best is not None
        xl, lower_result = lower_best
        upper_result = problem.evaluate_upper(xu, xl)
        lower_validation = problem.validate_lower_solution(xu, xl)
        _append_jsonl(
            trace_path,
            {
                "phase": "upper_evaluation",
                "upper_evaluation": outer_index + 1,
                "xu": xu,
                "xl": xl,
                "lower_incumbent": lower_result.to_dict(),
                "lower_validation": lower_validation,
                **upper_result.to_dict(),
            },
        )
        candidate = (xu, xl, upper_result, lower_validation)
        if best is None:
            best = candidate
        else:
            candidate_key = (not bool(lower_validation["valid"]),) + _quality(upper_result)
            best_key = (not bool(best[3]["valid"]),) + _quality(best[2])
            if candidate_key < best_key:
                best = candidate

    return {
        "upper_evaluations": problem.upper_evaluations,
        "lower_evaluations": problem.lower_evaluations,
        "best": None
        if best is None
        else {
            "xu": best[0],
            "xl": best[1],
            **best[2].to_dict(),
            "lower_validation": best[3],
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def _run_source_bilevel(
    problem: BilevelProblem,
    seed: int,
    upper_budget: int,
    inner_budget: int,
    trace_path: Path,
) -> dict[str, Any]:
    """BLEAQ-II-style leader search with the strongest available follower solver."""
    from scipy.optimize import minimize

    problem.reset(seed)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    outer_index = 0
    best: tuple[np.ndarray, np.ndarray, EvaluationResult, dict[str, Any]] | None = None

    def solve_follower(xu: np.ndarray, current_outer: int) -> tuple[np.ndarray, EvaluationResult]:
        # SMD publishes the exact lower-level optimal mapping. Using it is the
        # source-provided plug-and-play follower model used by BLEAQ-II tests.
        if hasattr(problem, "analytic_lower_response"):
            xl = np.asarray(problem.analytic_lower_response(xu), dtype=float)
            lower = problem.evaluate_lower(xu, xl)
            _append_jsonl(
                trace_path,
                {
                    "phase": "lower_optimization",
                    "outer_evaluation": current_outer,
                    "inner_evaluation": 1,
                    "method": "published_analytic_follower",
                    "xu": xu,
                    "xl": xl,
                    **lower.to_dict(),
                },
            )
            return xl, lower

        evaluated: list[tuple[np.ndarray, EvaluationResult]] = []

        def lower_objective(xl: np.ndarray) -> float:
            if len(evaluated) >= inner_budget:
                return 1.0e15
            candidate = np.clip(
                np.asarray(xl, dtype=float),
                problem.lower_bounds[:, 0],
                problem.lower_bounds[:, 1],
            )
            result = problem.evaluate_lower(xu, candidate)
            evaluated.append((candidate.copy(), result))
            _append_jsonl(
                trace_path,
                {
                    "phase": "lower_optimization",
                    "outer_evaluation": current_outer,
                    "inner_evaluation": len(evaluated),
                    "method": "slsqp_follower",
                    "xu": xu,
                    "xl": candidate,
                    **result.to_dict(),
                },
            )
            return _merit(result)

        def lower_feasibility(raw_xl: np.ndarray) -> np.ndarray:
            candidate = np.clip(
                np.asarray(raw_xl, dtype=float),
                problem.lower_bounds[:, 0],
                problem.lower_bounds[:, 1],
            )
            return -np.asarray(problem.lower_constraints(xu, candidate), dtype=float)

        constraints = {
            "type": "ineq",
            "fun": lower_feasibility,
        }
        starts = [
            np.mean(problem.lower_bounds, axis=1),
            rng.uniform(problem.lower_bounds[:, 0], problem.lower_bounds[:, 1]),
        ]
        allocation = max(1, inner_budget // len(starts))
        for start in starts:
            if len(evaluated) >= inner_budget:
                break
            minimize(
                lower_objective,
                start,
                method="SLSQP",
                bounds=[tuple(row) for row in problem.lower_bounds],
                constraints=constraints,
                options={"maxiter": allocation, "ftol": 1.0e-10, "disp": False},
            )
        if not evaluated:
            raise RuntimeError("Follower optimizer produced no evaluations.")
        feasible = [item for item in evaluated if item[1].feasible]
        pool = feasible or evaluated
        return min(pool, key=lambda item: _quality(item[1]))

    def leader_objective(raw_xu: np.ndarray) -> float:
        nonlocal outer_index, best
        xu = np.clip(
            np.asarray(raw_xu, dtype=float),
            problem.upper_bounds[:, 0],
            problem.upper_bounds[:, 1],
        )
        outer_index += 1
        xl, lower = solve_follower(xu, outer_index)
        upper = problem.evaluate_upper(xu, xl)
        validation = problem.validate_lower_solution(xu, xl)
        _append_jsonl(
            trace_path,
            {
                "phase": "upper_evaluation",
                "upper_evaluation": outer_index,
                "method": "cma_es_leader",
                "xu": xu,
                "xl": xl,
                "lower_incumbent": lower.to_dict(),
                "lower_validation": validation,
                **upper.to_dict(),
            },
        )
        candidate = (xu.copy(), xl.copy(), upper, validation)
        if best is None:
            best = candidate
        else:
            candidate_key = (not bool(validation["valid"]),) + _quality(upper)
            best_key = (not bool(best[3]["valid"]),) + _quality(best[2])
            if candidate_key < best_key:
                best = candidate
        if not validation["valid"]:
            return 1.0e12 + 1.0e6 * float(validation.get("optimality_gap", 1.0))
        return _merit(upper)

    run_numeric_optimizer("cma_es", leader_objective, problem.upper_bounds, upper_budget, seed)
    return {
        "upper_evaluations": problem.upper_evaluations,
        "lower_evaluations": problem.lower_evaluations,
        "best": None
        if best is None
        else {
            "xu": best[0],
            "xl": best[1],
            **best[2].to_dict(),
            "lower_validation": best[3],
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_campaign(
    problem_id: str,
    optimizer: str,
    *,
    seed: int = 0,
    budget: int | None = None,
    inner_budget: int | None = None,
    output_root: str | Path = "results/raw",
) -> dict[str, Any]:
    """Run one isolated campaign and return the persisted summary."""
    problem_id = problem_id.upper()
    if problem_id not in PROBLEMS:
        raise KeyError(f"Unknown problem ID {problem_id!r}.")
    if optimizer not in OPTIMIZERS:
        raise KeyError(f"Unknown optimizer {optimizer!r}.")

    run_id, run_dir = _make_run_dir(output_root, problem_id, optimizer, seed)
    trace_path = run_dir / "trace.jsonl"
    started_at = datetime.now(timezone.utc).isoformat()
    selected_budget: int | None = None
    selected_inner: int | None = None
    try:
        problem = create_problem(problem_id, seed=seed)
        allowed, reason = applicability(problem, optimizer)
        if not allowed:
            result: dict[str, Any] = {"reason": reason}
            status = "not_applicable"
        elif isinstance(problem, BilevelProblem):
            selected_budget = problem.upper_budget if budget is None else int(budget)
            selected_inner = (
                problem.lower_budget_per_upper_candidate if inner_budget is None else int(inner_budget)
            )
            if selected_budget < 1 or selected_inner < 1:
                raise ValueError("Both bilevel budgets must be positive.")
            result = _run_bilevel(problem, optimizer, seed, selected_budget, selected_inner, trace_path)
            status = "completed"
        else:
            selected_budget = problem.budget if budget is None else int(budget)
            if selected_budget < 1:
                raise ValueError("The evaluation budget must be positive.")
            result = _run_single(problem, optimizer, seed, selected_budget, trace_path)
            status = "completed"
    except Exception as exc:
        summary = {
            "run_id": run_id,
            "problem_id": problem_id,
            "optimizer": optimizer,
            "seed": seed,
            "status": "failed",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "versions": _versions(),
        }
        _write_json(run_dir / "summary.json", summary)
        raise

    summary = {
        "run_id": run_id,
        "problem_id": problem_id,
        "optimizer": optimizer,
        "seed": seed,
        "status": status,
        "budget_limit": selected_budget,
        "inner_budget_limit": selected_inner,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "task": problem.agent_spec(),
        "sota_method": sota_method(problem_id).to_dict()
        if optimizer == sota_method(problem_id).optimizer
        else None,
        "regret_reference": objective_reference(problem).to_dict(),
        "result": result,
        "versions": _versions(),
        "artifacts": {"trace": "trace.jsonl", "summary": "summary.json"},
    }
    _write_json(run_dir / "summary.json", summary)
    return _jsonable(summary)


def build_matrix_plan(
    problem_ids: list[str] | None = None,
    optimizers: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Build a backend-free campaign plan; `auto` covers every task once per seed."""
    selected_problems = sorted(PROBLEMS) if not problem_ids or problem_ids == ["all"] else problem_ids
    selected_optimizers = ["auto"] if not optimizers else optimizers
    selected_seeds = [0] if not seeds else seeds
    plan: list[dict[str, Any]] = []
    for raw_problem_id in selected_problems:
        problem_id = raw_problem_id.upper()
        if problem_id not in PROBLEMS:
            raise KeyError(f"Unknown problem ID {problem_id!r}.")
        for requested_optimizer in selected_optimizers:
            optimizer = requested_optimizer
            if optimizer == "auto":
                optimizer = sota_method(problem_id).optimizer
            if optimizer not in OPTIMIZERS:
                raise KeyError(f"Unknown optimizer {optimizer!r}.")
            for seed in selected_seeds:
                plan.append({"problem_id": problem_id, "optimizer": optimizer, "seed": int(seed)})
    return plan


def run_matrix(
    plan: list[dict[str, Any]],
    *,
    output_root: str | Path = "results/raw",
    budget: int | None = None,
    inner_budget: int | None = None,
) -> list[dict[str, Any]]:
    """Execute a plan sequentially and continue after independently logged failures."""
    outcomes: list[dict[str, Any]] = []
    for item in plan:
        try:
            summary = run_campaign(
                item["problem_id"],
                item["optimizer"],
                seed=item["seed"],
                budget=budget,
                inner_budget=inner_budget,
                output_root=output_root,
            )
            outcomes.append(
                {
                    **item,
                    "status": summary["status"],
                    "run_id": summary["run_id"],
                }
            )
        except Exception as exc:
            outcomes.append(
                {
                    **item,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return outcomes
