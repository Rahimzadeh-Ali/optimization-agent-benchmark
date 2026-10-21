from __future__ import annotations

import json
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PALETTE = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
    "#898781",
    "#17a2b8",
)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#686762"
GRID = "#e1e0d9"
INVALID = "#b5b3ad"


@dataclass(frozen=True)
class RunArtifact:
    summary_path: Path
    summary: dict[str, Any]

    @property
    def trace_path(self) -> Path:
        relative = self.summary.get("artifacts", {}).get("trace", "trace.jsonl")
        return self.summary_path.parent / relative


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires `python -m pip install -e \".[visualization]\"`."
        ) from exc

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": TEXT_MUTED,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "text.color": TEXT,
            "savefig.facecolor": SURFACE,
        }
    )
    return plt


def discover_runs(root: str | Path) -> list[RunArtifact]:
    runs: list[RunArtifact] = []
    for path in sorted(Path(root).resolve().rglob("summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runs.append(RunArtifact(path, summary))
    return runs


def _latest_unique_runs(runs: list[RunArtifact]) -> tuple[list[RunArtifact], int]:
    """Keep only the latest completed artifact for each campaign key.

    A resumed campaign can leave more than one successful directory for the
    same problem, optimizer, and seed.  Treating those directories as separate
    seeds would give the duplicate campaign extra statistical weight.
    """
    latest: dict[tuple[str, str, int], RunArtifact] = {}
    for run in runs:
        key = (
            str(run.summary.get("problem_id", "")),
            str(run.summary.get("optimizer", "")),
            int(run.summary.get("seed", 0)),
        )
        previous = latest.get(key)
        ordering = (str(run.summary.get("finished_at", "")), str(run.summary_path))
        if previous is None:
            latest[key] = run
            continue
        previous_ordering = (
            str(previous.summary.get("finished_at", "")),
            str(previous.summary_path),
        )
        if ordering > previous_ordering:
            latest[key] = run
    selected = sorted(latest.values(), key=lambda run: str(run.summary_path))
    return selected, len(runs) - len(selected)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _valid_result(record: dict[str, Any]) -> bool:
    lower_valid = record.get("lower_validation", {}).get("valid", True)
    return bool(record.get("feasible", True) and lower_valid)


def _total_violation(record: dict[str, Any]) -> float:
    violation = float(record.get("constraint_violation", 0.0))
    lower = record.get("lower_validation", {})
    if not lower.get("valid", True):
        gap = lower.get("optimality_gap")
        if gap is not None and np.isfinite(float(gap)):
            violation += max(0.0, float(gap))
        elif violation == 0.0:
            violation = 1.0
    return violation


def _trajectory(run: RunArtifact) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points: list[tuple[int, float, bool, float]] = []
    for row in _read_jsonl(run.trace_path):
        if row.get("phase") == "optimization":
            index = int(row["evaluation"])
        elif row.get("phase") == "upper_evaluation":
            index = int(row["upper_evaluation"])
        else:
            continue
        points.append(
            (
                index,
                float(row["objective"]),
                _valid_result(row),
                _total_violation(row),
            )
        )

    points.sort(key=lambda item: item[0])
    indices: list[int] = []
    objectives: list[float] = []
    violations: list[float] = []
    best_objective = np.inf
    best_violation = np.inf
    for index, objective, valid, violation in points:
        if valid:
            best_objective = min(best_objective, objective)
            best_violation = 0.0
        else:
            best_violation = min(best_violation, violation)
        indices.append(index)
        objectives.append(best_objective if np.isfinite(best_objective) else np.nan)
        violations.append(best_violation if np.isfinite(best_violation) else np.nan)
    return np.asarray(indices), np.asarray(objectives), np.asarray(violations)


def _regret_trajectory(run: RunArtifact) -> tuple[np.ndarray, np.ndarray, str]:
    indices, objectives, _ = _trajectory(run)
    reference_data = run.summary.get("regret_reference", {})
    reference = reference_data.get("objective")
    kind = str(reference_data.get("kind", "run_best"))
    if reference is None:
        finite = objectives[np.isfinite(objectives)]
        reference = float(np.min(finite)) if len(finite) else 0.0
        kind = "run_best"
    regrets = objectives - float(reference)
    # Numerical noise and stochastic observations may cross a latent reference.
    regrets = np.where(np.isfinite(regrets), np.maximum(regrets, 1.0e-12), np.nan)
    return indices, regrets, kind


def _aggregate_series(
    runs: Iterable[RunArtifact], series_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trajectories = [_trajectory(run) for run in runs]
    max_index = max((int(values[0][-1]) for values in trajectories if len(values[0])), default=0)
    matrix = np.full((len(trajectories), max_index), np.nan)
    for row_index, values in enumerate(trajectories):
        indices, series = values[0], values[series_index]
        if len(indices):
            matrix[row_index, indices.astype(int) - 1] = series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(matrix, axis=0)
        lower = np.nanpercentile(matrix, 25, axis=0)
        upper = np.nanpercentile(matrix, 75, axis=0)
    return np.arange(1, max_index + 1), median, lower, upper


def _is_constraint_sensitive(run: RunArtifact) -> bool:
    task = run.summary.get("task", {})
    return bool(
        task.get("direction") == "true_bilevel"
        or task.get("number_of_inequality_constraints", 0)
        or task.get("upper_inequality_constraints", 0)
        or task.get("lower_inequality_constraints", 0)
    )


def _group_by_optimizer(runs: Iterable[RunArtifact]) -> dict[str, list[RunArtifact]]:
    grouped: dict[str, list[RunArtifact]] = defaultdict(list)
    for run in runs:
        grouped[str(run.summary["optimizer"])].append(run)
    return dict(sorted(grouped.items()))


def _style_axis(axis: Any) -> None:
    axis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_problem_convergence(runs: list[RunArtifact], output: Path) -> None:
    plt = _matplotlib()
    problem_id = str(runs[0].summary["problem_id"])
    constrained = _is_constraint_sensitive(runs[0])
    if constrained:
        figure, (objective_axis, violation_axis) = plt.subplots(
            2, 1, figsize=(10.5, 7.0), dpi=150, sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0]},
        )
    else:
        figure, objective_axis = plt.subplots(figsize=(10.5, 5.8), dpi=150)
        violation_axis = None

    any_valid = False
    for color_index, (optimizer, optimizer_runs) in enumerate(_group_by_optimizer(runs).items()):
        color = PALETTE[color_index % len(PALETTE)]
        x, median, lower, upper = _aggregate_series(optimizer_runs, 1)
        finite = np.isfinite(median)
        if finite.any():
            any_valid = True
            objective_axis.plot(x[finite], median[finite], color=color, linewidth=2.0, label=optimizer)
            objective_axis.fill_between(
                x, lower, upper, where=np.isfinite(lower) & np.isfinite(upper),
                color=color, alpha=0.14, linewidth=0,
            )
        if violation_axis is not None:
            vx, vmedian, vlower, vupper = _aggregate_series(optimizer_runs, 2)
            valid_v = np.isfinite(vmedian)
            if valid_v.any():
                violation_axis.plot(vx[valid_v], vmedian[valid_v], color=color, linewidth=1.8)
                violation_axis.fill_between(
                    vx, vlower, vupper,
                    where=np.isfinite(vlower) & np.isfinite(vupper),
                    color=color, alpha=0.12, linewidth=0,
                )

    if not any_valid:
        objective_axis.text(
            0.5, 0.5, "No feasible, lower-level-valid objective was observed.",
            transform=objective_axis.transAxes, ha="center", va="center", color=TEXT_MUTED,
        )
    objective_axis.set_ylabel("Best valid objective")
    objective_axis.set_title(
        f"{problem_id}: best valid objective vs. evaluation",
        loc="left", fontsize=13, pad=12,
    )
    _style_axis(objective_axis)
    handles, legend_labels = objective_axis.get_legend_handles_labels()
    if handles:
        objective_axis.legend(
            handles, legend_labels, frameon=False,
            bbox_to_anchor=(1.01, 0.5), loc="center left",
        )
    if violation_axis is not None:
        violation_axis.set_xlabel("Upper/objective evaluation")
        violation_axis.set_ylabel("Best total violation")
        violation_axis.set_yscale("symlog", linthresh=1e-8)
        _style_axis(violation_axis)
    else:
        objective_axis.set_xlabel("Evaluation")
    figure.text(
        0.01, 0.01,
        "Line = median across seeds; band = interquartile range. Invalid points never improve the objective curve.",
        fontsize=8, color=TEXT_MUTED,
    )
    figure.tight_layout(rect=(0, 0.03, 0.86, 1))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_problem_regret(runs: list[RunArtifact], output: Path) -> None:
    plt = _matplotlib()
    problem_id = str(runs[0].summary["problem_id"])
    figure, axis = plt.subplots(figsize=(10.5, 5.8), dpi=150)
    kinds: set[str] = set()
    any_regret = False
    for color_index, (optimizer, optimizer_runs) in enumerate(_group_by_optimizer(runs).items()):
        color = PALETTE[color_index % len(PALETTE)]
        trajectories = [_regret_trajectory(run) for run in optimizer_runs]
        kinds.update(item[2] for item in trajectories)
        max_index = max((int(item[0][-1]) for item in trajectories if len(item[0])), default=0)
        matrix = np.full((len(trajectories), max_index), np.nan)
        for row, (indices, regrets, _) in enumerate(trajectories):
            if len(indices):
                matrix[row, indices.astype(int) - 1] = regrets
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(matrix, axis=0)
            lower = np.nanpercentile(matrix, 25, axis=0)
            upper = np.nanpercentile(matrix, 75, axis=0)
        x = np.arange(1, max_index + 1)
        finite = np.isfinite(median)
        if finite.any():
            any_regret = True
            axis.plot(x[finite], median[finite], color=color, linewidth=2.0, label=optimizer)
        axis.fill_between(
            x,
            lower,
            upper,
            where=np.isfinite(lower) & np.isfinite(upper),
            color=color,
            alpha=0.14,
            linewidth=0,
        )
    if any_regret:
        axis.set_yscale("log")
    else:
        axis.text(
            0.5,
            0.5,
            "No feasible regret trajectory was observed.",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color=TEXT_MUTED,
        )
    axis.set_xlabel("Evaluation")
    axis.set_ylabel("Best feasible regret")
    axis.set_title(f"{problem_id}: iteration vs regret", loc="left", fontsize=13, pad=12)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, frameon=False)
    _style_axis(axis)
    figure.text(
        0.01,
        0.01,
        "Reference type: " + ", ".join(sorted(kinds)) + ". Regret is clipped at 1e-12 for log display.",
        fontsize=8,
        color=TEXT_MUTED,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_suite_regret_dashboard(
    by_problem: dict[str, list[RunArtifact]], output: Path
) -> None:
    plt = _matplotlib()
    ordered = sorted(by_problem, key=_problem_sort_key)
    figure, axes = plt.subplots(6, 5, figsize=(18, 19), dpi=150)
    for axis, problem_id in zip(axes.flat, ordered):
        runs = by_problem[problem_id]
        any_regret = False
        for color_index, (optimizer, optimizer_runs) in enumerate(_group_by_optimizer(runs).items()):
            color = PALETTE[color_index % len(PALETTE)]
            for run in optimizer_runs:
                x, regret, _ = _regret_trajectory(run)
                finite = np.isfinite(regret)
                if finite.any():
                    any_regret = True
                    axis.plot(x[finite], regret[finite], color=color, linewidth=1.35, alpha=0.9)
            axis.text(
                0.98,
                0.96,
                optimizer,
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=6.5,
                color=TEXT_MUTED,
            )
        if any_regret:
            axis.set_yscale("log")
        else:
            axis.text(
                0.5,
                0.5,
                "No valid\nregret",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=7,
                color=TEXT_MUTED,
            )
        axis.set_title(problem_id, loc="left", fontsize=10, pad=4)
        axis.tick_params(labelsize=7)
        _style_axis(axis)
    for axis in axes[-1, :]:
        axis.set_xlabel("Evaluation", fontsize=8)
    for axis in axes[:, 0]:
        axis.set_ylabel("Regret", fontsize=8)
    figure.suptitle("SOTA optimizer profiles: iteration versus regret", fontsize=17, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.988))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _summary_value(run: RunArtifact) -> tuple[bool, float, float] | None:
    best = run.summary.get("result", {}).get("best")
    if not best:
        return None
    valid = _valid_result(best)
    noisy = run.summary.get("result", {}).get("noisy_validation")
    objective = float(noisy["mean"]) if noisy else float(best["objective"])
    return valid, objective, _total_violation(best)


def plot_problem_final(runs: list[RunArtifact], output: Path) -> None:
    plt = _matplotlib()
    problem_id = str(runs[0].summary["problem_id"])
    grouped = _group_by_optimizer(runs)
    labels = list(grouped)
    medians: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    valid_counts: list[int] = []
    total_counts: list[int] = []
    for optimizer_runs in grouped.values():
        values = [_summary_value(run) for run in optimizer_runs]
        values = [value for value in values if value is not None]
        valid_objectives = [value[1] for value in values if value[0]]
        total_counts.append(len(values))
        valid_counts.append(len(valid_objectives))
        if valid_objectives:
            q25, median, q75 = np.percentile(valid_objectives, [25, 50, 75])
            medians.append(float(median))
            lower_errors.append(float(median - q25))
            upper_errors.append(float(q75 - median))
        else:
            medians.append(np.nan)
            lower_errors.append(0.0)
            upper_errors.append(0.0)

    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(max(7.5, 1.2 * len(labels)), 5.2), dpi=150)
    for index, (label, median) in enumerate(zip(labels, medians)):
        if np.isfinite(median):
            axis.bar(index, median, color=PALETTE[index % len(PALETTE)], width=0.62, zorder=3)
            axis.errorbar(
                index, median,
                yerr=np.array([[lower_errors[index]], [upper_errors[index]]]),
                fmt="none", ecolor=TEXT, capsize=4, linewidth=1.1, zorder=4,
            )
            axis.annotate(
                f"{median:.5g}\nvalid {valid_counts[index]}/{total_counts[index]}",
                (index, median), xytext=(0, 8), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, color=TEXT_MUTED,
            )
        else:
            axis.scatter(index, 0.0, marker="x", s=70, color="#e34948", zorder=4)
            axis.annotate(
                f"no valid result\n0/{total_counts[index]}", (index, 0.0),
                xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8,
                color="#a52c2c",
            )
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylabel("Final valid objective (lower is better)")
    axis.set_title(f"{problem_id}: final performance by optimizer", loc="left", fontsize=13, pad=12)
    _style_axis(axis)
    figure.text(
        0.01, 0.01,
        "Bar = median across valid seeds; error bar = interquartile range. No-valid-result markers are not objective values.",
        fontsize=8, color=TEXT_MUTED,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _problem_sort_key(problem_id: str) -> tuple[int, int]:
    family_order = {letter: index for index, letter in enumerate("DVNSCB")}
    return family_order.get(problem_id[0], 99), int(problem_id[1:])


def _optimizer_ranks(runs: list[RunArtifact]) -> tuple[list[str], list[str], np.ndarray]:
    grouped: dict[tuple[str, str], list[tuple[bool, float, float]]] = defaultdict(list)
    for run in runs:
        value = _summary_value(run)
        if value is not None:
            grouped[(str(run.summary["problem_id"]), str(run.summary["optimizer"]))].append(value)
    problems = sorted({key[0] for key in grouped}, key=_problem_sort_key)
    optimizers = sorted({key[1] for key in grouped})
    matrix = np.full((len(problems), len(optimizers)), np.nan)
    for row, problem_id in enumerate(problems):
        quality: list[tuple[tuple[float, float, float], str]] = []
        for optimizer in optimizers:
            values = grouped.get((problem_id, optimizer), [])
            valid_objectives = [value[1] for value in values if value[0]]
            if valid_objectives:
                key = (0.0, 0.0, float(np.median(valid_objectives)))
            elif values:
                key = (
                    1.0,
                    float(np.median([value[2] for value in values])),
                    float(np.median([value[1] for value in values])),
                )
            else:
                continue
            quality.append((key, optimizer))
        quality.sort(key=lambda item: item[0])
        for rank, (_, optimizer) in enumerate(quality, start=1):
            matrix[row, optimizers.index(optimizer)] = rank
    return problems, optimizers, matrix


def plot_suite_rank_heatmap(runs: list[RunArtifact], output: Path) -> None:
    plt = _matplotlib()
    problems, optimizers, matrix = _optimizer_ranks(runs)
    figure, axis = plt.subplots(
        figsize=(max(7.5, 1.0 * len(optimizers)), max(4.5, 0.36 * len(problems) + 2.0)),
        dpi=150,
    )
    masked = np.ma.masked_invalid(matrix)
    image = axis.imshow(masked, cmap="viridis_r", aspect="auto", vmin=1, vmax=max(2, len(optimizers)))
    axis.set_xticks(np.arange(len(optimizers)), optimizers, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(problems)), problems)
    for row in range(len(problems)):
        for column in range(len(optimizers)):
            value = matrix[row, column]
            axis.text(
                column, row, "—" if np.isnan(value) else str(int(value)),
                ha="center", va="center", fontsize=8,
                color=TEXT_MUTED if np.isnan(value) else ("white" if value > len(optimizers) / 2 else TEXT),
            )
    axis.set_title("Per-task optimizer rank", loc="left", fontsize=13, pad=12)
    axis.set_xlabel("Optimizer")
    axis.set_ylabel("Problem ID")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.8)
    colorbar.set_label("Rank (1 = best)")
    figure.text(
        0.01, 0.01,
        "Ranks are computed within each task; feasibility and lower-level validity take priority over objective value.",
        fontsize=8, color=TEXT_MUTED,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def visualize_results(
    input_root: str | Path = "results/raw",
    output_root: str | Path = "results/figures",
    problem_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Create GitHub-style comparison figures from persisted campaign artifacts."""
    discovered = discover_runs(input_root)
    completed_all = [run for run in discovered if run.summary.get("status") == "completed"]
    completed, duplicates_ignored = _latest_unique_runs(completed_all)
    if not completed:
        raise ValueError(f"No completed campaign summaries found under {Path(input_root).resolve()}.")

    requested = None if not problem_ids or problem_ids == ["all"] else {item.upper() for item in problem_ids}
    selected = [run for run in completed if requested is None or run.summary.get("problem_id") in requested]
    if not selected:
        raise ValueError("No completed runs match the requested problem IDs.")

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    by_problem: dict[str, list[RunArtifact]] = defaultdict(list)
    for run in selected:
        by_problem[str(run.summary["problem_id"])].append(run)

    for problem_id in sorted(by_problem, key=_problem_sort_key):
        problem_dir = output_root / problem_id
        problem_dir.mkdir(parents=True, exist_ok=True)
        convergence = problem_dir / "convergence.png"
        regret = problem_dir / "regret_profile.png"
        final = problem_dir / "final_performance.png"
        plot_problem_convergence(by_problem[problem_id], convergence)
        plot_problem_regret(by_problem[problem_id], regret)
        plot_problem_final(by_problem[problem_id], final)
        generated.extend([str(convergence), str(regret), str(final)])

    heatmap = output_root / "suite_rank_heatmap.png"
    plot_suite_rank_heatmap(selected, heatmap)
    generated.append(str(heatmap))
    dashboard = output_root / "iteration_vs_regret.png"
    plot_suite_regret_dashboard(by_problem, dashboard)
    generated.append(str(dashboard))
    manifest = {
        "input_root": str(Path(input_root).resolve()),
        "output_root": str(output_root),
        "discovered_summaries": len(discovered),
        "duplicate_completed_runs_ignored": duplicates_ignored,
        "completed_runs_visualized": len(selected),
        "problem_ids": sorted(by_problem, key=_problem_sort_key),
        "generated_files": generated,
        "semantics": {
            "objective": "best feasible and lower-level-valid objective",
            "aggregation": "median and interquartile range across seeds",
            "suite_comparison": "within-task ranks; raw objectives are never compared across tasks",
            "regret": "best valid objective minus exact, published, empirical, or run-best reference",
        },
    }
    manifest_path = output_root / "plot_manifest.json"
    manifest["generated_files"].append(str(manifest_path))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
