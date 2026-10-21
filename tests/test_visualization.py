from pathlib import Path
import json
import shutil

import pytest

from optbench.runner import run_campaign


def _artifact(tmp_path: Path, summary: dict, rows: list[dict]):
    from optbench.visualization import RunArtifact

    run_dir = tmp_path / summary["run_id"]
    run_dir.mkdir()
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return RunArtifact(summary_path, summary)


def _row(evaluation: int, objective: float, *, feasible: bool = True, violation: float = 0.0) -> dict:
    return {
        "phase": "optimization",
        "evaluation": evaluation,
        "objective": objective,
        "feasible": feasible,
        "constraint_violation": violation,
    }


def test_profile_metrics_do_not_manufacture_log_floor(tmp_path: Path) -> None:
    from optbench.visualization import _profile_trajectory

    base = {
        "run_id": "run",
        "problem_id": "X01",
        "optimizer": "test",
        "seed": 0,
        "artifacts": {"trace": "trace.jsonl"},
        "task": {"noisy": False},
        "regret_reference": {"objective": None, "kind": "run_best"},
    }
    objective = _profile_trajectory(
        _artifact(tmp_path, base, [_row(1, 5.0), _row(2, 3.0)])
    )
    assert objective.metric == "Best valid objective"
    assert objective.values.tolist() == [5.0, 3.0]


def test_profile_metrics_preserve_signed_gaps_and_infeasibility(tmp_path: Path) -> None:
    from optbench.visualization import _profile_trajectory

    empirical_summary = {
        "run_id": "empirical",
        "problem_id": "X02",
        "optimizer": "test",
        "seed": 0,
        "artifacts": {"trace": "trace.jsonl"},
        "task": {"noisy": False},
        "regret_reference": {"objective": 5.0, "kind": "empirical_500k"},
    }
    empirical = _profile_trajectory(
        _artifact(tmp_path, empirical_summary, [_row(1, 6.0), _row(2, 4.0)])
    )
    assert empirical.metric == "Signed gap to empirical reference"
    assert empirical.values.tolist() == [1.0, -1.0]

    infeasible_summary = {
        **empirical_summary,
        "run_id": "infeasible",
        "regret_reference": {"objective": 1.0, "kind": "published_global"},
    }
    infeasible = _profile_trajectory(
        _artifact(
            tmp_path,
            infeasible_summary,
            [_row(1, 10.0, feasible=False, violation=4.0), _row(2, 8.0, feasible=False, violation=2.0)],
        )
    )
    assert infeasible.metric == "Best constraint violation"
    assert infeasible.values.tolist() == [4.0, 2.0]


def test_noisy_profile_uses_rolling_observations_not_best_sample(tmp_path: Path) -> None:
    from optbench.visualization import _profile_trajectory

    summary = {
        "run_id": "noisy",
        "problem_id": "N99",
        "optimizer": "test",
        "seed": 0,
        "artifacts": {"trace": "trace.jsonl"},
        "task": {"noisy": True},
        "regret_reference": {"objective": 0.0, "kind": "latent_global"},
    }
    profile = _profile_trajectory(
        _artifact(tmp_path, summary, [_row(1, -100.0), _row(2, 100.0), _row(3, -20.0)])
    )
    assert profile.metric == "Rolling median observed objective"
    assert profile.values.tolist() == [-100.0, 0.0, -20.0]


def test_visualization_creates_reference_style_comparisons(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("opfunu")
    raw = tmp_path / "raw"
    figures = tmp_path / "figures"
    for optimizer in ("random_search", "sobol"):
        for seed in (0, 1):
            run_campaign("S04", optimizer, seed=seed, budget=6, output_root=raw)

    # Simulate a successful rerun of the same campaign key.  Visualization
    # must not mistake it for an additional independent seed.
    original = next(raw.glob("S04_random_search_seed0_*"))
    duplicate = raw / f"{original.name}_duplicate"
    shutil.copytree(original, duplicate)
    summary_path = duplicate / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["run_id"] = duplicate.name
    summary["finished_at"] = "9999-12-31T23:59:59+00:00"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    from optbench.visualization import visualize_results

    manifest = visualize_results(raw, figures, ["S04"])
    expected = [
        figures / "S04" / "convergence.png",
        figures / "S04" / "optimization_profile.png",
        figures / "S04" / "regret_profile.png",
        figures / "S04" / "final_performance.png",
        figures / "suite_rank_heatmap.png",
        figures / "iteration_profiles.png",
        figures / "iteration_vs_regret.png",
        figures / "plot_manifest.json",
    ]
    assert manifest["completed_runs_visualized"] == 4
    assert manifest["duplicate_completed_runs_ignored"] == 1
    assert manifest["problem_ids"] == ["S04"]
    assert all(path.exists() and path.stat().st_size > 0 for path in expected)
