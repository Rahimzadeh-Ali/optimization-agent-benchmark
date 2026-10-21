from pathlib import Path
import json
import shutil

import pytest

from optbench.runner import run_campaign


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
        figures / "S04" / "regret_profile.png",
        figures / "S04" / "final_performance.png",
        figures / "suite_rank_heatmap.png",
        figures / "iteration_vs_regret.png",
        figures / "plot_manifest.json",
    ]
    assert manifest["completed_runs_visualized"] == 4
    assert manifest["duplicate_completed_runs_ignored"] == 1
    assert manifest["problem_ids"] == ["S04"]
    assert all(path.exists() and path.stat().st_size > 0 for path in expected)
