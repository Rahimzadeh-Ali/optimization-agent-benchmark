from pathlib import Path

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

    from optbench.visualization import visualize_results

    manifest = visualize_results(raw, figures, ["S04"])
    expected = [
        figures / "S04" / "convergence.png",
        figures / "S04" / "final_performance.png",
        figures / "suite_rank_heatmap.png",
        figures / "plot_manifest.json",
    ]
    assert manifest["completed_runs_visualized"] == 4
    assert manifest["problem_ids"] == ["S04"]
    assert all(path.exists() and path.stat().st_size > 0 for path in expected)
