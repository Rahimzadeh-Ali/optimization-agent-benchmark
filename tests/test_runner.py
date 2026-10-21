import json
from pathlib import Path

from optbench.runner import build_matrix_plan, run_campaign


def test_random_campaign_logs_every_budgeted_query(tmp_path: Path) -> None:
    summary = run_campaign(
        "C04", "random_search", seed=7, budget=8, output_root=tmp_path
    )
    run_dir = tmp_path / summary["run_id"]
    lines = (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert summary["status"] == "completed"
    assert summary["result"]["optimization_evaluations"] == 8
    assert len(lines) == 8
    assert all(json.loads(line)["phase"] == "optimization" for line in lines)


def test_inapplicable_pair_is_recorded_not_silently_run(tmp_path: Path) -> None:
    summary = run_campaign("B01", "de", seed=0, budget=2, output_root=tmp_path)
    assert summary["status"] == "not_applicable"
    assert "bilevel" in summary["result"]["reason"]
    assert (tmp_path / summary["run_id"] / "summary.json").exists()


def test_bilevel_budgets_are_separate(tmp_path: Path) -> None:
    summary = run_campaign(
        "B01", "nested_random", seed=2, budget=3, inner_budget=4, output_root=tmp_path
    )
    assert summary["result"]["upper_evaluations"] == 3
    assert summary["result"]["lower_evaluations"] == 12


def test_auto_matrix_covers_every_task_without_loading_backends() -> None:
    plan = build_matrix_plan(seeds=[0, 1])
    assert len(plan) == 60
    assert {row["problem_id"] for row in plan} == {
        *(f"D{i:02d}" for i in range(1, 6)),
        *(f"V{i:02d}" for i in range(1, 6)),
        *(f"N{i:02d}" for i in range(1, 6)),
        *(f"S{i:02d}" for i in range(1, 6)),
        *(f"C{i:02d}" for i in range(1, 6)),
        *(f"B{i:02d}" for i in range(1, 6)),
    }
    from optbench.sota import sota_method

    assert all(row["optimizer"] == sota_method(row["problem_id"]).optimizer for row in plan)
