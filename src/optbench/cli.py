from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from optbench.optimizers import OPTIMIZERS
from optbench.registry import create_problem, list_problems
from optbench.runner import build_matrix_plan, run_campaign, run_matrix
from optbench.scoring import load_summaries, rank_completed
from optbench.sota import list_sota_methods


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="optbench", description="Function-visible optimization benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list all task records without loading optional backends")
    commands.add_parser("optimizers", help="list function-aware optimization methods")
    commands.add_parser("sota-methods", help="list the one SOTA method selected for each task")

    inspect = commands.add_parser("inspect", help="show the public interface for one task")
    inspect.add_argument("problem_id")
    inspect.add_argument("--seed", type=int, default=0)

    run = commands.add_parser("run", help="run one optimizer/task/seed campaign")
    run.add_argument("--problem", required=True)
    run.add_argument("--optimizer", required=True, choices=sorted(OPTIMIZERS))
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--budget", type=int)
    run.add_argument("--inner-budget", type=int)
    run.add_argument("--output", type=Path, default=Path("results/raw"))

    score = commands.add_parser("score", help="rank completed summaries per task")
    score.add_argument("--input", type=Path, default=Path("results/raw"))

    matrix = commands.add_parser("matrix", help="plan or execute several isolated campaigns")
    matrix.add_argument("--problems", nargs="+", default=["all"])
    matrix.add_argument("--optimizers", nargs="+", default=["auto"])
    matrix.add_argument("--seeds", nargs="+", type=int, default=[0])
    matrix.add_argument("--budget", type=int)
    matrix.add_argument("--inner-budget", type=int)
    matrix.add_argument("--output", type=Path, default=Path("results/raw"))
    matrix.add_argument(
        "--execute",
        action="store_true",
        help="run the plan; without this flag matrix is a safe dry-run",
    )

    plot = commands.add_parser("plot", help="create comparison figures from campaign results")
    plot.add_argument("--input", type=Path, default=Path("results/raw"))
    plot.add_argument("--output", type=Path, default=Path("results/figures"))
    plot.add_argument("--problems", nargs="+", default=["all"])

    sota = commands.add_parser(
        "sota",
        help="run one source-backed SOTA method for each task and plot regret profiles",
    )
    sota.add_argument("--problems", nargs="+", default=["all"])
    sota.add_argument("--seed", type=int, default=0)
    sota.add_argument("--budget", type=int, default=2000)
    sota.add_argument("--inner-budget", type=int, default=100)
    sota.add_argument("--output", type=Path, default=Path("results/sota/raw"))
    sota.add_argument("--figures", type=Path, default=Path("results/sota/figures"))
    sota.add_argument("--no-plot", action="store_true")
    sota.add_argument(
        "--resume",
        action="store_true",
        help="skip matching completed runs that already reached the requested budget",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        _print_json(list_problems())
    elif args.command == "optimizers":
        _print_json([{"name": item.name, "description": item.description} for item in OPTIMIZERS.values()])
    elif args.command == "sota-methods":
        _print_json(list_sota_methods())
    elif args.command == "inspect":
        _print_json(create_problem(args.problem_id, seed=args.seed).agent_spec())
    elif args.command == "run":
        _print_json(
            run_campaign(
                args.problem,
                args.optimizer,
                seed=args.seed,
                budget=args.budget,
                inner_budget=args.inner_budget,
                output_root=args.output,
            )
        )
    elif args.command == "score":
        _print_json(rank_completed(load_summaries(args.input)))
    elif args.command == "matrix":
        plan = build_matrix_plan(args.problems, args.optimizers, args.seeds)
        _print_json(
            run_matrix(
                plan,
                output_root=args.output,
                budget=args.budget,
                inner_budget=args.inner_budget,
            )
            if args.execute
            else {"execute": False, "campaign_count": len(plan), "plan": plan}
        )
    elif args.command == "plot":
        from optbench.visualization import visualize_results

        _print_json(visualize_results(args.input, args.output, args.problems))
    elif args.command == "sota":
        plan = build_matrix_plan(args.problems, ["auto"], [args.seed])
        requested_keys = {
            (row["problem_id"], row["optimizer"], row["seed"])
            for row in plan
        }
        if args.resume:
            completed_keys = set()
            for summary in load_summaries(args.output):
                result = summary.get("result", {})
                evaluations = result.get(
                    "optimization_evaluations",
                    result.get("upper_evaluations"),
                )
                stored_limit = summary.get("budget_limit")
                budget_matches = (
                    stored_limit == args.budget
                    or (stored_limit is None and evaluations is not None)
                )
                if summary.get("status") == "completed" and budget_matches:
                    completed_keys.add(
                        (summary["problem_id"], summary["optimizer"], summary["seed"])
                    )
            plan = [
                row
                for row in plan
                if (row["problem_id"], row["optimizer"], row["seed"])
                not in completed_keys
            ]
        outcomes = run_matrix(
            plan,
            output_root=args.output,
            budget=args.budget,
            inner_budget=args.inner_budget,
        )
        completed_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        for summary in load_summaries(args.output):
            key = (summary["problem_id"], summary["optimizer"], summary["seed"])
            result = summary.get("result", {})
            evaluations = result.get(
                "optimization_evaluations",
                result.get("upper_evaluations"),
            )
            stored_limit = summary.get("budget_limit")
            budget_matches = (
                stored_limit == args.budget
                or (stored_limit is None and evaluations is not None)
            )
            if (
                key in requested_keys
                and summary.get("status") == "completed"
                and budget_matches
            ):
                completed_by_key[key] = {
                    "problem_id": summary["problem_id"],
                    "optimizer": summary["optimizer"],
                    "seed": summary["seed"],
                    "status": "completed",
                    "run_id": summary["run_id"],
                }
        unresolved = [
            row
            for row in outcomes
            if (row["problem_id"], row["optimizer"], row["seed"])
            not in completed_by_key
        ]
        runs = sorted(
            [*completed_by_key.values(), *unresolved],
            key=lambda row: row["problem_id"],
        )
        manifest: dict[str, Any] = {
            "budget": args.budget,
            "inner_budget": args.inner_budget,
            "seed": args.seed,
            "campaign_count": len(runs),
            "completed": sum(row["status"] == "completed" for row in runs),
            "failed": sum(row["status"] == "failed" for row in runs),
            "not_applicable": sum(row["status"] == "not_applicable" for row in runs),
            "runs": runs,
        }
        if not args.no_plot and manifest["completed"]:
            from optbench.visualization import visualize_results

            manifest["figures"] = visualize_results(
                args.output,
                args.figures,
                args.problems,
            )
        manifest_path = args.output.parent / "sota_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifest["manifest"] = str(manifest_path.resolve())
        _print_json(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
