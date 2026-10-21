from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from optbench.runner import run_campaign
from optbench.sota import sota_method


def run_task(problem_id: str, argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=f"python -m optbench.tasks.task{TASK_NUMBERS[problem_id]}",
        description=f"Run the selected SOTA optimizer for {problem_id}.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--inner-budget", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results/sota/raw"))
    args = parser.parse_args(argv)
    method = sota_method(problem_id)
    summary = run_campaign(
        problem_id,
        method.optimizer,
        seed=args.seed,
        budget=args.budget,
        inner_budget=args.inner_budget,
        output_root=args.output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


TASK_NUMBERS = {
    "D01": "001",
    "D02": "002",
    "D03": "003",
    "D04": "004",
    "D05": "005",
    "V01": "006",
    "V02": "007",
    "V03": "008",
    "V04": "009",
    "V05": "010",
    "N01": "011",
    "N02": "012",
    "N03": "013",
    "N04": "014",
    "N05": "015",
    "S01": "016",
    "S02": "017",
    "S03": "018",
    "S04": "019",
    "S05": "020",
    "C01": "021",
    "C02": "022",
    "C03": "023",
    "C04": "024",
    "C05": "025",
    "B01": "026",
    "B02": "027",
    "B03": "028",
    "B04": "029",
    "B05": "030",
}
