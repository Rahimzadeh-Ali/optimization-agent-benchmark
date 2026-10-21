# Implementation status

All 30 function-visible tasks from `optimization_agent_benchmark_suite.md` are
registered. "Implemented" means the adapter/equation exists; an optional
backend must still be installed for rows that name one.

| IDs | Count | Source/backend | Status |
|---|---:|---|---|
| D01-D03 | 3 | COCO `bbob` | Implemented; native backend required |
| D04-D05 | 2 | Direct CEC equations + `opfunu` data | Implemented |
| V01-V03 | 3 | COCO `bbob-mixint` | Implemented; native backend required |
| V04-V05 | 2 | YAHPO Gym | Implemented; data setup required |
| N01-N03 | 3 | COCO `bbob-noisy` | Implemented; native backend required |
| N04-N05 | 2 | Nevergrad artificial functions | Implemented |
| S01-S03 | 3 | COCO `bbob` | Implemented; native backend required |
| S04-S05 | 2 | Direct CEC equations + `opfunu` data | Implemented |
| C01-C03 | 3 | COCO `bbob-constrained` | Implemented; native backend required |
| C04-C05 | 2 | pymoo G-suite | Implemented |
| B01-B03 | 3 | Direct SMD equations | Implemented |
| B04-B05 | 2 | Direct BOLIB equations/data | Implemented |

## One SOTA optimizer per task

| Numbered scripts | IDs | Method |
|---|---|---|
| task001-task003 | D01-D03 | CMA-ES |
| task004-task005 | D04-D05 | Nevergrad NGOpt |
| task006-task008 | V01-V03 | Nevergrad PortfolioDiscreteOnePlusOne |
| task009-task010 | V04-V05 | SMAC HPO |
| task011-task015 | N01-N05 | Nevergrad TBPSA |
| task016-task017 | S01-S02 | CMA-ES |
| task018-task020 | S03-S05 | Nevergrad NGOpt |
| task021-task025 | C01-C05 | pymoo ISRES |
| task026-task030 | B01-B05 | Source-structured bilevel optimization |

AI/LLM methods are deliberately outside the current scope. Each numbered
module can be run with, for example,
`python -m optbench.tasks.task001 --budget 2000`.

## Visualization

`optbench sota --budget 2000` runs all selected methods and creates per-task
convergence and task-valid optimization profiles plus `iteration_profiles.png`.
The profile metric is exact regret only when a trustworthy optimum exists;
otherwise it is explicitly labelled as objective, signed empirical gap, noisy
rolling median, or constraint violation. Raw magnitudes are never compared
across different tasks.

## Definition of a completed campaign

A run is complete only when its isolated result directory contains
`summary.json` with `status: completed` and a `trace.jsonl` record for every
optimization query. Unsupported method/task combinations are saved with
`status: not_applicable`; missing packages or execution defects are saved with
`status: failed`.
