# Implementation status

All 30 function-visible tasks from `optimization_agent_benchmark_suite.md` are
registered. “Implemented” means the adapter/equation exists; an optional
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

## Function-aware methods

The suite includes random search, scrambled Sobol, CMA-ES, differential
evolution, GP Bayesian optimization, TuRBO, BOBYQA, Nelder-Mead, L-BFGS-B,
and nested random search for true bilevel tasks. AI/LLM methods are deliberately
outside the current scope.

## Visualization

`optbench plot` creates per-task convergence and final-performance figures plus
a suite-level optimizer-rank heatmap. The figures use feasibility-first scoring
and never compare raw objective magnitudes across different tasks.

## Definition of a completed campaign

A run is complete only when its isolated result directory contains
`summary.json` with `status: completed` and a `trace.jsonl` record for every
optimization query. Unsupported method/task combinations are saved with
`status: not_applicable`; missing packages or execution defects are saved with
`status: failed`.
