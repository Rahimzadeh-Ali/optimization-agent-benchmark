# Optimization-Agent Benchmark

This repository implements the 30 function-visible tasks defined in
`optimization_agent_benchmark_suite.md`. It intentionally contains no LLM,
Claude, OpenAI, prompt, or blind-agent implementation. Optimizers receive a
callable objective (and constraints where appropriate), matching the ordinary
`src/` methods in the reference project.

## What is implemented

- One registry for all IDs `D01` through `B05`.
- A shared result type with the constraint convention `g(x) <= 0`.
- Evaluation-budget accounting and JSONL query logs.
- COCO BBOB, MixInt, Noisy, and Constrained adapters.
- Direct CEC-2017 F5/F6/F8 implementations using packaged official
  shift/rotation data from `opfunu` (the package is used only as a data source).
- YAHPO adapters with fixed `task_id`, `trainsize=1.0`, and `repl=1`.
- Nevergrad noisy artificial-function adapters.
- pymoo G04 and G07 adapters.
- Direct SMD2, SMD3, and SMD6 equations with separate upper/lower calls.
- Direct BOLIB Calamai-Vicente 1994b and Allende-Still 2013 equations.
- Nine function-aware baselines inspired by the reference repository, plus a
  nested-random baseline for true bilevel tasks.
- Pinned problem-backend versions so benchmark definitions do not drift when
  a dependency publishes a new release.
- GitHub-style benchmark visualization: convergence curves, final-performance
  bars, and a cross-suite rank heatmap.

## Install

Use Python 3.11 for the complete suite. YAHPO's current dependency stack is not
reliably compatible with Python 3.13. With Anaconda on Windows:

```powershell
conda create -n optbench python=3.11 -y
conda activate optbench
python -m pip install --upgrade pip
python -m pip install -e ".[full]"
```

COCO is a compiled extension. If pip cannot find a wheel on Windows, install
Microsoft C++ Build Tools, then repeat the install command. The non-COCO core
and tests do not require a compiler.

YAHPO also needs its v1.0.2 model/data bundle:

```powershell
git clone --depth 1 --branch v1.0.2 https://github.com/slds-lmu/yahpo_data.git external/yahpo_data
python -c "from yahpo_gym import local_config; local_config.init_config(); local_config.set_data_path(r'external/yahpo_data')"
```

## Inspect and run

```powershell
# List all 30 registered tasks without importing optional backends.
optbench list

# Show exactly what an optimizer is allowed to see.
optbench inspect D01

# Small smoke campaign.
optbench run --problem C04 --optimizer random_search --seed 0 --budget 20

# True bilevel smoke campaign. The outer and inner budgets are separate.
optbench run --problem B01 --optimizer nested_random --seed 0 --budget 4 --inner-budget 20

# Safely preview one baseline across all 30 tasks; this does not execute.
optbench matrix --optimizers auto --seeds 0

# Execute a small all-task smoke matrix. "auto" selects random_search for
# single-level tasks and nested_random for true bilevel tasks.
optbench matrix --optimizers auto --seeds 0 --budget 3 --inner-budget 5 --execute

# Visualize every completed campaign. Figures are written to results/figures.
optbench plot --input results/raw --output results/figures

# Visualize selected tasks only.
optbench plot --input results/raw --problems D01 C04 B01

# Test the implementation.
python -m pytest
```

Results are written under `results/raw/<run_id>/` as `trace.jsonl` and
`summary.json`. Every objective query is retained.

## Visualizing benchmark results

The plotting command follows the comparison style used by the reference
repository while respecting the additional scientific rules required by this
suite:

- `convergence.png` shows the median best-valid objective and interquartile
  range across seeds. Constrained and bilevel tasks receive a second panel for
  total constraint/lower-level-optimality violation.
- `final_performance.png` compares final valid objectives. An optimizer with no
  valid result is marked explicitly instead of displaying an invalid objective
  as if it were successful.
- `suite_rank_heatmap.png` compares within-task ranks. Raw objective values are
  never compared across unrelated task families.
- `plot_manifest.json` records exactly which runs and semantics produced the
  figures.

For reliable uncertainty bands, run the same optimizer with several shared
seeds before plotting, for example:

```powershell
optbench matrix --problems C04 --optimizers random_search sobol de `
  --seeds 0 1 2 3 4 --execute
optbench plot --problems C04
```

## Important benchmark choices

The source specification leaves three details undefined. They are frozen here
so different agents are compared on the same problem.

1. **YAHPO target:** minimize `logloss`. Training fraction is fixed to `1.0`
   and replicate ID to `1`.
2. **BOLIB search boxes:** B04 uses `[-5, 5]` for all variables; B05 uses
   `[0, 2]`. These are benchmark search domains, not new mathematical
   constraints. Published BOLIB constraints remain unchanged.
3. **Nevergrad box:** N04/N05 use `[-5, 5]^20`, because the Markdown specifies
   no finite domain but the selected function-aware methods require one.

The Markdown combines CEC-2017 IDs with function names that do not match the
official CEC-2017 ID-to-function map. This implementation follows the *named
equation* in the Markdown (Rastrigin, Expanded Schaffer F6, or Non-Continuous
Rastrigin) and uses the shift/rotation data for its stated ID. It does not call
`opfunu`'s evaluators. This policy is explicit because silently choosing either
the ID or the name would define a different benchmark.

## Adding a task yourself

1. Subclass `BenchmarkProblem` or `BilevelProblem` in `src/optbench/problems/`.
2. Return `EvaluationResult`; convert every inequality to `g <= 0`.
3. Add one lazy factory in `src/optbench/registry.py`.
4. Add a known-point and budget test under `tests/`.
5. Run `python -m pytest`, then a small `optbench run` smoke campaign.

## Definition sources

- Reference layout and function-aware optimizers: https://github.com/Mahyar-rajabi94/Cheapest_opt_lunch
- COCO suites and Python interface: https://coco-platform.org/getting-started/index.html
- CEC source data: https://github.com/P-N-Suganthan/CEC2017-BoundContrained
- YAHPO Gym and surrogate data: https://github.com/slds-lmu/yahpo_gym and https://github.com/slds-lmu/yahpo_data
- SMD definitions: https://faculty.iima.ac.in/asinha/bilevel/resources.html
- BOLIB definitions: https://www.bolib.org/problems/quadratic_bilevel/ and https://www.bolib.org/problems/quadratically_constrained_quadratic_bilevel/

