# Optimization-Agent Benchmark Suite
## 30 fast benchmark problems across 6 optimization challenges

**Purpose:** Benchmark optimization agents that must choose/configure optimization methods across different problem characteristics.

**Design target:**  
- ~30 total problems
- Above-average but not extreme difficulty
- Cheap enough to run a full first-pass benchmark in one day
- Two benchmark families per direction
- Prefer callable functions/surrogates rather than expensive simulations
- True bilevel problems for the nested category

---

# 1. Benchmark directions

The suite covers six directions:

1. **Dimensionality**
2. **Variable type**
3. **Noise**
4. **Smoothness / landscape regularity**
5. **Constraints**
6. **True bilevel optimization**

The suite contains:

- **6 directions**
- **2 benchmark families per direction**
- **5 concrete problem configurations per direction**
- **30 total benchmark problems**

---

# 2. Important terminology

## 2.1 Function vs dataset

Most problems here are **not tabular datasets**.

For most benchmarks, the interaction is:

```python
y = problem.evaluate(x)
```

rather than:

```python
row = csv.lookup(x)
```

Continuous optimization spaces contain infinitely many possible `x`, so a CSV lookup is generally not appropriate.

### Benchmark types in this suite

| Family | What it actually is | Query style |
|---|---|---|
| COCO BBOB | Mathematical functions | `f(x)` |
| COCO BBOB-MixInt | Mathematical mixed-variable functions | `f(x)` |
| COCO BBOB-Noisy | Mathematical functions + stochastic noise | `f_noisy(x)` |
| COCO BBOB-Constrained | Objective + constraint functions | `f(x), g(x)` |
| CEC-2017 | Mathematical transformed functions | `f(x)` |
| Nevergrad | Callable benchmark functions | `f(x)` |
| YAHPO Gym | Fast surrogate models trained from real HPO experiments | `surrogate(config)` |
| pymoo G-series | Mathematical objective + constraints | `f(x), g(x)` |
| SMD | Mathematical upper/lower bilevel functions | `F(xu,xl), f(xu,xl)` |
| BOLIB | Structured bilevel problem definitions, often JSON | objectives + constraints |

---

## 2.2 Bounds are not counted as explicit constraints

A variable domain such as

```text
-5 <= x_i <= 5
```

is treated as the **search domain**.

A genuine optimization constraint is something such as

```text
g_1(x) <= 0
h_1(x) = 0
```

The dedicated **Constraints** direction below tests explicit nonlinear constraints.

---

# 3. Full 30-problem suite

## Summary table

| ID | Direction | Family | Problem | Configuration | Type | Suggested budget |
|---:|---|---|---|---|---|---:|
| D01 | Dimensionality | COCO BBOB | Rosenbrock f8 | D=10 | Continuous | 250 |
| D02 | Dimensionality | COCO BBOB | Rosenbrock f8 | D=20 | Continuous | 500 |
| D03 | Dimensionality | COCO BBOB | Rosenbrock f8 | D=40 | Continuous | 1000 |
| D04 | Dimensionality | CEC-2017 | Expanded Schaffer F6, F6 | D=30 | Continuous | 750 |
| D05 | Dimensionality | CEC-2017 | Expanded Schaffer F6, F6 | D=50 | Continuous | 1250 |
| V01 | Variable type | BBOB-MixInt | Rosenbrock f8 | D=10 | Mixed | 500 |
| V02 | Variable type | BBOB-MixInt | Rastrigin f15 | D=10 | Mixed | 500 |
| V03 | Variable type | BBOB-MixInt | Schaffer F7 f17 | D=10 | Mixed | 500 |
| V04 | Variable type | YAHPO Gym | rbv2_svm | OpenML dataset 40981 | Mixed + hierarchical | 150 |
| V05 | Variable type | YAHPO Gym | rbv2_xgboost | OpenML dataset 40981 | Mixed + hierarchical | 150 |
| N01 | Noise | BBOB-Noisy | Rosenbrock f110 | D=5 | Gaussian noise | 500 |
| N02 | Noise | BBOB-Noisy | Rosenbrock f111 | D=5 | Uniform noise | 500 |
| N03 | Noise | BBOB-Noisy | Rosenbrock f112 | D=5 | Seldom-Cauchy noise | 500 |
| N04 | Noise | Nevergrad | Noisy Rosenbrock | D=20, noise=10 | Noisy continuous | 500 |
| N05 | Noise | Nevergrad | Noisy Cigar | D=20, noise=10 | Noisy ill-conditioned | 500 |
| S01 | Smoothness | COCO BBOB | Step Ellipsoid f7 | D=10 | Plateau / discontinuous-like | 500 |
| S02 | Smoothness | COCO BBOB | Sharp Ridge f13 | D=10 | Nondifferentiable ridge | 500 |
| S03 | Smoothness | COCO BBOB | Weierstrass f16 | D=10 | Rugged / multimodal | 500 |
| S04 | Smoothness | CEC-2017 | Rastrigin F5 | D=30 | Smooth + multimodal | 750 |
| S05 | Smoothness | CEC-2017 | Non-Continuous Rastrigin F8 | D=30 | Non-continuous + multimodal | 750 |
| C01 | Constraints | BBOB-Constrained | Rotated Ellipsoid ID 20 | D=10, 3 constraints | Continuous constrained | 500 |
| C02 | Constraints | BBOB-Constrained | Different Powers ID 38 | D=10, 3 constraints | Continuous constrained | 500 |
| C03 | Constraints | BBOB-Constrained | Rastrigin ID 44 | D=10, 3 constraints | Constrained + multimodal | 500 |
| C04 | Constraints | pymoo G-series | G04 | 5 variables, 6 constraints | Constrained | 500 |
| C05 | Constraints | pymoo G-series | G07 | 10 variables, 8 constraints | Constrained | 500 |
| B01 | True bilevel | SMD | SMD2 | small scalable instance | Bilevel, conflict | 40 outer + inner budget |
| B02 | True bilevel | SMD | SMD3 | small scalable instance | Bilevel, lower multimodality | 40 outer + inner budget |
| B03 | True bilevel | SMD | SMD6 | small scalable instance | Bilevel, multiple LL optima | 40 outer + inner budget |
| B04 | True bilevel | BOLIB | Calamai-Vicente 1994b | xu=4, xl=2 | Constrained bilevel | 40 outer + inner budget |
| B05 | True bilevel | BOLIB | Allende-Still 2013 | xu=2, xl=2 | QCQP bilevel | 40 outer + inner budget |

---

# 4. Direction A — Dimensionality

## Goal

Test whether an optimization agent adapts when the number of decision variables grows while keeping the basic optimization structure controlled.

The important design principle is:

```text
same family / same landscape
different dimension
```

Otherwise dimensionality becomes confounded with landscape difficulty.

---

## Dataset family A1 — COCO BBOB

### D01 — Rosenbrock f8, D=10

```yaml
id: D01
family: coco_bbob
function_id: 8
name: Rosenbrock Original
dimension: 10
variables: continuous
constraints: none
noise: none
budget: 250
```

### D02 — Rosenbrock f8, D=20

```yaml
id: D02
family: coco_bbob
function_id: 8
name: Rosenbrock Original
dimension: 20
variables: continuous
constraints: none
noise: none
budget: 500
```

### D03 — Rosenbrock f8, D=40

```yaml
id: D03
family: coco_bbob
function_id: 8
name: Rosenbrock Original
dimension: 40
variables: continuous
constraints: none
noise: none
budget: 1000
```

### Why Rosenbrock?

It is not a trivial Sphere problem. It contains a narrow curved valley and nontrivial variable interactions while remaining cheap to evaluate.

### Source

- COCO BBOB overview: https://coco-platform.org/testsuites/bbob/overview.html
- Rosenbrock f8: https://coco-platform.org/testsuites/bbob/functions/f08.html

---

## Dataset family A2 — CEC-2017

### D04 — Shifted/Rotated Expanded Schaffer F6, D=30

```yaml
id: D04
family: cec2017
function_id: 6
name: Shifted and Rotated Expanded Schaffer F6
dimension: 30
variables: continuous
constraints: none
noise: none
budget: 750
```

### D05 — Shifted/Rotated Expanded Schaffer F6, D=50

```yaml
id: D05
family: cec2017
function_id: 6
name: Shifted and Rotated Expanded Schaffer F6
dimension: 50
variables: continuous
constraints: none
noise: none
budget: 1250
```

### Why this pair?

The same transformed multimodal function is used at two larger dimensions. It is harder than basic separable functions but still computationally trivial.

### Note

Use the **official CEC-2017 shift/rotation data** or a carefully verified implementation. Do not silently replace the transformed function with the textbook unshifted version.

---

# 5. Direction B — Variable type

## Goal

Test whether the agent recognizes that continuous-only optimization methods are not always appropriate.

We want exposure to:

- continuous variables
- binary variables
- finite-cardinality discrete variables
- categorical variables
- hierarchical/conditional variables

---

## Dataset family B1 — COCO BBOB-MixInt

The official mixed-integer transformation keeps roughly:

- 20% continuous
- 20% binary
- 20% 4-ary
- 20% 8-ary
- 20% 16-ary

Source:  
https://coco-platform.org/testsuites/bbob-mixint/overview.html

### V01 — Mixed-integer Rosenbrock f8

```yaml
id: V01
family: bbob_mixint
function_id: 8
dimension: 10
variables: mixed
budget: 500
```

### V02 — Mixed-integer Rastrigin f15

```yaml
id: V02
family: bbob_mixint
function_id: 15
dimension: 10
variables: mixed
budget: 500
```

### V03 — Mixed-integer Schaffer F7 f17

```yaml
id: V03
family: bbob_mixint
function_id: 17
dimension: 10
variables: mixed
budget: 500
```

### Why these three?

They progressively introduce:

```text
mixed variables + curved valley
mixed variables + strong multimodality
mixed variables + nonseparability / multimodality
```

---

## Dataset family B2 — YAHPO Gym

YAHPO is different from BBOB.

It is based on real hyperparameter-optimization experiments, but exposes fast learned surrogate models. From the optimizer's perspective it is still queried like a function.

Source:

- https://github.com/slds-lmu/yahpo_gym
- https://slds-lmu.github.io/yahpo_gym/scenarios.html

### V04 — rbv2_svm, OpenML dataset 40981

```yaml
id: V04
family: yahpo
scenario: rbv2_svm
instance: "40981"
search_space_dimension: 6
variables: mixed
hierarchical: true
fidelity: fixed_maximum
budget: 150
```

### V05 — rbv2_xgboost, OpenML dataset 40981

```yaml
id: V05
family: yahpo
scenario: rbv2_xgboost
instance: "40981"
search_space_dimension: 14
variables: mixed
hierarchical: true
fidelity: fixed_maximum
budget: 150
```

### Important

For this benchmark direction, **fix the fidelity parameter at its maximum**.

Otherwise the task simultaneously tests:

- variable types
- multi-fidelity optimization

and becomes harder to interpret.

### Query example

```python
from yahpo_gym import benchmark_set

bench = benchmark_set.BenchmarkSet("rbv2_svm")
bench.set_instance("40981")

result = bench.objective_function(configuration)
```

YAHPO's documentation confirms that `40981` is an available instance for both selected scenarios.

---

# 6. Direction C — Noise

## Goal

Test whether the agent handles unreliable observations rather than blindly trusting the best observed value.

---

## Dataset family C1 — COCO BBOB-Noisy

The BBOB-Noisy suite includes Gaussian, uniform, and seldom-Cauchy noise.

Source:  
https://numbbo.github.io/coco/testsuites/bbob-noisy

Use the same underlying Rosenbrock landscape so that the major change is the **noise distribution**.

### N01 — f110 Rosenbrock + Gaussian noise

```yaml
id: N01
family: bbob_noisy
function_id: 110
dimension: 5
noise: gaussian
budget: 500
```

### N02 — f111 Rosenbrock + uniform noise

```yaml
id: N02
family: bbob_noisy
function_id: 111
dimension: 5
noise: uniform
budget: 500
```

### N03 — f112 Rosenbrock + seldom-Cauchy noise

```yaml
id: N03
family: bbob_noisy
function_id: 112
dimension: 5
noise: seldom_cauchy
budget: 500
```

### Why the Cauchy case matters

Rare extreme observations can make "take the best observed point" a very poor strategy.

An agent may need to consider:

- replication
- averaging
- robust statistics
- noise-aware Bayesian optimization
- stochastic/evolutionary methods

---

## Dataset family C2 — Nevergrad noisy problems

Nevergrad's published noisy benchmark includes Sphere, Rosenbrock, Cigar, and a highly multimodal function with noise level 10 at several dimensions.

Source:  
https://facebookresearch.github.io/nevergrad/benchmarks.html

### N04 — Noisy Rosenbrock

```yaml
id: N04
family: nevergrad
function: Rosenbrock
dimension: 20
noise_level: 10
budget: 500
```

### N05 — Noisy Cigar

```yaml
id: N05
family: nevergrad
function: Cigar
dimension: 20
noise_level: 10
budget: 500
```

The Cigar problem adds strong conditioning to the noise challenge.

---

# 7. Direction D — Smoothness / landscape regularity

## Goal

Distinguish among:

```text
smooth
nondifferentiable
plateaued
non-continuous
rugged / multimodal
```

These are different properties.

**Multimodal** means that the landscape contains many local optima.  
It does **not** automatically mean nonsmooth.

For example:

```text
Rastrigin = smooth + multimodal
```

while a step function can be:

```text
plateaued / discontinuous-like
```

---

## Dataset family D1 — COCO BBOB

### S01 — Step Ellipsoid f7

```yaml
id: S01
family: coco_bbob
function_id: 7
dimension: 10
characteristic: plateaus
budget: 500
```

### S02 — Sharp Ridge f13

```yaml
id: S02
family: coco_bbob
function_id: 13
dimension: 10
characteristic: nondifferentiable_ridge
budget: 500
```

### S03 — Weierstrass f16

```yaml
id: S03
family: coco_bbob
function_id: 16
dimension: 10
characteristic: rugged_multimodal
budget: 500
```

Source:  
https://coco-platform.org/testsuites/bbob/overview.html

---

## Dataset family D2 — CEC-2017

### S04 — Shifted/Rotated Rastrigin F5

```yaml
id: S04
family: cec2017
function_id: 5
dimension: 30
characteristic: smooth_multimodal
budget: 750
```

### S05 — Shifted/Rotated Non-Continuous Rastrigin F8

```yaml
id: S05
family: cec2017
function_id: 8
dimension: 30
characteristic: noncontinuous_multimodal
budget: 750
```

This pair is useful because it compares related Rastrigin-type landscapes with a major change in regularity.

---

# 8. Direction E — Explicit constraints

## Goal

Test whether the agent can:

- maintain or discover feasibility
- balance objective improvement and constraint violation
- choose an optimizer that supports nonlinear constraints
- avoid treating infeasible low-objective points as valid solutions

---

## Dataset family E1 — COCO BBOB-Constrained

The current COCO constrained suite defines **54 constrained functions** from nine objective families and six constraint-count settings.

Source:

- https://coco-platform.org/testsuites/bbob-constrained/test-suite.html
- Function-definition PDF: https://numbbo.github.io/coco-doc/bbob-constrained/functions.pdf

We use problems with **3 explicit constraints** to keep the constraint burden meaningful but not excessive.

### C01 — Rotated Ellipsoid, ID 20

From the COCO table:

- IDs 19–24 correspond to Rotated Ellipsoid
- ID 20 uses 3 constraints

```yaml
id: C01
family: bbob_constrained
problem_id: 20
dimension: 10
objective: rotated_ellipsoid
constraints: 3
budget: 500
```

### C02 — Different Powers, ID 38

```yaml
id: C02
family: bbob_constrained
problem_id: 38
dimension: 10
objective: different_powers
constraints: 3
budget: 500
```

### C03 — Rastrigin, ID 44

```yaml
id: C03
family: bbob_constrained
problem_id: 44
dimension: 10
objective: rastrigin
constraints: 3
multimodal: true
budget: 500
```

This gives three distinct constrained geometries without moving to the very high-constraint variants.

---

## Dataset family E2 — pymoo G-series

pymoo exposes standard constrained single-objective benchmark problems directly.

Source:  
https://pymoo.org/problems/test_problems.html

### C04 — G04

```yaml
id: C04
family: pymoo
problem: g4
dimension: 5
constraints: 6
budget: 500
```

### C05 — G07

```yaml
id: C05
family: pymoo
problem: g7
dimension: 10
constraints: 8
budget: 500
```

Example access pattern:

```python
from pymoo.problems import get_problem

problem = get_problem("g4")
result = problem.evaluate(x)
```

Check the exact return options used by your installed pymoo version so that objective and constraint arrays are exposed consistently.

---

# 9. Direction F — True bilevel optimization

## Goal

These must remain **true bilevel problems**.

The generic form is:

\[
\min_{x_u,x_l} F(x_u,x_l)
\]

subject to

\[
x_l \in \arg\min_z f(x_u,z)
\]

plus any upper- or lower-level constraints.

An upper-level candidate is not valid merely because it has a good `F`.  
Its associated lower-level decision must satisfy the follower's optimality requirement.

---

# 10. Recommended agent-facing bilevel API

For the bilevel tasks, do **not** hide the lower-level solve inside a single ordinary black-box evaluator if the purpose is to benchmark true bilevel reasoning.

Recommended interface:

```python
problem.evaluate_upper(xu, xl)
problem.evaluate_lower(xu, xl)

problem.upper_constraints(xu, xl)
problem.lower_constraints(xu, xl)

problem.upper_bounds
problem.lower_bounds
```

The optimization agent is responsible for deciding how to solve the nested problem.

The benchmark harness should independently verify the final lower-level response.

---

## Dataset family F1 — SMD

Source:

- https://faculty.iima.ac.in/asinha/bilevel/
- Problem description: https://faculty.iima.ac.in/asinha/bilevel/files/Description-SMD.pdf

The SMD suite was explicitly designed to generate different bilevel difficulties including:

- conflict/cooperation between levels
- lower-level multimodality
- multiple lower-level optima
- scalable dimensions
- constrained and unconstrained cases

The three selected problems are chosen because they provide different genuine bilevel difficulties without using the heaviest constrained SMD variants.

---

### B01 — SMD2

Main property:

```text
upper/lower-level conflict
```

Recommended small configuration:

```yaml
id: B01
family: SMD
problem: SMD2
p: 1
q: 1
r: 1
upper_budget: 40
lower_budget_per_upper_candidate: 100
```

Why useful:

An inaccurate follower solution can mislead the leader.

---

### B02 — SMD3

Main property:

```text
multimodal lower-level problem
```

Recommended:

```yaml
id: B02
family: SMD
problem: SMD3
p: 1
q: 1
r: 1
upper_budget: 40
lower_budget_per_upper_candidate: 100
```

This tests whether the agent can avoid accepting a local follower solution as if it were the true lower-level optimum.

---

### B03 — SMD6

Main property:

```text
multiple lower-level global optima / set-valued follower response
```

Recommended small configuration:

```yaml
id: B03
family: SMD
problem: SMD6
p: 1
q: 1
r: 1
s: 2
upper_budget: 40
lower_budget_per_upper_candidate: 100
```

This is a distinctly bilevel challenge: multiple follower-optimal solutions can have different implications for the upper level.

---

## Dataset family F2 — BOLIB

BOLIB provides published continuous bilevel problem definitions, including dimensions, objectives, constraints, and known solutions.

Source:  
https://www.bolib.org/

---

### B04 — Calamai-Vicente 1994b

BOLIB reports:

- upper variables: 4
- lower variables: 2
- upper constraints: 0
- lower constraints: 6
- known global solution available

Source:  
https://www.bolib.org/problems/quadratic_bilevel/

```yaml
id: B04
family: BOLIB
problem_file: calamai_vicente_1994b.json
upper_dimension: 4
lower_dimension: 2
upper_constraints: 0
lower_constraints: 6
upper_budget: 40
lower_budget_per_upper_candidate: 100
```

---

### B05 — Allende-Still 2013

BOLIB reports:

- upper variables: 2
- lower variables: 2
- upper constraints: 5
- lower constraints: 2
- known global solution available

Source:  
https://www.bolib.org/problems/quadratically_constrained_quadratic_bilevel/

```yaml
id: B05
family: BOLIB
problem_file: allende_still_2013.json
upper_dimension: 2
lower_dimension: 2
upper_constraints: 5
lower_constraints: 2
upper_budget: 40
lower_budget_per_upper_candidate: 100
```

This gives a compact but nontrivial quadratically constrained bilevel problem.

---

# 11. Runtime strategy for a one-day benchmark

The proposed first-pass budgets are deliberately much smaller than the budgets sometimes used in benchmark papers.

The goal of this hackathon benchmark is:

```text
method discrimination under limited budget
```

not full convergence proofs or publication-grade optimizer ranking.

Approximate total per agent:

## Ordinary problems

Roughly:

```text
15k–20k direct function/surrogate evaluations
```

depending on exact dimensionality budgets.

These evaluations are generally cheap.

## Bilevel problems

With:

```text
40 upper candidates
100 lower evaluations per upper candidate
```

the worst-case nested work is roughly:

```text
4,000 lower evaluations per bilevel task
x 5 tasks
= ~20,000 lower-level evaluations
```

The SMD evaluations themselves are mathematically cheap.

The BOLIB problems are low-dimensional, so the main cost is algorithm overhead rather than function evaluation.

---

# 12. Seed policy

For the first one-day screening:

## Deterministic problems

Use:

```text
1 fixed official instance
```

for every agent.

## Noisy problems

Use:

```text
3 noise seeds
```

if time allows.

If not, use one optimization seed and perform a robust final reevaluation.

## Stochastic optimizers

Use exactly the same seed list across agents.

Example:

```python
SEEDS = [11, 29, 47]
```

Do not let different agents receive different random instances.

---

# 13. How to score noisy problems

Do **not** score using:

```text
lowest noisy value ever observed
```

That rewards lucky noise.

Instead:

1. Let the optimizer return its final recommended point `x_final`.
2. Reevaluate `x_final` many times independently.
3. Score using the mean or a robust estimator.

Recommended quick protocol:

```text
30 reevaluations of x_final
```

For heavy-tailed Cauchy noise, also store the median.

Example:

```python
samples = [problem(x_final) for _ in range(30)]

score_mean = np.mean(samples)
score_median = np.median(samples)
```

Where access to the hidden noiseless base objective is available in your implementation, store that as an additional evaluator-only metric.

---

# 14. Constraint scoring

Use **feasible-first** comparison.

Recommended ordering:

```text
1. Feasible beats infeasible.
2. Between feasible solutions: smaller objective wins.
3. Between infeasible solutions: smaller total violation wins.
```

Define total positive violation:

\[
CV(x)=\sum_i \max(0,g_i(x))
\]

with equality constraints handled using a tolerance if any are present.

Suggested result:

```python
{
    "objective": float,
    "feasible": bool,
    "constraint_violation": float
}
```

---

# 15. Bilevel scoring

Bilevel tasks need separate cost accounting.

Record:

```text
N_upper
N_lower
```

Do not report only a single evaluation number.

An agent performing:

```text
40 upper x 100 lower = 4,000 follower evaluations
```

is not equivalent in cost to one performing:

```text
40 upper x 10,000 lower = 400,000 follower evaluations
```

Recommended final result:

```python
{
    "upper_objective": float,
    "upper_feasible": bool,
    "lower_optimality_error": float,
    "upper_evaluations": int,
    "lower_evaluations": int
}
```

A final bilevel solution should be considered valid only when lower-level optimality is within a predeclared tolerance.

For SMD, exploit the known analytical structure for evaluator-side validation where practical.

For BOLIB, use the known published solutions as an additional sanity check.

---

# 16. Aggregate scoring across very different tasks

Do not directly average raw objective values because the scales differ enormously.

Recommended hackathon aggregation:

## Per task

Record:

1. final best valid objective
2. best-so-far trajectory
3. success/failure
4. runtime
5. evaluation count
6. invalid proposals
7. agent/tool overhead

## Across tasks

Use **per-task ranks**.

Example:

```text
Agent A ranks: 1, 3, 2, 1, ...
Agent B ranks: 2, 1, 1, 3, ...
```

Then compute:

```text
mean rank
median rank
number of wins
number of failures
```

This avoids one large-scale objective dominating the aggregate score.

A useful second metric is normalized regret/AUC, but rank aggregation is easiest to implement correctly for the hackathon.

---

# 17. Suggested common Python interface

Create a benchmark abstraction rather than exposing every library directly to every agent.

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationResult:
    objective: float
    feasible: bool = True
    constraint_violation: float = 0.0
    cost: float = 1.0
    metadata: dict[str, Any] | None = None


class BenchmarkProblem:
    problem_id: str
    direction: str
    variable_types: list[str]
    bounds: Any
    budget: int

    def evaluate(self, x) -> EvaluationResult:
        raise NotImplementedError
```

---

# 18. Bilevel interface

Use a separate interface for true bilevel problems.

```python
@dataclass
class BilevelEvaluation:
    objective: float
    feasible: bool
    constraint_violation: float = 0.0


class BilevelProblem:
    problem_id: str

    upper_bounds: Any
    lower_bounds: Any

    upper_budget: int
    lower_budget_per_upper_candidate: int

    def evaluate_upper(self, xu, xl) -> BilevelEvaluation:
        raise NotImplementedError

    def evaluate_lower(self, xu, xl) -> BilevelEvaluation:
        raise NotImplementedError

    def validate_lower_solution(self, xu, xl) -> dict:
        # Evaluator-side validation. Do not expose hidden solution data to the agent.
        raise NotImplementedError
```

---

# 19. Agent-visible problem specification

Each agent should receive the same information.

Example:

```python
{
    "problem_id": "C03",
    "direction": "constraints",
    "objective": "minimize",
    "dimension": 10,
    "variable_types": ["continuous"] * 10,
    "bounds": [...],
    "number_of_inequality_constraints": 3,
    "budget": 500
}
```

The agent should **not** receive:

```text
global optimum coordinates
hidden shift vectors
hidden rotations
evaluation history from other agents
benchmark-family answer keys
```

unless your experiment explicitly tests optimizer selection with benchmark identity revealed.

---

# 20. Two benchmark modes worth keeping

For the hackathon, consider running two variants later.

## Mode A — problem characteristics visible

Tell the agent:

```text
continuous
10D
3 nonlinear constraints
no noise
```

This tests:

```text
method selection
```

## Mode B — characteristics hidden

Only expose the search interface and let the agent infer properties from evaluations.

This tests:

```text
diagnosis + method selection
```

For the first implementation, **Mode A is recommended** because it is easier to interpret and debug.

---

# 21. Recommended project layout

```text
optimization-agent-benchmark/
|
|-- README.md
|-- pyproject.toml
|
|-- benchmark/
|   |-- __init__.py
|   |-- base.py
|   |-- registry.py
|   |-- runner.py
|   |-- scoring.py
|   |
|   |-- dimensionality/
|   |   |-- bbob.py
|   |   `-- cec2017.py
|   |
|   |-- variable_type/
|   |   |-- bbob_mixint.py
|   |   `-- yahpo.py
|   |
|   |-- noise/
|   |   |-- bbob_noisy.py
|   |   `-- nevergrad_noise.py
|   |
|   |-- smoothness/
|   |   |-- bbob.py
|   |   `-- cec2017.py
|   |
|   |-- constraints/
|   |   |-- bbob_constrained.py
|   |   `-- pymoo_g.py
|   |
|   `-- bilevel/
|       |-- smd.py
|       `-- bolib.py
|
|-- configs/
|   `-- problems.yaml
|
|-- agents/
|   |-- random_search.py
|   |-- scipy_agent.py
|   |-- nevergrad_agent.py
|   `-- ...
|
|-- results/
|   |-- raw/
|   `-- summaries/
|
`-- tests/
    |-- test_problem_api.py
    |-- test_known_optima.py
    `-- test_budget_accounting.py
```

---

# 22. Suggested problem registry

The 30 problems should be registered centrally.

```python
PROBLEMS = {
    "D01": {...},
    "D02": {...},
    "D03": {...},
    "D04": {...},
    "D05": {...},

    "V01": {...},
    "V02": {...},
    "V03": {...},
    "V04": {...},
    "V05": {...},

    "N01": {...},
    "N02": {...},
    "N03": {...},
    "N04": {...},
    "N05": {...},

    "S01": {...},
    "S02": {...},
    "S03": {...},
    "S04": {...},
    "S05": {...},

    "C01": {...},
    "C02": {...},
    "C03": {...},
    "C04": {...},
    "C05": {...},

    "B01": {...},
    "B02": {...},
    "B03": {...},
    "B04": {...},
    "B05": {...},
}
```

---

# 23. Recommended result record

Save **every query**, not only the final answer.

Example JSONL row:

```json
{
  "run_id": "agentA_C03_seed11",
  "agent": "agentA",
  "problem_id": "C03",
  "seed": 11,
  "evaluation": 127,
  "x": [0.2, -1.1, 0.7],
  "objective": 14.31,
  "feasible": false,
  "constraint_violation": 0.042,
  "wall_time_s": 0.0031
}
```

For bilevel:

```json
{
  "run_id": "agentA_B02_seed11",
  "problem_id": "B02",
  "upper_evaluation": 17,
  "lower_evaluations_total": 1284,
  "xu": [0.3, -0.4],
  "xl": [1.2, 0.1],
  "upper_objective": 2.81,
  "lower_objective": 0.19,
  "lower_optimality_valid": true
}
```

JSONL is recommended over CSV for raw logs because:

- variable vectors have different lengths
- metadata differs by benchmark
- bilevel records are nested
- categorical variables are easier to preserve

Produce CSV summaries afterward.

---

# 24. Baselines

At minimum, include:

1. **Random search**
2. One strong general derivative-free baseline
3. One local/gradient-based baseline where applicable
4. The optimization agent being evaluated

Why random search matters:

If an agent cannot consistently beat random search under the same budget, method selection is not adding value.

Possible later baselines:

```text
Random Search
CMA-ES
Differential Evolution
Nelder-Mead
L-BFGS-B
SLSQP / trust-constr
TPE
Bayesian Optimization
mixed-variable evolutionary method
```

Do not run algorithms on tasks where their input assumptions are invalid merely to create a failure.

Instead record:

```text
not_applicable
```

separately from:

```text
attempted_and_failed
```

---

# 25. Initial implementation order

For fastest progress in VS Code, implement in this order.

## Phase 1 — easiest callable benchmarks

1. COCO BBOB
2. pymoo G04/G07
3. Nevergrad
4. BBOB-Noisy

This establishes:

```text
common API
budget counting
logging
scoring
```

## Phase 2 — mixed/hierarchical

5. BBOB-MixInt
6. YAHPO

## Phase 3 — CEC-2017

7. Add a verified CEC implementation
8. Run sanity checks against known optimum values

## Phase 4 — true bilevel

9. Implement SMD equations
10. Add BOLIB parser/wrapper
11. Add separate upper/lower evaluation counters
12. Add lower-level optimality validation

---

# 26. Sanity tests before benchmarking agents

Every problem wrapper should pass tests like:

```python
def test_budget_counter():
    ...

def test_bounds():
    ...

def test_repeatability_deterministic():
    ...

def test_noise_changes_observation():
    ...

def test_constraint_sign_convention():
    ...

def test_known_optimum_if_available():
    ...

def test_bilevel_lower_optimality():
    ...
```

Also manually test:

```text
random point
known/near-known optimum
out-of-bounds point
invalid categorical configuration
infeasible constrained point
```

before running the full benchmark.

---

# 27. Constraint sign convention

Standardize all constraints internally to:

\[
g_i(x) \leq 0
\]

Do not let some libraries use:

```text
g >= 0
```

and others:

```text
g <= 0
```

without converting them.

A single sign error can invalidate the entire constrained benchmark.

---

# 28. Optimization-agent decision logging

Because the hackathon is about **agents for optimization methods**, store not only performance but also the method decisions.

Example:

```json
{
  "problem_id": "S02",
  "agent": "agentA",
  "selected_method": "CMA-ES",
  "reason_code": [
    "continuous",
    "non_smooth",
    "no_gradient_available"
  ],
  "method_switches": [
    {
      "evaluation": 200,
      "from": "CMA-ES",
      "to": "Nelder-Mead"
    }
  ]
}
```

This lets you answer two different questions:

```text
Did the agent get a good solution?
```

and:

```text
Did the agent make a sensible optimization decision?
```

---

# 29. What each direction is intended to diagnose

| Direction | Main diagnostic question |
|---|---|
| Dimensionality | Does method selection scale sensibly with D? |
| Variable type | Does the agent respect discrete/categorical/hierarchical structure? |
| Noise | Does it recognize uncertainty and avoid overreacting to noisy observations? |
| Smoothness | Does it avoid methods relying on inappropriate local smoothness assumptions? |
| Constraints | Does it manage feasibility correctly? |
| True bilevel | Does it correctly handle follower optimality and nested cost? |

---

# 30. Source index

## COCO / BBOB

Main BBOB suite:  
https://coco-platform.org/testsuites/bbob/overview.html

Rosenbrock f8:  
https://coco-platform.org/testsuites/bbob/functions/f08.html

BBOB-MixInt:  
https://coco-platform.org/testsuites/bbob-mixint/overview.html

BBOB-Noisy:  
https://numbbo.github.io/coco/testsuites/bbob-noisy

BBOB-Constrained:  
https://coco-platform.org/testsuites/bbob-constrained/test-suite.html

BBOB-Constrained function definitions:  
https://numbbo.github.io/coco-doc/bbob-constrained/functions.pdf

COCO data archive:  
https://coco-platform.org/data-archive/

---

## YAHPO Gym

Repository:  
https://github.com/slds-lmu/yahpo_gym

Scenarios and instances:  
https://slds-lmu.github.io/yahpo_gym/scenarios.html

Getting started:  
https://slds-lmu.github.io/yahpo_gym/getting_started.html

Data repository:  
https://github.com/slds-lmu/yahpo_data

---

## Nevergrad

Benchmark documentation:  
https://facebookresearch.github.io/nevergrad/benchmarks.html

Repository:  
https://github.com/facebookresearch/nevergrad

---

## pymoo

Test problems:  
https://pymoo.org/problems/test_problems.html

Constraint handling:  
https://pymoo.org/constraints/index.html

---

## SMD bilevel benchmarks

Project page:  
https://faculty.iima.ac.in/asinha/bilevel/

SMD description PDF:  
https://faculty.iima.ac.in/asinha/bilevel/files/Description-SMD.pdf

Reference:

> A. Sinha, P. Malo, and K. Deb.  
> Test problem construction for single-objective bilevel optimization.  
> Evolutionary Computation, 22(3):439–477, 2014.

---

## BOLIB

Main site:  
https://www.bolib.org/

Quadratic bilevel problems:  
https://www.bolib.org/problems/quadratic_bilevel/

Quadratically constrained quadratic bilevel problems:  
https://www.bolib.org/problems/quadratically_constrained_quadratic_bilevel/

---

# 31. Final benchmark checklist

Before freezing the benchmark:

- [ ] All 30 problem wrappers run.
- [ ] Every problem has a fixed budget.
- [ ] Every deterministic problem uses a fixed instance.
- [ ] Noise seeds are fixed and shared.
- [ ] Bounds use a common representation.
- [ ] Constraint sign is standardized to `g <= 0`.
- [ ] YAHPO fidelity is fixed at maximum.
- [ ] Raw query logs are saved.
- [ ] Noisy final points are reevaluated.
- [ ] Bilevel upper and lower calls are counted separately.
- [ ] Lower-level optimality is independently checked.
- [ ] Known optima are hidden from agents.
- [ ] Random search baseline works.
- [ ] Scores are aggregated by per-task rank, not raw-objective averaging.
- [ ] `attempted_and_failed` and `not_applicable` are distinguished.

---

# 32. Recommended next coding step

Do **not** start by implementing all 30 wrappers independently.

Start with:

```text
benchmark/base.py
benchmark/registry.py
benchmark/runner.py
benchmark/scoring.py
```

and make one simple problem, for example `D01`, run end-to-end:

```text
problem registry
    ->
agent
    ->
evaluate()
    ->
budget counter
    ->
JSONL log
    ->
final score
```

Only after that works should the remaining wrappers be added.

This prevents 30 slightly different interfaces from appearing across the project.
