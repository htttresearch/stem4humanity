# Benchmarking and scientific protocol

An ETSP benchmark is credible only when objective semantics, reference status,
budgets, hardware, and postprocessing are aligned. This protocol is designed to
distinguish solver progress from metric mistakes, data leakage, or mismatched
search effort.

This file is the execution/publication checklist. The rationale and statistical
theory are in [BENCHMARK_RESEARCH.md](BENCHMARK_RESEARCH.md), the concrete data
plan is in [BENCHMARK_CATALOG.md](BENCHMARK_CATALOG.md), and the future build
contracts and phased plan are in
[BENCHMARK_IMPLEMENTER_SPEC.md](BENCHMARK_IMPLEMENTER_SPEC.md).

## 1. Questions, not one leaderboard

Maintain separate views:

1. **Exact:** solved count, proof time, search nodes, bound trajectory and peak
   memory.
2. **Anytime quality:** gap and success probability as functions of wall time.
3. **Fast inference:** cold and warm latency, throughput, and quality after a
   strictly fixed postprocessing budget.
4. **Scaling:** quality, time and peak resident/device memory against `n`.
5. **Robustness:** performance by distribution/family and under unseen shift.
6. **Component research:** matched-engine ablations for candidates, models,
   move types, perturbations, population and decomposition.

Mixing these into one mean rank obscures the engineering decision.

## 2. Instance corpus

### 2.1 Public standards

- **TSPLIB95:** classic named instances and published optima/best knowns. Parse
  each `EDGE_WEIGHT_TYPE` exactly according to the
  [specification](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf).
  Do not reinterpret `EUC_2D` as raw float L2.
- **DIMACS 8th Challenge:** includes large TSPLIB, uniform random, clustered,
  national, VLSI and other families, with machine-normalized timing conventions.
  See the [challenge description](https://dimacs.rutgers.edu/archive/Challenges/TSP/about.html).
- **Waterloo TSP data:** national, World, VLSI, art and other large instances;
  see the [data index](https://www.math.uwaterloo.ca/tsp/data/index.html).
- **World TSP:** 1,904,711 locations under `GEOM`, useful for huge-scale methods
  and upper/lower-bound context, but not a planar raw-L2 benchmark.
- **Hard/adversarial families:** include tetrahedron instances from
  [Hougardy and Zhong](https://doi.org/10.1007/s12532-020-00184-5) and any
  available structured instances designed to stress candidates or exact bounds.

Record the original file checksum and parser version. Store reference tours as
well as objective values when licensing permits, then independently rescore
them.

### 2.2 Synthetic factorial suite

Generate multiple seeds across a crossed design:

| Axis | Levels |
|---|---|
| size | 10–25, 50, 100, 200, 500, 1k, 5k, 10k, 100k, 1m as feasible |
| process | uniform, Gaussian mixtures, Poisson-disc, grid+jitter, rings, curves, corridors/strips |
| aspect ratio | 1, 2, 10, 100 |
| clusters | 1, 2–5, 10–50; balanced and heavy-tailed sizes |
| density contrast | mild through extreme |
| outliers/holes | none, sparse outliers, central/multiple holes |
| degeneracy | duplicates, near-duplicates, collinear runs, cocircular points |
| coordinates | float, small integer, large integer, quantized |
| transforms | rotation, reflection, translation and scale |

Use nested but independent generator seeds. Reserve entire generator parameter
regions and families for out-of-distribution test, not merely unseen random
draws from the same setting.

### 2.3 Split discipline for ML

- `train_iid`: sizes/processes used in training;
- `test_iid`: unseen instances under the same generator distribution;
- `test_size`: larger and smaller `n`;
- `test_distribution`: held-out spatial families and parameters;
- `test_metric`: coordinate quantization/rounding shifts where supported;
- `test_public`: TSPLIB/DIMACS/Waterloo, never used for hyperparameter search;
- `test_adversarial`: hard constructions and mutation-selected instances.

Rigid transforms of one point set stay in the same split. So do differently
permuted city IDs and derived subproblems. Deduplicate by a transform-aware
coordinate signature where practical.

## 3. References and gaps

Let `U` be solver tour length, `U*` a reference upper bound, and `L` a valid
lower bound under the same metric.

```text
gap_to_optimum        = (U - OPT) / OPT           only if OPT certified
gap_to_best_known     = (U - U*) / U*             can be negative
certified_gap         = (U - L) / L
improved_best_known   = U < U* after exact rescoring
```

Never clamp a negative best-known gap to zero: it may indicate a genuine new
tour or, more often, a metric/parser/reference bug. Investigate before publishing.

For synthetic small cases, Held–Karp supplies exact references. For medium
cases, Concorde or a verified branch-and-cut run supplies optima. For large
cases, keep best upper and lower bounds as independently versioned artifacts.

## 4. Baseline matrix

### 4.1 Required classical baselines

- nearest neighbor and multi-fragment greedy;
- Christofides implementation for a guaranteed metric floor;
- full 2-opt and candidate 2-opt;
- chained 2-opt with double-bridge under multiple seeds;
- official LKH release, with parameter file archived;
- official/public GA-EAX where build and license permit;
- Concorde for exact/small-to-moderate cases;
- the proposed solver with every new component disabled in turn.

Use the same coordinate file and metric. When an external solver expects integer
costs, do not compare it directly to raw-L2 results unless an explicit conversion
creates a new named benchmark.

### 4.2 ML baselines

Choose baselines appropriate to the claimed role:

- constructor: attention model/POMO plus identical local search;
- heatmap: distance and alpha heatmaps through the identical decoder;
- improvement: random/best-immediate move selection through the same move set;
- candidate model: kNN/sector/Delaunay/alpha and a simple tree model;
- hybrid: identical LKH/ACO engine without learned guidance;
- large scale: classical partitioning with the same local solver;
- diffusion: same sample count, feasibility projection and 2-opt budget.

Re-running a published checkpoint is not enough if its decoder or post-hoc
search differs. Factor results into `representation score + decoder + local
search`.

## 5. Budget policy

### 5.1 Wall-clock checkpoints

Report a logarithmic sequence appropriate to scale, for example:

```text
10 ms, 30 ms, 100 ms, 300 ms, 1 s, 3 s, 10 s, 30 s,
1 min, 3 min, 10 min, 1 h
```

Not every solver runs at every checkpoint. A process emits incumbents during one
run, or independent capped runs are used when interruption changes behavior.

### 5.2 Fair accounting

Charge:

- parsing and metric construction according to declared end-to-end and
  solve-only views;
- candidate preprocessing;
- JIT compilation in cold results and separately cached in warm results;
- model load/device transfer in cold results;
- augmentation, sampling, decoding, repair, and local search;
- all parallel CPU cores and GPUs;
- external solver startup consistently.

Report hardware model, OS, compiler flags, math/LP libraries, thread counts,
power mode when material, and model precision. DIMACS-style normalized timing
can supplement, not replace, raw wall time.

### 5.3 Work-normalized views

Wall time is the user-facing metric. Also report algorithmic work where possible:

- edge/move evaluations;
- accepted moves by type/depth;
- LKH trials;
- EAX children and repair attempts;
- neural samples and forward passes;
- branch-and-cut nodes, LP iterations, cuts and priced edges.

This explains hardware-sensitive results and catches an implementation that
quietly did much more search.

## 6. Replication and statistics

- deterministic solvers: one verified run, plus repeat timing samples;
- stochastic solvers: at least 10 seeds for routine experiments and 30+ for
  close claims, or enough to give useful confidence intervals;
- pair seeds/instances between configurations;
- report median, mean, standard deviation, 10/90th percentiles and worst-case
  tail by instance family;
- bootstrap paired confidence intervals for gap/time differences;
- use survival/time-to-target curves for hitting a known objective;
- report probability of reaching optimum/target by deadline;
- correct for multiple comparisons when exploring many variants;
- publish per-instance rows, not only aggregates.

A performance profile or data profile is more informative than a single
average. Anytime empirical cumulative distributions show whether one solver is
consistently good or wins through a few lucky seeds.

## 7. Quality-time summaries

Recommended primary plots/tables:

- median and 90th-percentile gap versus log wall time;
- fraction reaching certified optimum or target gap versus time;
- primal `U(t)`, lower `L(t)` and certified gap for bounded/exact modes;
- quality versus peak memory and `n`;
- per-family heatmap of paired gap improvement;
- candidate recall/degree/time versus final tour gap;
- Pareto frontier of latency, quality, memory and energy if measured.

For each instance, compute area under a clipped log-time/gap curve only as a
secondary scalar. Always retain the curve because scalarization embeds arbitrary
weights.

## 8. Ablation matrix

### 8.1 Candidate layer

```text
kNN only
kNN + sectors
+ Delaunay
+ alpha-nearness
+ elite edges
+ learned quota
learned replacing vs augmenting classical candidates
candidate budgets 8/16/32/64/128
widening off/on
```

Measure build time, memory, average/max degree, optimum/elite-edge recall,
connectivity, search evaluations and final anytime quality.

### 8.2 Search layer

```text
2-opt -> +Or-opt -> +3-opt -> shallow LK -> full variable-depth LK
no kick / double bridge / adaptive kick
one start / independent multistart / elite tour merge
flat array / segment representation
first / best improvement
```

### 8.3 EAX

```text
AB-cycle selection rules
single vs multi-cycle crossover
repair variants
local vs full post-crossover search
quality-only vs diversity-aware replacement
population size and stage schedule
LKH elite exchange off/on
```

### 8.4 Decomposition

```text
closed local tours vs path subproblems
partition family and shifted boundaries
leaf size/overlap
port count and selection
boundary reoptimization radius
global exception candidates
global final LK budget
```

### 8.5 Learning

```text
same decoder/search with no model
distance/tree/GNN scores
geometry only / +alpha / +elite / +embeddings
single-optimum vs elite-frequency labels
random negatives vs hard negatives
IID vs multi-distribution training
confidence/OOD widening off/on
inference and training compute charged separately
```

## 9. Exact-solver protocol

For every “optimal” row retain:

- incumbent tour and exact recomputed value;
- final lower bound and integrality tolerance;
- objective convention;
- branch/cut log and solver build;
- proof/check artifact when supported;
- independent verification result.

Measure root relaxation gap, cuts by family, edge reduction, priced columns,
branch nodes, peak tree memory, time to best incumbent and time from incumbent
to proof. The last distinction is crucial: a heuristic may find the optimal tour
instantly while certification takes days.

## 10. Adversarial and metamorphic testing

Every solver should preserve objective/tour quality under:

- permutation of city IDs;
- translation, rotation and reflection for raw L2;
- positive uniform scaling, with objective scaled accordingly;
- reversal/rotation of an input/reference tour;
- insertion of exact duplicates under the documented expansion policy.

Use fuzzing to generate small instances and compare all exact solvers. Mutate
coordinates to maximize disagreement between candidate variants or heuristic
gap. For every accepted local move, compare the O(1) delta with a full objective
recompute in an instrumented build.

## 11. Artifact schema

One run record should include:

```json
{
  "instance": {"id": "...", "sha256": "...", "n": 1000,
               "d": 2, "metric": "TSPLIB_EUC_2D", "family": "..."},
  "solver": {"id": "...", "build": "...", "config_sha256": "...",
             "model_sha256": null},
  "execution": {"seed": 7, "threads": 1, "device": "cpu",
                "budget_s": 10.0, "wall_s": 10.01, "peak_rss_bytes": 0},
  "result": {"status": "feasible", "objective": 12345,
             "reference_upper": 12340, "lower_bound": 12200,
             "gap_to_best_known": 0.000405, "certified_gap": 0.011885,
             "tour_sha256": "..."},
  "trace": [{"t": 0.03, "objective": 13001},
            {"t": 0.72, "objective": 12410}]
}
```

Use JSON strings or integers for objective values whose exact representation
would be lost in a JSON number. Archive config/model/source digests and raw rows;
rendered leaderboards should be reproducible from the records alone.

## 12. Publication checklist

- [ ] Objective mode exactly named and tested.
- [ ] Reference identified as optimum, upper bound, or lower bound.
- [ ] End-to-end budget and hardware disclosed.
- [ ] Threads, seeds, samples and augmentations equalized or explicitly shown.
- [ ] Postprocessing included and ablated.
- [ ] Official LKH/EAX/Concorde configuration archived.
- [ ] IID, size-shift, distribution-shift, public and adversarial tests present.
- [ ] Per-instance and tail results published.
- [ ] Statistical uncertainty reported.
- [ ] Peak memory and failures/timeouts included.
- [ ] New best-known tours independently rescored and saved.
- [ ] Exact claims independently verified.
