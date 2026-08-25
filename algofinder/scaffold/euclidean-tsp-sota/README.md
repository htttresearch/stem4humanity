# Euclidean TSP: state of the art and solver-building dossier

Research cutoff: **2026-08-02**. Scope: the full symmetric Euclidean traveling
salesperson problem (ETSP), not merely the instance sizes or distributions
currently represented in Algofinder.

“Full” here means the unconstrained point-set problem across exact,
approximation, heuristic, population, learned, hybrid and extreme-scale methods,
including fixed dimension and important geometric special regimes. TSP with
time windows, neighborhoods, multiple salespeople, obstacles, Dubins motion,
time-dependent/asymmetric road cost, prize collection, or vehicle capacities
are separate variants; they appear only where their techniques illuminate ETSP.

This dossier separates claims that are often—and misleadingly—collapsed under
the label “state of the art”:

1. proving an optimum;
2. obtaining a mathematically guaranteed approximation;
3. finding the best tour in a practical time budget;
4. producing a good tour at very low latency or high batch throughput;
5. scaling to hundreds of thousands or millions of points;
6. generalizing across dimensions, distributions, coordinate conventions, and
   adversarial instances.

A system may lead one of these regimes while being a poor choice in another.

## The short answer

- **Exact, practical:** Concorde-style branch-and-cut remains the reference:
  degree and subtour LPs, families of stronger cuts, pricing, branching, strong
  primal heuristics, and exact post-verification. Held–Karp subset DP is useful
  for small instances and labels, not as a general exact engine. Fixed-
  dimensional geometry yields a theoretically optimal subexponential exact
  algorithm, but that separator/rank-based-DP result is not the practical
  champion.
- **Best tours in practice:** a serious solver should start from two mature,
  complementary lineages: **LKH** (candidate-restricted variable-depth
  Lin–Kernighan search, alpha-nearness, penalties, kicks and tour merging) and
  **GA-EAX** (edge-assembly crossover plus strong local optimization). Generic
  simulated annealing, ACO, PSO, or a plain genetic algorithm is not a credible
  SOTA core unless it incorporates search machinery of comparable strength.
- **Huge instances:** never materialize the complete distance matrix. Generate
  geometric/cost-aware candidate graphs, partition spatially, solve and stitch
  local path problems, then run boundary and global refinement. POPMUSIC and
  LKH's large-instance machinery are important precedents. The 1,904,711-city
  World TSP record is a vivid demonstration of an LKH/GA-EAX portfolio, not of
  an end-to-end neural solver.
- **Approximation theory:** planar ETSP has a PTAS. Recent results sharpen this
  to near-linear time with essentially tight exponential dependence on
  `1/epsilon`; this is a theoretical landmark, not normally the practical route
  to the best tours.
- **Machine learning:** neural construction, improvement, heatmap, diffusion,
  and hierarchical/decomposition models are active and can be compelling at
  fixed low latency or batched GPU inference. The most robust engineering bet
  is **learning-augmented search**: predict candidate edges, node penalties,
  partitions, move priorities, or portfolios, while retaining a classical
  feasibility-preserving search and a deterministic fallback. Results trained
  and tested only on uniform random points of one size do not establish general
  ETSP superiority.
- **What to build:** a portfolio with a shared geometry/candidate layer, a fast
  chained-LK engine, an EAX population engine, exact lower bounds and small-
  instance certification, spatial decomposition for large `n`, and optional
  learned policies that can be ablated without disabling the solver.

## Documents

- [Landscape](LANDSCAPE.md) — definitions, structural facts, complexity,
  approximation, exact algorithms, local search, population/metaheuristics,
  large-scale techniques, learning, and emerging work.
- [ETSP subproblem taxonomy](SUBPROBLEM_TAXONOMY.md) — the cross-cutting lattice
  of dimension, intrinsic geometry, hull/layer structure, width, distributions,
  density, degeneracy, numerical semantics, scale, and restricted-tour tasks.
- [Implementation blueprint](IMPLEMENTATION_BLUEPRINT.md) — data contracts,
  candidate generation, tour structures, LK/EAX/exact/decomposition engines,
  learning integration, pseudocode, defaults, and staged delivery.
- [Benchmark research report](BENCHMARK_RESEARCH.md) — theory of good ETSP
  benchmarks, public-suite strengths and limits, feature/instance-space design,
  generator experiments, references, anytime/statistical methodology, ML
  leakage, governance, and anti-patterns.
- [Benchmark catalog](BENCHMARK_CATALOG.md) — public sources, a proposed
  eight-layer corpus, controlled generators, size ladders, splits, references,
  features, weights, and artifact acceptance criteria.
- [Benchmark implementer specification](BENCHMARK_IMPLEMENTER_SPEC.md) —
  decision-ready schemas, invariants, subsystem boundaries, workflows, metrics,
  release contracts, implementation phases, and definition of done.
- [Benchmark literature](BENCHMARK_LITERATURE.md) — benchmark-focused annotated
  bibliography with full citations, primary links, publication status,
  implementer relevance, interpretation limits, and licensing notes.
- [Benchmarking run protocol](BENCHMARKING.md) — baselines, budgets,
  quality/time statistics, generalization, ablation, exact-solver records,
  metamorphic validation, and publication checklist.
- [Algofinder gap analysis](ALGOFINDER_GAP_ANALYSIS.md) — what the repository
  currently implements, what its results do and do not show, and a prioritized
  path from the current 2-opt stack toward a competitive ETSP laboratory.
- [Annotated sources](SOURCES.md) — primary papers, official implementations,
  benchmark repositories, benchmark-science literature, licensing cautions,
  and frontier/preprint labels for the full solver dossier.

## A decision table

| Need | Recommended starting point | What “done” means |
|---|---|---|
| Exact optimum, small `n` | Held–Karp DP; branch-and-bound with 1-tree bound | Tour plus independently checked bound/certificate |
| Exact optimum, serious instances | Concorde or Concorde-like branch-and-cut | Zero integral gap and exact LP verification |
| Best tour, seconds to hours | LKH and GA-EAX portfolio | Best-of-seeds anytime curve, not one terminal result |
| Millisecond/batched construction | POMO/attention-style constructor or learned heatmap, followed by tiny 2-opt budget | End-to-end wall time including augmentation/decoding |
| `10^4–10^6+` points | Sparse candidates + spatial decomposition + local path solves + global refinement | Linear/near-linear memory and reported residual gap/bound |
| Research on learned guidance | NeuroLKH/DeepACO-style hybrid with safe fallback | Gains over the same classical engine and same budget |
| Approximation guarantee | Euclidean PTAS implementation/research | Declared metric/precision model and `(1+epsilon)` proof |

## Non-negotiable scientific cautions

1. **Name the metric.** True floating Euclidean length is not TSPLIB's
   `EUC_2D`, which rounds every edge to the nearest integer. `GEO`, `ATT`, and
   raw floating distances are different problems. A tour optimal for one need
   not be optimal for another.
2. **Do not call best-known “optimal.”** A reference tour without a matching
   lower bound is an upper bound. Report `gap_to_best_known` separately from a
   certified optimality gap.
3. **Count all work.** Neural inference timing must include augmentation,
   sampling, decoding, feasibility repair, local search, device transfer, and
   model loading according to a declared warm/cold protocol.
4. **Do not use `n` as a difficulty proxy.** Small constructed Euclidean
   instances can be brutal for exact solvers; huge benign geometric instances
   can yield excellent tours quickly.
5. **Heuristic sparsification is not proof-preserving.** k-nearest-neighbor,
   Delaunay, learned, and alpha-candidate graphs are excellent search devices,
   but excluding an edge is only exact when a valid elimination argument proves
   it cannot belong to an optimum.
6. **Generalization is part of the problem.** Evaluate unseen sizes and unseen
   spatial processes, coordinate scales, aspect ratios, rotations, density
   shifts, degeneracies, and adversarial families.

## Current record context

The [official World TSP page](https://www.math.uwaterloo.ca/tsp/world/) reports,
as of its 2025-10-24 update, a best tour of **7,515,755,912** for the
1,904,711-city instance, found by Yuichi Nagata by continuing GA-EAX from the
previous LKH record tour. The best lower bound is **7,512,218,268**, leaving the
tour at most about **0.0471%** above optimum. The metric is a special rounded
great-circle `GEOM` norm, so this is a TSP scale record and methodological case
study, not literally planar `EUC_2D`.

For exact solution, [Concorde](https://www.math.uwaterloo.ca/tsp/concorde.html)
has solved all 110 TSPLIB instances and proved a tour for the 85,900-node
`pla85900` VLSI instance optimal. These records explain why practical SOTA must
be discussed in terms of both upper-bound search and lower-bound/certification
machinery.
