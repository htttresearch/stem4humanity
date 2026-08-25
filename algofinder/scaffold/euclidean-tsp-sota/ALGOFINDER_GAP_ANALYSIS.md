# Algofinder Euclidean TSP gap analysis

This audit describes the repository state inspected on 2026-08-02. It is not the
scope of the research: the broader target is the full ETSP landscape described
in the other documents.

## 1. Current problem semantics

`algofinder/problems/tsp.py` represents 2D coordinates as float64 and eagerly
constructs the complete `n x n` raw Euclidean distance matrix with
`numpy.linalg.norm`. Tours are permutations and the closing edge is implicit.

Consequences:

- objective semantics are clear for current synthetic instances: unrounded raw
  L2 under float64;
- TSPLIB `EUC_2D` files cannot be compared to published optima through this
  class without implementing TSPLIB's per-edge rounding;
- memory and preprocessing become quadratic, blocking serious large-instance
  work;
- the generic symmetric-matrix base is useful, but the Euclidean subclass needs
  a sparse/on-demand path rather than always inheriting a dense matrix.

## 2. Current instances and evidence

The synthetic generator provides:

- uniform points;
- separated Gaussian clusters;
- noisy sinusoidal corridors;
- random rotation, reflection, scaling and translation.

Test size bands are 9–13, 18–24, 38–46 and 95–105, six records per band;
training uses 12 instances of size 11–15. Exact `best_known` annotations are
generated only up to 16 cities.

This is a reasonable smoke-test suite, but it supports no conclusion about
full ETSP SOTA. Missing are:

- public TSPLIB/DIMACS/Waterloo cases;
- sizes above 105 and sparse-memory scaling;
- grids, rings, manifolds, extreme rectangles, many-cluster/imbalanced mixtures,
  holes, outliers, duplicates and quantization;
- exact-hard tetrahedron/adversarial families;
- held-out distribution and size generalization;
- lower bounds and certified gaps on the larger rows.

The current leaderboard's near-zero displayed TSP gaps are mostly uninformative:
larger test instances lack a reference value, while tiny referenced instances
are easy for the existing methods. A dash should not be read as zero, and
“best on” over this suite is not a broad quality claim.

## 3. Current solver stack

### 3.1 Heuristic core

`chained-2opt-euclidean`:

- convex-hull initialization;
- maximum-regret insertion of interior points;
- candidate union of 12 nearest and nearest in 8 angular sectors, symmetrized;
- first-improvement candidate 2-opt;
- random double-bridge perturbations;
- eight restarts and at most 2,000 accepted 2-opt moves per local search.

This is a good first Euclidean ILS baseline. It is not LKH-class: there is no
1-tree, alpha-nearness, penalty ascent, 3/5-opt or variable-depth exchange,
efficient tour segment structure, active/don't-look queue, tour merging,
candidate widening, or strong elite mechanism.

Implementation bottlenecks:

- dense distances and per-city full `argsort` make candidate construction
  quadratic or worse in practical memory traffic;
- regret insertion scans every remaining city against every tour edge, roughly
  cubic in the direct implementation;
- accepted 2-opt reverses a Python list slice and rebuilds a full city-to-
  position dictionary each pass;
- candidates are sorted repeatedly inside search;
- budget seconds are accepted by the solver interface but not used to stop the
  fixed restart/iteration loops;
- the candidate graph can omit decisive edges and has no widening/fallback
  triggered by stagnation.

`distance-ranked-2opt` reduces each geometric list to the closest eight choices.
It is the correct kind of matched control for candidate learning, but selecting
directed top-k lists after initial symmetrization can make the final search
neighborhood asymmetric. That is not invalid, but it should be deliberate and
measured.

`general-2opt` is useful as a nongeometric baseline.

### 3.2 Learned candidate solver

The learned ranker is a 250-tree balanced random-forest classifier trained on
exact Held–Karp tour edges from 11–15-city synthetic instances. Its 12 hand-
designed invariant features cover relative length, neighbor ranks, mutual
candidate status, local/global density, hull membership, midpoint radius and
angular gaps. At inference it ranks only edges already produced by the smaller
8-neighbor/4-sector graph and retains five per city before running the same
chained 2-opt.

Good design choices:

- feasibility remains classical;
- features are translation/rotation/scale invariant;
- a distance-ranked matched control exists;
- candidate recall during label construction is recorded;
- model preprocessing/search time is separated in metadata.

Limits:

- only 12 tiny training instances cover a very small distribution;
- one exact tour labels alternate optimal edges as negatives;
- no alpha/1-tree, Delaunay, elite-frequency, bridge, or relaxation features;
- the model can only reorder a pre-pruned graph and cannot recover an omitted
  edge;
- retaining learned top-5 can delete strong classical candidates rather than
  adding a safe learned quota;
- edge-classification probability is not necessarily search utility;
- no held-out size/distribution/OOD evaluation or calibration;
- the leaderboard currently shows greater mean runtime and no convincing
  quality advantage over the distance control.

The right next learning experiment is not a larger end-to-end Transformer. It is
a broad-corpus candidate ranker that augments mixed classical candidates and is
tested through the same improved LK engine.

### 3.3 Exact solvers

`held-karp` is a compiled subset DP with a default maximum of 18 cities. It uses
O(n 2^n) storage for costs/parents and the standard O(n² 2^n) recurrence. It is
valuable as an oracle and labeler.

`incremental-exact` is a substantial assignment-relaxation branch-and-bound,
default maximum 25. It incrementally repairs the Hungarian assignment after
branching and is a meaningful research baseline. On the current leaderboard it
times out on some Euclidean cells under the 30-second hard cap. Its assignment
cycle-cover bound is useful, but the solver is not a replacement for 1-tree/
subtour branch-and-cut, geometric edge elimination, or Concorde.

Both exact solvers operate on float64 semantics. `exact=True` means exhaustive
optimization of those stored coefficients, not a formal bit-model proof about
an ideal sum of radicals.

## 4. Priority changes

### P0 — make results meaningful

1. Add an explicit metric enum and TSPLIB parser/rounding tests.
2. Distinguish certified optimum, best-known upper bound, and lower bound in
   manifests and reports.
3. Add time-quality traces and enforce cooperative deadlines in all heuristics.
4. Add a sparse/on-demand Euclidean problem path; retain dense matrices only
   below a configured threshold.
5. Import a stratified public and synthetic benchmark suite, with per-family
   reporting and exact-hard cases.

### P1 — establish a strong classical baseline

1. Replace dense neighbor sorting with kd-tree/spatial candidate construction.
2. Add multi-fragment construction and a faster incremental regret constructor.
3. Implement active candidate 2-opt, Or-opt and 3-opt with an efficient tour
   representation.
4. Add a minimum 1-tree, subgradient penalties and alpha-nearness.
5. Build staged variable-depth LK, adaptive kicks, elite candidates and tour
   merging.
6. Benchmark the official LKH release through a native/external adapter.

This step is prerequisite to credible “best solver” or ML claims.

### P2 — robustness and breadth

1. Implement candidate widening and global exception edges.
2. Add independent seeded trials, elite pool and parallel portfolio.
3. Integrate/implement EAX and share elite tours with LK.
4. Add POPMUSIC/overlapping path decomposition for 10k–million scale.
5. Track 1-tree lower bounds so heuristic runs can return an honest bounded gap.

### P3 — learning that can matter

1. Train on exact and multi-elite labels from a broad curriculum.
2. Add alpha, penalties, candidate provenance, elite frequency and bridge
   features.
3. Reserve learned candidate slots instead of replacing structural candidates.
4. Add confidence/OOD detection and widening.
5. Compare a calibrated tree ranker, sparse invariant GNN and no-model baseline.
6. Explore portfolio, partition-port and EAX-repair guidance after the candidate
   experiment succeeds.

### P4 — exact/certified path

1. Strengthen the in-house exact small solver with heuristic incumbents and
   1-tree/MST bounds.
2. Add exact-safe geometric edge elimination.
3. Integrate Concorde for reference generation and certification under its
   licensing constraints.
4. Only if exact-solver research is itself a goal, build native DFJ branch-and-
   cut with separation, pricing and exact verification.

## 5. Concrete first research release

A contained but meaningful next release would include:

```text
etsp metric contract + sparse coordinate oracle
TSPLIB + expanded synthetic manifests
mixed kNN/sector/Delaunay candidates
1-tree lower bound + alpha candidates
multi-fragment and hull-regret seeds
native active 2/3-opt ILS with deadlines and anytime trace
official LKH adapter
per-family quality-time report
```

That release would transform Algofinder from a small 2-opt demonstration into a
credible ETSP experimentation platform. The subsequent LKH-class, EAX,
decomposition and learning stages should be admitted only through matched
ablation gates defined in [the benchmarking protocol](BENCHMARKING.md).

## 6. What should remain in Python and what should not

Keep in Python:

- manifests, metric parsing, orchestration and trusted verification;
- benchmark/report generation;
- data generation, feature analysis and ML training;
- small reference implementations.

Move to native/compiled code:

- large spatial candidate construction;
- 1-tree/alpha computation;
- tour sequence operations and variable-depth move search;
- EAX cycle/repair loops;
- large exact and decomposition hot paths.

The existing `scaffold/cpp-solvers-design.md` subprocess boundary aligns with
this split. For ETSP benchmarking, a persistent worker is preferable to one
process per cell because short local-search trials make spawn overhead visible.
Python should still independently verify every returned tour and objective.

## 7. Acceptance gates

Do not promote a new solver because it wins the current 24-row synthetic test.
Require:

- no metric/verifier discrepancies;
- matched budgets and end-to-end timing;
- improvement in anytime curves over the existing and official baselines;
- no material tail regression across public, shifted, and adversarial families,
  or a portfolio rule that avoids it;
- bounded memory at the advertised scale;
- for learning, positive results after model inference and postprocessing cost,
  with the identical classical search;
- for exact claims, a matching valid lower bound and independent check.
