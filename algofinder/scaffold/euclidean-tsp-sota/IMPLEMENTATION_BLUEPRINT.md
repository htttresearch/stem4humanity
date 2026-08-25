# Implementation blueprint for a best-in-class Euclidean TSP system

This is a design for a solver laboratory and an eventual competitive solver,
not a recommendation to reimplement every paper before measuring anything. The
architecture keeps exact, heuristic, learned, and large-scale paths on a shared
problem contract while allowing native implementations where constant factors
matter.

## 1. Product targets

Expose explicit modes rather than one ambiguous `solve`:

```text
FAST       latency-first construction and shallow improvement
QUALITY    multi-start LK + optional EAX portfolio
HUGE       linear-memory spatial decomposition + sparse global refinement
EXACT      branch-and-cut or external Concorde; certified status only
BOUNDED    heuristic incumbent + lower bound, returning a real gap
RESEARCH   instrumented component/ablation mode
```

Every mode takes a deadline and remains anytime: it returns the best valid tour
seen even when interrupted. `EXACT` returns `optimal` only after a verified
matching lower bound.

## 2. Core problem and result contracts

### 2.1 Instance

```text
EuclideanInstance:
    id: string
    coordinates: contiguous float64[n,d] or integer[n,d]
    dimension: int
    metric: RAW_L2 | TSPLIB_EUC_2D | CEIL_2D | ATT | GEO | GEOM | ...
    scale/rounding metadata
    duplicate policy
    coordinate digest and canonical instance digest
```

Do not precompute a dense matrix in the base class. Give solvers a cost oracle:

```text
cost(i,j) -> objective-domain edge value
squared_geometric_distance(i,j) -> only for safe spatial ranking
cost_batch(edge_pairs) -> vectorized evaluation
```

For rounded integer metrics, accumulate tour cost in a wide integer. For raw
Euclidean, sum float64 with compensated or pairwise summation and re-evaluate a
final contender in extended precision if deltas are nearly tied. A move is
accepted only under a documented tolerance policy; log near-zero comparisons.

### 2.2 Result

```text
TspResult:
    canonical_tour: int[n]
    objective: int64 | float64
    lower_bound: optional number
    certified_gap: optional ratio
    status: feasible | optimal | timeout | inapplicable | error
    metric and instance digest
    solver/config/build/model digests
    seed, threads, devices
    wall/cpu time, evaluations, trials, accepted moves
    time-quality trace
```

The Python harness—or a separate trusted checker—must validate permutation,
recompute the objective under the instance metric, and reject inconsistent
native results.

## 3. Layered architecture

```text
coordinates + metric
        |
        +--> geometry index ----> candidate graph ------------------+
        |        |                       |                           |
        |        +--> hull/partitions    +--> LK / local search ----+--> elite pool
        |                                +--> constructors ---------+       |
        |                                +--> EAX population <------+       |
        |                                +--> learned ranking               |
        |                                                                    v
        +--> relaxation engine --> penalties / alpha / lower bounds --> incumbent
        |                                                       |            |
        +--> exact engine <-------------------------------------+------------+
        |
        +--> decomposition: region paths -> stitch -> boundary/global search
```

Stable seams:

- `DistanceOracle`
- `CandidateProvider`
- `Tour` and `MoveEngine`
- `Constructor`
- `Improver`
- `Perturbation`
- `PopulationEngine`
- `LowerBoundEngine`
- `Partitioner` / `PathSubsolver`
- `GuidanceModel` (optional)
- `StopController` and `EventSink`

This makes every learned component replaceable with a deterministic baseline
and every native engine auditable through the same result contract.

## 4. Geometry and candidate subsystem

### 4.1 Preprocessing

1. Validate finite coordinates and metric metadata.
2. Group exact duplicate coordinates. Either keep zero-cost chains explicitly
   or solve representatives and expand duplicates deterministically; expansion
   must preserve all city IDs.
3. Compute robust convex hull with stable collinearity policy.
4. Build a kd-tree or grid; use a robust Delaunay library only if available.
5. Compute scale features: bounding box, median k-NN distance, density estimates,
   hull membership/depth, angular gaps and cluster labels.
6. Detect trivial/special paths: one-dimensional, all-collinear, all-on-hull,
   or very small `n`.

### 4.2 Candidate graph construction

Recommended initial per-node budget for planar quality mode: 20–40 undirected
edges, tuned by experiment. Build a union before truncating:

- 12–20 nearest neighbors;
- 1–3 closest in each of 8–16 angular sectors;
- Delaunay edges where available;
- 5–20 alpha-near edges from an optimized 1-tree;
- edges from current and elite tours (never evict current-tour edges);
- a small learned quota, e.g. 4–8;
- cross-cluster/partition bridge candidates;
- symmetric closure: if `j in C(i)`, normally expose `i in C(j)`.

Score example:

```text
score(i,j) = w_a * normalized_alpha(i,j)
           + w_d * log1p(distance(i,j)/local_scale(i))
           - w_e * elite_frequency(i,j)
           - w_l * calibrated_log_odds(i,j)
           + w_s * sector_redundancy(i,j).
```

Keep provenance bits per candidate. This supports ablations and prevents a
learned model from silently crowding out structural edges.

Candidate levels:

```text
L0: 8–12 ultra-local, latency path
L1: 20–40 mixed candidates, normal LK
L2: 60–100 widened candidates after stagnation
L3: spatial/exhaustive fallback for selected active vertices
```

Measure recall against exact optimal edges on small instances and against the
best-known elite union on larger instances. Report both mean recall and the
fraction of instances with 100% recall; the latter exposes catastrophic misses.

### 4.3 1-tree and alpha-nearness

Implement a minimum 1-tree engine:

```text
min_one_tree(root, transformed_cost):
    T = MST(V - {root})
    add two cheapest root edges
    return T, cost(T), degrees(T)
```

Use Prim with dense O(n²) scans for moderate dense instances or a sparse
geometric graph plus edge-generation fallback for large instances. Subgradient
optimization:

```text
pi = 0
best_bound = -infinity
step = initial_step
repeat until iteration/deadline/stagnation:
    T = min_one_tree(c_ij + pi_i + pi_j)
    bound = transformed_cost(T) - 2 * sum(pi)
    best_bound = max(best_bound, bound)
    g_i = degree_T(i) - 2
    if all g_i == 0: T is a tour; update incumbent and stop
    pi_i += step * g_i
    adapt/reduce step when progress stalls
```

A Polyak-style step can use `(upper_bound - bound) / ||g||²`, scaled and
clamped. Carry penalties between trials but reset when they become pathological.
Compute alpha values by tree sensitivity/replacement-edge queries rather than
rerunning an MST per edge.

## 5. Tour representation and move engine

### 5.1 Representation

For a prototype:

- `next[n]`, `prev[n]` for adjacency;
- `position[n]` and a flat order array for cheap membership/orientation;
- lazy segment reversal with periodic rebuild.

For a production variable-depth search:

- two-level tree: tour split into parent segments; segments have orientation,
  endpoints and sizes;
- or a balanced sequence tree/rope supporting split, join, reverse, predecessor,
  successor and between queries in logarithmic time;
- optimized special cases for the dominant sequential moves.

Keep move evaluation separate from mutation. A move object lists deleted and
added undirected edges plus the required reconnection. Validate degree 2 and one
cycle in debug/research mode.

### 5.2 2-opt kernel

For tour edges `(a,b)` and `(c,d)` in compatible cyclic order:

```text
gain = cost(a,b) + cost(c,d) - cost(a,c) - cost(b,d)
if gain > tolerance:
    reverse segment b..c
```

Candidate-restricted first improvement is fast; best improvement is more
reproducible but evaluates more. Use active vertices/don't-look bits. Reset bits
for endpoints and spatially affected neighbors after an accepted move.

A crossing-removal pass can query segment bounding boxes rather than inspect all
pairs. It is a useful validation invariant but not a substitute for 2-opt.

### 5.3 LK search skeleton

Represent an alternating sequence

```text
x1=(t1,t2) delete, y1=(t2,t3) add,
x2=(t3,t4) delete, y2=(t4,t5) add, ...
```

Maintain cumulative gain and connectivity constraints. Search outline:

```text
for t1 in active vertices:
  for incident tour edge x1=(t1,t2):
    DFS(depth=1, endpoint=t2, gain=cost(x1))

DFS(depth, endpoint, gain):
  for y=(endpoint,u) in ordered_candidates(endpoint):
    skip if y is a tour edge, repeats an endpoint, or violates rules
    partial_gain = gain - cost(y)
    prune if configured positive-gain criterion fails

    if closing edge (u,t1) yields a feasible tour and positive total gain:
        record/apply best admissible move

    for tour edge x incident to u that continues the alternating path:
        next_gain = partial_gain + cost(x)
        recurse subject to depth, feasibility, exclusion and backtracking limits

  at selected depths, enumerate nonsequential reconnections/submoves
```

Engineering features required before calling this LKH-class:

- alpha-ordered candidates and penalties;
- correct enumeration of 2–5-opt reconnections;
- restricted backtracking at early levels;
- exclusion sets and tour-feasibility tests;
- breadth/depth schedules and gain criteria;
- active queue/don't-look bits;
- multiple trials and kicks;
- tour merging and candidate augmentation from elite tours.

Start with chained candidate 2-opt/3-opt, then add LK depth incrementally. A
buggy “LK-inspired” recursive search can be slower and worse than tuned 2-opt;
property-test every move against a slow reference implementation outside the
production harness before relying on it.

### 5.4 Perturbations and acceptance

Implement at least:

- random double bridge with spatial separation;
- candidate-aware bridge exchange;
- kick strength selected from recent improvement/stagnation;
- penalized kick that removes high-frequency elite edges;
- destroy/repair window for clusters or partition boundaries.

Default ILS accepts the locally optimized kicked tour for continuation while
retaining a global incumbent. Avoid a Metropolis layer until it beats this
simple policy under an equal move engine.

## 6. Constructors

Implement a diversified seed portfolio:

1. convex-hull plus maximum-regret insertion;
2. multi-fragment greedy over candidate edges using union-find and endpoint
   degree constraints;
3. nearest/farthest insertion variants;
4. space-filling-curve order for huge instances;
5. randomized nearest neighbor;
6. optional neural constructor.

### 6.1 Multi-fragment greedy

Sort sparse candidate edges by cost/score. Add `(i,j)` if both degrees are below
2, it does not create a premature cycle, and it joins compatible path fragments.
Use disjoint-set state plus fragment endpoints. When candidates cannot complete
the cycle, widen locally or use a global cheapest compatible edge heap. Finish
with LK/2-opt.

### 6.2 Hull-regret insertion

Initialize with the convex hull in cyclic order. For every interior point `v`
and candidate tour edge `(a,b)`, compute

```text
delta(v;a,b) = c(a,v) + c(v,b) - c(a,b).
```

Insert the point with maximum regret (`second_best_delta - best_delta`) at its
best position. Maintain a heap with lazy invalidation and recompute entries
whose chosen edge was changed. Candidate edge positions can be seeded by nearest
tour vertices, but an exact scan fallback is needed when no candidate is valid.

## 7. GA-EAX engine

### 7.1 Population state

Store tours compactly as adjacency pairs and flat order when needed. Maintain:

- objective and edge hash;
- edge-frequency table across population;
- nearest-neighbor diversity or edge entropy;
- age/improvement counters;
- global elite archive distinct from the active population.

Seed with independently optimized constructors and LKH trials, not raw random
permutations.

### 7.2 Crossover pipeline

1. Build the 4-regular multigraph union of parents A and B, marking edge origin.
2. Decompose the symmetric difference into alternating AB-cycles. Common edges
   are inherited directly.
3. Select cycles using both expected gain and diversity; support single-cycle
   EAX and multi-cycle/block variants.
4. Apply exchanges to A, yielding an intermediate set of subtours.
5. Merge subtours through minimal-damage 4-edge reconnections, evaluating
   candidate links first and widening if needed.
6. Run focused 2/3/LK improvement around modified vertices, then optionally a
   full local optimum.
7. Replace a parent only under a quality/diversity rule; periodically restart
   or shift from diversity to convergence.

The repair step is performance-critical. Cache fragment endpoints, build a
nearest-fragment candidate structure, and compare multiple merge orders rather
than greedily locking the first reconnection. Instrument the raw crossover gain,
repair damage, local-search recovery and final child contribution separately.

### 7.3 LKH/EAX cooperation

- feed LKH incumbents into the EAX population;
- add EAX elite edges to LK candidates;
- use EAX when many strong tours disagree on backbone edges;
- use LKH intensification after population convergence;
- share solutions at coarse intervals across processes, never on every move.

## 8. Exact and bounded modes

### 8.1 First deliverable: trustworthy small exact solver

Keep Held–Karp as the oracle for tiny instances. Add an exact branch-and-bound
for medium-small instances with:

- best heuristic incumbent from the quality engine;
- partial path representation with symmetry breaking;
- MST completion and optimized Held–Karp 1-tree bounds;
- nearest candidates first for incumbent discovery but full exact branching;
- forced degree propagation and subtour prevention;
- transposition/dominance cache where state representation allows;
- monotonic deadline checks.

This is valuable for labels and testing even if it never competes with Concorde.

### 8.2 Serious exact path

Do not casually rebuild Concorde in Python. Choose one:

1. integrate Concorde as an external reference solver under its license;
2. build a native branch-and-cut research engine over an LP solver;
3. export the DFJ model with iterative/lazy SECs to a commercial/open MILP
   solver for constrained experimentation.

For path 2, the incremental milestones are:

- degree LP on a sparse priced graph;
- connected-component SEC separator;
- global min-cut separator for fractional solutions;
- reduced-cost pricing against all geometric edges;
- heuristic incumbent callback using sparse LK;
- branching and tree management;
- comb/blossom-style cuts and cut pool;
- exact/rational final verification.

Never claim exactness when pricing only a heuristic candidate set. A lower bound
from a restricted primal edge set is generally not a valid lower bound for the
complete minimization problem unless the dual pricing conditions establish that
no omitted negative-reduced-cost edge exists.

### 8.3 Bounded anytime mode

For users who need an honest quality statement but cannot wait for proof:

```text
parallel:
    quality portfolio improves upper bound U
    1-tree/subtour relaxation improves lower bound L
return tour, U, L, (U-L)/L
```

The gap is only certified if `L` is valid under the same objective convention.
A 1-tree Lagrangian bound is cheap enough to be useful even when full branch-
and-cut is unavailable.

## 9. Huge-instance path

### 9.1 Memory budget

Target O(nk) storage with `k` candidates per node. At `n=10^6`, even a few
64-bit arrays matter. Use 32-bit vertex IDs while `n < 2^32`, structure-of-arrays
candidate storage, and avoid Python objects per node/edge. Keep model embeddings
local/streamed rather than O(n²).

### 9.2 Hierarchical algorithm

```text
order points by Hilbert key
create balanced overlapping leaves of B points
for several shifted partitions:
    derive candidate entry/exit ports for each leaf
    solve leaf path variants with sparse LK or exact DP when tiny
    solve a coarse TSP/path over regions and port choices
    stitch leaf paths
    optimize boundary windows
take best stitched tour
run global sparse LK/2-opt with L1/L2 candidates
periodically select bad/high-regret windows and reoptimize them
```

Typical experimental starting points, not universal constants:

- leaf size `B = 500–5,000` depending memory/deadline;
- 5–20 ports per side/region;
- 5–15% overlap;
- 2–4 shifted partitions;
- global candidates 12–30 per node;
- exact path DP only on very small boundary sets.

Global edge exceptions are essential: retain hull links, cluster bridges,
alpha-near long edges, and elite edges even when they cross distant partitions.

### 9.3 Streaming and updates

For online point additions, first insert by minimum/regret delta near spatial
neighbors, then repair a local window and schedule global background LK. For
deletions, join predecessor/successor and repair. Dynamic optimality is a
different research problem; advertise these as maintained heuristic tours.

## 10. Learning integration

### 10.1 Highest-return first model: candidate ranker

Use a sparse edge set generated independently of the model. Predict either:

- probability an edge belongs to an optimal/near-optimal elite tour;
- ranking within a source vertex;
- expected search utility, measured by accepted moves or final improvement.

Features:

- relative length divided by local k-NN scale;
- source/destination neighbor ranks;
- alpha-nearness and 1-tree membership;
- Delaunay and sector provenance;
- hull flags/depth;
- local density ratio and angular vacancy;
- cluster relation and bridge score;
- current/elite edge frequency;
- node penalties and degree in relaxation;
- embedding score from an E(n)-invariant/equivariant sparse GNN.

Supervision should include **all** optimal edges across tied optima when feasible,
or soft labels from elite frequency. A single canonical optimal tour incorrectly
labels alternative optimal edges negative. Use pairwise/listwise ranking loss or
class-balanced focal/BCE loss. Calibrate probabilities per distribution.

Hard-negative mining should emphasize short plausible edges, alpha-near edges
not in elites, and nonlocal bridges—not uniformly random edges that are trivial
to reject.

### 10.2 Safe deployment

Partition candidate slots:

```text
40% alpha/1-tree
30% geometry (kNN, sector, Delaunay)
15% elite/current tour
15% learned
```

Tune the split; keep non-learned minima. Widen on low confidence, OOD score,
candidate-recall proxy failure, or stagnation. Log what fraction of accepted
moves used learned-only edges and the counterfactual rank under classical scores.

### 10.3 Other useful models

- **Penalty warm start:** predict node potentials, then run classical subgradient
  ascent; never trust the prediction as a bound without optimizing/verifying it.
- **Move ordering:** score promising candidate additions or active vertices;
  search still checks exact gain and feasibility.
- **Kick policy:** contextual bandit chooses perturbation type/strength from
  stagnation and tour structure.
- **Partition/ports:** predict boundary connectors while always including
  geometric fallbacks.
- **Algorithm selector:** allocate time among LKH parameters, EAX, decomposition,
  and exact attempts from cheap instance features. This is low risk and can beat
  a single static configuration.
- **Branch/cut selection:** high upside in exact search but requires careful
  distribution shift handling and must never affect proof validity.

### 10.4 Training corpus

Generate a curriculum spanning:

- uniform square/rectangle with varied aspect ratios;
- clustered mixtures with varied counts, variances and imbalance;
- corridors/strips, grids plus jitter, rings/annuli, curves/manifolds;
- holes, outliers, duplicate/near-duplicate points;
- rotations, translations, reflections, scales and coordinate quantization;
- TSPLIB/DIMACS/real geography;
- tetrahedron and other hard/adversarial constructions;
- sizes both below and far above training `n`.

Labels:

- exact tours and lower bounds for small/medium solvable instances;
- many high-quality LKH/EAX elites for large instances;
- accepted-move and candidate-usage traces;
- partition connector alternatives;
- time-to-target for portfolio learning.

Prevent leakage by splitting at generator seed/family/real-instance identity,
not by permuting or augmenting the same coordinates across train and test.

## 11. Deadline and determinism

Use a monotonic clock and a hierarchical stop token. Each engine checks at
bounded intervals; native subprocesses also receive a hard outer deadline.
Reserve a small handoff margin to canonicalize and emit the last incumbent.

Deterministic mode fixes:

- PRNG algorithm and seed derivation;
- thread count and reduction order;
- tie-breaking on `(score, cost, city ids)`;
- model and native build digests.

Fast mode may use nondeterministic GPU kernels, but record that fact.

## 12. Native implementation plan

Python is suitable for orchestration, data generation, training, analysis and
trusted verification. Put these hot paths in C++/Rust or compiled kernels:

- spatial/candidate construction at scale;
- 1-tree/MST and alpha sensitivity;
- tour sequence structure;
- LK move enumeration and mutation;
- EAX AB-cycle/repair loops;
- exact DP and branch-and-bound;
- huge-instance partition refinement.

A persistent subprocess gives crash isolation and hard timeouts while avoiding
per-instance spawn overhead. Use a versioned binary protocol containing metric,
coordinate dtype, endianness, instance digest, deadline, seed and config. The
native process returns intermediate incumbents through framed messages so a hard
kill does not erase all progress. Python recomputes and verifies the chosen
result.

For external Concorde/LKH/EAX, preserve source attribution and review licenses.
Concorde and LKH downloads are not equivalent to permissive production
dependencies.

## 13. Observability

Emit sampled structured events rather than one event per evaluated move:

```text
candidate_stats: source counts, degree quantiles, elite/optimal recall
one_tree: iteration, bound, step, degree-norm, penalty spread
local_search: move type, depth, gain, candidates examined, active queue
trial: constructor, initial/final length, kick, elapsed, incumbent
eax: AB cycles, selected cycles, subtours, repair damage, child diversity
partition: sizes, ports, local path costs, stitch damage, boundary gain
model: latency, confidence/OOD, learned-slot utilization
exact: LP bound, cuts by family, priced edges, nodes, incumbent, gap
```

The time-quality trace is the primary artifact. A final objective without its
budget trajectory hides the main difference between solvers.

## 14. Staged build plan

### Stage 0 — objective and benchmark integrity

- distance modes and canonical instance/result hashes;
- sparse-capable problem representation;
- independent verifier;
- correct best-known versus certified-optimal semantics;
- diverse benchmark generator and anytime traces.

Exit: the same tour is scored identically by independent code, and TSPLIB
rounding tests pass against published examples/reference values.

### Stage 1 — strong nonlearned baseline

- mixed candidate graph;
- hull-regret and multi-fragment constructors;
- fast candidate 2-opt, 3-opt/Or-opt, double-bridge ILS;
- multiple seeded trials and elite candidate union.

Exit: clear improvement over the current chained 2-opt on every distribution
class at matched time, with no regression caused by dense memory.

### Stage 2 — LKH-class engine

- optimized 1-tree and penalties;
- alpha-nearness;
- segment tour structure;
- correct variable-depth sequential/nonsequential k-opt;
- trials, kicks and tour merging.

Exit: competitive anytime curves against the official LKH executable under
documented configurations; discrepancies are explained by features/config, not
hidden timing.

### Stage 3 — EAX and portfolio

- EAX crossover/repair/population diversity;
- island parallelism;
- LKH/EAX elite exchange;
- portfolio scheduler.

Exit: portfolio improves robustness or best-of-budget quality over each engine
alone on heterogeneous and hard suites.

### Stage 4 — scale

- sparse-only storage path;
- overlapping spatial path decomposition;
- port selection, stitching and boundary reoptimization;
- million-node streaming/telemetry.

Exit: bounded peak memory and competitive quality-time on DIMACS 10k–10m and
Waterloo large instances.

### Stage 5 — learning augmentation

- broad labeled corpus and simple ranker baseline;
- sparse invariant GNN candidate/penalty model;
- safe slot allocation and OOD fallback;
- portfolio/partition learning.

Exit: statistically significant gains over the identical classical engine on
unseen distributions and sizes; learned component wins after charging model
cost and postprocessing.

### Stage 6 — certification

- robust 1-tree bound in normal modes;
- external Concorde integration;
- optional native branch-and-cut research track and exact verifier.

Exit: optimal status cannot be emitted without a checked certificate/matching
bound; bounded mode reports a valid gap.

## 15. Suggested experimental defaults

These are starting configurations to tune, not literature constants:

```text
FAST:
  candidates=12 nearest/sector union
  seeds=2 (multi-fragment, hull-regret)
  candidate 2-opt + shallow 3-opt until deadline

QUALITY:
  candidates=30 mixed, widen to 60
  optimized 1-tree + alpha-nearness
  8–64 independent LK trials by budget
  double-bridge/variable kicks
  elite pool=8–32; EAX when budget and n justify it

HUGE:
  candidates=16–24 sparse
  leaves=500–5000, overlap=10%
  2 shifted hierarchies
  local path LK, boundary windows, global sparse refinement

LEARNED:
  no more than 15–25% learned-only candidate slots initially
  classical fallback always enabled
  inference batch over sparse O(nk) edges
```

Tune by racing/configuration on training instances, then freeze before test.
Do not hand-tune on the reported test set.

## 16. Research ideas with unusually good leverage

1. **Candidate recall under shift.** Optimize not mean edge classification AUC,
   but probability that every critical/optimal bridge edge survives, with
   uncertainty-triggered widening.
2. **Bound-aware learned candidates.** Combine alpha sensitivity, LP reduced
   costs, geometry and elite frequency in a calibrated ranker.
3. **EAX repair as a learned decision with exact deltas.** Predict merge order
   or candidate reconnections; retain exact feasibility and cost computation.
4. **Adaptive partition portals.** Learn a distribution over entry/exit ports,
   then solve a small exact/global connector problem over the retained options.
5. **Anytime portfolio control.** A contextual bandit chooses the next trial,
   engine and configuration from improvement-rate/posterior features.
6. **Hardness-aware curricula.** Generate instances selected for low candidate
   recall, high edge entropy, fractional bound gap, or disagreement between LKH
   and EAX—not just random coordinates.
7. **Exact/heuristic feedback.** Use branch-and-cut fractional edges and reduced
   costs to seed LK/EAX; use elite incumbents to improve pruning and branching.
8. **Safe learned elimination.** Let a model prioritize which expensive exact
   edge-elimination tests to run; only the mathematical test deletes an edge.

## 17. Failure checklist

Before believing an improvement, rule out:

- scoring raw L2 tours against rounded TSPLIB references;
- omitting the closing edge;
- accepting duplicate/missing city IDs;
- integer overflow or float tolerance accepting a worsening move;
- candidate graph disconnectivity or missing reconnection fallback;
- calling a restricted-edge optimum a complete-graph optimum;
- data leakage through augmented copies of a test instance;
- comparing GPU batches with single-thread CPU without end-to-end budgets;
- timing LKH startup one way and model startup another;
- comparing many neural samples with one classical seed;
- using a weak classical parameterization or stopping it before its first trial;
- reporting only uniform random means and hiding per-instance tails;
- counting best-known solutions as certified optima;
- using final gap alone instead of the anytime curve and success probability.
