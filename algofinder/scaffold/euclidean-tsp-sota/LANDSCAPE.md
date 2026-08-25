# The Euclidean TSP landscape

## 1. Problem contract: “Euclidean TSP” is not one numerical problem

Given points `p_0, ..., p_{n-1}` in `R^d`, find a Hamiltonian cycle minimizing

```text
L(pi) = sum_i ||p_{pi_i} - p_{pi_(i+1 mod n)}||_2.
```

The common research problem fixes `d = 2`, but fixed `d > 2`, variable
dimension, and rounded coordinate-derived metrics have materially different
algorithmic behavior.

### 1.1 Distance semantics

Keep these modes distinct in APIs, instance hashes, reference values, and
leaderboards:

- **Raw Euclidean:** IEEE-754 or higher-precision evaluation of square roots.
- **TSPLIB `EUC_2D`:** each edge is `nint(sqrt(dx^2 + dy^2))`, as defined by
  the [TSPLIB95 specification](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf).
- **`CEIL_2D`, `ATT`, `GEO`, `GEOM`:** separate norms/rounding rules.
- **Scaled fixed point:** coordinates or edge lengths converted to integers
  with a documented scale and rounding mode.

For raw rational coordinates, comparing sums of square roots exactly is itself
subtle. The usual theoretical exact algorithms adopt a real-RAM or comparable
exact-comparison model. In production, robust predicates and extended-
precision final comparison are preferable to assuming that float64 equality is
mathematical equality. The [sum-of-square-roots problem](https://arxiv.org/abs/cs/0603002)
is the reason one should avoid the casual claim that unrounded Euclidean TSP's
decision version is simply known to be NP-complete; NP-hardness is established,
but certificate comparison in the bit model has this additional issue.

### 1.2 Outputs and symmetries

A tour is invariant under rotation and reversal. Canonicalize for storage by
rotating the smallest city ID to position zero and choosing the lexicographically
smaller direction. Solvers should return:

```text
tour, objective_under_declared_metric, status,
best_lower_bound?, certified_gap?, wall_time, seed, provenance
```

Duplicate coordinates are legal unless the benchmark forbids them. They create
zero-length edges and multiple optimal tours. Collinearities and cocircularities
create tie cases in hulls, triangulations, and move comparisons; deterministic
tie-breaking is part of reproducibility.

## 2. Geometry that should shape every solver

### 2.1 Noncrossing optimum

In the plane, two properly crossing tour edges can be uncrossed by a 2-opt move
without increasing length, and normally with a strict decrease. Therefore an
optimal tour can be chosen non-self-crossing. This yields:

- an immediate correctness check and cleanup move;
- spatial filters for 2-opt and candidate generation;
- the fact that convex-hull vertices occur in cyclic hull order in an optimum;
- useful decomposition boundaries.

It does **not** mean that optimizing over triangulation edges is automatically
exact. In particular, an optimal tour need not be contained in the Delaunay
triangulation; [Eppstein gives an explicit counterexample](https://ics.uci.edu/~eppstein/junkyard/dt-not-tsp.html).

### 2.2 Edge length and locality

Most useful tour edges in typical planar data are local. This supports k-NN,
quadrant/sector neighbors, Delaunay edges, alpha-nearness, and learned edge
scores. Yet the exceptional nonlocal edges frequently determine global
connectivity and search difficulty. The design rule is:

> use sparse candidates to make search fast, but retain candidate widening,
> bridge candidates, and an occasional exhaustive or proof-safe path.

### 2.3 Scale of random instances

For iid points under regular densities, optimum length grows on the order of
`sqrt(n)` in two dimensions (more generally `n^((d-1)/d)`) after normalizing a
fixed-area domain. This is useful for anomaly detection and normalized learning
targets. It is not a per-instance lower bound and says little about clustered,
manifold, or adversarial input.

### 2.4 Special regimes

- `d = 1`: sorting gives the cyclic optimum immediately.
- All points in convex position: the boundary cycle is optimal after hull
  ordering, modulo degeneracies.
- Few interior points: parameterized algorithms exploit the number of points
  inside the convex hull; see the [few-inner-points study](https://arxiv.org/abs/1406.2154).
- Narrow strips and bounded-width geometry admit stronger dynamic programs;
  see recent work on [Euclidean TSP in a narrow strip](https://doi.org/10.1007/s00454-023-00609-7).
- Fixed dimension admits geometric separators; variable dimension loses much of
  the planar advantage.
- The bitonic-tour dynamic program solves a **restricted** x-monotone tour
  problem. It is not an exact algorithm for unrestricted planar ETSP.

## 3. Complexity and approximation theory

### 3.1 Exact complexity

Planar ETSP is NP-hard. A landmark geometric result by de Berg, Bodlaender,
Kisfaludi-Bak, and Kolay gives an exact fixed-dimensional algorithm running in
`2^{O(n^(1-1/d))}` time, including `2^{O(sqrt(n))}` in the plane, and a matching
ETH lower bound in the exponent. It combines a geometric separator with packing
arguments and rank-based dynamic programming for connectivity. See
[An ETH-Tight Exact Algorithm for Euclidean TSP](https://arxiv.org/abs/1807.06933).

This is the right theoretical SOTA statement. It is not evidence that a current
implementation will beat Concorde on ordinary benchmark sizes: constants,
separator states, exact geometric handling, and connectivity DP are formidable.

### 3.2 Approximation algorithms

Because Euclidean distance is metric, generic metric baselines apply:

- doubled MST plus shortcutting: factor 2;
- Christofides–Serdyukov: factor 3/2, by adding a minimum-weight perfect
  matching on the MST's odd-degree vertices, taking an Euler tour and
  shortcutting repeated vertices;
- metric-TSP improvements slightly below 3/2 exist, but ETSP's stronger
  structure supports a PTAS and is the more relevant asymptotic story.

Arora's geometric PTAS uses a randomly shifted recursive dissection, portals on
cell boundaries, patching to limit crossings, and dynamic programming over
boundary states. It produces a `(1 + epsilon)` approximation for fixed
dimension; see the [primary paper](https://graphics.stanford.edu/courses/cs468-06-winter/Papers/arora-tsp.pdf).
Mitchell independently obtained a PTAS through `m`-guillotine subdivisions:
augment a geometric network to have a recursively splittable structure with
only a `(1 + O(1/m))` length increase, then use dynamic programming; see
[Guillotine Subdivisions](https://doi.org/10.1137/S0097539796309764).

The modern frontier sharpens the dependence on accuracy. A 2025 JACM result by
Kisfaludi-Bak, Nederlof, and Węgrzycki gives, for fixed `d`, a randomized scheme
with leading term `2^{O(1/epsilon^(d-1))} n`, plus lower-order near-linear work,
with essentially matching Gap-ETH dependence; see
[A Gap-ETH-Tight Approximation Scheme for Euclidean TSP](https://doi.org/10.1145/3766548).
A 2024 planar preprint gives a particularly clean randomized
`2^{O(1/epsilon)} n` result; see [Mömke and Zhou](https://arxiv.org/abs/2411.02585).

Why PTASes rarely win practical tour-quality contests:

- their constants and boundary-state complexity can dominate;
- implementation is substantially harder than candidate-restricted local search;
- for a fixed wall-clock budget, LKH/EAX usually target much smaller empirical
  gaps without a worst-case guarantee.

They matter when a formal approximation guarantee is a requirement and as a
source of decomposition/patching ideas.

## 4. Exact practical solution

### 4.1 The hierarchy

| Method | Useful regime | Main weakness |
|---|---|---|
| Brute-force permutations | teaching, tiny `n` | factorial |
| Held–Karp subset DP | small `n`, labels, oracle tests | `Theta(n 2^n)` memory and `Theta(n^2 2^n)` time in the standard form |
| Branch-and-bound with MST/1-tree/assignment bound | small/medium, custom exact research | bound and branching quality dominate; hard families explode |
| Generic MILP/CP-SAT with lazy SECs | integration convenience, constrained variants | usually weaker/slower than specialized TSP machinery |
| Specialized branch-and-cut | practical exact SOTA | sophisticated implementation and LP dependency |
| Geometric separator DP | theoretical exact SOTA in fixed `d` | not currently a mainstream practical solver |

### 4.2 Held–Karp dynamic programming

Fix root 0 and define

```text
DP[S, j] = minimum cost of a path 0 -> ... -> j
           visiting exactly S subset of {1,...,n-1}, j in S.
DP[{j}, j] = c(0,j)
DP[S,j] = min_{i in S\{j}} DP[S\{j},i] + c(i,j)
OPT = min_j DP[all,j] + c(j,0).
```

Implementation choices:

- store only consecutive subset cardinality layers to reduce objective-only
  memory; retaining all parents is required for direct reconstruction;
- use bit masks, combinations grouped by cardinality, contiguous arrays, and
  compiled kernels;
- exploit tour reversal/root symmetry only where it simplifies rather than
  complicates indexing;
- do not construct an `n x n` float matrix if coordinates can be read cheaply,
  although for this small-`n` regime the matrix is generally beneficial;
- a meet-in-the-middle variant or GPU parallelism shifts constants, not the
  exponential wall.

### 4.3 Lower bounds

The lower-bound stack is as important as the search:

1. **MST completion bounds:** cheap but weak. For a partial path, add a minimum
   connection from each endpoint plus an MST over unvisited vertices.
2. **Assignment/cycle-cover relaxation:** degree constraints without global
   connectivity; stronger than a naive bound, but produces subtours.
3. **Minimum 1-tree:** choose a root, compute an MST on the other vertices, and
   add the two cheapest root edges. Every tour is a 1-tree.
4. **Held–Karp/Lagrangian 1-tree bound:** transform edge costs to
   `c'(i,j)=c(i,j)+pi_i+pi_j`, find a minimum 1-tree, and subtract
   `2 sum_i pi_i`. Update node penalties by subgradient ascent according to
   degree violation `deg(i)-2`. This produces both a strong bound and the
   alpha-nearness candidate measure central to LKH. See
   [Held and Karp's original relaxation](https://doi.org/10.1287/opre.18.6.1138).
5. **Subtour LP:** degree equations plus subtour-elimination inequalities. Its
   separation problem finds violated cuts, commonly through connected
   components and minimum cuts.
6. **Stronger TSP cuts:** comb, clique-tree, domino-parity and locally generated
   cuts drive a specialized branch-and-cut engine.

For raw floating distances, a numerical LP bound is not automatically a proof.
Certification needs controlled rounding/rational reconstruction or an exact
LP verifier such as the workflow built around
[QSopt_ex](https://www.math.uwaterloo.ca/~bico/qsopt/).

### 4.4 Branch-and-cut anatomy

The symmetric edge formulation is

```text
minimize  sum_{i<j} c_ij x_ij
subject to
          sum_{j != i} x_ij = 2                 for every i
          sum_{i in S, j not in S} x_ij >= 2    for every nontrivial S
          x_ij in {0,1}.
```

A competitive engine repeatedly:

1. solves the current LP;
2. separates violated subtour cuts, then stronger cut families;
3. prices missing edges when using a restricted edge set;
4. runs primal heuristics on fractional/incumbent structure;
5. fixes or eliminates edges when reduced costs or valid geometric arguments
   permit;
6. branches on edges or derived structures using strong/pseudocost branching;
7. manages a best-bound tree, warm starts and cut pools;
8. verifies the final bound exactly.

[Concorde](https://www.math.uwaterloo.ca/tsp/concorde.html) is still the
reference implementation. Its source organization exposes modules for cuts,
edge generation, Held–Karp/1-trees, kd-trees, linkern local search, LP, TSP tree
management and verification. It has proved all 110 TSPLIB instances and the
85,900-point `pla85900` tour optimal. This achievement is a stack, not a single
cutting-plane trick.

### 4.5 Edge elimination

Exact elimination proves an edge cannot occur in any optimal tour, unlike
heuristic candidate pruning. It can dramatically shrink the graph before or
during branch-and-cut. Hougardy and Schroeder describe an `O(n^2 log n)` exact
test and report substantial Concorde speedups; see
[Edge Elimination in TSP](https://arxiv.org/abs/1402.7301). More recent local
elimination/fixing work targets instances from thousands to over one hundred
thousand vertices; see the [2024 Mathematical Programming Computation paper](https://doi.org/10.1007/s12532-024-00262-y).

For random planar inputs with bounded density, analysis shows the
Hougardy–Schroeder test leaves only a linear expected number of edges while a
classic nonrecursive test can leave quadratic many; see
[Zhong's probabilistic analysis](https://arxiv.org/abs/1809.10469). This helps
explain why geometric preprocessing can be exceptional on benign instances but
does not remove worst-case hardness.

### 4.6 Difficulty is structural

The tetrahedron family of Hougardy and Zhong contains only a few hundred
Euclidean points yet can require days for Concorde. See
[Hard to Solve Instances of the Euclidean TSP](https://doi.org/10.1007/s12532-020-00184-5).
Benchmarks consisting only of uniform random points badly under-sample exact
difficulty. Instance features that matter include near-degenerate alternative
edges, fractional LP structure, backbone ambiguity, cluster bridges, hull/interior
organization, and symmetry.

## 5. Constructive heuristics

Constructors matter as seeds, not usually as final solvers.

| Constructor | Character | Best use |
|---|---|---|
| nearest neighbor, multi-start NN | very fast, myopic | diverse cheap seeds |
| cheapest/nearest/farthest insertion | incremental and easy to maintain | robust initialization |
| maximum-regret insertion | protects difficult-to-insert vertices | clustered/heterogeneous seeds |
| savings | route-merging perspective | seed diversity |
| sweep/space-filling order | geometry-linearithmic | enormous inputs, partitions |
| MST preorder / Christofides | guaranteed metric baseline | sanity and quality floor |
| greedy multi-fragment | builds short disjoint paths while preventing premature cycles | strong fast candidate-aware seed |

For insertion, maintain each unvisited point's best and second-best insertion
deltas. Regret is their difference. Use spatial candidate positions first, but
retain a repair path that scans all tour edges if candidates fail. Diversify by
randomized tie-breaking, restricted candidate lists, hull-first versus fragment
starts, and penalties on overused edges.

## 6. Local and variable-depth search

### 6.1 Move families

- **2-opt:** remove two edges and reverse a segment. It removes all proper
  crossings and is the indispensable minimum baseline.
- **Or-opt / relocate:** move one or a short chain of consecutive vertices.
- **swap:** useful in a general move engine, though often subsumed by k-opt.
- **3-opt:** reconnect three deleted edges through multiple patterns.
- **double bridge / 4-opt kick:** nonsequential perturbation that escapes a
  2/3-opt basin without destroying the whole tour.
- **k-opt:** exchange `k` edges; full enumeration is prohibitive, so candidate
  restriction and gain pruning are essential.
- **ejection chains / stem-and-cycle:** explore compound moves through an
  intermediate infeasible or structured representation.

Maintain `next`, `prev`, and a segment structure; computing a 2-opt delta should
be O(1). Reversing a long segment naively makes each accepted move O(n). LKH-
style implementations use specialized two-level/three-level trees or similar
sequence representations to make reversals and adjacency queries cheap.

### 6.2 Candidate sets

A robust union is stronger than any single source:

```text
C(i) = alpha-near edges
     U k nearest neighbors
     U Delaunay / quadrant-sector neighbors
     U edges in elite tours
     U learned high-score edges
     U mandatory bridge/widening edges.
```

Deduplicate and cap by a score that mixes alpha value, scaled distance,
geometric coverage, elite frequency and learned logit. Candidate recall of
known optimum/elite edges is a first-class metric. Directional sectors prevent
all candidates being consumed by one dense cluster. Candidate widening is
triggered after stagnation or an unsuccessful feasibility/reconnection search.

Alpha-nearness is stronger than pure distance. For a minimum 1-tree, it measures
the increase required to force an edge into that structure (implemented with
appropriate sensitivity calculations). Edges with small alpha are structurally
near the relaxation, even if they are not the nearest geometrically.

### 6.3 Lin–Kernighan–Helsgaun

Lin–Kernighan searches variable-depth alternating sequences of deleted and
added edges, choosing depth adaptively and closing when cumulative gain is
positive. Helsgaun's implementation made it a practical standard through:

- alpha-nearness candidates based on an optimized minimum 1-tree;
- node penalties found by subgradient ascent;
- sequential and nonsequential k-opt moves, with a strong 5-opt base;
- positive-gain and feasibility pruning;
- don't-look/active-node management;
- candidate restoration and restricted backtracking;
- double-bridge and stronger kicks between trials;
- multiple starts, elite tours, tour merging and partitioning.

The foundational implementation paper is
[An Effective Implementation of the Lin–Kernighan TSP Heuristic](https://doi.org/10.1016/S0377-2217(99)00284-2).
The author's [LKH page](https://webhotel4.ruc.dk/~keld/research/LKH/) lists
LKH 2.0.11 (June 2025) and its C source. Treat the source's noncommercial/
research licensing terms as a dependency constraint; do not silently vendor it
into a differently licensed product.

A 2026 Computers & Operations Research paper carefully relaxes LKH's usual
positive-gain condition—allowing a restricted temporary nonpositive prefix in
an alternating exchange—and reports essentially unchanged tour quality with an
average 13.6% time reduction on large instances over the compared LKH setup.
This is a particularly relevant current classical frontier because it improves
the dominant search kernel rather than replacing it; see
[A Speed-up for Helsgaun's TSP Heuristic](https://doi.org/10.1016/j.cor.2026.107443).

LKH is an anytime system. Report the time to first tour, best-so-far trajectory,
trial count, and time to a target gap. One run/seed is not representative.

### 6.4 What theory says about local optima

Excellent empirical behavior is not a constant worst-case guarantee. For every
fixed `k >= 2`, the worst-case approximation ratio of k-opt local optima for
two-dimensional ETSP is `Theta(log n / log log n)`; see
[Brodowsky, Hougardy, and Zhong](https://doi.org/10.1137/21M146199X). Variable-
depth LK, restarts and recombination are precisely attempts to avoid being
trapped in one fixed-k local optimum, but do not convert the practical solver
into a PTAS.

Smoothed analyses explain part of the gap between hostile worst cases and normal
performance: after Gaussian perturbation, expected 2-opt path length/running
time and approximation behavior admit polynomial bounds under specified noise
models. Two current references are
[Improved Smoothed Analysis of 2-Opt](https://doi.org/10.1007/s00453-025-01309-9)
and [Smoothed Analysis under Gaussian Noise](https://doi.org/10.1007/s00453-025-01335-7).
These are distributional analyses, not a claim that every implementation or
unperturbed instance terminates quickly.

### 6.5 Iterated local search and POPMUSIC

The most effective ILS pattern is simple:

```text
construct -> local optimum -> kick -> local optimum -> accept/update elite -> repeat
```

The engineering sophistication lives in candidate sets, move engine, kick
strength, edge-frequency memory, and tour recombination. POPMUSIC repeatedly
optimizes overlapping subpaths or spatial subproblems, and can also generate
high-quality candidates. The [POPMUSIC TSP paper](https://doi.org/10.1016/j.ejor.2018.06.039)
shows why this local-subproblem view is effective on very large instances.

## 7. Population methods and metaheuristics

### 7.1 GA-EAX

Edge Assembly Crossover operates on two parent tours:

1. superimpose parent A and B;
2. decompose their symmetric difference into alternating AB-cycles;
3. select one or more cycles and exchange the corresponding edge sets;
4. obtain one or more subtours;
5. reconnect/repair subtours with minimal damage;
6. locally optimize and apply diversity-aware replacement.

This respects the fact that good tours are better represented by their edges
than by permutation positions. Variants differ in AB-cycle selection, entropy/
diversity management, repair, two-stage population schedules, and localized
application. Sources include the
[original EAX paper](https://www.jstage.jst.go.jp/article/jjsai/14/5/14_848/_article/-char/en),
[localized EAX](https://www.jstage.jst.go.jp/article/tjsai/22/5/22_5_542/_article/),
and Nagata's [GA-EAX implementation](https://github.com/nagata-yuichi/GA-EAX).
A 2025 preprint re-examines repair and AB-cycle choices over 10,000 instances;
see [To Repair or Not to Repair?](https://arxiv.org/abs/2505.00803).

EAX and LKH have complementary failure modes. A best-of-both portfolio and an
elite pool shared through tours/edge frequencies is a stronger engineering
starting point than declaring one universally superior. The 2025 World TSP
record used precisely this complementarity: GA-EAX continued from an LKH tour.

### 7.2 Other metaheuristics

Simulated annealing, tabu search, guided local search, GRASP, ant colony
optimization, genetic algorithms, particle-swarm-inspired methods, and variable
neighborhood search can all solve ETSP. Their role should be judged by what
they add around the move engine:

- Does the method discover candidates or long-range bridge edges?
- Does it manage diversity more effectively than independent ILS trials?
- Does it yield a better anytime curve after equalizing all local search?
- Does it adapt across distribution shifts?

Without strong k-opt/LK/EAX-quality exploitation, generic nature-inspired
methods are normally not competitive with the practical SOTA. “New metaphor +
2-opt” should be compared against “same 2-opt + a simple restart/ILS policy” to
isolate the claimed innovation.

## 8. Scaling beyond dense memory

An `n x n` float64 matrix costs `8n^2` bytes: about 0.8 GB at 10,000 points,
80 GB at 100,000, and is impossible at million scale. A large solver stores
coordinates, sparse candidates, the tour, spatial indices, and compact local
state.

### 8.1 Spatial infrastructure

- kd-tree for nearest/range queries in low dimension;
- Delaunay triangulation where robust libraries and memory allow;
- grids/geohashes for uniform-ish planar data;
- space-filling-curve order (Hilbert/Morton) for partitions and cache locality;
- bounding boxes and segment spatial indexes for crossing detection;
- on-demand distance evaluation, optionally caching only candidate edges.

Squared distance can rank neighbors, but move gains require the declared actual
edge cost. Vectorize batches; avoid `sqrt` only when comparisons remain valid.

### 8.2 Partition, solve, stitch, refine

A scalable hierarchy:

1. normalize only for geometry/model features, never silently for objective;
2. create balanced spatial cells with overlaps/halos;
3. choose boundary ports and solve each cell as a Hamiltonian **path** subproblem
   with entry/exit conditions, not blindly as a closed tour;
4. solve a coarse region-order/connector problem;
5. stitch paths and repair degree/connectivity;
6. optimize wider boundary windows;
7. run sparse global LK-like refinement;
8. occasionally repartition with shifted boundaries.

Closed local cycles are a common decomposition mistake: breaking each cycle
later can lose most of the local quality. Multiple candidate ports and overlap
reduce boundary myopia. Maintain a global exception-edge budget for nonlocal
bridges.

### 8.3 Parallelism

The best low-risk parallelism is portfolio parallelism: independent LKH trials,
EAX islands, shifted partitions, or alternative candidate policies. Share only
improved incumbents, elite tours, and edge frequencies at coarse intervals.
Within-run move evaluation can be parallelized, but fine-grained synchronization
often harms a branchy, memory-sensitive LK search. GPUs fit batched distance,
candidate scoring, neural inference and massively parallel construction better
than irregular tour mutation.

For an extreme low-memory/low-quality operating point, Taillard's randomized
POPMUSIC-based linearithmic heuristic was tested beyond two billion cities,
with roughly 10% deviation on the special huge toroidal random instances; see
[A linearithmic heuristic for the TSP](https://doi.org/10.1016/j.ejor.2021.05.034).
A separate 2024 pair-center heuristic reports empirical `O(n log n)` time and
linear space, with average gaps of 0.94% below 1,001 points and 4.57% on its
larger benchmark collection; see
[Quasi-linear time heuristic for ETSP with low gap](https://doi.org/10.1016/j.jocs.2024.102424).
These occupy a throughput/memory frontier, not the best-tour frontier.

## 9. Neural and learning-augmented optimization

### 9.1 Taxonomy

| Family | Output | Strength | Main risk |
|---|---|---|---|
| autoregressive constructor | next-node distribution | simple feasible decoding | O(n²) attention/decoding, exposure bias |
| non-autoregressive heatmap | edge probabilities | parallel and useful for candidates | heatmap-to-tour search does much of the work |
| learned improvement | move/action policy | works on complete tours | long horizons, size shift |
| diffusion/consistency | distribution over edge/tour structures | diversity and test-time refinement | expensive decoding, feasibility projection |
| hierarchical/decomposition | partitions and local paths | large-instance scaling | irreversible boundary errors |
| learned classical guidance | candidates, penalties, heuristics, branching | keeps mature search and fallback | training labels/cost, coupling |
| algorithm/portfolio selector | solver/config/budget allocation | low integration risk | needs broad instance corpus |

### 9.2 Construction lineage

Pointer Networks introduced sequence-to-sequence attention for combinatorial
outputs; REINFORCE-based neural combinatorial optimization then trained tour
constructors without exact labels. The attention model of Kool, van Hoof and
Welling became a standard baseline; see
[Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475).

POMO exploits multiple symmetric starting points and augmentation, obtaining a
reported 0.14% gap on its 100-node random-Euclidean protocol; see the
[NeurIPS 2020 paper](https://proceedings.neurips.cc/paper/2020/hash/f231f2107df69eab0a3862d50018a9b2-Abstract.html).
EAS adapts a small part of a pretrained model per test instance
([Efficient Active Search](https://arxiv.org/abs/2106.05126)); simulation-guided
beam search combines learned probabilities and rollouts
([SGBS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/39b9b60f0d149eabd1fff2d7c5afc4-Abstract-Conference.html)).

LEHD deliberately uses a light encoder and heavy decoder to improve size
generalization, with supervised subpath reconstruction and repeated route
reconstruction; see [LEHD](https://arxiv.org/abs/2310.07985). Such systems can
be useful rapid constructors, but gaps typically grow on heterogeneous TSPLIB
instances compared with matched uniform-random tests.

### 9.3 Learned improvement and heatmap search

Learning 2-opt policies can prioritize moves or regions; see
[Learning 2-opt Heuristics for the TSP](https://proceedings.mlr.press/v129/costa20a.html).
DPDP uses a learned edge policy to restrict and guide a dynamic-programming
beam; it is an insightful hybrid, but pruning removes exactness unless the
discarded states/edges are certified safe. See
[DPDP](https://openreview.net/forum?id=OAMrSPRRxJx) and its
[implementation](https://github.com/wouterkool/dpdp).

A crucial negative result for interpretation is the ICML 2024 position paper
[Rethinking the Interface Between Neural and Classical Combinatorial Optimization](https://proceedings.mlr.press/v235/xia24f.html):
simple heatmaps can match or beat more complicated learned heatmaps when the
same post-hoc search is used, while the full heatmap-plus-search system still
trails LKH-3 in the studied settings. Always ablate the learned score against
distance, alpha-nearness, and simple frequency/geometry scores under the same
decoder.

### 9.4 Learning inside mature search

This is the most convincing direction for a best solver.

- **NeuroLKH** uses a sparse GNN to predict candidate-edge scores and node
  penalties, then lets LKH perform feasibility-preserving variable-depth
  search. It reports generalization to much larger problems than training;
  see the [NeurIPS 2021 paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/3d863b367aa379f71c7afc0c9cdca41d-Abstract.html).
- **DeepACO** learns the heuristic measure used by ant construction and couples
  it to local search; see the [NeurIPS 2023 paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/883105b282fe15275991b411e6b200c5-Abstract-Conference.html)
  and [code](https://github.com/henry-yeh/DeepACO).
- **Graph Convolutional Branch and Bound** augments exact 1-tree and Concorde
  search with an unsupervised GNN-derived optimality score for prioritizing tree
  decisions, reporting fewer explored nodes and lower compute time; see the
  [2026 EJOR paper](https://doi.org/10.1016/j.ejor.2026.03.036). Because the
  model changes search order rather than pruning validity, exactness can remain
  with the classical solver.
- Reinforced LKH variants learn candidate/backbone or exploration choices. The
  scientific test is improvement over the *same* LKH build, parameterization,
  seeds, and time budget—not over a weakly configured executable.

High-value prediction targets are calibrated edge recall, candidate ordering,
node penalties, kick strength, partitions/ports, move ordering, branch choice,
and per-instance algorithm configuration. Preserve all classical candidates and
allocate a fixed fraction of slots to learned edges. If the model is absent,
uncertain, or out of distribution, performance should degrade gracefully to the
classical engine.

### 9.5 Diffusion, consistency, and frontier models

DIFUSCO applies diffusion to edge adjacency and reported gaps of roughly 0.46%,
1.17%, and 2.58% on its 500-, 1,000-, and 10,000-node protocols; see
[NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/0ba520d93c3df592c83a611961314c98-Abstract-Conference.html).
Fast T2T uses consistency models plus test-time gradient search
([NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/352b13f01566ae34affacc60e98c16af-Abstract-Conference.html)).
StruDiCO instead uses variable-absorption structured denoising plus gradient-free,
objective-aware refinement. On its uniform-random TSP500/TSP1000 protocol it
reports 0.168%/0.261% gaps with four-sample decoding, guided refinement and
2-opt, while LKH reaches approximately zero at a larger reported latency; see
[NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/file/6728fcf94660c59c938319a6833a6073-Paper-Conference.pdf).
Later work attempts
structure-aware distributions and cheaper feasibility projection.

As of the research cutoff, the following are useful frontier signals, not settled
baselines:

- [PCI](https://arxiv.org/abs/2606.09343), a June 2026 preprint, replaces costly
  gradient refinement with a structural projection/valid-tour decoder and
  2-opt, reporting 0.17%/0.31% gaps on its TSP500/TSP1000 tests and a rapid-
  generation advantage over LKH-3.
- [GeoRouteNet](https://arxiv.org/abs/2606.22776), also a June 2026 preprint,
  adds geometric structure to non-autoregressive inference and reports a 3.60%
  aggregate gap on 27 stratified TSPLIB `EUC_2D` instances—an improvement over
  its neural baseline, but not high-quality classical tour SOTA.
- [GES-TSP](https://arxiv.org/abs/2607.09708) learns aggressive graph
  sparsification with small reported gaps. Such sparsification is approximate
  unless paired with exact elimination or pricing.
- A July 2026 preprint studies GNN loss functions for selecting among Chained
  LK, EAX, LKH, MAOS and Concorde under distinct budgets; see
  [GNN-based TSP solver selection](https://arxiv.org/abs/2607.18632). The result
  is too new to treat as established, but budget-aware portfolio prediction is
  a lower-risk use of learning than replacing the component solvers.

Claims that a neural method “beats LKH” usually describe a particular short
budget, hardware, instance distribution, and LKH configuration. They do not
imply domination at long budgets, on TSPLIB/adversarial distributions, or in
certified quality.

### 9.6 Generalization failure modes

Neural policies can fail under changes in:

- size and density;
- uniform versus clustered/mixture/manifold point processes;
- aspect ratio, scale, rotation, translation, or coordinate quantization;
- number and geometry of hull points;
- outliers, holes, barriers implicit in the point pattern, and bridge structure;
- distance convention;
- decoding/search budget.

The peer-reviewed study
[Generalization in Neural Combinatorial Optimization](https://doi.org/10.1007/s10601-022-09327-y)
documents brittleness masked by same-distribution tests. Architectures should
use invariant/equivariant features, relative scales, sparse neighborhoods, and
multi-distribution training, but only cross-family evaluation establishes the
result.

## 10. Hierarchical learned solvers

GLOP partitions a large TSP and solves local shortest-Hamiltonian-path tasks;
see [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/30009) and
[code](https://github.com/henry-yeh/GLOP). H-TSP uses hierarchical reinforcement
learning up to 10,000-node experiments; see
[H-TSP](https://arxiv.org/abs/2304.09395). GELD uses a global encoder/local
decoder with region-averaged linear attention and claims scaling to 744,710
nodes, while acknowledging quality below LKH in key comparisons; see the
[2025 preprint](https://arxiv.org/abs/2506.06634).

Their most reusable ideas are not necessarily their complete policies:
learned boundary ports, global context for local decisions, path rather than
cycle subproblems, overlapping regions, and repeated cross-boundary refinement.
These can be combined with exact small path DP, LK, or EAX inside each region.

## 11. Alternative formulations

- **MILP:** DFJ with lazy subtour cuts is the appropriate basic formulation.
  MTZ is compact but its relaxation is weak and it is rarely the choice for a
  serious pure TSP solver. Modern commercial solvers are valuable for custom
  side constraints and as controlled baselines.
- **Constraint programming / CP-SAT:** a circuit constraint plus propagation is
  convenient, especially when ETSP is embedded in scheduling or routing
  constraints. Pure ETSP generally favors specialized algorithms.
- **SAT/SMT/answer-set encodings:** useful for solver research and hybrid
  constraints; a 2026 survey/encoding paper is
  [New Encodings of ETSP in Constraint ASP](https://doi.org/10.1093/logcom/exaf072).
- **QUBO/Ising/quantum annealing:** one-hot position encodings use `Theta(n^2)`
  binary variables before embedding and require large penalty scales. Current
  hardware and hybrid demonstrations do not compete with LKH/EAX/Concorde on
  meaningful unrestricted ETSP quality-scale-time tradeoffs. Treat them as
  experimental platforms, not production SOTA.
- **Differentiable relaxations and optimal transport:** useful for learning
  representations or soft assignments, but rounding plus connectivity repair is
  still a combinatorial solver.

## 12. What “current SOTA” means in 2026

| Regime | Defensible leader/reference | Confidence |
|---|---|---|
| exact practical symmetric TSP/ETSP | Concorde-style specialized branch-and-cut | high |
| exact fixed-dimensional asymptotics | geometric separator + rank-based DP, `2^{Theta(n^(1-1/d))}` under ETH | high, theoretical |
| worst-case approximation | Euclidean PTAS; near-linear Gap-ETH-tight modern schemes | high, mainly theoretical |
| top-quality heuristic tours | LKH family and GA-EAX family, often in portfolio | high |
| massive-instance heuristic | sparse candidates, POPMUSIC/LKH partitioning, EAX/LK portfolio, hierarchical refinement | high as a design family |
| fixed-latency GPU/batch | POMO/attention/heatmap/diffusion families depending budget | benchmark-dependent |
| robust ML contribution | learned guidance inside classical search/decomposition | medium-high directionally |
| end-to-end neural domination over classical search | not established across broad ETSP | high confidence in this caution |

The central synthesis is that the best system is layered. Geometry proposes a
small but diverse action set; relaxations quantify global structure; local and
population search exploit it; exact machinery supplies labels, bounds, and
certificates; decomposition controls scale; learning allocates scarce search
effort. Each layer must remain measurable and replaceable.
