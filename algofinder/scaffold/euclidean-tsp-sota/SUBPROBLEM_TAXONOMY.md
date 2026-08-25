# A taxonomy of Euclidean TSP subproblems

Research cutoff: **2026-08-02**. This document concerns the canonical symmetric
Euclidean traveling salesperson problem (ETSP): a finite point set, Euclidean
edge lengths, and a minimum-length Hamiltonian cycle. It distinguishes genuine
ETSP regimes from nearby routing variants that change the problem definition.

## 1. The right abstraction is a lattice, not a tree

There is no single mathematically natural division of ETSP. The useful regimes
cross-cut one another: an instance can be three-dimensional, have low intrinsic
dimension, contain a few inner points, be highly clustered, use rounded edge
costs, and be evaluated in an exact-proof regime simultaneously. Treating each
named family as a disjoint class discards exactly the interactions a solver or
benchmark should expose.

Use a faceted signature instead:

```text
ETSP(
  ambient_dimension,
  intrinsic_dimension,
  distance_semantics,
  hull_size / inner_point_count / onion_depth,
  strip_or_manifold_width,
  density_and_spread,
  cluster_separation_and_bridge_entropy,
  degeneracy,
  distribution_class,
  scale,
  exactness_requirement
)
```

Some coordinates are exact parameters; others are empirical descriptors. The
signature is useful even when fields are unknown. It is also the correct bridge
from mathematical taxonomy to benchmark metadata.

## 2. Ambient and intrinsic dimension

Let `n` be the number of points and `d` the ambient Euclidean dimension.

### 2.1 One dimension

For points on a line, sorting solves the problem. If the smallest and largest
coordinates are `xmin` and `xmax`, an optimal closed tour has length
`2(xmax - xmin)` under raw Euclidean length. Duplicates create zero-cost edges
but do not alter the value. This is a valuable parser and correctness stratum,
not a search challenge.

### 2.2 The plane

Planar ETSP is the central practical regime. Every optimal tour can be chosen
without proper crossings, convex-hull points occur in their cyclic hull order,
planar separators become available, and Delaunay/nearest-neighbor geometry is
algorithmically useful. None of these facts says that a fixed heuristic sparse
graph contains an optimum; Delaunay containment, for example, is false in
general.

### 2.3 Fixed `d >= 3`

The Euclidean structure remains powerful but planar visualization and several
two-dimensional tools disappear. For every fixed `d`, ETSP has a PTAS. The
exact-complexity frontier is
`2^{Theta(n^(1-1/d))}` up to constants in the exponent under ETH, according to
the separator/rank-based algorithm and matching lower bound of
[de Berg et al.](https://arxiv.org/abs/1807.06933). Benchmarking only `d=2`
therefore misses a theoretically and practically distinct scaling axis.

### 2.4 Growing or high dimension

When dimension grows with the input, fixed-dimensional geometric guarantees no
longer transfer uniformly. Trevisan's
[high-dimensional hardness result](https://doi.org/10.1137/S0097539799352735)
shows that Euclidean TSP becomes Max-SNP-hard to approximate when dimension is
allowed to grow (the construction uses logarithmic dimension). High-dimensional
nearest-neighbor search, distance concentration, and candidate-graph quality
also become separate practical problems.

### 2.5 Low intrinsic dimension

Points embedded in a large `d` may lie near a curve, surface, low-rank affine
subspace, or a metric of small doubling dimension. Ambient and intrinsic
dimension must therefore be separate fields. Approximation schemes in doubling
metrics, such as the work of
[Bartal, Gottlieb, and Krauthgamer](https://arxiv.org/abs/1112.0699),
motivate tests in which ambient dimension changes but intrinsic dimension and
noise are controlled.

## 3. Hull, layers, and topological organization

### 3.1 Convex position

If every point is a vertex of the convex hull, the hull boundary is an optimal
tour. This is a polynomially trivial but important endpoint. It tests whether a
solver preserves obvious geometric structure and establishes a controlled
continuation from easy convex instances to hard interior-rich ones.

### 3.2 Few inner points

Let `k` be the number of points strictly inside the convex hull. ETSP is fixed-
parameter tractable in this parameter. Gawrychowski and Rusak give a linear-
space algorithm with running time
`O(n k^2 + k^{O(sqrt(k))})` and a quadratic bikernel
([paper](https://arxiv.org/abs/1406.2154)). Thus `n` and `k` define a meaningful
two-axis regime: a very large instance may be structurally simple when `k` is
small.

### 3.3 Onion depth and nested geometry

Repeatedly removing the convex hull yields convex layers. Hull fraction,
interior count, and onion depth capture different structure: two instances may
have the same number of interior points but arrange them as one dense core,
many nested rings, or scattered outliers. Onion depth is not by itself a known
universal complexity parameter, but it is an excellent explanatory benchmark
axis for candidate generation, crossings, and local-search basins.

### 3.4 Holes, outliers, and cluster topology

Other useful descriptors are the number and scale of empty regions, isolated
outliers, components of a multiscale proximity graph, and the topology of a
cluster adjacency graph. These are not separate problem definitions; they are
structural ETSP regimes. They control which long bridge edges must be selected
and how easily a decomposition method can repair boundary choices.

## 4. Width, lines, curves, and manifolds

### 4.1 Narrow strips and hypercylinders

For points in an infinite strip of width `delta`, complexity can be
parameterized by width. For sparse point sets, Alkema et al. give runtime
`2^{O(sqrt(delta))} n + O(delta^2 n^2)`; for an explicit random model the
expected runtime is `2^{O(sqrt(delta))} n`. With distinct integer
`x`-coordinates, a shortest bitonic tour is globally optimal when
`delta <= 2 sqrt(2)`, a tight threshold. Their results extend to
`d`-dimensional hypercylinders with an exponential factor
`2^{O(delta^(1-1/d))}`
([paper](https://arxiv.org/abs/2003.09948)).

This suggests a particularly clean benchmark continuation: line -> thin strip
-> wide strip -> unrestricted plane, crossing a proved structural threshold.

### 4.2 A fixed number of parallel lines

When all points lie on `N` parallel lines, Rote's dynamic program takes `n^N`
time for fixed `N`; a controlled relaxation to certain almost-parallel segments
is also possible
([paper and corrected report](https://page.mi.fu-berlin.de/rote/Papers/abstract/The%2BN-line%2Btraveling%2Bsalesman%2Bproblem)).
This differs from a narrow-strip parameter: many points may cover a very wide
region while still occupying only a small number of lines.

### 4.3 Axes and other special supports

Points constrained to the coordinate axes admit specialized polynomial
structure; see
[Cambazard and Catusse](https://www.sciencedirect.com/science/article/pii/S0377221712004912).
Older work also studies a convex hull plus a fixed number of lines
([reference](https://www.sciencedirect.com/science/article/pii/0020019096001251)).
More broadly, points on circles, several rings, polylines, smooth curves,
surfaces, or noisy manifolds create low-dimensional ordering structure without
necessarily satisfying the assumptions of those exact algorithms.

## 5. Distributional regimes

### 5.1 Random Euclidean instances

The canonical random model samples IID points from a density `f` in a bounded
region. The Beardwood-Halton-Hammersley theorem gives the large-sample law

```text
OPT_n / n^((d-1)/d) -> beta_d * integral f(x)^((d-1)/d) dx
```

under the theorem's conditions; Steele's
[treatment](https://epubs.siam.org/doi/10.1137/1.9781611970029.ch2) explains the
subadditive Euclidean-functional framework. Uniform-square RUE is only one
point in this space. Nonuniform smooth densities, mixtures, anisotropy,
heavy-tailed cluster sizes, boundary effects, and stochastic point processes
such as Poisson-disc sampling create meaningfully different geometric graphs.

The asymptotic law is a generator-validation and normalization tool, not a
per-instance lower bound or an excuse to label an approximate tour optimal.

### 5.2 Smoothed instances

Smoothed analysis starts from an arbitrary configuration and perturbs it using
an explicit noise model. It interpolates between adversarial and random
instances and asks which bad behavior survives realistic imprecision. The 2025
smoothed analyses of 2-opt under Gaussian perturbations sharpen runtime and
approximation results
([Manthey and van Rhijn](https://doi.org/10.1007/s00453-025-01309-9),
[Künnemann et al.](https://doi.org/10.1007/s00453-025-01335-7)). They make
perturbation scale a theoretically motivated benchmark factor.

### 5.3 Clustered and hierarchical instances

“Clustered” is too vague to be a reproducible class. Record at least cluster
count, within-cluster diameter or covariance, between-cluster separation,
size imbalance, hierarchy depth, outlier rate, and density contrast. For solver
research also estimate **bridge entropy**: whether only a few geometrically
plausible inter-cluster ports exist or many near-equivalent connections compete.
Low bridge entropy favors decomposition; high bridge entropy stresses stitching
and global candidate widening.

### 5.4 Adversarial and solver-discriminating instances

Adversarial regimes include mathematical constructions, evolutionary mutation,
and instances selected to maximize a performance difference between solvers.
Hougardy and Zhong's tetrahedron family has subtour-LP integrality ratio tending
to `4/3` and makes Concorde more than a million times slower than similarly
sized TSPLIB examples in their comparison
([paper](https://doi.org/10.1007/s12532-020-00184-5)). It proves that point count
is not a sufficient hardness coordinate.

Evolved instances are useful for finding algorithmic blind spots, but an
instance evolved against solver A is evidence about A and its neighborhood in
configuration space—not a solver-independent definition of ETSP hardness.

## 6. Density, spread, and degeneracy

### 6.1 Density and spread

Bounded-density point sets, minimum separation, coordinate spread, and packing
properties matter for both geometric data structures and candidate elimination.
For example, one exact geometric elimination test leaves only linearly many
edges in expectation under a bounded-density random planar model
([Zhong](https://arxiv.org/abs/1809.10469)). Benchmarks should separately vary
global coordinate spread, local crowding, and the ratio of largest to smallest
nonzero distance.

### 6.2 Degenerate geometry

Duplicates, near-duplicates, collinear runs, cocircular sets, lattice points,
repeated distances, nearly crossing alternatives, and extreme aspect ratios
are first-class subregimes. They stress tie breaking, orientation predicates,
zero-cost edges, triangulations, normalization, and numerical repeatability.
They should not be silently removed unless the benchmark definition forbids
them.

## 7. Numerical and objective semantics

The following are different optimization problems even on identical
coordinates:

- raw real-valued Euclidean sums;
- IEEE floating evaluation under a declared precision/order;
- TSPLIB `EUC_2D`/`EUC_3D` per-edge rounding;
- `CEIL_2D` and special TSPLIB functions;
- fixed-point or scaled-integer approximations.

Coordinate bit length matters for exact arithmetic, and comparison of sums of
square roots is itself subtle
([Qian and Wang](https://arxiv.org/abs/cs/0603002)). A benchmark must name the
distance oracle and objective convention, not simply say “Euclidean.” `GEO` and
the World TSP's `GEOM` are valuable geometric TSP cases but are not planar raw
ETSP.

## 8. Scale and required output

Scale changes the feasible methodology even when the mathematical problem is
unchanged:

- micro instances support exhaustive and exact cross-checks;
- small/medium instances support certified optima and rich ablations;
- large instances emphasize anytime upper bounds and candidate quality;
- huge instances emphasize streaming, linear memory, decomposition, and
  throughput.

The required output also defines a subproblem of the *solver task*:

- exact tour plus proof/certified bound;
- `(1+epsilon)`-guaranteed tour;
- best upper bound by a deadline;
- first feasible tour at minimal latency;
- candidate edges, heatmaps, or decompositions for another engine;
- a portfolio decision rather than a tour.

These output regimes must not share a single undifferentiated leaderboard.

## 9. Restricted-tour research problems

Restrictions on the admissible tour can form useful ETSP subproblems or solver
components:

- bitonic or pyramidal tours;
- Hamiltonian paths with fixed/free endpoints;
- tours constrained to a candidate graph;
- tours respecting a fixed backbone or set of edges;
- portal-respecting or guillotine tours in approximation schemes;
- local subproblems with prescribed boundary ports.

Their optimum is generally an upper bound for unrestricted ETSP, not the ETSP
optimum. A benchmark must state when it evaluates the restricted problem versus
using the restriction only inside a solver.

## 10. Nearby variants are not ETSP subproblems

The following are related routing problems, not strata of the canonical ETSP:
TSP with neighborhoods, generalized/cluster TSP, bottleneck or maximum TSP,
multiple-salesperson TSP, prize collecting and orienteering, Steiner TSP,
geodesic/obstacle TSP, Dubins/curvature-constrained TSP, time windows,
stochastic/online TSP, and asymmetric or time-dependent travel. Techniques may
transfer, but results cannot be placed on an ETSP leaderboard without a formal
reduction and identical objective.

## 11. Most promising cross-regime research programs

1. **Ambient versus intrinsic dimension:** vary embedding dimension, manifold
   dimension, noise, and candidate recall independently.
2. **Few-inner and onion structure:** continue from convex position through
   controlled interior counts and nested layers.
3. **Width/manifold/line structure:** line, `N` lines, narrow strip, wide strip,
   curves, surfaces, and unrestricted clouds.
4. **Hierarchical clusters and ports:** vary separation and bridge entropy to
   understand decomposition failure and global repair.
5. **Random, smoothed, and adversarial continuations:** perturb known hard
   configurations through a noise ladder and compare them with matched random
   controls.

These programs produce explanatory solver science: they reveal *where and why*
a method works, rather than only whether it wins on a fixed historical list.
