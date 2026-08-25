# ETSP benchmark catalog and proposed corpus

Research cutoff: **2026-08-02**. This is a sourcing and suite-construction guide,
not a redistribution bundle. Verify every upstream license/usage condition at
acquisition time. Where redistribution rights are unclear, retain an upstream
URL, source checksum, and fetch instructions instead of copying the artifact.

The catalog deliberately separates **canonical ETSP** from broader geometric
TSP instances with another distance convention. They can share infrastructure
but not objective aggregates.

## 1. Public benchmark sources

| Source | Contents and scale | Best use | Critical cautions |
|---|---|---|---|
| [TSPLIB95](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) | over 100 classic TSP examples; full library also contains other problem types and several edge-weight modes | parser/oracle validation, historical comparison, certified named cases | public and tuneable; mixed metrics; `EUC_2D` rounds each edge; sparse replication by regime; inspect upstream redistribution terms |
| [DIMACS 8th TSP Challenge](https://dimacs.rutgers.edu/archive/Challenges/TSP/about.html) | 34 large TSPLIB STSP cases, 26 uniform-square cases from 1k to 10m nodes, 22 clustered cases from 1k to 100k, plus nongeometric random matrices | historical large-scale quality/time/memory; standard generators; scaling | exclude random matrices from ETSP; aged protocol/generators; old machine-normalization code is only a supplemental control |
| [Waterloo data index](https://www.math.uwaterloo.ca/tsp/data/index.html) | national, VLSI, World, Mona Lisa, USA, and other collections; VLSI reaches 744,710 and World 1,904,711 nodes | real/application geometry, huge-scale engineering, public tour/bound context | objectives differ; World uses `GEOM`; reference status can change; preserve status date |
| [National TSP collection](https://www.math.uwaterloo.ca/tsp/world/countries.html) | country-scale geographic point sets; official index describes 27 examples from 28 to 71,009 cities | geographic/nonuniform structure, exact and heuristic cases | check each objective; the separate status table may cover a different count/version; do not assume all optimal |
| [VLSI collection](https://www.math.uwaterloo.ca/tsp/vlsi/index.html) | 102 circuit-derived cases, 131 to 744,710 nodes | structured application geometry and scale | application-derived does not mean representative of every deployment; inspect per-instance metric/reference |
| [World TSP](https://www.math.uwaterloo.ca/tsp/world/) | 1,904,711 populated places with maintained best tour and lower bound | huge-scale portfolio, decomposition, memory and record context | `GEOM`, not planar raw L2; one instance is a case study, not a statistical sample |
| [Mona Lisa](https://www.math.uwaterloo.ca/tsp/data/ml/monalisa.html) | 100,000 points arranged for continuous-line art | nonuniform image-derived geometry and large-scale visualization | one highly specialized construction; reference and distance semantics must be captured |
| [TSP Algorithm Selection data](https://tspalgsel.github.io/) | clustered, morphed, and RUE files with Concorde-computed tour lengths; features and solver ecosystem | feature-performance modeling and portfolio research | verify exact objective and certification artifacts; group morph/parent lineages in splits |
| [Tetrahedron instances](https://doi.org/10.1007/s12532-020-00184-5) | parameterized rounded-Euclidean constructions, about 200 vertices already exact-hard in reported experiments | exact proof stress, subtour-LP stress, hard-family continuations | hardness is solver/configuration and metric dependent; include construction parameters and modified/unmodified status |
| [`tspgen`](https://jakobbossek.github.io/tspgen/) | generator/mutations for explosions, clusters, axis projection, grids, and other topological changes | feature-space diversity, evolved stress, algorithm-selection data | GPL-3 software; generated outputs need lineage and fitness records; avoid final-test evolution against evaluated solvers |
| [`netgen`](https://github.com/jakobbossek/netgen) | RUE, clustered network generation, morphing, TSPLIB IO | controlled RUE/cluster/morph studies | verify package version/license; replace vague defaults with stored parameters |

### 1.1 What to import from TSPLIB

Do not bulk-label all `TYPE: TSP` files “ETSP.” Create objective strata from
the [specification](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf):

- canonical rounded Euclidean: `EUC_2D`, and separately `EUC_3D`;
- related Euclidean convention: `CEIL_2D`;
- noncanonical geometric controls: `GEO`, `ATT`, special norms;
- explicit/general STSP: exclude from ETSP aggregates, but retain if a generic
  TSP engine is being checked separately.

For every import, retain the original header, bytes checksum, parsed dimension,
edge-weight type/format, coordinate digest, reference-tour digest, and locally
rescored objective. Reject a reference whose local score disagrees until the
semantic discrepancy is understood.

### 1.2 What to import from DIMACS

The official challenge reports:

- 34 TSPLIB symmetric instances with at least 1,000 cities, sizes 1,000–85,900;
  all but one are rounded planar Euclidean;
- 26 RUE point sets in a `1,000,000 x 1,000,000` square, sizes
  1,000–10,000,000;
- 22 randomly clustered point sets in the same square, sizes 1,000–100,000;
- seven random distance-matrix problems, which are not ETSP.

Import the first three groups into separate strata. Retain the historical
multi-fragment benchmark program/results only for rough hardware continuity;
modern reports still need raw wall time and hardware. The original protocol's
end-to-end time and peak resident-memory definitions should be preserved.

### 1.3 Waterloo reference snapshots

Treat a Waterloo page as a changing upstream record. On acquisition, capture:

```text
instance ID and checksum
metric and coordinate format
tour artifact and its rescored objective
lower-bound value and validity scope
status = optimal / nonzero gap / upper only / unknown
source page and retrieval date
attribution for tour and bound
```

The [national status table](https://www.math.uwaterloo.ca/tsp/world/summary.html)
explicitly lists tours, Concorde lower bounds, and gaps; it must not be reduced
to a stale list of supposed optima. The World record is likewise a bound pair,
not a certified optimum at the research cutoff.

## 2. Proposed benchmark release structure

Use eight top-level suites. The counts below are initial targets, not a
scientific constant. Scale them according to reference-generation cost while
maintaining multiple independent instances per cell.

| Suite | Purpose | Initial composition |
|---|---|---|
| `semantic_micro` | objective, parsing, tour validity, invariance | 500–2,000 cases, mostly `n=1..18`, exact reference for all valid definitions |
| `exact_structure` | exact bounds/proof and geometry-sensitive exact methods | 1,000–3,000 generated cases `n=20..300`, plus TSPLIB and tetrahedron subsets |
| `anytime_quality` | strong heuristic and portfolio quality/time | 2,000–10,000 cases `n=100..10,000`, balanced across controlled/evolved/public geometry |
| `latency_batch` | constructors, heatmaps, ML and tiny-budget search | large generated banks at fixed sizes, plus held-out family/parameter shifts |
| `scale_sparse` | time/memory/candidate/decomposition scaling | `n=10k..10m` DIMACS and generated strata; VLSI/national cases |
| `huge_case_studies` | extreme memory and decomposition | selected 100k–1.9m+ Waterloo/DIMACS and generated streaming cases |
| `adversarial_coverage` | falsification and solver complementarity | tetrahedron, structured traps, feature-space/evolved instances, perturbation ladders |
| `hidden_generalization` | tuning-resistant evaluation | sealed seeds, parameter ranges, families, dimensions, and mutation descendants |

Every suite is partitioned by objective mode. Never compute one quality mean
across raw L2, `EUC_2D`, `CEIL_2D`, or `GEOM`.

## 3. `semantic_micro`: the correctness corpus

This corpus is intentionally easy to solve and difficult to parse incorrectly.

### 3.1 Base configurations

- empty/one/two-city inputs only if the API defines them; otherwise explicit
  expected rejection;
- triangle, square, rectangle, regular polygons, and convex random points;
- line and vertical-line sets with distinct points;
- all-coincident and partially duplicated points;
- collinear and nearly collinear runs;
- cocircular points with many optimal tours;
- integer grids with repeated distances;
- one interior point, a few inner points, and nested convex layers;
- extreme aspect ratios and coordinates near accepted numeric limits;
- examples whose optimum differs between raw L2, rounded nearest integer, and
  ceiling distance;
- hand-constructed ties and alternative optima.

### 3.2 Required metamorphic siblings

For each eligible parent generate city-ID permutation, cycle reversal, tour
rotation, translation, reflection, axis permutation, and uniform scale. For raw
L2, rotations preserve the ideal objective; in finite arithmetic compare under
the declared tolerance. Integer-coordinate rotations that do not preserve the
lattice should not be treated as bit-identical `EUC_2D` siblings.

Expected relations:

- point/city permutation leaves optimum unchanged;
- translation/reflection/orthogonal transform preserves raw L2;
- positive scale multiplies raw-L2 optimum by the same factor;
- adding an exact duplicate does not necessarily leave the cycle encoding
  unchanged but permits a zero-cost adjacency and leaves the geometric optimum
  value unchanged;
- every accepted output is a permutation of all city IDs and closes implicitly
  or explicitly according to one declared convention.

These siblings share one lineage group and must never cross ML splits.

## 4. Controlled synthetic family library

The implementer should expose each family as a pure versioned generator with an
explicit parameter domain. Default coordinates can be normalized to `[0,1]^d`
and transformed/quantized only in a later named stage.

### 4.1 Baseline densities

**Uniform hyperrectangle**

```text
d in {2,3,4,8,16,32}
side spectrum / aspect ratio controlled
IID uniform coordinates
optional affine rotation before bounding normalization
```

The square/cube is a calibration family. Varying aspect ratio prevents it from
becoming the entire notion of “random ETSP.”

**Smooth nonuniform IID density**

- product beta densities with logged shape parameters;
- linear/radial density gradients;
- bounded mixtures whose analytic density is known when possible.

These permit BHH-normalization checks and distinguish nonuniform density from
explicit clustering labels.

**Poisson-disc / repulsive process**

Control minimum separation or target packing fraction. Record the precise
sampling/rejection algorithm; different algorithms do not define identical
point processes. This family reduces near-duplicates and produces locally
regular spacing.

### 4.2 Clustered and hierarchical families

**Gaussian or elliptical mixture**

```text
cluster_count
mixture weights or imbalance exponent
center process and minimum separation
within-cluster covariance eigenvalues/orientation
truncation/boundary rule
outlier/background weight
```

Sample separation relative to within-cluster RMS radius; absolute coordinates
alone are not portable.

**Disk/ball clusters** use bounded support and make inter-cluster separation
unambiguous. **Hierarchical mixtures** recursively sample subclusters and expose
depth, branching factor, scale ratio, and weight imbalance.

**Bridge-entropy constructions** place two or more clusters with a controlled
number of near-equivalent port regions. Low-entropy cases have clear closest
ports; high-entropy cases arrange arcs/faces so many bridge pairs have similar
length. This is a diagnostic generator for decomposition and EAX/LK global
edge choices, not a standard stochastic process.

### 4.3 Lattice and degeneracy families

- rectangular/triangular/hexagonal grids where applicable;
- grid plus IID or Gaussian jitter over a logarithmic noise ladder;
- coordinate quantization at several bit depths;
- duplicate and near-duplicate injection with controlled multiplicity;
- collinear, cocircular, coplanar, or repeated-distance fractions;
- random small-integer coordinates conditioned on collision rate.

Keep the unjittered parent and all noise descendants in one lineage. Grid
dimensions, cropping rule, and boundary deletion must be stored.

### 4.4 Boundary and layer families

**Convex plus `k` inner points**

- choose `n`, exact `k`, hull shape, inner density, and distance to boundary;
- include fixed `k` with growing `n`, fixed ratio `k/n`, and transitions from
  `k=0` to interior-dominated cases;
- verify the realized hull count after finite-precision generation.

**Onion layers**

- nested regular or perturbed convex polygons/ellipses;
- layer count, points per layer, angular phase, radial gap, and jitter;
- balanced versus concentrated layer sizes.

**Holes and outliers**

- one/multiple excluded disks or general voids;
- hole radius/separation and density near boundaries;
- isolated outlier count and distance relative to diameter;
- outliers arranged as a chain, ring, or independent background.

### 4.5 Lines, strips, corridors, and manifolds

**Line / `N` parallel lines** records `N`, spacing, longitudinal distribution,
and endpoint alignment. **Narrow strip** records width `delta` in the paper's
coordinate convention, point intensity, integer/distinct-x conditions, and
sparsity. Include values below, at, and above `2 sqrt(2)` when testing the
bitonic theorem's assumptions.

**Corridor networks** place points along a polyline/tree-like set of corridors,
with junction density, width, and cycles controlled. They stress partitioning
but do not inherit the exact narrow-strip theorem.

**Curves and surfaces** include circles, multiple rings, ellipses, spirals,
splines, tori/spheres in higher ambient dimensions, and low-rank affine
subspaces. Store intrinsic dimension, parameter distribution, reach/curvature
proxy, orthogonal noise, and ambient embedding transform.

### 4.6 Anisotropy, scale, and numeric stages

Apply these stages after geometry generation:

- affine anisotropy with stored singular values;
- translation and uniform scaling;
- random orthogonal embedding;
- integer rounding, fixed-point quantization, or TSPLIB edge semantics;
- large offsets designed to reveal cancellation/precision issues;
- coordinate permutation and file-order permutation.

The transformed instance retains `geometry_parent_id`; changing distance
semantics also creates a new `objective_sibling_group`.

## 5. Adversarial and feature-space strata

### 5.1 Mathematical hard families

Start with tetrahedron instances and their modified unique-optimum variants from
[Hougardy and Zhong](https://doi.org/10.1007/s12532-020-00184-5). Store `T[n,m]`
parameters and construction revision. Add a perturbation ladder that preserves
the unperturbed official files and explicitly states whether per-edge rounding
still applies.

Include known worst-case constructions for the component being studied only
when their Euclidean coordinates and objective are reproducibly specified.
Label the intended stress (`subtour_lp`, `k_opt`, `candidate`, `decomposition`)
instead of a universal `hard=true`.

### 5.2 Feature-diverse evolution

Use `tspgen`-style topology mutations or an equivalent documented generator to
fill sparse feature cells. Candidate mutation categories include:

- local point displacement/replacement;
- explosion/void creation;
- cluster creation and compression;
- axis/line projection;
- grid regularization;
- scaling/rotation of subsets;
- point-group translation;
- adding/removing structural noise while keeping `n` fixed.

Record the initial parent, every accepted mutation or a reproducible event log,
fitness components, feature scaler, solver portfolio/configurations, evaluation
budget, and evolutionary seed.

Create three frozen products:

1. `feature_fill`: selected without evaluated-solver performance fitness;
2. `portfolio_discrimination`: selected using multiple control solvers;
3. `solver_specific_stress`: openly labeled with its target solver/config.

Only the first two belong in broad leaderboard aggregates.

### 5.3 Counterexample bank

Maintain small geometric counterexamples for unsafe assumptions: optimal edges
outside a chosen Delaunay/kNN candidate set, disconnected candidate graphs,
nearest-neighbor pathologies, decomposition boundary failures, tied or
ambiguous edges, and numerical ordering reversals. These are component and
correctness tests, not representative performance samples.

## 6. Size ladders by task

Use logarithmic size ladders but do not require every family at every size.

### 6.1 Exact/control ladder

```text
n = 10, 12, 15, 18                    exhaustive/Held-Karp cross-check
n = 20, 25, 30, 40, 50                dense exact labeling
n = 75, 100, 150, 200, 300            selective exact/proof stress
n > 300                               only where certified upstream or feasible
```

Hard-family sizes require their own denser progression around the runtime
transition. Use adaptive pilot runs to set caps, but freeze the final set before
solver comparison.

### 6.2 Heuristic/ML ladder

```text
n = 20, 50, 100, 200, 500, 1k, 2k, 5k, 10k
```

Small sizes connect to optima; larger sizes expose candidate and generalization
behavior. For neural training, avoid claiming size generalization if adjacent
sizes or descendants leaked into the training generator.

### 6.3 Scale ladder

```text
n = 10k, 30k, 100k, 300k, 1m, 3m, 10m
```

Use DIMACS at its official sizes and generated counterparts under a fixed
coordinate/intensity convention. At huge scale, record whether coordinates and
candidates are streamed, and report per-node memory as well as peak RSS.

## 7. Sampling and split policy

### 7.1 Minimum independent sampling

For controlled small/medium experiments, target at least 30 independent point
sets per primary factor cell when affordable; 10 is a pilot, not a robust final
sample for close effects. Use more instances rather than excessive solver seeds
once seed-level uncertainty is small. Exact-hard cells may necessarily contain
fewer instances and should report that limitation.

Do not mechanically create the full Cartesian product of every factor. Select
interactions that correspond to hypotheses, then add space-filling samples over
continuous parameters.

### 7.2 Group-aware splits

Assign a `lineage_group_id` before splitting. Group together:

- rigid/scale transforms and city permutations;
- raw/rounded objective siblings if model input includes the same coordinates;
- subinstances or supersets from one parent;
- morph paths and perturbation ladders;
- mutation/evolution descendants;
- repeated coordinate files from multiple public mirrors.

Recommended public split roles:

```text
train_iid
validation_iid
test_iid
test_size_ood
test_parameter_ood
test_family_ood
test_metric_ood
test_dimension_ood
test_public_untuned
test_adversarial
hidden_release_N
```

### 7.3 Hidden-suite construction

Publish schema, broad family proportions, objective modes, caps, and validation
rules. Keep seeds, exact parameter draws, selected public-like private inputs,
and reference artifacts sealed. Before evaluation, freeze the solver container,
model digest, and parameter policy. Release stratified results and eventually a
subset of artifacts for audit; create a new hidden release rather than reusing
exposed cases indefinitely.

## 8. Reference-generation matrix

| Regime | Preferred reference process | Stored evidence |
|---|---|---|
| `n <= 25` | at least two independent exact routes where feasible (Held–Karp and branch-and-cut/brute force for tiny cases) | optimum, tour, logs/digests, cross-check status |
| small/medium ordinary | Concorde or independently verified branch-and-cut | incumbent, final lower bound, exact status, solver/version/log |
| exact-hard | long capped Concorde plus strong independent bounds; accept nonzero certified gap | best tour, valid bound, full status; never relabel timeout as optimum |
| public certified | import official tour/proof/bound and independently rescore | source/date/checksum and local verification |
| large heuristic | reference portfolio including official LKH and GA-EAX configurations | each tour, best upper, run compute, reference snapshot |
| huge | upstream/public record plus local portfolio and available lower bound | upper/lower separately; case-study status |
| random asymptotic | BHH diagnostic only | normalized distribution summaries, never `OPT_reference` |

For portfolio-generated best knowns, use a compute ledger. A “best known” found
after 10,000 GPU/CPU hours is valid as an upper reference but is not a fair
baseline runtime result for a 10-second solver.

## 9. Feature catalog

The manifest should distinguish cost tiers.

### Tier 0: direct and linear

- `n`, `d`, coordinate/objective type and bit depth;
- bounding-box sides, diameter estimate, covariance eigenvalues;
- duplicate/quantization/tie samples;
- per-axis and radial quantiles;
- coordinate entropy/occupancy at several grids.

### Tier 1: near-linear geometric

- convex hull count/fraction in 2D; inner count;
- exact or sampled onion depth;
- kNN-distance summaries for fixed `k` values;
- MST normalized weight, degree, edge quantiles, max-edge ratio;
- 2D Delaunay degree/edge summaries where predicates are robust;
- multiscale component/cluster and MST edge-gap summaries;
- intrinsic-dimension estimates and residuals to line/plane/manifold models.

### Tier 2: charged probing

- nearest-neighbor/greedy initial gap;
- fixed-work 2-opt improvement and local optimum;
- fixed-trial LKH/EAX probe curves;
- minimum 1-tree and optimized Held–Karp bound/gap;
- candidate graph connectivity, reference-edge recall, and widening depth;
- elite-edge frequency, backbone fraction, and entropy;
- subtour-LP root gap, fractionality, cuts, and pricing statistics.

Every feature has `definition_version`, runtime, memory, status, and missingness
reason. Tier 2 is included in selector budgets when used at inference.

## 10. Corpus weighting and reports

Publish at least four views:

1. **per-source:** TSPLIB, DIMACS RUE/cluster, Waterloo categories;
2. **balanced diagnostic:** equal or declared weights across structural strata;
3. **target population:** weights based on a stated deployment distribution;
4. **stress:** adversarial/hard and worst-tail outcomes, with no claim of
   representativeness.

Within each, stratify by objective, size band, dimension, and reference status.
Never let the numerous cheap generated cases drown the smaller but meaningful
public/adversarial strata.

## 11. Acceptance checklist for a benchmark artifact

- [ ] Coordinates parse under the declared schema and dimension.
- [ ] Distance semantics have an exact versioned identifier.
- [ ] City IDs are unique; duplicate coordinates are represented intentionally.
- [ ] `n`, coordinate digest, file digest, and lineage IDs are present.
- [ ] Generator/source provenance and usage/redistribution status are present.
- [ ] Split assignment respects all lineage groups.
- [ ] Reference status is one of the defined hierarchy values.
- [ ] Every stored tour is feasible and independently rescored.
- [ ] Every lower bound identifies its proof/solver semantics.
- [ ] Feature values include version, cost, and missingness.
- [ ] Public and generated selection procedures are documented.
- [ ] No non-ETSP objective enters a canonical ETSP aggregate.
