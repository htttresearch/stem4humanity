# Euclidean TSP benchmarking: research report

Research cutoff: **2026-08-02**. This report explains what an ETSP benchmark is
supposed to establish, why the usual datasets are insufficient by themselves,
and how to design experiments from which a solver builder can draw reliable
conclusions. [BENCHMARK_CATALOG.md](BENCHMARK_CATALOG.md) specifies the corpus;
[BENCHMARK_IMPLEMENTER_SPEC.md](BENCHMARK_IMPLEMENTER_SPEC.md) turns the design
into implementation contracts; [BENCHMARK_LITERATURE.md](BENCHMARK_LITERATURE.md)
is the focused annotated bibliography; [BENCHMARKING.md](BENCHMARKING.md)
contains the run and publication protocol.

## 1. Executive conclusions

A serious ETSP benchmark is not one leaderboard and not one distribution. It is
a versioned experimental system with five simultaneous responsibilities:

1. **semantic correctness:** all tours, references, and bounds use exactly the
   same distance definition;
2. **construct validity:** the measured response actually matches the solver
   claim—proof, quality, latency, scale, generalization, or component behavior;
3. **instance validity:** the corpus spans relevant geometry and difficulty,
   instead of equating ETSP with uniform random points or TSPLIB;
4. **measurement validity:** budgets, failures, randomness, preprocessing, and
   postprocessing are accounted for consistently;
5. **inferential validity:** conclusions are made over independent instances
   and declared target populations, with uncertainty and distribution shift
   visible.

The recommended benchmark is therefore layered:

- **public historical/realistic collections** for continuity and recognizable
  case studies;
- **factorial synthetic families** for controlled causal questions;
- **feature-space and evolved instances** to fill coverage holes and reveal
  solver complementarity;
- **mathematical/adversarial families** for known structural stress;
- **hidden frozen tests** for honest model/configuration generalization;
- **metamorphic and micro instances** for objective and implementation
  correctness.

No single aggregate score should erase the separate frontiers for exact proof,
anytime upper bounds, low latency, memory-limited scale, and ML generalization.

## 2. What is the scientific object being benchmarked?

### 2.1 Instance, representation, task, solver, and run

Keep five objects separate:

- an **abstract instance** is a point multiset plus a distance function;
- an **encoding** is a coordinate/file representation of that instance;
- a **task** specifies the requested output and budget;
- a **solver configuration** includes code, parameters, hardware use, models,
  preprocessing, and postprocessing;
- a **run** combines all of the above with a seed and produces a trace.

This distinction prevents common mistakes. Permuting city IDs creates a new
encoding, not a new independent point set. Changing raw L2 to rounded
`EUC_2D` creates a new objective, not a harmless serialization change. Running
the same constructor with a larger 2-opt budget creates a different configured
solver. Measuring only the neural forward pass does not measure an end-to-end
neural solver if decoding and repair are required.

### 2.2 Claims define benchmark tasks

The same corpus can support several tasks, but their responses differ.

| Claim | Primary responses | Required reference |
|---|---|---|
| exact optimizer | solved/proved status, proof time, nodes, memory, bound trace | certified optimum or independently valid bounds |
| best upper-bound heuristic | objective/gap trajectory, success probability, work, memory | certified optimum or versioned best upper/lower bounds |
| fast constructor | cold/warm latency, throughput, immediate quality, tiny matched postprocessing | reference upper/optimum where available |
| huge-scale method | objective or certified gap, peak memory, throughput/scaling exponent | valid reference/bound when available; otherwise comparative upper bounds |
| learned guidance | matched-engine improvement and inference cost | identical classical engine/control |
| portfolio selector | regret to virtual best, feature cost, timeout loss | full solver-performance matrix on held-out instances |
| approximation implementation | empirical quality plus conformance to the proof assumptions | proof-level invariant checks and valid optimum/bound sample |

Calling all of these “TSP solution quality” destroys construct validity.

## 3. Foundations of empirical algorithmics

Hooker argued for an empirical science that builds explanatory theories of
algorithm behavior, not only competitive tables
([1994](https://doi.org/10.1287/opre.42.2.201)); his follow-up warns that
competitive testing alone tells us little about *why* methods differ
([1995](https://doi.org/10.1007/BF02430364)). Johnson's
[experimental-analysis guide](https://dimacs.rutgers.edu/archive/Challenges/TSP/papers/experguide.pdf)
and Johnson and McGeoch's
[STSP chapter](https://dimacs.rutgers.edu/archive/Challenges/TSP/papers/stspchap.pdf)
translate that philosophy into algorithm-engineering practice.

For ETSP, there are three complementary experiment types:

1. **comparison experiments** ask which complete solver/configuration wins for
   a stated target population and budget;
2. **controlled experiments** vary one factor or component to explain a
   mechanism;
3. **characterization experiments** model performance as a function of
   instance features and discover regimes or solver footprints.

The benchmark must support all three. A stable public suite is excellent for
historical comparison but weak for controlled inference. A synthetic factorial
design is excellent for estimating effects but may be unrepresentative. An
evolved hard set is excellent for falsification but biased toward the solvers
and objectives used to evolve it.

## 4. What makes an ETSP benchmark good?

### 4.1 Relevance

The corpus should represent the population to which the final claim refers. A
solver for planar raw-L2 industrial point clouds cannot establish that claim on
only TSPLIB `EUC_2D`, because the latter rounds each edge. A claim about
unrestricted ETSP cannot be inferred from 100-node IID uniform squares. If the
deployment population is unknown, the honest target is broad robustness and the
benchmark should explicitly diversify structural axes.

### 4.2 Coverage and boundary cases

Coverage means more than many files. It requires variation in features that can
change solver behavior: dimension, intrinsic dimension, size, hull fraction,
strip/manifold width, density, aspect ratio, clustering, bridge ambiguity,
degeneracy, coordinate precision, and objective semantics. Include both typical
regions and boundaries at which algorithmic behavior changes—for example the
narrow-strip bitonic threshold and a ladder of perturbations away from convex
position.

### 4.3 Controlled difficulty

`n` is neither a universal difficulty measure nor a sufficient stratification.
Hougardy and Zhong report a rounded-Euclidean 200-vertex tetrahedron instance
requiring several CPU days in Concorde and more than one million times the
runtime of similar-sized TSPLIB examples
([paper](https://doi.org/10.1007/s12532-020-00184-5)). Difficulty should be
represented as a response vector, not baked into a dataset name:

```text
H_exact       = proof time/nodes and root/bound behavior under fixed exact controls
H_upper(B)    = portfolio gap distribution at budget B
H_target(eps) = probability and time to reach an epsilon target
H_candidate(k)= recall/connectivity and widening needed at candidate budget k
H_scale       = time and peak memory growth
H_shift       = degradation under size/family/metric/dimension shift
```

Do not select instances as “hard” using only the solver later being evaluated.
Use multiple controls or label the result solver-conditional.

### 4.4 Discrimination without artificiality

A suite where all modern solvers instantly find the optimum is saturated; one
where all time out without informative bounds is also weak. Good instances
spread the relevant response and distinguish algorithmic choices. But pure
solver discrimination can produce exotic artifacts. Maintain separate labels
for representative, controlled, and adversarial strata so a win in one cannot
masquerade as universal superiority.

### 4.5 Independence and sufficient replication

Thousands of stochastic runs on one point set are not thousands of independent
ETSP instances. The unit of generalization is normally the instance (or the
family, for conclusions spanning families), while seeds are repeated
measurements nested inside it. Generate multiple independent point sets per
factor cell and multiple solver seeds per point set. Keep transformed copies,
subsamples, and mutated descendants grouped to avoid leakage and
pseudo-replication.

### 4.6 Reproducibility, immutability, and extensibility

Every artifact should have a content digest; every generator should have a
version, full parameter record, PRNG algorithm, and seed. Published suites are
immutable releases. Corrections produce a new release and an erratum, not an
in-place replacement. The schema must permit new instance families, metrics,
reference updates, and solver traces without invalidating old results.

### 4.7 Neutrality and leakage resistance

Public instances are useful but can be overfit through years of manual tuning.
The DIMACS protocol already prohibited per-instance hand tuning and required
automatic adaptation costs to be included
([challenge](https://dimacs.rutgers.edu/archive/Challenges/TSP/about.html)). A
modern benchmark should additionally freeze hidden generator seeds/parameter
regions, declare whether public instances entered training or tuning, and keep
all symmetry-derived representations within the same split.

### 4.8 Auditability

Publish per-run raw records, incumbent traces, tours or tour digests, failure
states, configuration files, environment facts, and reference provenance.
Aggregate tables must be reproducible from raw records. A benchmark whose only
artifact is a rounded average cannot be audited.

## 5. Theory that should shape ETSP benchmark design

### 5.1 The random Euclidean scaling law

For IID points with density `f` in a bounded region of `R^d`, the
Beardwood-Halton-Hammersley law has the form

```text
OPT_n / n^((d-1)/d)
  -> beta_d * integral f(x)^((d-1)/d) dx
```

under its regularity assumptions. Steele's
[account](https://epubs.siam.org/doi/10.1137/1.9781611970029.ch2) places it in
the theory of subadditive Euclidean functionals. Consequences for benchmarking:

- raw tour length is not comparable across `n`, dimension, scale, or density;
- normalized MST/tour length can validate generator behavior and detect
  objective/parser bugs;
- a large RUE suite estimates average-case behavior for one distribution, not
  unrestricted ETSP behavior;
- the asymptotic expression is not a certified lower bound for a particular
  instance.

For a uniform unit square in 2D, length is order `sqrt(n)`; for a uniform unit
cube in 3D it is order `n^(2/3)`. If coordinates occupy a square of side `s`,
length scales by `s`. These invariances should be checked before timing any
solver.

### 5.2 Dimension changes both theory and engineering

For fixed `d`, the ETH-tight exact exponent is `n^(1-1/d)`
([de Berg et al.](https://arxiv.org/abs/1807.06933)). This does not predict
Concorde runtime per instance, but it explains why `n`-only scaling plots across
dimensions are conceptually incomplete. High dimension also changes distance
concentration, geometric index effectiveness, candidate recall, and memory.

Benchmarks should use separate curves by `d`, plus low-intrinsic-dimensional
embeddings that disentangle ambient representation from real geometry.

Empirical runtime models remain solver- and distribution-specific. Hoos and
Stützle fitted historical Concorde scaling on RUE instances and found a model
exponential in `sqrt(n)` to describe their observed regime
([paper](https://doi.org/10.1016/j.ejor.2014.04.042)); the associated later
[supplementary testbed](https://iridia.ulb.ac.be/supp/IridiaSupp2017-010/index.html)
contains large independent banks across sizes. This is useful calibration and a
model-checking precedent, not a theorem about all ETSP or all exact solvers.

### 5.3 Worst-case, random, and smoothed regimes answer different questions

Fixed-`k` local optima can have worst-case approximation ratio
`Theta(log n / log log n)` in planar ETSP
([Brodowsky et al.](https://doi.org/10.1137/21M146199X)), while recent smoothed
analyses prove polynomial behavior for 2-opt under explicit perturbation models,
including complementary 2025 results for Gaussian noise
([Manthey and van Rhijn](https://doi.org/10.1007/s00453-025-01309-9),
[Künnemann, Manthey, and Veenstra](https://doi.org/10.1007/s00453-025-01335-7)). These are not
contradictory results. They motivate a three-part benchmark:

- representative/random inputs for typical behavior;
- adversarial constructions for failure modes;
- a noise ladder around adversarial and structured seeds to measure how robust
  the failure is.

Always log the perturbation distribution and scale relative to nearest-neighbor
spacing or bounding-box diameter.

### 5.4 Geometry can create exact sparsity only under proof conditions

Euclidean candidates are usually sparse in practice. Zhong's probabilistic
analysis shows linear expected survivors for one exact edge-elimination rule
under bounded-density random inputs
([paper](https://arxiv.org/abs/1809.10469)). That result is distribution- and
rule-specific. A benchmark should measure candidate degree, build cost, graph
connectivity, reference-edge recall, and widening behavior; it must not infer
exact safety from good random-case recall.

### 5.5 Structural parameters can dominate `n`

Few-inner-point ETSP has an FPT algorithm in inner count `k`
([Gawrychowski and Rusak](https://arxiv.org/abs/1406.2154)); narrow-strip ETSP
has width-parameterized algorithms and a tight bitonic threshold
([Alkema et al.](https://arxiv.org/abs/2003.09948)); points on a fixed number
`N` of parallel lines admit an `n^N` dynamic program
([Rote](https://page.mi.fu-berlin.de/rote/Papers/abstract/The%2BN-line%2Btraveling%2Bsalesman%2Bproblem)).
These results tell the benchmark designer which exact control parameters are
more meaningful than a visually assigned family label.

## 6. Why the famous public suites are necessary but insufficient

### 6.1 TSPLIB95

TSPLIB standardized file formats, reference instances, and many objective
functions; Reinelt's original library paper is
[here](https://doi.org/10.1287/ijoc.3.4.376) and the authoritative format and
distance rules are in the
[TSPLIB95 specification](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf).
It is indispensable for historical comparability, exact-parser validation, and
named case studies.

Its limitations are equally important:

- it is a small, fixed, public collection and therefore tuneable;
- it mixes edge-weight semantics and application origins;
- most planar `EUC_2D` objectives round each edge and are not raw L2;
- structural regimes have few or no independent replicates;
- the collection is not a probability sample from a stated deployment
  population;
- many entries are now too easy for strong solvers at common budgets.

TSPLIB should be a protected public-test stratum, never the whole benchmark and
never the default neural training set.

### 6.2 DIMACS 8th TSP Challenge

DIMACS was explicitly designed to create a reproducible picture of heuristic
effectiveness, robustness, and scalability. Its geometric core contains 34
TSPLIB instances of at least 1,000 nodes (all but one planar rounded Euclidean),
26 uniform-square instances from 1,000 to 10,000,000 nodes, and 22 clustered
instances from 1,000 to 100,000 nodes. It requested per-instance tour length,
end-to-end user time, and memory, and supplied multi-fragment greedy for rough
machine normalization
([official protocol](https://dimacs.rutgers.edu/archive/Challenges/TSP/about.html)).

This remains an excellent scale and historical-control suite. Its principal
limitations are age, narrow random generators, old hardware normalization, and
the absence of modern anytime traces, energy/GPU accounting, hidden test
splits, higher dimensions, or feature-space coverage.

### 6.3 Waterloo collections

The official [Waterloo data index](https://www.math.uwaterloo.ca/tsp/data/index.html)
lists national instances, 102 VLSI problems from 131 to 744,710 cities, the
1,904,711-city World instance, the 100,000-city Mona Lisa instance, and other
large geometric cases. These are valuable for real/application geometry and
scale, with public tours and bounds for many instances.

Do not silently merge their metrics: the World instance uses `GEOM`, not planar
raw L2. Reference status is also dynamic. Store separately the best tour, valid
lower bound, status date, source, and objective. The
[national status table](https://www.math.uwaterloo.ca/tsp/world/summary.html),
for example, distinguishes optimal instances from nonzero certified gaps.

### 6.4 Hard and evolved collections

The tetrahedron family supplies mathematically explained exact-solver stress.
The TSP Algorithm Selection project publishes clustered, morphed, and RUE
instances with Concorde-computed tour lengths, along with feature software and
solver-performance research
([project](https://tspalgsel.github.io/)). `tspgen` supplies topology-changing
mutations such as explosion, clustering, axis projection, and grid formation
under GPL-3
([documentation](https://jakobbossek.github.io/tspgen/)).

Bossek et al. designed mutations to increase feature, topology, and performance
diversity rather than repeatedly nudging uniform points
([FOGA 2019 paper](https://doi.org/10.1145/3299904.3340307)). These instances
are particularly useful for solver complementarity and algorithm selection,
but their selection process must be logged: evolved test data can leak the
identity and configuration of the solvers used as fitness functions.

## 7. Instance-space design instead of dataset accumulation

### 7.1 Why named-family balancing is not enough

Two different generators can produce nearly identical geometry, while one
generator can span several behavioral regimes. Smith-Miles and Bowly's
[instance-space methodology](https://doi.org/10.1016/j.cor.2015.04.022) uses
measurable features and algorithm footprints to identify coverage gaps. The TSP
feature literature shows that characteristics beyond `n` predict local-search
quality
([Mersmann et al. 2012](https://doi.org/10.1007/978-3-642-34413-8_9),
[expanded 2013 study](https://doi.org/10.1007/s10472-013-9341-2)).

Use generator-factor coverage and feature-space coverage together:

1. define interpretable generator strata;
2. compute solver-independent cheap features;
3. inspect sparse and overrepresented regions;
4. add targeted samples or evolved instances;
5. verify that additions change coverage rather than merely file count;
6. retain a realistic weighting scheme separately from a stress weighting.

Projection to two dimensions is a visualization, not the definition of
coverage. Quantify coverage in a standardized higher-dimensional feature space
using distances, cells, density, or space-filling criteria; assess sensitivity
to the selected features and scaling.

### 7.2 Core solver-independent feature groups

Compute only when well-defined under the metric and dimension:

- identity: `n`, ambient dimension, objective semantics, coordinate precision;
- scale/shape: diameter, bounding volume, aspect ratios, covariance spectrum,
  spread and minimum nonzero separation;
- degeneracy: duplicate, collinearity/coplanarity, tie, and quantization rates;
- boundary: convex-hull fraction, inner count, sampled onion depth;
- neighborhoods: kNN-distance quantiles, coefficient of variation and angular
  gaps at several `k`;
- sparse graphs: MST normalized weight, edge-length distribution, degree and
  bottleneck statistics; Delaunay features in 2D where robustly defined;
- clustering: multiscale component counts, MST edge gaps, density contrast,
  cluster imbalance, outlier fraction, separation/diameter ratios;
- dimension: intrinsic-dimension or doubling estimators with estimator settings
  and uncertainty;
- topology: holes/void proxies, ring/manifold residuals, line/strip residuals.

Features that depend on a run—1-tree penalties, LP fractionality, probe-solver
improvement, elite-edge entropy—are valuable **probing features**, but their
cost must be charged and they must not be mislabeled as free instance metadata.

### 7.3 Solver complementarity is a benchmark property

Kerschke et al. directly compared LKH, EAX, their restart variants, and MAOS and
found complementary instance-wise behavior
([paper](https://doi.org/10.1162/EVCO_a_00215)). A good portfolio benchmark must
therefore preserve per-instance performance matrices. Averaging first can make
the virtual-best opportunity invisible and can reward a corpus dominated by one
easy regime.

## 8. Generator design and experimental controls

### 8.1 A generator is a probability model, not a name

For each generated instance record:

- generator family and immutable version;
- all parameters and their sampling distribution;
- PRNG name/version and seed;
- rejection or conditioning rules;
- normalization and coordinate quantization;
- parent artifact for mutations or perturbations;
- intended split and selection procedure.

“Gaussian clusters” is insufficient without covariance, separation, count,
weights, truncation, and bounding behavior. “Random” is insufficient without a
density and domain.

### 8.2 Crossed, nested, and response-surface designs

Use a crossed design for a manageable set of major factors—size, family,
dimension, aspect, cluster separation, and noise—when interactions matter. Use
nested factors where definitions are family-specific. For continuous
parameters, space-filling designs or response-surface sampling are often more
efficient than arbitrary low/medium/high bins. Always include multiple point-
set seeds within each cell.

The corpus should have two weightings:

- **balanced diagnostic weighting** gives comparable influence to regimes;
- **target-population weighting** estimates expected deployment performance.

Do not call the balanced mean a deployment expectation.

### 8.3 Matched continuations are unusually informative

Construct sequences in which one structural parameter changes smoothly:

- uniform -> increasingly separated mixtures;
- convex boundary -> increasing inner-point fraction;
- line -> widening strip;
- grid -> increasing jitter;
- hard construction -> increasing perturbation noise;
- low-dimensional manifold -> increasing orthogonal noise;
- low bridge entropy -> many near-equivalent inter-cluster ports.

These continuations support change-point and sensitivity analyses that a bag of
unrelated instances cannot.

### 8.4 Evolving instances safely

Evolution can target feature gaps, high variance, portfolio regret, candidate
failure, or performance differences. Avoid optimizing only terminal runtime:
timeouts produce plateaus and machine noise. Prefer multiobjective fitness that
includes feature novelty, performance discrimination, validity, and distance
from existing artifacts. Freeze evolved outputs before the final comparison
and evaluate against solvers/configurations not used in evolution.

## 9. Reference values and objective correctness

### 9.1 Reference hierarchy

Every instance must have one explicit status:

1. `certified_optimum`: optimum and verification provenance;
2. `bounds_equal`: independently valid upper and lower bounds coincide;
3. `best_known_upper_with_lower`: feasible best-known tour plus a valid lower
   bound;
4. `best_known_upper_only`: feasible reference tour, no certified gap;
5. `lower_bound_only`;
6. `unknown`;
7. `asymptotic_expectation`: generator-level theory only, never a reference
   objective.

Store upper and lower references as separate versioned records. A new shorter
tour should not overwrite history. Rescore every imported tour using the local
declared oracle; for integer objectives use integer accumulation.

### 9.2 Gap definitions

For solver upper bound `U`, certified optimum `OPT`, best-known feasible upper
bound `Uref`, and valid lower bound `L` under identical semantics:

```text
gap_to_optimum    = (U - OPT) / OPT          only with certified OPT
gap_to_best_known = (U - Uref) / Uref        may be negative
certified_gap     = (U - L) / L
```

Do not clamp negative best-known gaps. They signal a new record or, more often,
a metric/parser/tour bug. If `L=0` because all points coincide, use exact status
and absolute error instead of division.

### 9.3 Raw Euclidean numerical policy

Raw L2 tour lengths are sums of square roots. Define coordinate storage,
distance precision, summation order, and comparison tolerance. Use a stricter
independent rescoring path for claimed improvements. Never promise a bit-exact
raw-real optimum merely because two IEEE sums match; the complexity of
comparing sums of square roots is discussed by
[Qian and Wang](https://arxiv.org/abs/cs/0603002).

For benchmark certification, integer edge-weight modes are much easier to
audit. Retain both raw-L2 and rounded variants only as separately named
instances sharing a coordinate-parent ID.

## 10. Performance measurement

### 10.1 Fixed-budget and fixed-target views are dual necessities

- **fixed budget:** best valid objective achieved by deadline `B`;
- **fixed target:** time or work required to reach target quality `q`.

Fixed-budget views answer deployment questions; fixed-target views expose
reliability and speed across quality levels. COCO's performance-assessment
methodology develops target-based anytime comparison, runlength targets,
empirical distributions, and simulated restarts
([paper](https://arxiv.org/abs/1605.03560)). ETSP should adapt these ideas using
objective gaps and exact targets.

At minimum capture incumbent changes on a logarithmic clock. A terminal value
cannot distinguish a fast starter from a late improver or show solver
crossovers; anytime comparisons of LKH/EAX variants confirm that rankings can
depend on the requested quality
([Bossek et al.](https://arxiv.org/abs/2005.13289)).

### 10.2 Time, work, memory, and compute

Wall time is primary for users but hardware-dependent. Supplement it with
algorithm-specific work:

- evaluated edges/moves and accepted moves;
- LK trials, kicks, and candidate expansions;
- EAX children, AB cycles, repairs, and local-search work;
- neural forward passes, samples, augmentations, and decoded candidates;
- exact LP iterations, cuts, priced edges, nodes, and bound events.

Report CPU model, core/thread affinity, GPU, precision, RAM, compiler/libraries,
OS, power policy when material, and cold/warm state. Peak resident memory is a
first-class response. The DIMACS requirement to include reading and data-
structure construction in end-to-end time remains an excellent default.

### 10.3 Parallel and accelerator fairness

Equal wall time with unequal resources is not equal compute. Publish both the
wall-time frontier and resource facts such as CPU core-seconds, GPU-seconds,
device count, and energy if reliably measured. Do not collapse them to an
invented universal equivalent. A production user may rationally prefer lower
wall time at higher compute; a scientific comparison should make the tradeoff
visible.

### 10.4 Failure and censoring semantics

Distinguish timeout, memory limit, crash, invalid tour, numerical error,
unsupported metric, and no target reached. Time-to-target observations are
right-censored at the cap, not equal to the cap. Use success probabilities and
survival/time-to-target curves. Expected running time under restart assumptions
can be useful, but report its success probability, cap, and restart model; it is
undefined/uninformative with no successes.

## 11. Aggregation and statistical inference

### 11.1 Preserve the hierarchy

Runs are nested within instances; instances are nested within generated cells
or source families. Pair configurations on the same instance and, where random
interfaces permit, common seeds. For uncertainty over a broad corpus, use a
hierarchical/stratified bootstrap:

1. resample families or factor cells according to the declared estimand;
2. resample independent instances within them;
3. resample solver seeds within instances;
4. recompute the complete paired statistic.

This prevents a high-variance instance with many repeated runs from dominating
the inference. Publish median and tail behavior; means alone are fragile under
timeouts and heavy-tailed runtime distributions.

### 11.2 Performance profiles

For performance measure `t[p,s]` on problem `p` and solver `s`, define

```text
r[p,s]   = t[p,s] / min_s t[p,s]
rho_s(t) = fraction of problems with r[p,s] <= t
```

Dolan and Moré's
[performance profiles](https://arxiv.org/abs/cs/0102001) summarize efficiency
near `t=1` and robustness as `t` grows. Declare how failures are assigned a
ratio. Profiles are relative to the included solver set; adding/removing a
solver can change them, so retain absolute results too.

### 11.3 Data profiles and ECDFs

Data profiles show the fraction of problems solved within a normalized absolute
budget and are useful under resource constraints
([Moré and Wild](https://doi.org/10.1137/080724083)). Fixed-target empirical
cumulative distribution functions across instance-target pairs show how much
of the suite is solved at each budget. Normalize work only by a defensible unit
such as `n`, `n log n`, or a baseline measurement, and always show raw wall time
as well.

### 11.4 Multiple responses, targets, and comparisons

Quality at many times and targets is a correlated curve, not hundreds of
independent discoveries. Predeclare primary budgets/targets; use simultaneous
bands or control false discoveries for broad exploratory claims. Area under a
clipped log-time/gap curve can aid ranking but embeds arbitrary weights and
must remain secondary to the visible curve.

Use practically meaningful effect sizes: paired objective-gap change, speedup
at a target, success-probability change, and memory ratio with intervals. A tiny
statistically detectable difference on thousands of instances may be
irrelevant.

## 12. ML-specific benchmark theory

### 12.1 Split by generative lineage, not only row

The following must remain in one split: city permutations, rotations,
reflections, translations/scalings of a parent when treated as augmentations,
coordinate-rounding siblings when derived from one point set, subsamples, and
mutation descendants. Otherwise the model can recognize geometry rather than
generalize.

Use separate evaluations for:

- IID unseen seeds;
- size extrapolation/interpolation;
- parameter-range shift within a known family;
- unseen family/topology;
- metric/coordinate shift;
- ambient and intrinsic-dimension shift;
- public historical instances excluded from tuning;
- adversarial/evolved hidden tests.

Generalization failures in neural combinatorial optimization are documented by
[Bi et al.](https://doi.org/10.1007/s10601-022-09327-y). Do not report one OOD
average that hides which shift occurred.

### 12.2 Separate learned representation from search

Measure at least:

```text
model output alone
identical decoder with a nonlearned score
identical local search/postprocessing
complete end-to-end system
classical engine with and without learned guidance
```

Xia et al.'s controlled analysis finds simple heatmaps can rival learned ones
under the same post-hoc search and emphasizes strong classical controls
([ICML 2024](https://proceedings.mlr.press/v235/xia24f.html)). Charge training
compute separately from per-instance inference, but publish both. A pretrained
model amortizes training only under an explicitly assumed workload.

### 12.3 Hidden tests and benchmark saturation

Public TSPLIB results cannot prevent memorization or manual tuning. Keep a
versioned hidden suite whose generator code and broad strata may be public but
whose seeds/parameter draws remain sealed until an evaluation release. Rotate
only by creating new benchmark versions and preserve old results. Avoid a
single hidden scalar: return stratified results to expose failure modes.

Large row counts do not cure lineage leakage. A 2023 per-instance LKH
configuration study generated 100,000 2,000-node cases as local subsets of the
single World TSP
([paper](https://doi.org/10.1109/SSCI52147.2023.10372008)). This is valuable
data, but overlapping/local descendants cannot automatically be treated as
100,000 independent samples from all ETSP. The same paper obtained certified
Concorde optima for 99,998 cases and used best-of-100 EAX tours for the two
remaining cases; an implementer must preserve that reference-status
distinction rather than inheriting a convenient “optimal” label.

### 12.4 Algorithm-selection data

For learned portfolio selection, an ASlib-like scenario should include the
instance-feature matrix, feature costs/status, solver-performance matrix,
run/status matrix, cutoff, and cross-validation/group split
([ASlib](https://arxiv.org/abs/1506.02465),
[format repository](https://www.coseal.net/aslib/)). Evaluate regret to the
virtual best, improvement over the single best, timeout/penalty loss, and
feature-computation overhead. Avoid selecting only instances on which the
portfolio is known to be complementary.

## 13. Benchmark governance and lifecycle

### 13.1 Release contents

Each immutable release should include:

- manifests and cryptographic checksums;
- licenses/redistribution status and upstream links;
- generator definitions and seed/parameter records;
- point-set lineage/group IDs and split assignments;
- reference tours/bounds with provenance and validation status;
- feature definitions, values, cost, and computation version;
- task definitions, resource caps, failure semantics, and primary metrics;
- raw result schema and report-generation version;
- known issues and excluded/corrected artifacts.

If an upstream license does not allow redistribution, store metadata, checksum,
and a fetch instruction rather than copying the data.

### 13.2 Reference updates

Reference records are append-only. A new best tour changes
`best_known_upper`, not prior run objectives. Recompute derived gaps against a
chosen reference snapshot so historical tables remain reproducible. Certified
optimum status requires matching semantics and verification; solver reputation
is not a certificate.

### 13.3 Saturation policy

Track saturation by stratum: fraction of solvers hitting optimum, resolution of
quality differences, timeout rate, and feature-space coverage. Retain saturated
instances for correctness and historical trends but reduce their weight in
discriminating leaderboards. Add new diagnostic instances through declared
coverage gaps, not by quietly cherry-picking examples where a new solver wins.

## 14. Benchmark anti-patterns

Reject or qualify conclusions based on any of the following:

- only uniform random `[0,1]^2` instances at one or a few sizes;
- only TSPLIB, with it used repeatedly for hyperparameter tuning;
- mixing raw L2, `EUC_2D`, `CEIL_2D`, `GEO`, or `GEOM` results;
- calling a best-known tour optimal without a matching proof/bound;
- reporting best-of-many stochastic runs but charging one-run time;
- timing only neural inference while excluding decoding/repair/local search;
- comparing different postprocessing or candidate budgets as if only the model
  changed;
- treating repeated seeds as independent problem instances;
- averaging timeouts as if they finished at the cap;
- omitting invalid tours, memory failures, or unsupported instances;
- using `n` as the sole hardness explanation;
- optimizing the test suite against the method under evaluation;
- aggregating across families without per-family and tail results;
- presenting a rounded mean gap without raw per-instance rows;
- using transformed copies across train and test;
- claiming exactness after heuristic sparsification;
- changing reference values, generators, or instances in place.

## 15. Recommended research program

Build the benchmark in four scientific passes:

1. **semantic foundation:** objective oracles, micro/metamorphic corpus,
   reference hierarchy, and public-suite ingestion;
2. **explanatory design:** controlled family generators, structural features,
   matched continuations, and exact/heuristic control responses;
3. **coverage expansion:** feature-space audit, evolved/adversarial additions,
   higher/intrinsic dimension, and hidden OOD strata;
4. **longitudinal operation:** raw anytime records, immutable releases,
   reference updates, saturation monitoring, and portfolio/selection scenarios.

The result should answer more than “which solver has the smallest average
gap?” It should answer: which solver, for which Euclidean geometry, objective,
quality target, resource budget, and evidence level—and which mechanism causes
the difference.
