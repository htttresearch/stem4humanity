# Annotated literature for ETSP benchmark construction

Research cutoff: **2026-08-02**. This bibliography is the benchmark-focused
subset of [SOURCES.md](SOURCES.md). It prioritizes primary papers and official
project/data pages, records publication status when material, and explains what
each source should—and should not—justify in an implementation.

## 1. Experimental algorithmics and benchmark validity

### Hooker, J. N. (1994). Needed: An Empirical Science of Algorithms

- **Publication:** *Operations Research* 42(2), 201–212.
- **Link:** [DOI 10.1287/opre.42.2.201](https://doi.org/10.1287/opre.42.2.201)
- **Use:** Foundation for treating experiments as a way to develop explanatory
  theories of performance beyond worst- and average-case analysis.
- **Do not infer:** That empirical results become general merely by using many
  runs; target populations and design still matter.

### Hooker, J. N. (1995). Testing Heuristics: We Have It All Wrong

- **Publication:** *Journal of Heuristics* 1, 33–42.
- **Link:** [DOI 10.1007/BF02430364](https://doi.org/10.1007/BF02430364)
- **Use:** Justification for controlled mechanism experiments alongside
  competitive benchmarks; highlights realism, fair comparison, and explaining
  why algorithms differ.

### Johnson, D. S. (2002). A Theoretician's Guide to the Experimental Analysis of Algorithms

- **Publication:** DIMACS implementation-challenge methodology chapter.
- **Link:** [official DIMACS PDF](https://dimacs.rutgers.edu/archive/Challenges/TSP/papers/experguide.pdf)
- **Use:** Experiment purpose, implementation/reporting discipline, instance
  choice, reproducibility, and relating experiments to theory/literature.

### Johnson, D. S., and McGeoch, L. A. (2002). Experimental Analysis of Heuristics for the STSP

- **Publication:** chapter in *The Traveling Salesman Problem and Its
  Variations*, pp. 369–443.
- **Link:** [near-final 80-page DIMACS PDF](https://dimacs.rutgers.edu/archive/Challenges/TSP/papers/stspchap.pdf)
- **Use:** TSP-specific historical experimental methodology, heuristic classes,
  testbeds, quality measures, and engineering analysis.
- **Caution:** Hardware and practical SOTA are historical; methodology remains
  valuable.

### McGeoch, C. C. (2012). A Guide to Experimental Algorithmics

- **Publication:** Cambridge University Press book.
- **Link:** [DOI 10.1017/CBO9780511843747](https://doi.org/10.1017/CBO9780511843747)
- **Use:** Full methodology for experiments intended to be correct, general,
  informative, and useful.

## 2. Performance summaries, budgets, and algorithm selection

### Dolan, E. D., and Moré, J. J. (2002). Benchmarking Optimization Software with Performance Profiles

- **Publication:** *Mathematical Programming* 91, 201–213.
- **Link:** [author preprint](https://arxiv.org/abs/cs/0102001)
- **Use:** Distribution of per-problem performance ratios; exposes efficiency
  and robustness without a fragile single mean.
- **Caution:** Profiles are relative to the included solver set. Failure-ratio
  policy must be declared and absolute outcomes retained.

### Moré, J. J., and Wild, S. M. (2009). Benchmarking Derivative-Free Optimization Algorithms

- **Publication:** *SIAM Journal on Optimization* 20(1), 172–191.
- **Link:** [DOI 10.1137/080724083](https://doi.org/10.1137/080724083)
- **Use:** Data profiles for the fraction of problems solved under normalized
  absolute budgets. Adapt to ETSP targets and defensible work normalization.

### Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., and Brockhoff, D. (2021). COCO platform

- **Publication:** *Optimization Methods and Software*.
- **Links:** [journal DOI](https://doi.org/10.1080/10556788.2020.1808977),
  [preprint](https://arxiv.org/abs/1603.08785)
- **Use:** Benchmark automation, immutable problems, target values, runtime as
  a primary measure, and unified deterministic/stochastic recording.
- **Caution:** COCO targets continuous black-box optimization; adopt principles,
  not domain assumptions.

### Hansen, N., Auger, A., Brockhoff, D., Tušar, D., and Tušar, T. (2016). COCO: Performance Assessment

- **Publication:** methodological preprint.
- **Link:** [arXiv:1605.03560](https://arxiv.org/abs/1605.03560)
- **Use:** Fixed-target anytime views, ECDFs, runlength targets, expected runtime
  under simulated restarts, and target/budget duality.
- **Caution:** ERT needs success probability and restart/censoring semantics.

### Bischl et al. (2016). ASlib: A Benchmark Library for Algorithm Selection

- **Publication:** *Artificial Intelligence* 237, 41–58.
- **Links:** [preprint](https://arxiv.org/abs/1506.02465),
  [official format/repository page](https://www.coseal.net/aslib/)
- **Use:** Scenario representation for instance features and costs, solver
  performance/status, cutoffs, and folds. Preferred interoperability model for
  an ETSP portfolio dataset.

## 3. Canonical and large public data

### Reinelt, G. (1991). TSPLIB—A Traveling Salesman Problem Library

- **Publication:** *ORSA Journal on Computing* 3(4), 376–384.
- **Links:** [library paper](https://doi.org/10.1287/ijoc.3.4.376),
  [TSPLIB95 specification](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf)
- **Use:** Historical instances, file schema, and authoritative edge-weight
  definitions.
- **Cautions:** `EUC_2D`/`EUC_3D` are per-edge rounded; the public fixed set is
  tuneable and not a representative sample. Verify upstream redistribution
  terms rather than assuming a software/data license.

### Johnson, McGeoch, Glover, and Rego. DIMACS 8th TSP Implementation Challenge

- **Publication:** official challenge/testbed/protocol site.
- **Link:** [official description](https://dimacs.rutgers.edu/archive/Challenges/TSP/about.html)
- **Use:** 34 large TSPLIB STSP cases, 26 uniform-square cases from 1k–10m, 22
  clustered cases from 1k–100k, standard generators, end-to-end time, peak RSS,
  raw stochastic runs, and no manual per-instance tuning.
- **Cautions:** Seven random matrix cases are not ETSP. Generator diversity and
  hardware protocol are historical.

### Waterloo TSP data and record pages

- **Publication:** official maintained data/project pages.
- **Links:** [data index](https://www.math.uwaterloo.ca/tsp/data/index.html),
  [national instances](https://www.math.uwaterloo.ca/tsp/world/countries.html),
  [national bounds/status](https://www.math.uwaterloo.ca/tsp/world/summary.html),
  [World TSP](https://www.math.uwaterloo.ca/tsp/world/),
  [VLSI](https://www.math.uwaterloo.ca/tsp/vlsi/index.html)
- **Use:** Application/geographic geometry, official tours and bounds, and
  scale through 1,904,711 cities.
- **Cautions:** Store retrieval date and upper/lower records separately;
  references change. World uses `GEOM`, not planar raw L2. Verify usage terms
  for each collection/artifact.

### Hougardy, S., and Zhong, X. (2021). Hard to Solve Instances of the Euclidean TSP

- **Publication:** *Mathematical Programming Computation* 13, 51–74; open
  access.
- **Link:** [DOI 10.1007/s12532-020-00184-5](https://doi.org/10.1007/s12532-020-00184-5)
- **Use:** Parameterized tetrahedron benchmark, subtour-LP ratio approaching
  `4/3`, unique-optimum modifications, and exact-solver hardness despite small
  `n`.
- **Caution:** Reported runtime hardness is Concorde/configuration/hardware and
  rounded-metric conditional, not an intrinsic scalar for all solvers.

## 4. Instance features, diversity, and evolved data

### Mersmann et al. (2012). Local Search and the TSP: A Feature-Based Characterization of Problem Hardness

- **Publication:** LION 6, LNCS 7219, 115–129.
- **Link:** [DOI 10.1007/978-3-642-34413-8_9](https://doi.org/10.1007/978-3-642-34413-8_9)
- **Use:** Establishes feature-based analysis with achieved 2-opt approximation
  ratio as the response.
- **Caution:** This is conditional 2-opt hardness, not universal instance
  hardness.

### Mersmann et al. (2013). A Novel Feature-Based Approach to Characterize Algorithm Performance for the TSP

- **Publication:** *Annals of Mathematics and Artificial Intelligence* 69,
  151–182.
- **Links:** [journal DOI](https://doi.org/10.1007/s10472-013-9341-2),
  [manuscript](https://arxiv.org/abs/1208.2318)
- **Use:** Expanded geometric/MST/angle/distance feature precedent and
  predictive analysis.
- **Caution:** Feature definitions and computation cost must be versioned; do
  not copy an old selected subset as a universal sufficient statistic.

### Smith-Miles, K., and Bowly, S. (2015). Generating New Test Instances by Evolving in Instance Space

- **Publication:** *Computers & Operations Research* 63, 102–113.
- **Link:** [DOI 10.1016/j.cor.2015.04.022](https://doi.org/10.1016/j.cor.2015.04.022)
- **Use:** Feature-space coverage, algorithm footprints, locating gaps, and
  evolving instances toward target regions.
- **Caution:** The paper demonstrates the method on graph coloring, while
  explicitly building on TSP studies. Projection to 2D is exploratory, not the
  mathematical definition of coverage.

### Bossek, J., and Trautmann, H. (2016). Evolving Instances for Maximum Solver Performance Differences

- **Publication:** LION 10, LNCS 10079, 48–59.
- **Link:** [DOI 10.1007/978-3-319-50349-3_4](https://doi.org/10.1007/978-3-319-50349-3_4)
- **Use:** Solver-discriminating generation and algorithm-selection research.
- **Caution:** Outputs are selected against a particular solver pair/config and
  should be labeled accordingly.

### Bossek et al. (2019). Evolving Diverse TSP Instances by Novel and Creative Mutation Operators

- **Publication:** FOGA XV, 58–71; peer reviewed.
- **Links:** [DOI 10.1145/3299904.3340307](https://doi.org/10.1145/3299904.3340307),
  [author PDF](https://www.jakobbossek.de/bibpdf/bossek_evolving_2019.pdf),
  [`tspgen` docs](https://jakobbossek.github.io/tspgen/)
- **Use:** Mutations for feature, performance, topology, and visual diversity;
  reproducible software precedent.
- **Licensing:** `tspgen` documents GPL-3 for the software. Define license and
  lineage metadata for generated artifacts separately.

### Kerschke et al. (2018). Leveraging TSP Solver Complementarity through Machine Learning

- **Publication:** *Evolutionary Computation* 26(4), 597–620.
- **Link:** [DOI 10.1162/EVCO_a_00215](https://doi.org/10.1162/EVCO_a_00215)
- **Use:** Direct LKH/EAX/restart/MAOS comparison, per-instance
  complementarity, selector motivation, and evidence for keeping full
  performance matrices.

### TSP Algorithm Selection project

- **Publication:** official project/data/software index.
- **Link:** [tspalgsel.github.io](https://tspalgsel.github.io/)
- **Use:** RUE/cluster/morph data with reported Concorde optima, feature tools,
  generators, and algorithm-selection publications.
- **Caution:** Verify exact file/objective/reference version and group morph
  lineages before reuse.

### Bossek, Kerschke, and Trautmann (2020). Anytime Behavior of Inexact TSP Solvers

- **Publication:** preprint at the cutoff.
- **Link:** [arXiv:2005.13289](https://arxiv.org/abs/2005.13289)
- **Use:** Evidence that LKH/EAX variant ranking changes with target quality;
  motivates trace and empirical runtime-distribution storage.

## 5. Mathematical theory that determines benchmark axes

### Steele, J. M. (1997). BHH and subadditive Euclidean functionals

- **Publication:** Chapter 2 of *Probability Theory and Combinatorial
  Optimization*.
- **Link:** [SIAM chapter](https://epubs.siam.org/doi/10.1137/1.9781611970029.ch2)
- **Use:** `OPT_n / n^((d-1)/d)` scaling and density dependence for random
  Euclidean samples; generator diagnostics and normalization.
- **Do not infer:** A per-instance certified objective or lower bound.

### de Berg, Bodlaender, Kisfaludi-Bak, and Kolay (2020). ETH-Tight Exact ETSP

- **Publication:** FOCS/SIAM Journal on Computing lineage.
- **Link:** [arXiv:1807.06933](https://arxiv.org/abs/1807.06933)
- **Use:** Fixed-`d` exact exponent `2^{Theta(n^(1-1/d))}` under ETH; motivates
  dimension-aware scaling.

### Trevisan, L. (2000). When Hamming Meets Euclid

- **Publication:** *SIAM Journal on Computing* 30(2), 475–485.
- **Link:** [DOI 10.1137/S0097539799352735](https://doi.org/10.1137/S0097539799352735)
- **Use:** Max-SNP-hardness in dimension `log n`; theoretical boundary between
  fixed-dimensional schemes and growing dimension.

### Hoos and Stützle (2014). Empirical Scaling of Time to Optimal TSP Solutions

- **Publication:** *European Journal of Operational Research*.
- **Links:** [journal DOI](https://doi.org/10.1016/j.ejor.2014.04.042),
  [related RUE testbed/results](https://iridia.ulb.ac.be/supp/IridiaSupp2017-010/index.html)
- **Use:** Repeated-instance empirical runtime modeling for Concorde and later
  LKH/EAX studies; the supplemental corpus documents 1,000 RUE instances per
  size from 500–2,000 and 100 per size from 2,500–4,500.
- **Caution:** A fitted RUE scaling law is historical and solver/configuration/
  hardware/distribution conditional, not average-case theory for ETSP.

### Bartal, Gottlieb, and Krauthgamer (2012). Low-Dimensionality Implies a PTAS

- **Publication:** STOC 2012; later journal DOI listed by the preprint.
- **Link:** [arXiv:1112.0699](https://arxiv.org/abs/1112.0699)
- **Use:** PTAS in bounded-doubling/intrinsic-dimension metrics; separates
  ambient from intrinsic dimension as benchmark fields.

### Gawrychowski and Rusak (2014). Few Inner Points

- **Publication:** preprint/under submission on the source page.
- **Link:** [arXiv:1406.2154](https://arxiv.org/abs/1406.2154)
- **Use:** `O(n k^2 + k^{O(sqrt(k))})` linear-space algorithm for exact inner
  count `k`; motivates hull/inner-count continuations.

### Alkema, de Berg, van der Hofstad, and Kisfaludi-Bak (2024). ETSP in Narrow Strips

- **Publication:** *Discrete & Computational Geometry*.
- **Links:** [journal DOI](https://doi.org/10.1007/s00454-023-00609-7),
  [full preprint](https://arxiv.org/abs/2003.09948)
- **Use:** Width-parameterized regimes, sparse/random models, hypercylinders,
  and the tight `delta <= 2 sqrt(2)` bitonic threshold under distinct integer
  x-coordinates.

### Rote, G. (1992). The N-line TSP

- **Publication:** *Networks* 22, 91–108.
- **Link:** [author page, paper, and erratum](https://page.mi.fu-berlin.de/rote/Papers/abstract/The%2BN-line%2Btraveling%2Bsalesman%2Bproblem)
- **Use:** `n^N` dynamic program for fixed parallel-line count and correctly
  delimited almost-parallel generalization.

### Brodowsky, Hougardy, and Zhong (2023). Approximation Ratio of k-Opt for ETSP

- **Publication:** *SIAM Journal on Computing*.
- **Link:** [DOI 10.1137/21M146199X](https://doi.org/10.1137/21M146199X)
- **Use:** Tight `Theta(log n / log log n)` worst-case approximation ratio for
  every fixed `k`; reason to retain adversarial local-search stress.

### Manthey and van Rhijn; Künnemann, Manthey, and Veenstra (2025). Smoothed 2-opt

- **Publications:** *Algorithmica* 87, peer reviewed/open access.
- **Links:** [improved Gaussian runtime analysis](https://doi.org/10.1007/s00453-025-01309-9),
  [Gaussian runtime and approximation analysis](https://doi.org/10.1007/s00453-025-01335-7)
- **Use:** Theoretical reason for perturbation/noise ladders between adversarial
  and random cases.

### Zhong, X. (2025 journal lineage). Probabilistic Edge Elimination

- **Publication:** preprint link contains history and result; check final
  journal metadata when pinning the implementation bibliography.
- **Link:** [arXiv:1809.10469](https://arxiv.org/abs/1809.10469)
- **Use:** Linear expected surviving edges for a specific exact elimination
  test under bounded-density planar random models.
- **Caution:** Not a proof that kNN/Delaunay/learned candidates are exact-safe.

## 6. ML evaluation and leakage

### Seiler et al. (2020). Towards Feature-free TSP Solver Selection

- **Publication:** preprint/PPSN research lineage; verify the exact version used
  for any imported artifacts.
- **Link:** [arXiv:2006.00715](https://arxiv.org/abs/2006.00715)
- **Use:** A 6,000-instance solver-selection dataset and feature-free selection
  precedent.
- **Caution:** “largest” claims are time- and definition-dependent. Audit
  objective semantics, generation, exact references, and parent/morph groups.

### Seiler et al. (SSCI 2023). Reinforcement Learning for Per-Instance Algorithm Configuration on the TSP

- **Publication:** 2023 IEEE Symposium Series on Computational Intelligence.
- **Links:** [IEEE DOI 10.1109/SSCI52147.2023.10372008](https://doi.org/10.1109/SSCI52147.2023.10372008),
  [author manuscript](https://research.utwente.nl/files/346799492/Using_Reinforcement_Learning_for_Per-Instance_Algorithm_Configuration_on_the_TSP.pdf)
- **Use:** Large-data precedent: 100,000 instances of 2,000 nodes for LKH
  per-instance configuration.
- **Critical caution:** The paper constructs many local subsets from the single
  World TSP using nearby-city pools, so overlap/lineage grouping is essential.
  It reports 99,998 Concorde optima but substitutes best-of-100 EAX results for
  two unsolved cases; those two are best-known upper bounds, not certified
  optima, regardless of the downstream label.

### Bi et al. (2023). Learning the TSP Requires Rethinking Generalization

- **Publication:** *Constraints* 28.
- **Link:** [DOI 10.1007/s10601-022-09327-y](https://doi.org/10.1007/s10601-022-09327-y)
- **Use:** Controlled account of size/distribution generalization in neural
  combinatorial optimization; basis for separating OOD axes.

### Xia et al. (ICML 2024). Rethinking the Interface Between Neural and Classical CO

- **Publication:** Proceedings of Machine Learning Research 235.
- **Link:** [PMLR paper](https://proceedings.mlr.press/v235/xia24f.html)
- **Use:** Matched post-hoc-search comparisons, simple heatmap controls, and
  evidence that representation, decoder, and local search must be separated.

### Zhang et al. (AAAI 2022). Hardness-Adaptive Curriculum

- **Publication:** AAAI 2022, peer reviewed.
- **Links:** [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/20899),
  [preprint](https://arxiv.org/abs/2204.03236)
- **Use:** Learned/generative hardness-adaptive sampling precedent.
- **Caution:** Training curriculum and final benchmark must remain separate;
  generation against a model can leak its failure profile.

## 7. Implementation/licensing notes

- [Concorde's official download page](https://www.math.uwaterloo.ca/tsp/concorde/downloads/downloads.htm)
  states academic-research use for the public source and asks commercial users
  to contact the authors. Treat it as an external control unless terms for the
  target use are resolved.
- [GA-EAX's public repository](https://github.com/nagata-yuichi/GA-EAX) contains
  an MIT license at the research cutoff; pin a commit/release and archive its
  configuration.
- [LKH's official page](https://webhotel4.ruc.dk/~keld/research/LKH/) is the
  authoritative download/manual source. Review its actual distribution/license
  terms rather than inferring them from source availability.
- Dataset access does not automatically grant redistribution. The benchmark
  registry therefore needs `usage_status`, `redistribution_status`, source URL,
  retrieval date, and original checksum per artifact.
