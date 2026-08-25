# Annotated sources and implementations

Research cutoff: 2026-08-02. Primary papers and official project/benchmark pages
are preferred. “Frontier” marks recent preprints or unsettled comparisons; they
are included to map current directions, not treated as established dominance.

## 1. Exact solution, relaxations, and geometry

### Concorde and the cutting-plane lineage

- [Concorde home](https://www.math.uwaterloo.ca/tsp/concorde.html) — official
  summary, source/manual links, solved-instance claims, callable-library scope,
  parallel capabilities and licensing notice. Reference practical exact solver.
- [Concorde downloads](https://www.math.uwaterloo.ca/tsp/concorde/downloads/downloads.htm)
  — official source/executables. The public source is for academic research;
  contact the authors for commercial use. Do not assume a permissive license.
- [Concorde documentation](https://www.math.uwaterloo.ca/tsp/concorde/DOC/index.html)
  — APIs across cuts, edge generation, Held–Karp, kd-trees, linkern, LP, tree
  search and verification. Useful as an architectural index.
- [Concorde GitHub mirror](https://github.com/matthelb/concorde) — convenient
  source browser, not the authoritative licensing/download page.
- Applegate, Bixby, Chvátal, and Cook,
  [*The Traveling Salesman Problem: A Computational Study*, chapter sample](https://assets.press.princeton.edu/chapters/s8451.pdf)
  — definitive account of the branch-and-cut engineering stack.
- [TSP optimization methods](https://www.math.uwaterloo.ca/tsp/methods/opt/opt.htm)
  and [papers](https://www.math.uwaterloo.ca/tsp/methods/papers/index.html) —
  official Waterloo explanations/bibliography.
- [QSopt / QSopt_ex](https://www.math.uwaterloo.ca/~bico/qsopt/) — LP and exact
  rational verification infrastructure used in certified computations.

### Bounds and exact algorithms

- Held and Karp (1970),
  [The Traveling-Salesman Problem and Minimum Spanning Trees](https://doi.org/10.1287/opre.18.6.1138)
  — Lagrangian 1-tree lower bound and node penalties. Foundational to both exact
  bounds and LKH candidate generation.
- Valenzuela and Jones (1997),
  [Estimating the Held–Karp lower bound for the geometric TSP](https://doi.org/10.1016/S0377-2217(96)00214-7)
  — geometric/random-instance behavior and efficient estimation context.
- de Berg, Bodlaender, Kisfaludi-Bak, and Kolay (2020),
  [An ETH-Tight Exact Algorithm for Euclidean TSP](https://arxiv.org/abs/1807.06933)
  — fixed-dimensional `2^{O(n^(1-1/d))}` exact algorithm and matching ETH
  exponent lower bound; theoretical rather than a practical competitor.
- Hougardy and Schroeder (2014),
  [Edge Elimination in TSP](https://arxiv.org/abs/1402.7301) — exact geometric/
  graph-theoretic elimination and reported Concorde acceleration.
- Zhong (2025 journal publication; earlier preprint),
  [Probabilistic analysis of edge elimination](https://arxiv.org/abs/1809.10469)
  — linear expected surviving edges for one elimination test under bounded-
  density random planar models.
- Vercesi, Gualandi, and Malucelli (2024/2025),
  [Local elimination algorithms for TSP](https://doi.org/10.1007/s12532-024-00262-y)
  — exact elimination/fixing experiments on large instances.
- Hougardy and Zhong (2021),
  [Hard to Solve Instances of the Euclidean TSP](https://doi.org/10.1007/s12532-020-00184-5)
  — tetrahedron instances demonstrating that a few hundred geometric points can
  be exact-solver-hard.
- Hoos and Stützle (2014),
  [On the empirical scaling of run-time for finding optimal solutions to the TSP](https://doi.org/10.1016/j.ejor.2014.04.042)
  — empirical Concorde scaling on random uniform Euclidean instances. Historical
  empirical model, not a worst-case theorem.
- Qian and Wang (2006),
  [How to compare sums of square roots](https://arxiv.org/abs/cs/0603002) —
  numerical/complexity background for exact comparison of raw Euclidean tour
  lengths in the bit model.

## 2. Approximation and special geometric regimes

- Arora (1998),
  [Polynomial Time Approximation Schemes for Euclidean TSP and other Geometric Problems](https://graphics.stanford.edu/courses/cs468-06-winter/Papers/arora-tsp.pdf)
  — random dissection, portals, patching and dynamic programming; foundational
  Euclidean PTAS.
- Mitchell (1999),
  [Guillotine Subdivisions Approximate Polygonal Subdivisions](https://doi.org/10.1137/S0097539796309764)
  — independent PTAS based on augmenting tours into recursively structured
  `m`-guillotine networks and dynamic programming.
- Kisfaludi-Bak, Nederlof, and Węgrzycki (JACM 2025),
  [A Gap-ETH-Tight Approximation Scheme for Euclidean TSP](https://doi.org/10.1145/3766548)
  — near-linear fixed-dimensional scheme with essentially optimal dependence on
  accuracy under Gap-ETH. Peer reviewed.
- Mömke and Zhou (2024),
  [A linear-time Gap-ETH-tight approximation scheme for planar Euclidean TSP](https://arxiv.org/abs/2411.02585)
  — planar `2^{O(1/epsilon)} n` frontier result. Preprint at cutoff.
- Trevisan (2000),
  [When Hamming Meets Euclid: The Approximability of Geometric TSP and Steiner Tree](https://doi.org/10.1137/S0097539799352735)
  — high/growing-dimensional hardness boundary; rules out treating fixed-
  dimensional PTAS results as dimension-uniform.
- Bartal, Gottlieb, and Krauthgamer (STOC 2012),
  [Low-Dimensionality Implies a PTAS](https://arxiv.org/abs/1112.0699) —
  randomized PTAS for TSP in arbitrary metrics of bounded doubling/intrinsic
  dimension. Motivates separating ambient from intrinsic dimension.
- Gawrychowski and Rusak (2014),
  [Euclidean TSP with Few Inner Points in Linear Space](https://arxiv.org/abs/1406.2154)
  — `O(n k^2 + k^{O(sqrt(k))})` linear-space FPT algorithm and quadratic
  bikernel for `k` points strictly inside the hull. Preprint/under-submission
  status on the source page; distinguish it from earlier few-inner work.
- Alkema, de Berg, van der Hofstad, and Kisfaludi-Bak (2024),
  [Euclidean TSP in a narrow strip](https://doi.org/10.1007/s00454-023-00609-7)
  — special-width geometric structure and algorithms.
- Rote (1992),
  [The N-line Traveling Salesman Problem](https://page.mi.fu-berlin.de/rote/Papers/abstract/The%2BN-line%2Btraveling%2Bsalesman%2Bproblem)
  — `n^N` dynamic programming for fixed `N` parallel lines and a corrected
  technical-report link for the almost-parallel generalization.
- Çela, Deineko, and Woeginger (2012),
  [The x-and-y-axes Travelling Salesman Problem](https://doi.org/10.1016/j.ejor.2012.06.036)
  — quadratic-time exact algorithm when all cities lie on the two coordinate
  axes.
- Deineko and Woeginger (1996),
  [The Convex-hull-and-k-line TSP](https://doi.org/10.1016/0020-0190(96)00125-1)
  — polynomial case with hull points and cities on a fixed number of suitably
  arranged almost-parallel inner segments.
- Eppstein,
  [The Delaunay triangulation does not always contain an optimal TSP tour](https://ics.uci.edu/~eppstein/junkyard/dt-not-tsp.html)
  — explicit counterexample; crucial caution against treating Delaunay
  sparsification as exact.
- Österman (2008),
  [Good triangulations yield near-optimal restricted tours empirically](https://doi.org/10.1016/j.cor.2006.03.025)
  — evidence for triangulations as heuristic sparse graphs, not proof-safe
  supersets.

## 3. Classical heuristic and population search

### LKH, k-opt, and large-scale local optimization

- Helsgaun (2000),
  [An Effective Implementation of the Lin–Kernighan TSP Heuristic](https://doi.org/10.1016/S0377-2217(99)00284-2)
  — alpha-nearness, 1-tree penalties and implementation choices behind LKH.
- [LKH official page](https://webhotel4.ruc.dk/~keld/research/LKH/) — source and
  documentation; lists LKH 2.0.11 dated June 2025 at the cutoff. Review license
  terms before integration.
- Ammann, Ostermann, Stiller, and de Wolff (2026),
  [A Speed-up for Helsgaun's TSP Heuristic by Relaxing the Positive Gain Criterion](https://doi.org/10.1016/j.cor.2026.107443)
  — mild temporary relaxation of an LK pruning rule; reports an average 13.6%
  time reduction for large instances at virtually unchanged quality over the
  compared LKH setup. Current peer-reviewed classical frontier.
- Helsgaun (2009),
  [General k-opt submoves for the Lin–Kernighan heuristic](https://doi.org/10.1007/s12532-009-0004-6)
  — stronger move representation and search.
- Taillard and Helsgaun (2019),
  [POPMUSIC for the Traveling Salesman Problem](https://doi.org/10.1016/j.ejor.2018.06.039)
  — repeated local subproblem optimization and candidate generation for large
  instances.
- Taillard (2022),
  [A linearithmic heuristic for the traveling salesman problem](https://doi.org/10.1016/j.ejor.2021.05.034)
  — randomized POPMUSIC-derived method tested beyond two billion cities;
  evidence for extreme scale, with a quality target far below long-budget LKH.
- Formella (2024),
  [Quasi-linear time heuristic to solve ETSP with low gap](https://doi.org/10.1016/j.jocs.2024.102424)
  — pair-center construction with empirical `O(n log n)` time and linear space;
  reports sub-1% average gap below 1,001 nodes and larger gaps at huge scale.
- Brodowsky, Hougardy, and Zhong (2023),
  [Approximation Ratio of k-Opt for ETSP](https://doi.org/10.1137/21M146199X)
  — for every fixed `k`, tight `Theta(log n / log log n)` worst-case ratio in
  two dimensions.
- Manthey and van Rhijn (2025),
  [Improved Smoothed Analysis of 2-Opt](https://doi.org/10.1007/s00453-025-01309-9),
  and Künnemann, Manthey, and Veenstra (2025),
  [Gaussian-noise smoothed analysis](https://doi.org/10.1007/s00453-025-01335-7)
  — polynomial expected-behavior results under explicit perturbation models;
  bridge theory and typical empirical behavior without erasing worst cases.
- Johnson and McGeoch,
  [Experimental Analysis of Heuristics for the STSP](https://www.cs.ubc.ca/~hutter/previous-earg/EmpAlgReadingGroup/TSP-JohMcg97.pdf)
  — classic empirical methodology and local-search comparison.
- Glover and Kochenberger et al. (2010 survey),
  [TSP heuristics: leading methods, implementations and advances](https://leeds-faculty.colorado.edu/glover/fred%20pubs/429%20-%20TSP%20-%20problem%20heuristics%20-%20leading%20methods%2C%20implementations%2C%20latest%20advances.pdf)
  — broad pre-neural review, including LK and stem-and-cycle lineages.

### EAX and portfolios

- Nagata and Kobayashi (1999),
  [Edge Assembly Crossover](https://www.jstage.jst.go.jp/article/jjsai/14/5/14_848/_article/-char/en)
  — original alternating AB-cycle recombination.
- Nagata (2007),
  [Fast EAX using localized optimization](https://www.jstage.jst.go.jp/article/tjsai/22/5/22_5_542/_article/)
  — scaling crossover/local repair to large instances.
- Nagata,
  [GA-EAX source](https://github.com/nagata-yuichi/GA-EAX) — public reference
  implementation distributed under the MIT license in the repository.
- Nikfarjam et al. (2025),
  [To Repair or Not to Repair? Investigating AB-Cycles for EAX](https://arxiv.org/abs/2505.00803)
  — 10,000-instance empirical study of EAX variants. Preprint at cutoff.
- Kotthoff, Kerschke, Hoos, and Trautmann (2015),
  [Improving the state of the art in inexact TSP solving](https://www.eecs.uwyo.edu/~larsko/papers/kotthoff_improving_2015.pdf)
  — algorithm selection/configuration and complementary solver behavior.

## 4. Neural construction and improvement

- Vinyals, Fortunato, and Jaitly (2015),
  [Pointer Networks](https://arxiv.org/abs/1506.03134) — variable-length
  attention/pointer output, including small TSP experiments.
- Bello et al. (2016/2017),
  [Neural Combinatorial Optimization with Reinforcement Learning](https://arxiv.org/abs/1611.09940)
  — policy-gradient construction and active search.
- Kool, van Hoof, and Welling (2019),
  [Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475)
  — attention-model baseline used throughout neural routing research.
- Kwon et al. (NeurIPS 2020),
  [POMO](https://proceedings.neurips.cc/paper/2020/hash/f231f2107df69eab0a3862d50018a9b2-Abstract.html)
  — symmetry-aware multiple optima/starts and augmentation.
- Hottung, Kwon, and Tierney (2021),
  [Efficient Active Search](https://arxiv.org/abs/2106.05126) — instance-specific
  test-time adaptation of limited model components.
- Choo et al. (NeurIPS 2022),
  [Simulation-Guided Beam Search](https://proceedings.neurips.cc/paper_files/paper/2022/hash/39b9b60f0d149eabd1fff2d7c5afc4-Abstract-Conference.html)
  — learned beam search with simulations.
- Drakulic et al. (NeurIPS 2023),
  [BQ-NCO](https://openreview.net/forum?id=BRqlkTDvvm) — quotienting state
  symmetries for better generalization.
- Luo et al. (NeurIPS 2023),
  [LEHD](https://arxiv.org/abs/2310.07985) — light encoder/heavy decoder,
  supervised subpath training and route reconstruction for size generalization.
- Costa et al. (ACML 2020),
  [Learning 2-opt Heuristics for the TSP](https://proceedings.mlr.press/v129/costa20a.html)
  — reinforcement learning for improvement actions.
- Kool et al. (2022),
  [Deep Policy Dynamic Programming](https://openreview.net/forum?id=OAMrSPRRxJx)
  and [code](https://github.com/wouterkool/dpdp) — learned edge policy guides a
  pruned DP beam; powerful hybrid but not exact after heuristic pruning.

## 5. Learning-augmented classical search

- Xin et al. (NeurIPS 2021),
  [NeuroLKH](https://proceedings.neurips.cc/paper_files/paper/2021/hash/3d863b367aa379f71c7afc0c9cdca41d-Abstract.html)
  — sparse GNN candidate and penalty prediction feeding LKH. The clearest
  template for safe, high-quality learned ETSP guidance.
- Zheng et al. (2022),
  [Reinforced Lin–Kernighan–Helsgaun](https://arxiv.org/abs/2207.03876) — RL-
  guided aspects of LKH search. Evaluate against identical LKH configuration.
- Ye et al. (NeurIPS 2023),
  [DeepACO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/883105b282fe15275991b411e6b200c5-Abstract-Conference.html)
  and [code](https://github.com/henry-yeh/DeepACO) — learned heuristic measure
  inside ant-colony construction/local search.
- Sciandra et al. (EJOR 2026),
  [Graph Convolutional Branch and Bound](https://doi.org/10.1016/j.ejor.2026.03.036)
  — unsupervised GNN optimality scores guide exact 1-tree/Concorde tree search;
  evidence for ML reducing nodes/time without becoming the proof mechanism.
- Xia et al. (ICML 2024),
  [Rethinking the Interface Between Neural and Classical Combinatorial Optimization](https://proceedings.mlr.press/v235/xia24f.html)
  — critical controlled analysis: simple heatmaps can rival learned ones under
  the same post-hoc search, and classical LKH remains extremely strong.
- Bi et al. (Constraints 2023),
  [Generalization in Neural Combinatorial Optimization](https://doi.org/10.1007/s10601-022-09327-y)
  — size/distribution generalization failures and evaluation cautions.

## 6. Generative and diffusion solvers

- Sun and Yang (NeurIPS 2023),
  [DIFUSCO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/0ba520d93c3df592c83a611961314c98-Abstract-Conference.html)
  — diffusion over combinatorial edge structure with test-time search.
- Li et al. (NeurIPS 2024),
  [Fast T2T: Optimization Consistency Speeds Up Diffusion-Based Training-to-Testing Solving for Combinatorial Optimization](https://proceedings.neurips.cc/paper_files/paper/2024/hash/352b13f01566ae34affacc60e98c16af-Abstract-Conference.html)
  — consistency-style fast generation and test-time gradient search; its LKH
  comparison is specifically a limited-time-budget claim.
- Wang et al. (NeurIPS 2025),
  [StruDiCO: Structured Denoising Diffusion with Gradient-free Inference-stage Boosting](https://papers.neurips.cc/paper_files/paper/2025/file/6728fcf94660c59c938319a6833a6073-Paper-Conference.pdf)
  — variable-absorption diffusion, constrained consistency sampling, and
  gradient-free objective-aware refinement. Stronger/faster than Fast T2T on its
  uniform-random protocols; still relies on decoding/2-opt and does not displace
  long-budget LKH.
- Min et al. (2024),
  [Improving Diffusion Models for TSP with Implicit Differentiation](https://arxiv.org/abs/2412.13858)
  — structure-aware tour distribution/test-time method. Preprint.
- [Leveraging Structural Constraints for Diffusion-based Neural TSP Solvers](https://arxiv.org/abs/2606.09343)
  — June 2026 PCI preprint: structural projection and 2-opt replace costly
  gradient refinement; promising fixed-budget result, not broad settled SOTA.
- [GeoRouteNet](https://arxiv.org/abs/2606.22776) — June 2026 geometry-enhanced
  non-autoregressive solver. Its TSPLIB aggregate is useful evidence about
  generalization still lagging classical high-quality search.
- [GES-TSP](https://arxiv.org/abs/2607.09708) — July 2026 learned graph
  sparsification. Frontier preprint; pruning is not exact-safe by itself.
- [GNN-based Algorithm Selection for TSP](https://arxiv.org/abs/2607.18632) —
  July 2026 comparison of cost/rank losses for choosing a budget-conditioned
  portfolio including Chained LK, EAX, LKH, MAOS and Concorde. Frontier preprint.

## 7. Large-scale and hierarchical learning

- Ye et al. (AAAI 2024),
  [GLOP](https://ojs.aaai.org/index.php/AAAI/article/view/30009) and
  [implementation](https://github.com/henry-yeh/GLOP) — global partitioning and
  local shortest-Hamiltonian-path construction.
- Pan et al. (AAAI 2023),
  [H-TSP](https://arxiv.org/abs/2304.09395) and
  [implementation](https://github.com/Learning4Optimization-HUST/H-TSP) —
  hierarchical RL for large ETSP.
- [GELD](https://arxiv.org/abs/2506.06634) — global encoder/local decoder with
  region-averaged linear attention, evaluated at very large scale. 2025 preprint;
  emphasizes speed/scalability rather than beating long-budget LKH quality.

## 8. Benchmarks, records, and numerical definitions

- Reinelt (1991),
  [TSPLIB—A Traveling Salesman Problem Library](https://doi.org/10.1287/ijoc.3.4.376)
  — original peer-reviewed library paper. Establishes the historical role and
  design of TSPLIB; it does not make the finite collection a representative
  sample of all ETSP.
- Reinelt,
  [TSPLIB95 specification](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf)
  — authoritative edge-weight rules and file format.
- [DIMACS 8th TSP Challenge](https://dimacs.rutgers.edu/archive/Challenges/TSP/about.html)
  — broad large-instance families and benchmark protocol.
- [Waterloo TSP data](https://www.math.uwaterloo.ca/tsp/data/index.html) — World,
  national, VLSI and art instances.
- [World TSP status](https://www.math.uwaterloo.ca/tsp/world/) — official best
  tour/lower-bound record and historical methods. Updated 2025-10-24 at cutoff.
- [National instance bound summary](https://www.math.uwaterloo.ca/tsp/world/summary.html)
  — best tours, Concorde lower bounds and reported gaps.

### Public data and generators

- [DIMACS 8th Challenge full protocol](https://dimacs.rutgers.edu/archive/Challenges/TSP/about.html)
  — primary source for the corpus counts and ranges: 34 large TSPLIB STSP
  instances, 26 uniform Euclidean instances from 1,000 to 10,000,000 nodes, 22
  clustered Euclidean instances from 1,000 to 100,000 nodes, plus seven
  nongeometric random-matrix cases. Also defines end-to-end time, peak RSS,
  multiple raw randomized runs, and no per-instance hand tuning.
- [Waterloo TSP data index](https://www.math.uwaterloo.ca/tsp/data/index.html) —
  primary catalog for national, VLSI, World, Mona Lisa, and USA sets; reports
  102 VLSI cases up to 744,710 cities and the 1,904,711-city World instance.
- [TSP Algorithm Selection project](https://tspalgsel.github.io/) — data,
  feature implementations, RUE/cluster/morph generators, Concorde/EAX/LKH/MAOS
  links, and publication list. Its released instance lineages are useful for
  selection research; group parents and morphs when making ML splits.
- [`tspgen` documentation](https://jakobbossek.github.io/tspgen/) — reproducible
  R package for topology-changing TSP mutations including explosion, cluster,
  axis-projection, and grid operators. Software is GPL-3 according to the
  project page; generated-artifact redistribution and provenance still need an
  explicit project policy.
- Bossek et al. (FOGA 2019),
  [Evolving Diverse TSP Instances by Means of Novel and Creative Mutation Operators](https://doi.org/10.1145/3299904.3340307)
  and [author PDF](https://www.jakobbossek.de/bibpdf/bossek_evolving_2019.pdf) —
  peer-reviewed motivation and operators for feature, topology, and performance
  diversity; warns implicitly against relying on small Gaussian moves from RUE
  parents.
- Bossek and Trautmann (LION 2016),
  [Evolving Instances for Maximizing Performance Differences of State-of-the-Art Inexact TSP Solvers](https://doi.org/10.1007/978-3-319-50349-3_4)
  — solver-discriminating evolution. Such cases are conditional on the control
  solvers/configurations used as fitness functions.
- Hougardy and Zhong (2021),
  [Hard to Solve Instances of the Euclidean TSP](https://doi.org/10.1007/s12532-020-00184-5)
  — open-access tetrahedron construction, subtour-LP integrality-ratio analysis,
  TSPLIB-format files, and reported million-fold Concorde slowdown versus
  similar-sized TSPLIB cases. Strong exact-stress source; uses rounded Euclidean
  length in the experiments.

### Instance features, diversity, and solver complementarity

- Mersmann et al. (LION 2012),
  [Local Search and the TSP: A Feature-Based Characterization of Problem Hardness](https://doi.org/10.1007/978-3-642-34413-8_9)
  — peer-reviewed feature analysis with 2-opt approximation ratio as the
  conditional hardness response.
- Mersmann et al. (2013),
  [A Novel Feature-Based Approach to Characterize Algorithm Performance for the TSP](https://doi.org/10.1007/s10472-013-9341-2)
  and [author manuscript](https://arxiv.org/abs/1208.2318) — expanded feature
  set and empirical performance characterization. Useful feature precedent,
  not a timeless universal hardness model.
- Smith-Miles and Bowly (2015),
  [Generating New Test Instances by Evolving in Instance Space](https://doi.org/10.1016/j.cor.2015.04.022)
  — peer-reviewed framework for feature-space coverage, algorithm footprints,
  and targeted gap filling. The paper's demonstration is graph coloring but it
  explicitly builds on evolved TSP research and its methodology transfers.
- Kerschke et al. (2018),
  [Leveraging TSP Solver Complementarity through Machine Learning](https://doi.org/10.1162/EVCO_a_00215)
  — peer-reviewed direct comparison of LKH, EAX, restart variants, and MAOS;
  establishes material per-instance complementarity and the value of retaining
  a solver-performance matrix.
- Bossek, Kerschke, and Trautmann (2020),
  [Anytime Behavior of Inexact TSP Solvers](https://arxiv.org/abs/2005.13289) —
  preprint showing that solver ranking changes with target quality and arguing
  for empirical runtime distributions/anytime features. Treat publication
  status accordingly.
- Seiler et al. (2020),
  [Towards Feature-free TSP Solver Selection](https://arxiv.org/abs/2006.00715)
  — preprint introducing a 6,000-instance selection dataset. Useful scale/data
  precedent; verify the exact version, metric, instance lineage, and publication
  status before reuse.
- Seiler et al. (SSCI 2023),
  [Using Reinforcement Learning for Per-Instance Algorithm Configuration on the TSP](https://doi.org/10.1109/SSCI52147.2023.10372008)
  — peer-reviewed 100,000-instance, 2,000-node LKH-configuration corpus derived
  from local World-TSP subsets. Valuable scale precedent with two important
  audit issues: overlapping/related subsets require group-aware splits, and the
  two cases not solved by Concorde used best-of-100 EAX tours rather than
  certified optima.

## 9. Benchmark science and statistical methodology

- Hooker (1994),
  [Needed: An Empirical Science of Algorithms](https://doi.org/10.1287/opre.42.2.201)
  — foundational argument for rigorous experiments and empirically based
  explanatory theories beyond worst/average-case analysis.
- Hooker (1995),
  [Testing Heuristics: We Have It All Wrong](https://doi.org/10.1007/BF02430364)
  — argues for controlled scientific experiments and explains why polished
  competitive comparisons alone do not identify mechanisms.
- Johnson (2002),
  [A Theoretician's Guide to the Experimental Analysis of Algorithms](https://dimacs.rutgers.edu/archive/Challenges/TSP/papers/experguide.pdf)
  — primary practical guide to experiment purpose, implementation, instance
  choice, analysis, reproducibility, and reporting.
- Johnson and McGeoch (2002),
  [Experimental Analysis of Heuristics for the STSP](https://dimacs.rutgers.edu/archive/Challenges/TSP/papers/stspchap.pdf)
  — 80-page TSP-specific experimental review associated with the DIMACS
  challenge; historically essential even though hardware and leading solvers
  have advanced.
- McGeoch (2012),
  [A Guide to Experimental Algorithmics](https://doi.org/10.1017/CBO9780511843747)
  — book-length methodology for correct, general, informative, and useful
  computational experiments.
- Dolan and Moré (2002),
  [Benchmarking Optimization Software with Performance Profiles](https://arxiv.org/abs/cs/0102001)
  — peer-reviewed distribution-of-performance-ratios method. Profiles are
  relative to the included solver set and require declared failure penalties.
- Moré and Wild (2009),
  [Benchmarking Derivative-Free Optimization Algorithms](https://doi.org/10.1137/080724083)
  — introduces data profiles for budget-constrained comparison. ETSP can adapt
  the framework using target gaps and defensible work normalization.
- Hansen et al. (2021 journal version),
  [COCO: A Platform for Comparing Continuous Optimizers](https://doi.org/10.1080/10556788.2020.1808977)
  and [preprint](https://arxiv.org/abs/1603.08785) — benchmark-platform
  rationale: immutable problem definitions, targets, runtime measures, and
  automated reproducibility. The domain is continuous optimization, but the
  methodology is directly useful.
- Hansen et al. (2016),
  [COCO: Performance Assessment](https://arxiv.org/abs/1605.03560) — fixed-
  target anytime assessment, runlength-based targets, empirical distributions,
  and simulated restarts. Methodological preprint; adapt rather than copy its
  black-box evaluation unit.
- Bischl et al. (Artificial Intelligence, 2016),
  [ASlib: A Benchmark Library for Algorithm Selection](https://arxiv.org/abs/1506.02465)
  and [official repository/format page](https://www.coseal.net/aslib/) —
  standard scenario representation for features, costs, solver performances,
  run status, and evaluation folds. Appropriate export precedent for an ETSP
  portfolio benchmark.
- Steele (1997),
  [Concentration of Measure and the Classical Theorems](https://epubs.siam.org/doi/10.1137/1.9781611970029.ch2)
  — treatment of the Beardwood-Halton-Hammersley theorem and subadditive
  Euclidean functionals. Supports generator-scale normalization; not a
  per-instance optimum or lower-bound oracle.

## 10. Alternative solver paradigms

- Amendola and Ricca (Journal of Logic and Computation, 2026),
  [New encodings of the Euclidean TSP in constraint answer set programming](https://doi.org/10.1093/logcom/exaf072)
  — modern CASP/difference-logic formulation and comparison context. Useful for
  expressiveness, not evidence of pure-ETSP dominance.
- General MILP documentation should be consulted for the selected solver's lazy
  constraint callback, indicator and numerical behavior. For pure ETSP, compare
  DFJ separation—not only MTZ—against specialized exact methods.
- Quantum/QUBO papers are intentionally not elevated to a “SOTA” list here:
  no source found at the cutoff establishes a competitive unrestricted ETSP
  quality-scale-time frontier against LKH/EAX/Concorde under matched accounting.

## 11. Reading order for a solver builder

1. Helsgaun 2000 and the LKH source/manual.
2. Held–Karp 1970, then Concorde's architecture/docs and the Applegate et al.
   computational study.
3. EAX original/localized papers and source.
4. POPMUSIC for decomposition and candidate generation.
5. Arora plus the modern Gap-ETH-tight PTAS paper for the theory boundary.
6. Hard Euclidean instances and exact edge-elimination papers to avoid
   random-instance tunnel vision.
7. NeuroLKH, the ICML 2024 post-hoc-search critique, and the generalization
   paper before investing in a large neural model.
8. GLOP/GELD and the diffusion frontier only after the classical benchmark and
   accounting protocol are fixed.
9. For benchmark construction specifically: Hooker 1994/1995, Johnson 2002,
   Johnson–McGeoch, DIMACS, Mersmann 2013, Smith-Miles–Bowly, Dolan–Moré,
   COCO performance assessment, and ASlib.
