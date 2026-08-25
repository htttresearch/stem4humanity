# ETSP benchmark implementer specification

Research cutoff: **2026-08-02**. This is a build specification for a future
implementer agent. It intentionally contains no implementation. It defines
interfaces, invariants, artifact schemas, processing stages, reports, and a
phased plan derived from [BENCHMARK_RESEARCH.md](BENCHMARK_RESEARCH.md) and
[BENCHMARK_CATALOG.md](BENCHMARK_CATALOG.md).

## 1. Required product

Build a versioned ETSP benchmarking system that can:

1. acquire or generate immutable point-set instances;
2. assign exact objective semantics and lineage-aware data splits;
3. ingest tours and bounds without confusing their reference status;
4. run deterministic, stochastic, classical, exact, learned, and external
   solvers behind one trace-producing contract;
5. measure end-to-end time, work, memory, resources, failures, and anytime
   quality;
6. compute solver-independent and charged probing features;
7. produce stratified raw tables, uncertainty estimates, profiles, and
   auditable reports;
8. freeze benchmark releases so old results can always be regenerated.

The product is a laboratory and evidence store, not only a leaderboard UI.

## 2. Non-negotiable design decisions

### 2.1 One coordinate set may define multiple instances

Instance identity includes objective semantics. Raw L2 and TSPLIB `EUC_2D` on
the same coordinates receive different `instance_id` values and a common
`coordinate_parent_id`. `GEO`/`GEOM` do not enter canonical planar-ETSP
aggregates.

### 2.2 References are records, not fields to overwrite

Upper bounds, lower bounds, and certified optima are append-only records with
source and timestamp. Reports select a named `reference_snapshot_id`. A new
best tour updates a snapshot but does not mutate old run results.

### 2.3 A run emits an incumbent trace

Terminal-only wrappers are incomplete for anytime-capable solvers. At minimum,
each strict improvement records monotonic time, objective, and optional work
counters. The harness validates feasibility before accepting an incumbent.

### 2.4 End-to-end and solve-only are separate clocks

End-to-end includes input reading, model loading according to cold/warm policy,
distance/candidate preprocessing, decoding, repair, and postprocessing. A
solve-only clock may be reported additionally but never substituted silently.

### 2.5 Lineage groups precede splits

No row-random splitting. Coordinate transforms, city permutations, objective
siblings, morphs, subsamples, and mutations receive lineage/group metadata
before any train/validation/test assignment.

### 2.6 Raw results precede aggregates

The irreducible record is per run/per instance. Every chart or score is a pure,
versioned derivation from raw artifacts plus a reference snapshot.

## 3. Suggested subsystem boundaries

```text
benchmark registry
  -> source adapters / generators
  -> canonical instance store
  -> objective oracles and tour validator
  -> reference registry
  -> feature pipeline
  -> suite/split compiler
  -> solver adapters
  -> resource-controlled runner
  -> raw run/trace store
  -> analysis and report builder
  -> immutable release manifest
```

The objective oracle and validator are shared by acquisition, references,
running, and reporting. Solvers must not be trusted to score their own tours for
leaderboard purposes.

## 4. Identifiers and digests

Use content-addressed, human-readable IDs where practical.

```text
coordinate_parent_id = digest(canonical coordinate multiset + dimension)
instance_id          = digest(coordinate_parent_id + objective_spec_id)
encoding_id          = digest(original bytes or canonical serialized bytes)
lineage_group_id     = assigned root for transforms/morphs/descendants
objective_sibling_id = shared across metrics derived from the same coordinates
solver_id            = semantic solver family/version
config_id            = digest(solver code/model/parameters/resource policy)
run_id               = digest(release + instance + config + seed + repetition)
reference_id         = digest(instance + kind + value/tour/bound + provenance)
release_id           = semantic version plus manifest digest
```

Canonical coordinate hashing must specify:

- coordinate scalar encoding and endianness;
- treatment of `-0`, NaN, infinity, and decimal text;
- whether city order is normalized for point-set identity;
- duplicate multiplicities;
- dimension and coordinate-unit metadata.

Retain original byte digests independently; canonicalization must never erase
the ability to identify an upstream file.

## 5. Objective specification

Define an explicit `objective_spec` registry. Minimum modes:

```yaml
raw_l2_f64:
  family: euclidean
  dimension: dynamic
  edge: sqrt(sum((xi-xj)^2))
  accumulation: declared implementation plus independent audit path
  result_type: float64

tsplib_euc_2d:
  family: rounded_euclidean
  dimension: 2
  edge: nint(sqrt(dx^2 + dy^2))
  result_type: integer

tsplib_euc_3d:
  family: rounded_euclidean
  dimension: 3
  edge: nint(sqrt(dx^2 + dy^2 + dz^2))
  result_type: integer

tsplib_ceil_2d:
  family: ceiling_euclidean
  dimension: 2
  edge: ceil(sqrt(dx^2 + dy^2))
  result_type: integer
```

Implement other TSPLIB and Waterloo modes as distinct specs by the official
definition. Each spec has a semantic version and conformance examples. Do not
name an approximation `raw_l2_exact`.

## 6. Core data contracts

The syntax below is illustrative; the implementation may choose JSON, Parquet,
Arrow, or a relational catalog, provided lossless export is available.

### 6.1 Instance manifest

```json
{
  "schema_version": "etsp.instance.v1",
  "instance_id": "...",
  "display_name": "synthetic/gaussian_mix/n1000/seed42",
  "coordinate_parent_id": "...",
  "lineage_group_id": "...",
  "objective_sibling_group_id": "...",
  "n": 1000,
  "ambient_dimension": 2,
  "coordinate": {
    "storage": "float64",
    "units": "normalized",
    "file_digest": "sha256:...",
    "canonical_digest": "sha256:..."
  },
  "objective_spec_id": "raw_l2_f64@1",
  "class": "canonical_etsp",
  "source": {
    "kind": "generated",
    "name": "gaussian_mixture",
    "version": "1.0.0",
    "upstream_url": null,
    "retrieved_at": null,
    "license_spdx": "project-defined",
    "redistribution": "allowed"
  },
  "generator": {
    "prng": "named algorithm/version",
    "seed": "42",
    "parameters": {},
    "parent_instance_ids": []
  },
  "strata": {
    "family": "gaussian_mixture",
    "size_band": "1k",
    "dimension_band": "2d",
    "degeneracy": "none"
  },
  "split": "test_parameter_ood",
  "created_by_release": "..."
}
```

Coordinates themselves should live in an immutable blob/object store rather
than be repeated in every manifest.

### 6.2 Lineage event

```json
{
  "child_coordinate_parent_id": "...",
  "parent_coordinate_parent_ids": ["..."],
  "operation": "gaussian_perturbation",
  "operation_version": "1",
  "parameters": {"sigma_over_median_nn": 0.01},
  "seed": "...",
  "invertible": false
}
```

Lineage is a directed acyclic graph. The suite compiler rejects split
assignments that place connected lineage components in incompatible ML splits.

### 6.3 Tour artifact

```json
{
  "schema_version": "etsp.tour.v1",
  "tour_id": "sha256:...",
  "instance_id": "...",
  "encoding": "zero_based_city_permutation_without_repeated_start",
  "cities_blob": "...",
  "feasible": true,
  "local_objective": "12345",
  "audit_objective": "12345",
  "validator_version": "...",
  "producer": {"kind": "run", "id": "..."}
}
```

Integer objectives are decimal strings or arbitrary-precision integers in the
storage layer. Floating objectives retain a hexadecimal/bit representation in
addition to a display decimal if exact reproduction matters.

### 6.4 Reference record

```json
{
  "schema_version": "etsp.reference.v1",
  "reference_id": "...",
  "instance_id": "...",
  "kind": "certified_optimum | upper_bound | lower_bound | asymptotic_diagnostic",
  "value": "12345",
  "tour_id": "... or null",
  "status": "verified | imported_unverified | superseded | disputed",
  "proof": {
    "method": "concorde_branch_and_cut",
    "artifact_digest": "...",
    "independent_verifier": "..."
  },
  "provenance": {
    "source_url": "...",
    "source_name": "...",
    "retrieved_at": "...",
    "attribution": "..."
  },
  "created_at": "..."
}
```

An optimum is not represented by `kind=upper_bound` plus a comment. The
reference snapshot resolves the best currently accepted records without
destroying superseded history.

### 6.5 Solver configuration

```json
{
  "schema_version": "etsp.solver_config.v1",
  "solver_id": "lkh@upstream-version",
  "config_id": "sha256:...",
  "source_digest": "...",
  "binary_or_container_digest": "...",
  "model_digest": null,
  "parameters": {},
  "objective_modes_supported": ["tsplib_euc_2d@1"],
  "randomness": {"seed_interface": "...", "deterministic_claim": false},
  "threads": 1,
  "accelerators": [],
  "preprocessing_policy": "charged_end_to_end",
  "postprocessing": [],
  "warm_cold_policy": "cold_process_per_instance",
  "adapter_version": "..."
}
```

Archive upstream parameter files verbatim as blobs in addition to parsed
fields. If a solver internally changes parameters based on the instance, record
the automatic policy and charge its cost.

### 6.6 Run record and trace

```json
{
  "schema_version": "etsp.run.v1",
  "run_id": "...",
  "release_id": "...",
  "instance_id": "...",
  "config_id": "...",
  "seed": "7",
  "repetition": 0,
  "budget": {
    "wall_seconds": 10.0,
    "cpu_seconds": null,
    "memory_bytes": 8589934592,
    "work_limit": null
  },
  "environment_id": "...",
  "timing": {
    "start_monotonic": "opaque",
    "parse_seconds": 0.01,
    "preprocess_seconds": 0.12,
    "model_load_seconds": 0.0,
    "search_seconds": 9.84,
    "postprocess_seconds": 0.02,
    "end_to_end_seconds": 10.01,
    "cpu_seconds": 9.97
  },
  "resource": {
    "peak_rss_bytes": 0,
    "gpu_peak_bytes": null,
    "gpu_seconds": null,
    "energy_joules": null
  },
  "status": "success | timeout | memory_limit | crash | invalid_tour | numerical_error | unsupported",
  "best_tour_id": "...",
  "best_objective": "12345",
  "trace_blob": "...",
  "work_counters": {},
  "stdout_digest": "...",
  "stderr_digest": "..."
}
```

Trace events use monotonic elapsed time and never replace an incumbent with a
worse objective:

```json
{"t": 0.031, "objective": "13001", "tour_id": "...", "work": {}}
```

The wrapper can record raw solver events, but the accepted trace contains only
harness-validated feasible incumbents.

### 6.7 Feature record

```json
{
  "instance_id": "...",
  "feature_set_id": "etsp_geometry_tier1@1",
  "values": {"hull_fraction": 0.12, "mst_weight_bhh_norm": 0.91},
  "status_by_feature": {},
  "timing_seconds": 0.04,
  "peak_rss_bytes": 0,
  "extractor_digest": "...",
  "charged_at_inference": false
}
```

Tier-2 probing features set `charged_at_inference=true` when used by an online
selector or solver.

## 7. Acquisition and generation workflow

### 7.1 Source adapter contract

Each public-source adapter performs:

1. retrieve or locate an upstream artifact without changing it;
2. verify expected checksum if supplied;
3. record retrieval date, URL, and usage/redistribution status;
4. parse into coordinates/objective without discarding the original bytes;
5. create one manifest per actual objective instance;
6. import reference artifacts as unverified records;
7. validate and rescore tours through the shared oracle;
8. promote reference status only under the declared verification rule.

The adapter must never infer `EUC_2D` from filename or visual appearance.

### 7.2 Generator contract

A generator takes a fully materialized parameter record and seed and returns
coordinates plus realized diagnostics. It must not read wall clock, global RNG,
host name, or mutable defaults. Re-running the same generator version,
parameters, and PRNG version yields identical canonical coordinates.

Generators validate realized conditions. If a “convex plus `k` inner” draw
actually changes hull membership, either retry under a logged deterministic
rule or store the realized `k` and mark the intended condition failed; do not
silently mislabel it.

### 7.3 Suite compiler

The compiler consumes declarative cells, source filters, sample counts, split
rules, and a weighting definition. It emits an immutable list of `instance_id`
values with strata and weights. Compilation checks:

- no missing/ambiguous objective spec;
- no lineage leakage;
- no duplicate canonical instance in the same independent-sample count;
- no noncanonical objective in a canonical ETSP aggregate;
- adequate reference status for the declared task;
- family/cell count and coverage summary.

## 8. Solver-adapter contract

Adapters expose capabilities rather than pretending all solvers are identical:

```text
supported objective specs and dimensions
exact / upper-bound / lower-bound / constructor / selector roles
interruptible and incumbent-streaming capabilities
thread/GPU controls
seed control quality
input/output formats
warm-state/model-loading behavior
native work counters
proof/log artifacts
```

The harness owns resource caps and final scoring. Adapters translate, launch,
capture outputs, and expose incumbent events. Unsupported input is a recorded
status, not a bad objective or omitted row.

### 8.1 Exact solvers

Capture first-incumbent time, best-incumbent time, final lower bound, time from
best incumbent to proof, branch nodes, LP iterations, cuts by family when
available, priced/generated edges, peak tree memory, and proof/check artifact.
The distinction between finding and proving is mandatory.

### 8.2 Stochastic heuristics

Every seed is an individual run. A best-of-`r` configured solver is represented
as one aggregate configuration whose total time/compute includes all `r` runs,
or as a derived portfolio over raw runs—not as a free selection of the best row.

### 8.3 Learned solvers

Record model/training-data release, parameter count, precision, batch size,
augmentations, samples, device transfers, decoder, repair, local search, and
postprocessing. Maintain matched configurations for representation-only,
decoder-only, identical-search, and complete-system ablations.

## 9. Runner and resource policy

### 9.1 Clocks

Use a monotonic high-resolution wall clock. Capture process-tree CPU time where
available. Define whether timeout starts before process launch, file reading, or
adapter conversion; the default end-to-end benchmark starts before all
instance-specific work.

### 9.2 Memory

Measure peak resident memory over the complete process tree, plus accelerator
peak allocation/reservation when applicable. Classify a memory-limit kill
separately from crash/timeout. Avoid swap; record if the environment cannot
guarantee that.

### 9.3 Environment record

Store CPU/GPU model, logical/physical core availability, RAM, OS/kernel,
container/runtime, compiler, math/LP/CUDA libraries, frequency/power policy if
controlled, NUMA/affinity, filesystem/cache policy, and relevant environment
variables after redaction. Use an immutable `environment_id`.

### 9.4 Cold and warm protocols

Publish two separately named task modes where relevant:

- `cold_end_to_end`: fresh process/model and uncached instance-specific state;
- `warm_service`: preloaded model/binary service, steady-state requests, but all
  per-instance preprocessing/decoding remains charged.

Do not average cold and warm observations.

## 10. Metric engine

### 10.1 Per-run derived fields

Given a selected reference snapshot:

```text
feasible
objective
gap_to_certified_optimum, if certified
gap_to_best_known_upper, if available
certified_gap_from_lower, if available
target_hit_time for each target
quality_at each budget checkpoint
peak memory and resource use
failure/status category
```

If the run stops before a checkpoint, use the best incumbent already present;
if none exists, record missing/no-feasible status rather than infinity in raw
data. Aggregation defines its penalty explicitly.

### 10.2 Standard target grid

Targets should be relative gaps when a positive reference is valid:

```text
0 (certified optimum only), 1e-6, 1e-5, 1e-4, 5e-4,
1e-3, 2.5e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1
```

Adapt or omit targets that are below objective resolution for rounded small
instances. A “0” target means equality with a certified optimum under exact
integer semantics or a separately declared raw-L2 equality policy; it is not
gap-to-best-known zero.

### 10.3 Standard time grid

Use a logarithmic grid appropriate to each suite, for example:

```text
10 ms, 30 ms, 100 ms, 300 ms, 1 s, 3 s, 10 s, 30 s,
1 min, 3 min, 10 min, 30 min, 1 h, then task-specific long caps
```

Very small time points require clock-resolution and startup validation. Huge
instances may start at seconds/minutes; preserve input time.

## 11. Replication and analysis defaults

### 11.1 Run counts

- deterministic objective/proof: one result run, plus multiple timing repeats
  when timing claims matter;
- stochastic pilot: at least 10 seeds;
- close final claims: 30 or more seeds when instance cost permits, or an
  adaptive precision rule frozen before comparison;
- prioritize more independent instances over more seeds once within-instance
  variance is adequately estimated.

### 11.2 Inference units

Default broad-corpus intervals use paired hierarchical bootstrap resampling:
family/cell -> instance -> run seed. A report must state its estimand and
weights. For a fixed named public collection, resampling instances describes
uncertainty over that collection and should not be overinterpreted as sampling
from all ETSP.

### 11.3 Required report views

```text
exact: solved/proved fraction, proof-time survival, time-to-incumbent vs proof,
       nodes/cuts/bounds, memory
anytime: median and p90 gap vs log time, target-hit ECDF/survival,
         per-family heatmap, performance profile, raw tails
latency: cold/warm distributions, throughput, quality before/after matched LS
scale: time, work, peak/per-node memory vs n by family/dimension
ML: IID and each OOD axis separately, complete compute, matched decoder/search
candidate: build time/memory/degree/connectivity/recall/widening/final quality
portfolio: virtual-best, single-best, selector regret, feature cost, failures
```

Every report links to raw run IDs and the exact release/reference snapshot.

## 12. Benchmark release manifest

```yaml
schema_version: etsp.release.v1
release_id: etsp-bench-0.1.0+manifestdigest
research_cutoff: 2026-08-02
instance_manifests: sha256:...
suite_definitions: sha256:...
lineage_graph: sha256:...
objective_registry: sha256:...
reference_snapshot: sha256:...
feature_registry: sha256:...
licenses_and_sources: sha256:...
primary_tasks:
  - semantic_micro
  - exact_structure
  - anytime_quality
known_issues: []
supersedes: null
```

Release creation verifies every referenced blob exists and every digest
matches. Release objects are read-only. Corrections create `0.1.1` or a declared
new version, with a machine-readable difference report.

## 13. Phased implementation plan

This plan is ordered by epistemic dependency: later performance claims are not
trustworthy until earlier semantic layers exist.

### Phase 0 — decisions and repository fit

1. Locate existing instance, tour, solver, and benchmark abstractions.
2. Choose artifact storage (files/Parquet/database) and content-digest policy.
3. Define supported objective modes for the first release.
4. Define license/provenance fields and nonredistributable-source behavior.
5. Freeze the v1 schemas and naming conventions above.

**Exit:** design note maps every required field to a storage/API owner; no
objective is represented by an informal string.

### Phase 1 — semantic core

1. Implement objective registry and independent tour scoring.
2. Implement tour feasibility/normalization and objective audit path.
3. Implement instance/encoding digests and immutable blob handling.
4. Materialize `semantic_micro` and metamorphic lineages.
5. Produce a real pipeline report that imports, scores, and validates the micro
   corpus across all first-release objective modes.

**Exit:** all corpus tours and references rescore consistently; expected
metamorphic relations are reported, and failures are explicit.

### Phase 2 — sources and references

1. Add TSPLIB, DIMACS, and Waterloo adapters with checksums/provenance.
2. Implement append-only reference registry and snapshots.
3. Add exact-reference pipeline for small generated cases.
4. Add tetrahedron acquisition/generation with construction parameters.
5. Emit a public-source inventory and metric/reference-status audit.

**Exit:** no imported reference is called optimal without verified status; the
same coordinates under distinct metrics produce distinct instance IDs.

### Phase 3 — generator and suite compiler

1. Implement baseline, cluster, lattice, hull/layer, strip/line, manifold,
   degeneracy, and transformation stages from the catalog.
2. Add deterministic generator records and realized-condition validation.
3. Implement lineage graph and group-aware split compiler.
4. Compile pilot `exact_structure`, `anytime_quality`, and ML/OOD suites.
5. Audit factor and feature coverage before scaling counts.

**Exit:** rerunning a manifest reproduces coordinate digests; no lineage crosses
forbidden splits; suite counts/weights are reproducible.

### Phase 4 — solver runner

1. Define adapter capabilities and start with simple internal controls.
2. Add official external LKH, GA-EAX, and Concorde adapters as licensing/build
   permits, archiving exact parameter files and versions.
3. Add process-tree timing, memory caps, seed control, and status taxonomy.
4. Add validated incumbent trace ingestion.
5. Execute the complete generate/run/report pipeline on a small real subset.

**Exit:** raw records alone reproduce objective/time/status tables; invalid
outputs never enter incumbent traces.

### Phase 5 — statistics and reports

1. Implement fixed-budget/target derivations.
2. Implement stratified summaries, hierarchical paired intervals, performance
   profiles, ECDF/survival views, and scaling plots.
3. Make timeouts/censoring and missing values visible.
4. Add reference-snapshot selection and recomputation of derived gaps.
5. Publish machine-readable and human-readable reports from one result set.

**Exit:** every displayed number traces to release, reference snapshot, raw run
IDs, and analysis version.

### Phase 6 — features, coverage, and portfolio data

1. Implement feature tiers with cost/status records.
2. Audit public/generated corpus coverage and redundancy.
3. Add feature-fill and portfolio-discriminating evolved instances without
   using final evaluated configurations.
4. Export an ASlib-compatible or equivalent algorithm-selection scenario.
5. Quantify virtual-best opportunity, feature cost, and OOD performance.

**Exit:** additions demonstrably fill declared coverage gaps; selection data
uses leakage-safe folds and includes feature computation cost.

### Phase 7 — hidden and longitudinal releases

1. Establish sealed seed/parameter management and submission freeze protocol.
2. Publish release manifests and difference reports.
3. Add reference-update and dispute workflow.
4. Monitor saturation, coverage, and unsupported/failure rates by stratum.
5. Create new releases without mutating old ones.

**Exit:** a historical report remains bit-for-bit or numerically reproducible
after references and benchmark versions advance.

## 14. First useful release scope

Avoid attempting the complete catalog immediately. A credible `0.1` should
contain:

- raw L2 and TSPLIB `EUC_2D` objective specs;
- `semantic_micro` with exact references and metamorphic groups;
- correctly filtered TSPLIB planar Euclidean instances;
- DIMACS RUE and clustered subsets at feasible sizes;
- generated uniform, Gaussian-mixture, grid+jitter, convex-plus-inner,
  narrow-strip, rings, degeneracy, and tetrahedron strata;
- exact references for small cases and explicit bound status elsewhere;
- nearest neighbor, multi-fragment/greedy, full/candidate 2-opt, official LKH,
  GA-EAX, and Concorde controls as available;
- per-run incumbent traces, end-to-end timing, peak RSS, status, and raw rows;
- fixed-budget/target reports with family stratification and intervals.

Higher dimension, intrinsic manifolds, evolved feature filling, neural adapters,
huge instances, and hidden evaluation follow once the semantic and raw-record
foundation is stable.

## 15. Definition of done for the benchmark system

- Objective semantics are versioned and independently scored.
- All public artifacts have provenance, checksums, and usage status.
- Generated instances are reproducible from version/parameters/PRNG/seed.
- Lineage-aware splits prevent transform, morph, and descendant leakage.
- References distinguish certified optimum, feasible upper, and valid lower.
- Every run has explicit config, environment, seed, budget, status, and trace.
- End-to-end time includes all declared per-instance work.
- Invalid tours and failures are retained, not hidden.
- Reports show exact, anytime, latency, scale, robustness, and component claims
  separately.
- Statistical inference respects runs nested in instances/families.
- Public, controlled, adversarial, and hidden strata remain identifiable.
- Releases and reference snapshots are immutable and auditable.
- A new implementer can reproduce every aggregate from documented artifacts
  without private conventions or manual spreadsheet steps.
