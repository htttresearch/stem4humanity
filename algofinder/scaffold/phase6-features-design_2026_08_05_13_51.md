# Phase 6 design: per-instance features, coverage, and portfolio data

Date: 2026-08-05. Status: implemented (see `algofinder/features/` and the
runner subcommands `features`, `audit`, `export-aslib`, `portfolio`,
`generate --fill-features`).

## Scope

Per-INSTANCE feature system for algorithm/portfolio selection (spec
section 13, phase 6). The existing `algofinder/ml/` per-element learned
solvers are a separate concern and are untouched.

## Feature sets (v0)

- `params@1` (tier 0, all problems): derived from the instance descriptor
  (`n`, `seed`, size/type keys of `instance.data`). Near-free, no state
  build required.
- `etsp-geometry@1` (tier 1, Euclidean TSP only): hull fraction, MST
  weight normalized, nearest-neighbor statistics, candidate degree
  statistics (reusing `solvers/tsp/candidates.py`), bottleneck/span
  statistics. Deterministic, solver-independent.

Tier-2 probing features (`charged_at_inference=true`) are deferred until a
solver actually consumes them.

## Feature record (spec 6.7)

```json
{"instance_id", "feature_set_id", "values", "status_by_feature",
 "timing_seconds", "peak_rss_bytes", "extractor_digest", "charged_at_inference"}
```

`instance_id` is the content digest of the instance mapping (consistent
with the registry `instances` kind). `extractor_digest` is the sha256 of
the feature-set id + version + extractor source text, so a code change
invalidates old records rather than silently re-labelling them.

## Storage (Design A discipline)

- Canonical records: `public/data/features/<feature_set_id>.jsonl`
  (immutable, digest-checked, recomputable by the `features` step).
- Registry: `features` kind added to `private/registry/`.
- DuckDB: `v_features` view + `features.parquet` cache; joins with
  `v_instances` and `v_runs` power the portfolio and ASlib queries.

## Commands

- `features` — compute enabled sets for all instances, write records,
  update registry, refresh DB.
- `audit` — declared feature bins per set; per-family x bin coverage
  counts; near-duplicate redundancy report -> `private/reports/`.
- `export-aslib` — ASlib scenario (instance-features, feature-costs,
  algorithm-runtimes, run-v2) with leakage-safe train/test folds ->
  `public/benchmarks/aslib/<scenario>/`.
- `portfolio` — virtual-best, single-best, selector regret, feature cost,
  failures -> `public/results/portfolio.md`.
- `generate --fill-features` — deterministic parameter scan over the
  ETSP generators accepting instances that fill declared feature-bin
  gaps (solver-independent feature computation only; no final-evaluated
  configurations are used).

## Follow-ups

Full evolutionary portfolio-discriminating generation; tier-2 probing
features; phase-5 statistics; OOD axes; `feature_registry` digest pinning
in release manifests (phase 7).
