# algofinder

An algorithm finder for classic optimization problems (TSP, scheduling,
packing, shortest path, unit commitment) that pits **exact** solvers,
**heuristic** solvers, and **learned (ML)** solvers against each other
on generated instance suites, and renders a leaderboard of which family
wins where.

Part of stem4humanity step 0.0.1: a running environment that generates
instances, trains per-problem ML solvers, benchmarks every solver, and
reports results.

## Problems

| Problem | Variants | Instances |
|---|---|---|
| Euclidean TSP | 2D cities, 3D cities | `tsp2d_euclidean`, `tsp3d_euclidean` |
| General TSP | random distances | `tsp_general` |
| Knapsack | 0/1 | `knapsack_0_1` |
| Flow-shop scheduling | 2 machines | `flowshop_2` |
| Bin packing | vector (2D) | `binpack_vector` |
| Bin packing | large items | `binpack_large_items` |
| Parallel scheduling | general, forest | `parsched_general`, `parsched_forest` |
| Shortest path | DAG | `shortest_path` |
| Unit commitment | classic (thermal), storage-augmented | `unit_commitment_classic`, `unit_commitment_storage` |

## Solvers

Each problem ships with an exact solver (brute force or DP, with the
exact one verified against brute force on tiny instances), a
heuristic/priority solver, and a scikit-learn learned solver. Solver IDs
used in `data/results/leaderboard.md`:

- TSP / knapsack / scheduling / bin packing / parallel scheduling /
  shortest path: `*-exact-dfs`, `*-exact-*` (DP), `*-heuristic-*`,
  `*-learned-*` (e.g. `bp-learned`, `sp-learned`, `ps-learned`).
- Unit commitment: `uc-exact-dp`, `uc-priority-list`, `uc-learned-commitment`
  (classic); `uc-storage-dp`, `uc-storage-threshold`, `uc-storage-learned`
  (storage). Best-known exact answers are recorded on every generated
  instance by the DP solvers.

Current headline numbers (see the leaderboard): on classic UC, exact DP
is unbeaten (0% gap), the learned committer averages ~7.5% above it, the
priority list ~84%; on storage UC the DP is likewise unbeaten, with the
learned policy at ~0.9% and the threshold heuristic at ~5.7%.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .            # numpy, scikit-learn, joblib, networkx
pip install -e ".[dev]"    # optional: pytest
```

## Usage

```bash
python -m algofinder.harness.runner generate    # build all instance manifests
python -m algofinder.harness.runner train       # train the ML solvers
python -m algofinder.harness.runner benchmark   # run every applicable solver
python -m algofinder.harness.runner report      # render the leaderboard
python -m algofinder.harness.runner all         # generate + train + benchmark + report
```

`benchmark` and `all` accept `--manifest <file>` (repeatable) to scope a
run, `--budget-seconds`, `--timeout-seconds`, and `--out`.

## prod vs dev mode

Every run happens in one of two universal modes:

- `--mode prod` (default): run the solvers on the instances as fast as
  possible — no tracing, no extra I/O.
- `--mode dev`: additionally record the intermediate states of the
  problem and of the solver while a solve is happening, into a
  timestamped session directory `private/sessions/session-<timestamp>/`,
  one JSONL file per (instance, solver).

Example:

```bash
python -m algofinder.harness.runner benchmark --mode dev --manifest public/data/unit_commitment_classic_manifests.json
```

Currently traced in dev mode (one JSONL per instance+solver):

- `uc-exact-dp` — problem snapshot, per-hour DP frontier size, best
  partial cost and best commitment, result.
- `uc-priority-list` — per-hour commitment mask, power vs target,
  units started / decommitted, dispatch cost.
- `bp-exact-dfs` — search-tree nodes (depth, bins, lower bound, best
  incumbent; throttled on large searches) and incumbent improvements.
- `ps-learned-priority` — learned vs critical-path scores per task, and
  every scheduling step (task, machine, start, priority, available
  count).

Tracing is intentionally ad hoc until a universal problem/solver
abstraction is settled; new solvers can opt in via
`algofinder.util.tracing` (`open_trace`, `write_record`).

The proposed universal replacement is specified in
[`docs/dev-mode-observability-design.md`](docs/dev-mode-observability-design.md).

## Data layout

```
public/               committed, published artifacts
  data/               instance manifests (JSON)
  models/             trained models (joblib)
  results/            benchmark.json + leaderboard.md
  benchmarks/         published benchmark suites (future)
private/              gitignored, work-in-progress artifacts
  data/               experimental/richer data packages
  models/             unpublished models
  results/            unpublished results
  sessions/           dev-mode traces (JSONL), one dir per run
```

## Tests

`tests/` is reserved for smoke tests; run with `pytest` once populated.
