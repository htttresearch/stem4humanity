# Design: native-language solvers (C++ and friends) in algofinder

Status: decisions made below (opencode's choices). Implementation pending user go-ahead.

## Context

- Python is the harness language; every solver today is a `Solver` subclass with
  `solve(state, *, budget_seconds, context) -> SolverResult`
  (`algofinder/solvers/base.py`).
- The harness (registry, benchmark with per-cell hard timeouts, leaderboard,
  trace) treats solvers as opaque units, so a C++ solver only needs to satisfy
  that one interface.
- Goal: allow solvers to be written in C++ (or any native language) while
  keeping correctness, validation, trace, and ML integration in Python.
- Companion research: euclidean TSP (in progress). Euclidean TSP is the natural
  proving ground — 2-opt with kd-trees, geometric hashing, and the
  1,064-line `incremental_exact.py` exact solver are C++-shaped work.

## Guiding principle

Python stays the trust anchor. C++ computes; the harness audits
(`verify` / `objective_value` never leave Python). A wrong C++ solver shows up
as `invalid` in the leaderboard, never as silent garbage.

## Decision D1 — Speed vehicle

| Choice | Pros | Cons | Consequence |
|---|---|---|---|
| Numba (JIT) | ~90% of C++ speed, no new layer, works today, stays in harness | Only Python-shaped code; no C++ ecosystem | Fastest path to fast exact solvers; never "C++ capable" |
| C++ subprocess | True C++ capability, isolation, killable timeouts, language-agnostic | ~ms IPC overhead per cell; wire protocol + porting effort | The platform requested; higher cost, slower payoff |
| pybind11 in-process | Zero IPC overhead | Cannot hard-timeout a stuck call; segfault kills the benchmark | Speed now, architecture pain later |

Note: Numba and subprocess are not exclusive — Numba for exact solvers,
subprocess layer for C++-shaped algorithms. The decision to make is which to
build first.

**DECIDED: build the C++ subprocess layer** — the stated goal is "be able to
write algorithms in C++", and the contract is language-agnostic (Rust/Go/Julia
come free later). Numba is allowed opportunistically *inside* Python solvers
but is not part of this plan and not a prerequisite.

## Decision D2 — Wire protocol (for the subprocess route)

| Choice | Pros | Cons | Consequence |
|---|---|---|---|
| JSON | Zero dependency, debuggable, trivial in both languages | Slow parse for large matrices; numeric edge cases | Fine for ≤ ~100-city TSP; refactor later |
| msgpack | Fast, tiny, float64 round-trips exactly, trivial both sides | One pip dep + one C++ header dep | Right-sized default; not human-readable (trace covers that) |
| flatbuffers / protobuf | Schema'd, zero-copy, versioning story | Schema compiler in the build; overkill now | For the persistent-worker era, not one-shot v1 |

Envelope shape (v1): `{version, instance_sha, solver_id, problem, payload}`.

**DECIDED: msgpack** — fast, tiny, float64-exact round-trip, trivial on both
sides. One pip dep + one C++ header dep, both standard. Revisit flatbuffers
only if the persistent-worker era shows a real serialization bottleneck.

## Decision D3 — Process model

| Choice | Pros | Cons | Consequence |
|---|---|---|---|
| One-shot per cell | ~200 lines of Python, simplest possible | Spawn overhead (10-50 ms) swamps µs-scale exact solves; leaderboard wall times lie | Fine for proving the concept; must move to workers for exact-row competition |
| Persistent worker per problem | Spawn cost paid once, crash isolation (one instance fails, not the benchmark), cooperative budget per envelope | Framing/multiplexing code, worker lifecycle management | The "right" architecture; worth it from day one if C++ will run many cells |

**DECIDED: one-shot per cell for the first port, persistent workers as the
committed second step** — prove the layer cheaply, then upgrade before C++
competes on exact rows (µs-scale solves make spawn overhead the leaderboard
metric otherwise). The upgrade is scheduled, not "maybe later".

## Decision D4 — Trace integration

| Choice | Pros | Cons | Consequence |
|---|---|---|---|
| Coarse (wrapper-only) | Zero C++ work; run start/end + final quality events from Python wrapper | Dev-mode atomic-step traces have a C++ hole | Fine for benchmarking; bad for solver development |
| Native C++ trace emitter | Full observability parity, same `stem4humanity.trace-event/v1` envelope | Dual-language schema maintenance, forever | Correct endgame; expensive to retrofit |

**DECIDED: coarse (wrapper-only) now** — run start/end + final quality events
from the Python wrapper. Native C++ trace emitter is deferred with an explicit
trigger: the first time dev-mode step-level debugging of a C++ solver is
actually needed.

## Decision D5 — ML integration

| Choice | Pros | Cons | Consequence |
|---|---|---|---|
| Classical-only C++ first | Ships now | Learned solvers stay Python-only; C++ rows can't use learned candidates | Cleanest start |
| xgboost C API | Same model file loads natively in C++ | Couples build to model format/version; covers tree models only | More power, more coupling; addable later — it's inside the wrapper |

**DECIDED: classical-only C++ first.** xgboost C API deferred with an explicit
trigger: a learned solver is the best row for a problem *and* a C++ port wants
it.

## Decision D6 — First port target

| Choice | Pros | Cons | Consequence |
|---|---|---|---|
| Euclidean geometry heuristic | Small (~150 lines), exercises the whole layer quickly | Least impressive speed story | Fastest proof of the pipeline |
| Incremental exact (1,064 lines) | Biggest speed win, best C++-shaped algorithm | Longest port; parity debugging harder | Proves the layer and wins a leaderboard row |

**DECIDED: euclidean geometry heuristic first (validate the layer cheaply),
incremental exact second (the actual prize)** — both are committed ports;
sequencing is about de-risking, not about whether.

## Decision D7 — Acceptance gate

| Choice | Pros | Cons | Consequence |
|---|---|---|---|
| Parity check (same instances, compare vs Python original) | Real-pipeline verification per repo rules | Manual workflow step | Trust established before benchmarking |
| Benchmark only | Zero extra work | Subtle C++ port bugs surface as confusing leaderboard numbers | Trust problem deferred |

**DECIDED: parity check is mandatory** — same manifest set through the Python
original and the C++ port; compare solution equality/cost. Part of the
definition of done for every C++ port, executed via the real pipeline
(benchmark + report).

## Decided plan (synthesis of the choices above)

1. **Layer**: thin `CppSolver` Python wrapper + msgpack envelopes over
   subprocess stdin/stdout, one-shot per cell.
2. **First port**: euclidean geometry heuristic; parity-checked against the
   Python original on the full manifest set.
3. **Second port**: incremental exact solver; then upgrade to persistent
   workers before C++ rows compete on exact (µs-scale) solves.
4. **Integration**: register like any solver (not-built shows as `skipped`
   via `InapplicableError`); full `benchmark` + `report` on the real pipeline.
5. **Deferred with triggers**: native C++ trace emitter (first time C++ needs
   dev-mode step debugging), xgboost C API (learned solver best row + C++ wants
   it), flatbuffers (serialization bottleneck in worker era).

## Implementation sketch (for when the decision is made)

- `CppSolver` base class in `algofinder/solvers/`:
  1. encode `ProblemState` -> envelope bytes (canonical encoding in
     `problems/base.py`: `encode_state` / `decode_solution`; C++ mirrors it)
  2. `Popen(binary)`, write envelope to stdin, read envelope from stdout
  3. decode solution -> Python objects; Python-side verify/objective/trace
  4. return `SolverResult`
- `InapplicableError` gives "not built" a first-class state: registry registers
  C++ ids even when binaries aren't compiled; leaderboard shows `skipped`
  until `cmake --build` (or equivalent) runs. No code churn.
- Not-built binaries and build dirs are gitignored (no-delete rule applies:
  build into a fresh dir, never overwrite).
- Parity gate: same manifest set through Python original and C++ port; compare
  solution equality / cost. Run via the real pipeline (benchmark + report).

## Open questions

- Where do binaries live and how does the registry discover them
  (e.g. `algofinder/solvers/*/cpp/*.json` solver manifests)?
- Build tooling: CMake per problem vs single top-level build?
- Shared-memory / mmap data passing as a later optimization for large
  matrices (persistent-worker era)?

## Resolved questions (as part of these decisions)

- Binaries: `algofinder/solvers/<problem>/cpp/build/`, gitignored; discovery
  via a `solvers.json` manifest inside the cpp dir (solver ids + binary name).
- Build tooling: CMake per problem (each cpp dir is a self-contained target);
  top-level build script only if/when there are 3+ cpp dirs.
- Shared memory: out of scope until the persistent-worker upgrade shows a
  real serialization bottleneck on large distance matrices.
