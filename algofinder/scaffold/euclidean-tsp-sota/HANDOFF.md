# Handoff: Euclidean TSP — solver done, benchmark system next (Stage 2)

Research cutoff: **2026-08-02**. Dossier reading order: `README.md` →
`SUBPROBLEM_TAXONOMY.md` → `BENCHMARK_RESEARCH.md` → `BENCHMARK_CATALOG.md` →
`BENCHMARK_IMPLEMENTER_SPEC.md` → `BENCHMARKING.md` → `BENCHMARK_LITERATURE.md`
→ `IMPLEMENTATION_BLUEPRINT.md` → `SOURCES.md`.

## State

**Stage 1 (solver) — DONE, merged.** PR #1 → main `971213d`. `strong-ils-euclidean`
(mixed candidates + 3 constructors + 2/3-opt/Or-opt + double-bridge + elite
union + widening; `candidates.py`, `constructors.py`, `move_engine.py`,
`strong_ils.py`). Beats chained-2opt at matched 5 s on synthetic suite and on
TSPLIB (1.75 % vs 3.16 % mean gap). Residual (accepted): ~0.5 % on one n101
clustered at 0.5 s; single-seed stats.

**TSPLIB benchmark — DONE, PR #2 open** on `feat/tsplib-benchmark`
(commit `70267ba`, not yet merged). Spec-exact EUC_2D parser + independent
opt-tour rescoring + 8 instances (eil51..u574) + `config/runE_tsplib.toml` +
results. NOTE: Phase 2 work may depend on this branch (EUC_2D objective).

## Git

- Branches: main=`971213d`; `feat/etsp-stage1-strong-baseline` and
  `feat/tsplib-benchmark` left in place (local + remote).
- Push over HTTPS only (htttresearch token, `gh auth setup-git`); SSH key is
  HttTavares = no access. See AGENTS.md "GitHub accounts rule".
- venv `algofinder/.venv/bin/python`; runner: `python -m
  algofinder.harness.runner --config <cfg> generate|benchmark|report`
  (paths resolve relative to the config file's dir).
- Stash: stale `public/results/*` edits. `outD/` holds untracked junk
  (results_fixed, empty manifest dir) — not part of any PR.

## Stage 2 = benchmark system (normative: BENCHMARK_IMPLEMENTER_SPEC §13)

Phased plan (epistemic order — later claims depend on earlier semantics):

1. **Phase 0** decisions: objective modes (raw L2 + EUC_2D integer) as named
   specs; digests; storage; license/provenance; freeze v1 schemas.
2. **Phase 1 semantic core**: objective registry + independent tour scoring +
   feasibility/normalization + instance/tour digests + `semantic_micro`
   corpus with metamorphic lineages + pipeline report validating it.
3. **Phase 2 sources**: TSPLIB/DIMACS/Waterloo adapters w/ checksums +
   append-only reference registry + exact refs (Held-Karp) for small cases +
   tetrahedron instances.
4. **Phase 3 generators**: cluster/lattice/hull/strip/manifold/degeneracy
   stages, split compiler, pilot suites (exact_structure, anytime_quality, ML).
5. **Phase 4 runner**: adapter contracts; internal controls, then official
   LKH/GA-EAX/Concorde (license/build permitting); process timing, RSS, seeds,
   incumbent traces.
6. **Phase 5 stats**: fixed-budget/target, profiles, ECDF, hierarchical paired
   intervals, censoring-visible reports.
7. **Phase 6-7**: feature tiers, coverage audit, ASlib scenario; hidden +
   longitudinal releases.

v0.1 scope (§14): raw L2 + EUC_2D specs, semantic_micro with exact refs and
metamorphic groups, filtered TSPLIB planar EUC instances, DIMACS RUE/clustered
subsets, generated strata (uniform, GMM, grid+jitter, convex+inner, strips,
rings, degeneracy, tetrahedra), exact refs for small + bound status elsewhere,
baselines (NN, MF, full/candidate 2-opt, LKH, EAX, Concorde as available),
incumbent traces + end-to-end timing + RSS + status + raw rows.

Current work branch: `feat/benchmark-semantic-core` (off `feat/tsplib-benchmark`).
