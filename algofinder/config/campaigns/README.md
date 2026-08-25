# First AlgoFinder campaign: Euclidean TSP

This directory prepares the first controlled solver-improvement campaign.
It is deliberately a template until the research agent supplies its fixed
model, prompt package, and sampling configuration.

## Scope fixed here

- Candidate code may change only `algofinder/solvers/**`.
- Baseline: `distance-ranked-2opt` on the 12 `train` instances in
  `algofinder/data/instances/tsp_manifests.json`.
- Public smoke: the same manifest's `train` split.
- Metered validation and OOD: lineage-separated splits in
  `tsp_ml_ood_manifests.json`.
- Challenge: a separately materialized evaluator-only copy of the OOD corpus;
  its path must never appear in the final campaign spec.

## Before initializing the campaign

1. Replace `BASE_COMMIT` with this branch's committed infrastructure SHA.
2. Replace every `REPLACE_*` AgentSpec field with the agent's actual fixed
   configuration and calculate the SHA-256 of its prompt package.
3. Materialize evaluator-only validation/challenge manifests outside the agent
   worktree. The challenge suite must be given only by digest in the campaign
   spec.
4. Initialize with `python -m algofinder.agents.cli init`, then transition it
   to `active` only after the evaluator service is configured.

The template is intentionally not runnable until steps 1–3 are complete: a
campaign must not claim a model identity, prompt digest, or sealed suite that
does not exist.
