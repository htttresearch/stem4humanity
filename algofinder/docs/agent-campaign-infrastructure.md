# Agent campaign infrastructure

`algofinder.agents` contains both the trusted control plane for
solver-improvement campaigns and an untrusted research layer that operates
through its typed API. The boundary is explicit: research policies can propose
hypotheses, patches, experiments, and analyses, but only the evaluator authority
can execute benchmark gates or record fitness evidence.

## What is enforced

- A campaign fixes a base Git commit, solver-only path allowlist, suites,
  budgets, feedback policy, and agent specification digests.
- Evidence is immutable JSON/JSONL under `private/campaigns/<campaign-id>/`.
  Candidate source edits happen in disposable Git worktrees and freeze into a
  normalized patch before they may be evaluated.
- Candidate patches cannot change benchmark, verifier, suite, objective,
  policy, or dependency files unless a human deliberately expands the
  campaign allowlist.
- Public feedback may be detailed. Validation is aggregate-only and metered.
  Challenge execution fails closed unless an isolated sandbox backend is
  available; its suite paths are never included in agent-visible snapshots.
- Promotion is only an evidence-backed request. This package never merges a
  candidate into Git history.

## Minimal lifecycle

1. Write one `AgentSpec` JSON and one `CampaignSpec` JSON. The campaign spec
   must use the exact Git commit to be searched and allow only solver paths,
   such as `algofinder/solvers/**`.
2. Initialize and start the campaign:

   ```bash
   python -m algofinder.agents.cli init --spec campaign.json --agent-spec agent.json
   python -m algofinder.agents.cli start <campaign-id> --reason "approved baseline"
   ```

3. A research agent uses `AgentTools` to create a falsifiable hypothesis,
   obtain a disposable candidate worktree, apply an allowlisted patch, validate
   it, freeze the candidate, and pre-register an `ExperimentSpec`.
4. The evaluator-owned process uses `EvaluationAuthority`; agents have no API
   to set scores or read challenge instances.
5. Rebuild the campaign registry when desired:

   ```bash
   python -m algofinder.agents.cli index --registry-dir private/registry
   ```

The regular `index` command then exposes any campaign registry JSONL files as
DuckDB views (`v_campaigns`, `v_candidates`, `v_evaluations`, and so on).

## Research policies

Two complementary policies are implemented:

- `AIDETreeResearchAgent` performs optimistic tree search over immutable code
  lineages. It balances the campaign's primary metric against a durable
  exploration bonus derived from the number of child branches, and selects
  `repair`, `mutate`, or `tune` according to the branch evidence.
- `QualityDiversityResearchAgent` reconstructs a MAP-Elites archive from
  evaluator-authored records and uses UCB-style operator credit to preserve
  behavioral diversity while choosing invention, mutation, recombination,
  repair, tuning, or distillation.

Both use the provider-neutral `ResearchModel` protocol. A model returns a
structured hypothesis/patch pair and an evidence-linked analysis; the shared
runner performs every mutation through `AgentTools`. `AuthorityEvaluator` is
the only included execution bridge and calls `EvaluationAuthority.gate_0` and
`EvaluationAuthority.gate_1_public`; it never calls the benchmark harness.

The default research-method prompt and its SHA-256 digest are exported as
`DEFAULT_METHOD_PROMPT` and `DEFAULT_METHOD_PROMPT_DIGEST`. Campaigns pin that
digest in their `AgentSpec` so changing the research instructions creates a new
agent identity rather than silently changing an active campaign.

## Current evaluator boundary

Gate 0 (allowlist, patch integrity, and Python syntax) and isolated public gate
1 are implemented and recorded. The trusted execution runner remains
fail-closed for validation and challenge data. A later evaluator worker can
connect gates 2–6 only inside a real isolation backend; that is not delegated
to an agent, proposal model, or candidate workspace. The research layer is
also intentionally distinct from the future RSI layer: these policies optimize
solver candidates, never their own prompts, code, or selection hyperparameters
during an active campaign.

## Typed HIR lane for weak local models

`tsp-edge-formula-v1` remains the smallest pipeline canary. The broader
`tsp-hir-v1` template accepts a type-checked heuristic genome: tour
constructor, candidate-edge generator, conditional edge-score tree,
neighborhood, acceptance policy, diversification, and adaptation schedule.
The authority compiles that data into an immutable candidate module. A model
cannot edit the class, imports, solver identity, evaluator, or resource limits.

`algofinder.agents.tsp_hir` owns parsing, type checking, deterministic
compilation, numeric mutation, type-compatible crossover, and application of
one model-proposed edit. The model records an HIR edit and its hypothesis; it
never supplies Python source.

```bash
# Model-guided typed HIR
python -m algofinder.agents.pilot ... --representation hir --backend model

# Matched non-LLM mutation/crossover control
python -m algofinder.agents.pilot ... --representation hir --backend evolution
```

`python -m algofinder.agents.ablation` orchestrates matched formula, HIR
evolution, and HIR-model arms. Its fourth `full-code-strong` control remains
explicitly disabled until `--strong-model` is supplied; it is never replaced by
the local 1.5B model. Use the runner for a small assay first, then choose the
approved screening population and survivor/holdout schedule before consuming a
large evaluation budget.
