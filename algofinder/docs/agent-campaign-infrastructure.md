# Agent campaign infrastructure

`algofinder.agents` is the trusted control plane for solver-improvement
campaigns. It deliberately does **not** provide an algorithm-design agent,
prompt, search policy, or research methodology. Those components operate later
through its typed API.

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

## Current implementation boundary

Gate 0 (allowlist, patch integrity, and Python syntax) is implemented and
recorded. The trusted execution runner is intentionally fail-closed for
validation and challenge data. A later evaluator worker can connect existing
`harness.run_benchmark` calls to gates 1–6 only inside a real isolation backend;
that is not delegated to an agent or a candidate workspace.
