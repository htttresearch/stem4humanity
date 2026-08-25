"""Narrow, typed tool API exposed to future research agents.

The API deliberately has no shell, no evaluator-edit, no score-setting, and no
challenge-data read operation.  Search policies and agents can compose these
operations, but benchmark authority and promotion remain outside this surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from algofinder.agents.campaign import Campaign, CampaignError
from algofinder.agents.contracts import (
    Analysis,
    Candidate,
    Decision,
    ExperimentSpec,
    Hypothesis,
    new_id,
)
from algofinder.agents.ledger import LedgerError
from algofinder.agents.workspace import CandidateWorkspace, WorkspaceError, WorkspaceService


class ToolError(RuntimeError):
    """Raised for an invalid request through the agent-facing API."""


class AgentTools:
    """Versioned facade over a fixed campaign's permitted operations."""

    api_version = "algofinder-agent-tools@3"

    def __init__(self, campaign: Campaign, workspaces: WorkspaceService) -> None:
        self.campaign = campaign
        self.workspaces = workspaces

    # Read tools ---------------------------------------------------------
    def catalog_problems(self) -> list[dict[str, Any]]:
        from algofinder.problems.registry import all_problems

        return [
            {
                "id": problem_id,
                "display": problem.display,
                "subproblems": list(problem.subproblems),
                "minimize": problem.minimize,
            }
            for problem_id, problem in sorted(all_problems().items())
        ]

    def catalog_solvers(self) -> list[dict[str, Any]]:
        import algofinder.solvers.registry  # noqa: F401
        from algofinder.solvers.base import all_solvers

        return [
            {
                "id": solver_id,
                "display": cls.display,
                "tags": sorted(cls.tags),
                "applies_to": sorted(cls.applies_to),
                "module": cls.__module__,
            }
            for solver_id, cls in sorted(all_solvers().items())
        ]

    def contracts_problem(self, problem_id: str) -> dict[str, Any]:
        from algofinder.problems.registry import get_problem

        problem = get_problem(problem_id)
        return {
            "id": problem.id,
            "display": problem.display,
            "subproblems": list(problem.subproblems),
            "minimize": problem.minimize,
            "authority": "ProblemState.verify and ProblemState.objective_value",
        }

    def contracts_solver(self, solver_id: str) -> dict[str, Any]:
        import algofinder.solvers.registry  # noqa: F401
        from algofinder.solvers.base import all_solvers

        try:
            solver = all_solvers()[solver_id]()
        except KeyError as exc:
            raise ToolError(f"unknown solver {solver_id!r}") from exc
        return solver.describe()

    def archive_query(self, kind: str, **filters: Any) -> list[dict[str, Any]]:
        try:
            records = self.campaign.ledger.query(kind, **filters)
        except LedgerError as exc:
            raise ToolError(str(exc)) from exc
        if kind != "evaluation":
            return records
        # Evaluation.feedback contains authority-private raw runs. Generic
        # archive access must expose the same summary boundary as runs_compare.
        return [self._evaluation_summary(item) for item in records]

    def archive_lineage(self, candidate_id: str) -> list[dict[str, Any]]:
        """Return immutable candidate ancestors in parent-before-child order."""
        visited: set[str] = set()
        ordered: list[dict[str, Any]] = []

        def walk(node_id: str) -> None:
            if node_id in visited:
                return
            record = self.campaign.ledger.read("candidate", node_id)
            visited.add(node_id)
            for parent_id in record.get("parent_ids", []):
                walk(parent_id)
            ordered.append(record)

        try:
            walk(candidate_id)
        except LedgerError as exc:
            raise ToolError(str(exc)) from exc
        return ordered

    def budget_status(self) -> dict[str, Any]:
        return self.campaign.budget_status().to_mapping()

    def campaign_snapshot(self) -> dict[str, Any]:
        return self.campaign.snapshot(agent_visible=True)

    def source_read(
        self,
        paths: Iterable[str],
        *,
        max_total_bytes: int = 80_000,
    ) -> list[dict[str, str]]:
        """Read bounded solver sources from the campaign's fixed base commit."""
        try:
            return self.workspaces.read_base_sources(paths, max_total_bytes=max_total_bytes)
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    def candidate_patch_read(self, candidate_id: str, *, max_bytes: int = 80_000) -> str:
        """Read one frozen candidate patch for lineage-aware mutation."""
        if max_bytes < 1:
            raise ToolError("max_bytes must be positive")
        try:
            self.campaign.ledger.read("candidate", candidate_id)
        except LedgerError as exc:
            raise ToolError(str(exc)) from exc
        path = self.campaign.root / "candidates" / candidate_id / "change.patch"
        try:
            patch = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"candidate patch is unavailable: {candidate_id}") from exc
        if len(patch.encode("utf-8")) > max_bytes:
            raise ToolError(f"candidate patch exceeds {max_bytes} bytes")
        return patch

    def candidate_template_describe(self, workspace: CandidateWorkspace) -> dict[str, Any]:
        """Return authority-derived state and behavior descriptors for a template."""
        try:
            return self.workspaces.describe_template(workspace)
        except WorkspaceError as exc:
            raise ToolError(str(exc)) from exc

    def runs_compare(self, candidate_ids: Iterable[str]) -> dict[str, Any]:
        """Compare recorded evaluation summaries; raw benchmark data stays private."""
        wanted = set(candidate_ids)
        if not wanted:
            raise ToolError("runs_compare requires at least one candidate")
        evaluations = self.campaign.ledger.list("evaluation")
        rows = [
            self._evaluation_summary(item)
            for item in evaluations if item.get("candidate_id") in wanted
        ]
        return {"candidates": sorted(wanted), "evaluations": rows}

    def _evaluation_summary(self, item: dict[str, Any]) -> dict[str, Any]:
        zone = item["zone"]
        metric_vector = item.get("metric_vector", {})
        paired_baselines = item.get("paired_baselines", {})
        if zone == "challenge":
            metric_vector = {
                key: value
                for key, value in metric_vector.items()
                if isinstance(value, (int, float, bool)) or value is None
            } if isinstance(metric_vector, dict) else {}
            paired_baselines = {}
        return {
            "id": item.get("id"),
            "candidate_id": item["candidate_id"],
            "stage": item["stage"],
            "zone": item["zone"],
            "gate": item["gate"],
            "metric_vector": metric_vector,
            "paired_baselines": paired_baselines,
            "evaluator_build_digest": item.get("evaluator_build_digest"),
            "created_at": item.get("created_at"),
        }

    # Write tools --------------------------------------------------------
    def hypothesis_create(
        self,
        *,
        mechanism: str,
        affected_strata: Iterable[str],
        predicted_tradeoffs: str,
        falsification_test: str,
        supporting_sources: Iterable[str] = (),
        confidence: float = 0.5,
    ) -> Hypothesis:
        self._require_active()
        record = Hypothesis(
            hypothesis_id=new_id("hypothesis"),
            campaign_id=self.campaign.campaign_id,
            mechanism=mechanism,
            affected_strata=tuple(affected_strata),
            predicted_tradeoffs=predicted_tradeoffs,
            falsification_test=falsification_test,
            supporting_sources=tuple(supporting_sources),
            confidence=confidence,
        )
        self.campaign.ledger.write(record)
        return record

    def candidate_create(
        self,
        *,
        hypothesis_id: str,
        parent_ids: Iterable[str] = (),
        generation_operator: str = "invent",
        template_id: str | None = None,
    ) -> CandidateWorkspace:
        self._require_active()
        parents = tuple(parent_ids)
        try:
            self.campaign.ledger.read("hypothesis", hypothesis_id)
            for parent_id in parents:
                self.campaign.ledger.read("candidate", parent_id)
        except LedgerError as exc:
            raise ToolError(str(exc)) from exc
        return self.workspaces.create(
            hypothesis_id=hypothesis_id,
            parent_ids=parents,
            generation_operator=generation_operator,
            template_id=template_id,
        )

    def candidate_apply_patch(self, workspace: CandidateWorkspace, patch: str) -> None:
        self._require_active()
        self.workspaces.apply_patch(workspace, patch)

    def candidate_apply_template_values(
        self,
        workspace: CandidateWorkspace,
        *,
        values: dict[str, str],
    ) -> None:
        """Fill only the constrained, model-owned gaps in a trusted scaffold."""
        self._require_active()
        self.workspaces.apply_template_values(workspace, values=values)

    def candidate_validate(
        self,
        workspace: CandidateWorkspace,
        *,
        solver_entrypoints: Iterable[str] = (),
    ) -> dict[str, Any]:
        self._require_active()
        return self.workspaces.validate(
            workspace,
            solver_entrypoints=tuple(solver_entrypoints),
        ).to_mapping()

    def _candidate_discard_rejected(self, workspace: CandidateWorkspace) -> None:
        """Control-plane cleanup for an unfrozen rejected proposal."""
        self.workspaces.discard(workspace)

    def candidate_freeze(
        self,
        workspace: CandidateWorkspace,
        *,
        solver_entrypoints: Iterable[str] = (),
        descriptors: dict[str, Any] | None = None,
        producing_agent_run_id: str | None = None,
        producing_episode_id: str | None = None,
    ) -> Candidate:
        self._require_active()
        return self.workspaces.freeze(
            workspace,
            solver_entrypoints=tuple(solver_entrypoints),
            descriptors=descriptors,
            producing_agent_run_id=producing_agent_run_id,
            producing_episode_id=producing_episode_id,
        )

    def experiment_propose(
        self,
        *,
        candidate_id: str,
        baseline_candidate_ids: Iterable[str],
        suite_names: Iterable[str],
        stage: str,
        seeds: Iterable[int],
        budget_seconds: float | None,
        stopping_rule: str,
        expected_result: str,
        decision_rule: str,
    ) -> ExperimentSpec:
        self._require_active()
        baselines = tuple(baseline_candidate_ids)
        suites = tuple(suite_names)
        experiment_seeds = tuple(seeds)
        try:
            self.campaign.ledger.read("candidate", candidate_id)
            for baseline_id in baselines:
                self.campaign.ledger.read("candidate", baseline_id)
        except LedgerError as exc:
            raise ToolError(str(exc)) from exc
        record = ExperimentSpec(
            experiment_id=new_id("experiment"), campaign_id=self.campaign.campaign_id,
            candidate_id=candidate_id, baseline_candidate_ids=baselines,
            suite_names=suites, stage=stage,  # type: ignore[arg-type]
            seeds=experiment_seeds, budget_seconds=budget_seconds, stopping_rule=stopping_rule,
            expected_result=expected_result, decision_rule=decision_rule,
        )
        self.campaign.ledger.write(record)
        return record

    def analysis_record(
        self,
        *,
        evaluation_ids: Iterable[str],
        observation: str,
        interpretation: str,
        confidence: float,
        next_test: str,
        trace_regions: Iterable[str] = (),
    ) -> Analysis:
        self._require_active()
        record = Analysis(
            analysis_id=new_id("analysis"), campaign_id=self.campaign.campaign_id,
            evaluation_ids=tuple(evaluation_ids), observation=observation,
            interpretation=interpretation, confidence=confidence, next_test=next_test,
            trace_regions=tuple(trace_regions),
        )
        self.campaign.ledger.write(record)
        return record

    def archive_nominate(self, candidate_id: str, *, descriptors: dict[str, Any], rationale: str) -> Decision:
        self._require_active()
        try:
            self.campaign.ledger.read("candidate", candidate_id)
        except LedgerError as exc:
            raise ToolError(str(exc)) from exc
        decision = Decision(
            decision_id=new_id("decision"), campaign_id=self.campaign.campaign_id,
            kind="archive", target_id=candidate_id, evidence_ids=(),
            policy_version="agent-request@1", rationale=rationale + f"; descriptors={descriptors!r}",
        )
        self.campaign.ledger.decide(decision)
        return decision

    def promotion_request(self, candidate_id: str, *, evidence_ids: Iterable[str], rationale: str) -> Decision:
        self._require_active()
        decision = Decision(
            decision_id=new_id("decision"), campaign_id=self.campaign.campaign_id,
            kind="promotion_request", target_id=candidate_id, evidence_ids=tuple(evidence_ids),
            policy_version="agent-request@1", rationale=rationale,
        )
        self.campaign.ledger.decide(decision)
        return decision

    def _require_active(self) -> None:
        try:
            self.campaign.require_active()
        except CampaignError as exc:
            raise ToolError(str(exc)) from exc


__all__ = ["AgentTools", "ToolError"]
