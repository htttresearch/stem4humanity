"""Trusted evaluator-facing gates and evidence recording.

There is deliberately no agent-facing method for submitting a score.  The
authority owns the candidate build, calls this module after deterministic
benchmark execution, and records raw evidence privately before feedback is
redacted for the agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from algofinder.agents.campaign import Campaign, CampaignError
from algofinder.agents.contracts import Candidate, Evaluation, ExperimentSpec, new_id
from algofinder.agents.feedback import FeedbackPolicy
from algofinder.agents.workspace import CandidateWorkspace, ValidationReport, WorkspaceService


class EvaluationError(RuntimeError):
    """Raised when an evaluator request violates a campaign boundary."""


@dataclass(frozen=True)
class GateOutcome:
    evaluation: Evaluation
    agent_feedback: dict[str, Any]


class EvaluationAuthority:
    """Control-plane entry point used by evaluator-owned workers only."""

    def __init__(self, campaign: Campaign, workspaces: WorkspaceService) -> None:
        self.campaign = campaign
        self.workspaces = workspaces
        self.feedback = FeedbackPolicy(rounding=int(campaign.spec["feedback_rounding"]))

    def gate_0(self, workspace: CandidateWorkspace, experiment: ExperimentSpec) -> GateOutcome:
        """Run the policy/build gate and record its complete private report."""
        self._check_experiment(experiment, expected_stage="gate_0")
        report = self.workspaces.validate(workspace)
        candidate = self._candidate_from_workspace(workspace, report)
        raw = {
            "gate": "pass" if report.passed else "fail",
            "stage": "gate_0",
            "reason": "build" if not report.passed else "ok",
            "metric_vector": {
                "changed_paths": len(report.changed_paths),
                "compile_commands": len(report.commands),
            },
            "summary": {"errors": list(report.errors)},
        }
        return self._record(experiment, candidate, raw, evaluator_build_digest=report.build_digest)

    def record_worker_result(
        self,
        *,
        experiment: ExperimentSpec,
        candidate: Candidate,
        raw_result: dict[str, Any],
        evaluator_build_digest: str | None = None,
        benchmark_session_ids: tuple[str, ...] = (),
        benchmark_run_ids: tuple[str, ...] = (),
    ) -> GateOutcome:
        """Record a result supplied by a trusted evaluator worker, never an agent."""
        self._check_experiment(experiment)
        if candidate.campaign_id != self.campaign.campaign_id:
            raise EvaluationError("candidate belongs to another campaign")
        return self._record(
            experiment,
            candidate,
            raw_result,
            evaluator_build_digest=evaluator_build_digest,
            benchmark_session_ids=benchmark_session_ids,
            benchmark_run_ids=benchmark_run_ids,
        )

    def gate_1_public(
        self,
        *,
        workspace: CandidateWorkspace,
        candidate: Candidate,
        experiment: ExperimentSpec,
        adapter: "PublicBenchmarkAdapter",
        manifest_path: str,
        timeout_seconds: float,
    ) -> GateOutcome:
        """Run the public smoke gate through an evaluator-owned adapter."""
        self._check_experiment(experiment, expected_stage="gate_1")
        if self._zone("gate_1") != "public":
            raise EvaluationError("gate_1 must be bound to a public suite")
        if len(experiment.suite_names) != 1:
            raise EvaluationError("the local public adapter runs one suite per experiment")
        suite = next(
            (item for item in self.campaign.spec["suites"] if item["name"] == experiment.suite_names[0]),
            None,
        )
        if suite is None:
            raise EvaluationError("experiment references an unknown suite")
        raw = adapter.evaluate(
            workspace=workspace, candidate=candidate, experiment=experiment,
            manifest_path=manifest_path, splits=tuple(suite.get("splits", ("test",))),
            timeout_seconds=timeout_seconds,
        )
        return self._record(
            experiment, candidate, raw, evaluator_build_digest=candidate.build_digest,
        )

    def _record(
        self,
        experiment: ExperimentSpec,
        candidate: Candidate,
        raw: dict[str, Any],
        *,
        evaluator_build_digest: str | None,
        benchmark_session_ids: tuple[str, ...] = (),
        benchmark_run_ids: tuple[str, ...] = (),
    ) -> GateOutcome:
        zone = self._zone(experiment.stage)
        gate = raw.get("gate")
        if gate not in ("pass", "fail", "blocked"):
            raise EvaluationError("trusted evaluator result requires gate=pass|fail|blocked")
        evaluation = Evaluation(
            evaluation_id=new_id("evaluation"),
            campaign_id=self.campaign.campaign_id,
            experiment_id=experiment.experiment_id,
            candidate_id=candidate.candidate_id,
            stage=experiment.stage,
            zone=zone,  # type: ignore[arg-type]
            gate=gate,
            benchmark_session_ids=benchmark_session_ids,
            benchmark_run_ids=benchmark_run_ids,
            metric_vector=dict(raw.get("metric_vector", {})),
            paired_baselines=dict(raw.get("paired_baselines", {})),
            feedback=dict(raw),
            evaluator_build_digest=evaluator_build_digest,
        )
        self.campaign.ledger.write(evaluation)
        return GateOutcome(evaluation=evaluation, agent_feedback=self.feedback.redact(raw, zone))

    def _check_experiment(self, experiment: ExperimentSpec, expected_stage: str | None = None) -> None:
        try:
            self.campaign.require_active()
        except CampaignError as exc:
            raise EvaluationError(str(exc)) from exc
        if experiment.campaign_id != self.campaign.campaign_id:
            raise EvaluationError("experiment belongs to another campaign")
        if expected_stage is not None and experiment.stage != expected_stage:
            raise EvaluationError(f"expected {expected_stage}, got {experiment.stage}")
        suite_names = {suite["name"] for suite in self.campaign.spec["suites"]}
        missing = set(experiment.suite_names) - suite_names
        if missing:
            raise EvaluationError(f"experiment names unknown suites: {sorted(missing)}")
        seconds = experiment.budget_seconds or 0.0
        self.campaign.reserve_evaluation(experiment.stage, evaluation_seconds=seconds)

    def _zone(self, stage: str) -> str:
        zones = {suite["zone"] for suite in self.campaign.spec["suites"] if suite["stage"] == stage}
        if len(zones) != 1:
            raise EvaluationError(f"stage {stage} does not have exactly one feedback zone")
        return next(iter(zones))

    def _candidate_from_workspace(self, workspace: CandidateWorkspace, report: ValidationReport) -> Candidate:
        try:
            mapping = self.campaign.ledger.read("candidate", workspace.candidate_id)
        except Exception as exc:
            raise EvaluationError("gate_0 requires a frozen candidate") from exc
        if mapping.get("build_digest") != report.build_digest:
            raise EvaluationError("candidate build digest changed after freezing")
        # A trusted evaluator needs only the immutable identity fields here.
        return Candidate(
            candidate_id=mapping["id"], campaign_id=mapping["campaign_id"],
            hypothesis_id=mapping["hypothesis_id"],
            parent_ids=tuple(mapping["parent_ids"]),
            generation_operator=mapping["generation_operator"],
            solver_entrypoints=tuple(mapping.get("solver_entrypoints", ())),
            producing_agent_run_id=mapping.get("producing_agent_run_id"),
            patch_digest=mapping.get("patch_digest"),
            build_digest=mapping.get("build_digest"),
            descriptors=dict(mapping.get("descriptors", {})),
            created_at=mapping["created_at"],
        )


__all__ = ["EvaluationAuthority", "EvaluationError", "GateOutcome"]
