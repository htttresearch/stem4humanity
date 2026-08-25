"""Versioned, JSON-safe records for the AlgoFinder campaign control plane.

The records here are intentionally boring dataclasses rather than ORM models.
Campaign evidence must remain inspectable with a text editor and reproducible
without a service database.  Every record has an opaque id, an immutable
content digest, a schema identifier, and explicit causal links.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import re
from typing import Any, Literal
from uuid import uuid4

from algofinder.trace.serialize import canonical_dumps, to_json_value

SCHEMA_ID = "algofinder.agents"
SCHEMA_VERSION = "1"

CampaignLifecycle = Literal[
    "draft", "active", "exhausted", "stopped", "review", "closed"
]
FeedbackZone = Literal["public", "validation", "challenge"]
EvaluationStage = Literal[
    "gate_0", "gate_1", "gate_2", "gate_3", "gate_4", "gate_5", "gate_6"
]

_SAFE_ID = re.compile(r"^[a-z][a-z0-9-]{2,79}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")
_ZONES = frozenset({"public", "validation", "challenge"})
_STAGES = frozenset({f"gate_{number}" for number in range(7)})


class ContractError(ValueError):
    """Raised when a campaign record would be ambiguous or unsafe."""


def utc_now() -> str:
    """Return an explicit, timezone-aware timestamp suitable for a record."""
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    """Create an opaque stable id; content digests provide the integrity check."""
    prefix = prefix.rstrip("-").lower()
    if not re.fullmatch(r"[a-z][a-z0-9-]*", prefix):
        raise ContractError(f"unsafe id prefix {prefix!r}")
    return f"{prefix}-{uuid4().hex}"


def _require_id(value: str, label: str) -> None:
    if not _SAFE_ID.fullmatch(value):
        raise ContractError(f"{label} must be a lowercase slug, got {value!r}")


def _strings(values: tuple[str, ...], label: str) -> None:
    if any(not isinstance(value, str) or not value for value in values):
        raise ContractError(f"{label} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ContractError(f"{label} must not contain duplicates")


def _digest(mapping: dict[str, Any]) -> str:
    return sha256(canonical_dumps(mapping).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BudgetLimits:
    """Hard campaign ceilings. ``None`` means the dimension is not budgeted."""

    tokens: int | None = None
    wall_seconds: float | None = None
    cpu_seconds: float | None = None
    evaluation_seconds: float | None = None
    validation_submissions: int = 0
    challenge_submissions: int = 0

    def __post_init__(self) -> None:
        for name in ("tokens", "wall_seconds", "cpu_seconds", "evaluation_seconds"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ContractError(f"budget {name} must be >= 0")
        if self.validation_submissions < 0 or self.challenge_submissions < 0:
            raise ContractError("submission budgets must be >= 0")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "evaluation_seconds": self.evaluation_seconds,
            "validation_submissions": self.validation_submissions,
            "challenge_submissions": self.challenge_submissions,
        }


@dataclass(frozen=True)
class SuiteBinding:
    """A suite reference exposed through one feedback zone.

    ``manifest_digest`` is always recorded. ``manifest_path`` may be absent for
    a sealed suite so agents never receive a filesystem locator for it.
    """

    name: str
    stage: EvaluationStage
    zone: FeedbackZone
    manifest_digest: str
    manifest_path: str | None = None
    splits: tuple[str, ...] = ()
    repetitions: int = 1

    def __post_init__(self) -> None:
        _require_id(self.name, "suite name")
        if self.stage not in _STAGES:
            raise ContractError(f"unknown evaluation stage {self.stage!r}")
        if self.zone not in _ZONES:
            raise ContractError(f"unknown feedback zone {self.zone!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", self.manifest_digest):
            raise ContractError("manifest_digest must be a sha256 hex digest")
        if self.repetitions < 1:
            raise ContractError("suite repetitions must be >= 1")
        _strings(self.splits, "suite splits")
        if self.zone == "challenge" and self.manifest_path is not None:
            raise ContractError("challenge suite paths must not appear in campaign records")

    def to_mapping(self, *, include_path: bool = True) -> dict[str, Any]:
        result = {
            "name": self.name,
            "stage": self.stage,
            "zone": self.zone,
            "manifest_digest": self.manifest_digest,
            "splits": list(self.splits),
            "repetitions": self.repetitions,
        }
        if include_path and self.manifest_path is not None:
            result["manifest_path"] = self.manifest_path
        return result


@dataclass(frozen=True)
class AgentSpec:
    """Fixed agent configuration for a campaign; API secrets are references."""

    agent_spec_id: str
    provider: str
    model: str
    prompt_digest: str
    tool_api_version: str
    context_policy: str
    sampling: dict[str, Any] = field(default_factory=dict)
    permissions: tuple[str, ...] = ()
    secret_references: tuple[str, ...] = ()
    policy_id: str | None = None
    policy_digest: str | None = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_id(self.agent_spec_id, "agent_spec_id")
        if not self.provider or not self.model or not self.prompt_digest:
            raise ContractError("provider, model, and prompt_digest are required")
        if not re.fullmatch(r"[0-9a-f]{64}", self.prompt_digest):
            raise ContractError("prompt_digest must be a sha256 hex digest")
        _strings(self.permissions, "permissions")
        _strings(self.secret_references, "secret_references")
        if any("=" in ref or ref.lower().startswith(("sk-", "ghp_")) for ref in self.secret_references):
            raise ContractError("secret_references must be names/locators, never secret values")
        if self.policy_id is not None:
            _require_id(self.policy_id, "policy_id")
        if self.policy_digest is not None and not re.fullmatch(r"[0-9a-f]{64}", self.policy_digest):
            raise ContractError("policy_digest must be a sha256 hex digest")
        if (self.policy_id is None) != (self.policy_digest is None):
            raise ContractError("policy_id and policy_digest must be set together")
        to_json_value(self.sampling)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "kind": "agent_spec",
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "id": self.agent_spec_id,
            "provider": self.provider,
            "model": self.model,
            "prompt_digest": self.prompt_digest,
            "tool_api_version": self.tool_api_version,
            "context_policy": self.context_policy,
            "sampling": to_json_value(self.sampling),
            "permissions": list(self.permissions),
            "secret_references": list(self.secret_references),
            "policy_id": self.policy_id,
            "policy_digest": self.policy_digest,
            "created_at": self.created_at,
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True)
class CampaignSpec:
    """Immutable control-plane configuration for one campaign."""

    campaign_id: str
    base_commit: str
    allowed_paths: tuple[str, ...]
    suites: tuple[SuiteBinding, ...]
    budgets: BudgetLimits
    agent_spec_ids: tuple[str, ...]
    search_policy_version: str
    objective: dict[str, Any]
    goal: str
    problem_scope: tuple[str, ...] = ()
    feedback_rounding: int = 4
    network_policy: Literal["deny", "allowlist"] = "deny"
    human_approval_stages: tuple[EvaluationStage, ...] = ("gate_6",)
    distribution_profile_id: str | None = None
    distribution_profile_digest: str | None = None
    evidence_partition_id: str | None = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_id(self.campaign_id, "campaign_id")
        if not _COMMIT.fullmatch(self.base_commit):
            raise ContractError("base_commit must be a Git commit hash")
        _strings(self.allowed_paths, "allowed_paths")
        if any(path.startswith("/") or ".." in path.split("/") for path in self.allowed_paths):
            raise ContractError("allowed_paths must be relative paths without '..'")
        if not self.allowed_paths:
            raise ContractError("a campaign must have a non-empty path allowlist")
        _strings(self.agent_spec_ids, "agent_spec_ids")
        _strings(self.problem_scope, "problem_scope")
        if not self.goal.strip():
            raise ContractError("campaign goal is required")
        if self.feedback_rounding < 0 or self.feedback_rounding > 12:
            raise ContractError("feedback_rounding must be between 0 and 12")
        if self.network_policy not in ("deny", "allowlist"):
            raise ContractError("network_policy must be 'deny' or 'allowlist'")
        stages = [suite.stage for suite in self.suites]
        if len({suite.name for suite in self.suites}) != len(self.suites):
            raise ContractError("suite names must be unique")
        if any(stage not in _STAGES for stage in self.human_approval_stages):
            raise ContractError("unknown human approval stage")
        if self.distribution_profile_id is not None:
            _require_id(self.distribution_profile_id, "distribution_profile_id")
        if self.distribution_profile_digest is not None and not re.fullmatch(r"[0-9a-f]{64}", self.distribution_profile_digest):
            raise ContractError("distribution_profile_digest must be a sha256 hex digest")
        if (self.distribution_profile_id is None) != (self.distribution_profile_digest is None):
            raise ContractError("distribution profile id and digest must be set together")
        if self.evidence_partition_id is not None:
            _require_id(self.evidence_partition_id, "evidence_partition_id")
        to_json_value(self.objective)

    def to_mapping(self, *, agent_visible: bool = False) -> dict[str, Any]:
        suites = [
            suite.to_mapping(include_path=not (agent_visible and suite.zone == "challenge"))
            for suite in self.suites
        ]
        return {
            "kind": "campaign",
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "id": self.campaign_id,
            "base_commit": self.base_commit,
            "allowed_paths": list(self.allowed_paths),
            "suites": suites,
            "budgets": self.budgets.to_mapping(),
            "agent_spec_ids": list(self.agent_spec_ids),
            "search_policy_version": self.search_policy_version,
            "objective": to_json_value(self.objective),
            "goal": self.goal,
            "problem_scope": list(self.problem_scope),
            "feedback_rounding": self.feedback_rounding,
            "network_policy": self.network_policy,
            "human_approval_stages": list(self.human_approval_stages),
            "distribution_profile_id": self.distribution_profile_id,
            "distribution_profile_digest": self.distribution_profile_digest,
            "evidence_partition_id": self.evidence_partition_id,
            "created_at": self.created_at,
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    campaign_id: str
    mechanism: str
    affected_strata: tuple[str, ...]
    predicted_tradeoffs: str
    falsification_test: str
    supporting_sources: tuple[str, ...] = ()
    confidence: float = 0.5
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_id(self.hypothesis_id, "hypothesis_id")
        _require_id(self.campaign_id, "campaign_id")
        if not self.mechanism.strip() or not self.falsification_test.strip():
            raise ContractError("hypotheses require a mechanism and falsification test")
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractError("hypothesis confidence must be in [0, 1]")
        _strings(self.affected_strata, "affected_strata")
        _strings(self.supporting_sources, "supporting_sources")

    def to_mapping(self) -> dict[str, Any]:
        return _record_mapping("hypothesis", self.hypothesis_id, self.campaign_id, {
            "mechanism": self.mechanism,
            "affected_strata": list(self.affected_strata),
            "predicted_tradeoffs": self.predicted_tradeoffs,
            "falsification_test": self.falsification_test,
            "supporting_sources": list(self.supporting_sources),
            "confidence": self.confidence,
            "created_at": self.created_at,
        })

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    campaign_id: str
    hypothesis_id: str
    parent_ids: tuple[str, ...]
    generation_operator: Literal["invent", "mutate", "recombine", "repair", "tune", "distill"]
    solver_entrypoints: tuple[str, ...] = ()
    producing_agent_run_id: str | None = None
    producing_episode_id: str | None = None
    patch_digest: str | None = None
    build_digest: str | None = None
    descriptors: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_id(self.candidate_id, "candidate_id")
        _require_id(self.campaign_id, "campaign_id")
        _require_id(self.hypothesis_id, "hypothesis_id")
        _strings(self.parent_ids, "parent_ids")
        _strings(self.solver_entrypoints, "solver_entrypoints")
        if any(":" not in entrypoint for entrypoint in self.solver_entrypoints):
            raise ContractError("solver_entrypoints must use module:Class syntax")
        if self.generation_operator not in {"invent", "mutate", "recombine", "repair", "tune", "distill"}:
            raise ContractError(f"unsupported generation operator {self.generation_operator!r}")
        if self.producing_agent_run_id is not None:
            _require_id(self.producing_agent_run_id, "producing_agent_run_id")
        if self.producing_episode_id is not None:
            _require_id(self.producing_episode_id, "producing_episode_id")
        for digest in (self.patch_digest, self.build_digest):
            if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ContractError("candidate digests must be sha256 hex digests")
        to_json_value(self.descriptors)

    def to_mapping(self) -> dict[str, Any]:
        return _record_mapping("candidate", self.candidate_id, self.campaign_id, {
            "hypothesis_id": self.hypothesis_id,
            "parent_ids": list(self.parent_ids),
            "generation_operator": self.generation_operator,
            "solver_entrypoints": list(self.solver_entrypoints),
            "producing_agent_run_id": self.producing_agent_run_id,
            "producing_episode_id": self.producing_episode_id,
            "patch_digest": self.patch_digest,
            "build_digest": self.build_digest,
            "descriptors": to_json_value(self.descriptors),
            "created_at": self.created_at,
        })

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    campaign_id: str
    candidate_id: str
    baseline_candidate_ids: tuple[str, ...]
    suite_names: tuple[str, ...]
    stage: EvaluationStage
    seeds: tuple[int, ...]
    budget_seconds: float | None
    stopping_rule: str
    expected_result: str
    decision_rule: str
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_id(self.experiment_id, "experiment_id")
        _require_id(self.campaign_id, "campaign_id")
        _require_id(self.candidate_id, "candidate_id")
        _strings(self.baseline_candidate_ids, "baseline_candidate_ids")
        _strings(self.suite_names, "suite_names")
        if not self.suite_names:
            raise ContractError("an experiment requires at least one suite")
        if self.stage not in _STAGES:
            raise ContractError(f"unknown evaluation stage {self.stage!r}")
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise ContractError("experiments require non-negative seeds")
        if self.budget_seconds is not None and self.budget_seconds <= 0:
            raise ContractError("experiment budget_seconds must be positive")
        if not all((self.stopping_rule, self.expected_result, self.decision_rule)):
            raise ContractError("experiments must declare stopping, expected, and decision rules")

    def to_mapping(self) -> dict[str, Any]:
        return _record_mapping("experiment", self.experiment_id, self.campaign_id, {
            "candidate_id": self.candidate_id,
            "baseline_candidate_ids": list(self.baseline_candidate_ids),
            "suite_names": list(self.suite_names),
            "stage": self.stage,
            "seeds": list(self.seeds),
            "budget_seconds": self.budget_seconds,
            "stopping_rule": self.stopping_rule,
            "expected_result": self.expected_result,
            "decision_rule": self.decision_rule,
            "created_at": self.created_at,
        })

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True)
class Evaluation:
    evaluation_id: str
    campaign_id: str
    experiment_id: str
    candidate_id: str
    stage: EvaluationStage
    zone: FeedbackZone
    gate: Literal["pass", "fail", "blocked"]
    benchmark_session_ids: tuple[str, ...] = ()
    benchmark_run_ids: tuple[str, ...] = ()
    metric_vector: dict[str, Any] = field(default_factory=dict)
    paired_baselines: dict[str, Any] = field(default_factory=dict)
    feedback: dict[str, Any] = field(default_factory=dict)
    evaluator_build_digest: str | None = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_id(self.evaluation_id, "evaluation_id")
        for label, value in (("campaign_id", self.campaign_id), ("experiment_id", self.experiment_id), ("candidate_id", self.candidate_id)):
            _require_id(value, label)
        if self.stage not in _STAGES or self.zone not in _ZONES:
            raise ContractError("evaluation stage or feedback zone is invalid")
        if self.gate not in ("pass", "fail", "blocked"):
            raise ContractError("gate must be pass, fail, or blocked")
        if self.evaluator_build_digest is not None and not re.fullmatch(r"[0-9a-f]{64}", self.evaluator_build_digest):
            raise ContractError("evaluator_build_digest must be sha256 hex")
        to_json_value(self.metric_vector)
        to_json_value(self.paired_baselines)
        to_json_value(self.feedback)

    def to_mapping(self) -> dict[str, Any]:
        return _record_mapping("evaluation", self.evaluation_id, self.campaign_id, {
            "experiment_id": self.experiment_id,
            "candidate_id": self.candidate_id,
            "stage": self.stage,
            "zone": self.zone,
            "gate": self.gate,
            "benchmark_session_ids": list(self.benchmark_session_ids),
            "benchmark_run_ids": list(self.benchmark_run_ids),
            "metric_vector": to_json_value(self.metric_vector),
            "paired_baselines": to_json_value(self.paired_baselines),
            "feedback": to_json_value(self.feedback),
            "evaluator_build_digest": self.evaluator_build_digest,
            "created_at": self.created_at,
        })

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True)
class Analysis:
    analysis_id: str
    campaign_id: str
    evaluation_ids: tuple[str, ...]
    observation: str
    interpretation: str
    confidence: float
    next_test: str
    trace_regions: tuple[str, ...] = ()
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_id(self.analysis_id, "analysis_id")
        _require_id(self.campaign_id, "campaign_id")
        _strings(self.evaluation_ids, "evaluation_ids")
        if not self.evaluation_ids:
            raise ContractError("analysis must cite at least one evaluation")
        if not all((self.observation.strip(), self.interpretation.strip(), self.next_test.strip())):
            raise ContractError("analysis must contain observation, interpretation, and next_test")
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractError("analysis confidence must be in [0, 1]")

    def to_mapping(self) -> dict[str, Any]:
        return _record_mapping("analysis", self.analysis_id, self.campaign_id, {
            "evaluation_ids": list(self.evaluation_ids),
            "observation": self.observation,
            "interpretation": self.interpretation,
            "confidence": self.confidence,
            "next_test": self.next_test,
            "trace_regions": list(self.trace_regions),
            "created_at": self.created_at,
        })

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True)
class Decision:
    decision_id: str
    campaign_id: str
    kind: Literal["parent_selection", "early_stop", "archive", "validation_admission", "rejection", "promotion_request", "lifecycle"]
    target_id: str
    evidence_ids: tuple[str, ...]
    policy_version: str
    rationale: str
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_id(self.decision_id, "decision_id")
        _require_id(self.campaign_id, "campaign_id")
        if self.kind not in {"parent_selection", "early_stop", "archive", "validation_admission", "rejection", "promotion_request", "lifecycle"}:
            raise ContractError(f"unknown decision kind {self.kind!r}")
        if not self.target_id or not self.rationale.strip() or not self.policy_version:
            raise ContractError("decision target, policy_version, and rationale are required")
        _strings(self.evidence_ids, "evidence_ids")

    def to_mapping(self) -> dict[str, Any]:
        return _record_mapping("decision", self.decision_id, self.campaign_id, {
            "decision_kind": self.kind,
            "target_id": self.target_id,
            "evidence_ids": list(self.evidence_ids),
            "policy_version": self.policy_version,
            "rationale": self.rationale,
            "created_at": self.created_at,
        })

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True)
class AgentRun:
    agent_run_id: str
    campaign_id: str
    agent_spec_id: str
    campaign_snapshot_digest: str
    inputs: dict[str, Any]
    tool_calls: tuple[dict[str, Any], ...]
    produced_candidate_ids: tuple[str, ...]
    token_cost: int = 0
    monetary_cost: float | None = None
    terminal_outcome: Literal["completed", "failed", "cancelled"] = "completed"
    error: str | None = None
    episode_id: str | None = None
    policy_id: str | None = None
    attempt_ids: tuple[str, ...] = ()
    cost_vector: dict[str, float | int] = field(default_factory=dict)
    completion_reason: str | None = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for label, value in (("agent_run_id", self.agent_run_id), ("campaign_id", self.campaign_id), ("agent_spec_id", self.agent_spec_id)):
            _require_id(value, label)
        if not re.fullmatch(r"[0-9a-f]{64}", self.campaign_snapshot_digest):
            raise ContractError("campaign_snapshot_digest must be sha256 hex")
        if self.token_cost < 0 or (self.monetary_cost is not None and self.monetary_cost < 0):
            raise ContractError("agent costs must be non-negative")
        if self.terminal_outcome not in ("completed", "failed", "cancelled"):
            raise ContractError("agent terminal_outcome is invalid")
        _strings(self.produced_candidate_ids, "produced_candidate_ids")
        _strings(self.attempt_ids, "attempt_ids")
        if self.episode_id is not None:
            _require_id(self.episode_id, "episode_id")
        if self.policy_id is not None:
            _require_id(self.policy_id, "policy_id")
        if any(not isinstance(value, (int, float)) or value < 0 for value in self.cost_vector.values()):
            raise ContractError("agent cost_vector values must be non-negative numbers")
        to_json_value(self.inputs)
        to_json_value(self.tool_calls)

    def to_mapping(self) -> dict[str, Any]:
        mapping = _record_mapping("agent_run", self.agent_run_id, self.campaign_id, {
            "agent_spec_id": self.agent_spec_id,
            "campaign_snapshot_digest": self.campaign_snapshot_digest,
            "inputs": to_json_value(self.inputs),
            "tool_calls": to_json_value(self.tool_calls),
            "produced_candidate_ids": list(self.produced_candidate_ids),
            "token_cost": self.token_cost,
            "monetary_cost": self.monetary_cost,
            "terminal_outcome": self.terminal_outcome,
            "error": self.error,
            "episode_id": self.episode_id,
            "policy_id": self.policy_id,
            "attempt_ids": list(self.attempt_ids),
            "cost_vector": to_json_value(self.cost_vector),
            "completion_reason": self.completion_reason,
            "created_at": self.created_at,
        })
        mapping["schema_version"] = "2"
        return mapping

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping())


def _record_mapping(kind: str, record_id: str, campaign_id: str, fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": kind,
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "id": record_id,
        "campaign_id": campaign_id,
        **fields,
    }


RECORD_KINDS = frozenset({
    "agent_spec", "campaign", "hypothesis", "candidate", "experiment",
    "evaluation", "analysis", "decision", "agent_run", "agent_episode",
    "research_attempt", "agent_transition",
})


def record_digest(mapping: dict[str, Any]) -> str:
    """Digest a persisted record after rejecting non-JSON values."""
    if mapping.get("kind") not in RECORD_KINDS:
        raise ContractError(f"unknown record kind {mapping.get('kind')!r}")
    return _digest(mapping)


__all__ = [
    "AgentRun", "AgentSpec", "Analysis", "BudgetLimits", "CampaignLifecycle",
    "CampaignSpec", "Candidate", "ContractError", "Decision", "Evaluation",
    "EvaluationStage", "ExperimentSpec", "FeedbackZone", "Hypothesis", "RECORD_KINDS",
    "SCHEMA_ID", "SCHEMA_VERSION", "SuiteBinding", "new_id", "record_digest", "utc_now",
]
