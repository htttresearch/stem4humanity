"""Immutable records for agent-learning provenance and policy artifacts.

These records complement the campaign control-plane contracts.  Episode,
attempt, and transition records live in a campaign ledger; policy, dataset,
and evaluation records live in the learning artifact store.  In both cases
their payloads are JSON-safe and content-addressed by their writer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import re
from typing import Any, Literal

from algofinder.agents.contracts import ContractError, new_id, utc_now
from algofinder.trace.serialize import canonical_dumps, to_json_value


SCHEMA_ID = "algofinder.agents.learning"
SCHEMA_VERSION = "2"

_SAFE_ID = re.compile(r"^[a-z][a-z0-9-]{2,79}$")
_HEX = re.compile(r"^[0-9a-f]{64}$")
_BLOB = re.compile(r"^sha256:[0-9a-f]{64}$")
_EVIDENCE_ROLES = frozenset({
    "public_learning", "historical_training", "outer_validation", "outer_test", "sealed",
    # Compatibility names are accepted for callers migrating from the initial
    # experimental implementation, but writers use the names above.
    "train", "dev", "locked_test", "holdout", "shadow",
})
_BACKFILL = frozenset({
    "native_exact", "backfilled_exact", "backfilled_partial", "unresolved",
    "native", "synthetic", "unavailable",
})


def _id(value: str, label: str) -> None:
    if not _SAFE_ID.fullmatch(value):
        raise ContractError(f"{label} must be a lowercase opaque id")


def _hex(value: str, label: str) -> None:
    if not _HEX.fullmatch(value):
        raise ContractError(f"{label} must be a sha256 hex digest")


def _blob(value: str | None, label: str) -> None:
    if value is not None and not _BLOB.fullmatch(value):
        raise ContractError(f"{label} must use sha256:<hex>")


def _ids(values: tuple[str, ...], label: str) -> None:
    if len(set(values)) != len(values) or any(not isinstance(value, str) or not value for value in values):
        raise ContractError(f"{label} must contain unique non-empty ids")


def learning_digest(mapping: dict[str, Any]) -> str:
    """Return the canonical content digest for a learning artifact mapping."""
    return sha256(canonical_dumps(mapping).encode("utf-8")).hexdigest()


def _record(kind: str, record_id: str, fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": kind,
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "id": record_id,
        **to_json_value(fields),
    }


@dataclass(frozen=True)
class AgentEpisode:
    """Stable envelope for all actions taken during one agent invocation."""

    episode_id: str
    campaign_id: str
    agent_spec_id: str
    policy_id: str | None
    policy_digest: str | None
    campaign_snapshot_digest: str
    model_identity: dict[str, Any]
    distribution_profile_id: str | None = None
    evidence_role: str = "public_learning"
    rng_seed: int | None = None
    backfill_status: str = "native_exact"
    backfill_confidence: float = 1.0
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for label, value in (("episode_id", self.episode_id), ("campaign_id", self.campaign_id), ("agent_spec_id", self.agent_spec_id)):
            _id(value, label)
        if self.policy_id is not None:
            _id(self.policy_id, "policy_id")
        if (self.policy_id is None) != (self.policy_digest is None):
            raise ContractError("policy_id and policy_digest must be set together")
        if self.policy_digest is not None:
            _hex(self.policy_digest, "policy_digest")
        _hex(self.campaign_snapshot_digest, "campaign_snapshot_digest")
        if self.distribution_profile_id is not None:
            _id(self.distribution_profile_id, "distribution_profile_id")
        if self.evidence_role not in _EVIDENCE_ROLES or self.backfill_status not in _BACKFILL:
            raise ContractError("invalid evidence role or backfill status")
        if self.rng_seed is not None and self.rng_seed < 0:
            raise ContractError("rng_seed must be non-negative")
        if not 0.0 <= self.backfill_confidence <= 1.0:
            raise ContractError("backfill_confidence must be in [0, 1]")
        to_json_value(self.model_identity)

    def to_mapping(self) -> dict[str, Any]:
        return _record("agent_episode", self.episode_id, {
            "campaign_id": self.campaign_id,
            "agent_spec_id": self.agent_spec_id,
            "policy_id": self.policy_id,
            "policy_digest": self.policy_digest,
            "campaign_snapshot_digest": self.campaign_snapshot_digest,
            "model_identity": self.model_identity,
            "distribution_profile_id": self.distribution_profile_id,
            "evidence_role": self.evidence_role,
            "rng_seed": self.rng_seed,
            "backfill_status": self.backfill_status,
            "backfill_confidence": self.backfill_confidence,
            "created_at": self.created_at,
        })

    @property
    def digest(self) -> str:
        return learning_digest(self.to_mapping())


@dataclass(frozen=True)
class AgentTransition:
    """One replayable decision/response/validation transition in an episode."""

    transition_id: str
    campaign_id: str
    episode_id: str
    attempt_id: str
    ordinal: int
    state_before_digest: str
    context_blob_digest: str
    request_blob_digest: str | None
    response_blob_digest: str | None
    action: dict[str, Any]
    behavior_policy: dict[str, Any]
    validation: dict[str, Any]
    terminal_status: str
    state_after_digest: str | None = None
    candidate_id: str | None = None
    hypothesis_id: str | None = None
    evaluation_ids: tuple[str, ...] = ()
    rejection_code: str | None = None
    cost_vector: dict[str, float | int] = field(default_factory=dict)
    backfill_status: str = "native_exact"
    backfill_confidence: float = 1.0
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for label, value in (("transition_id", self.transition_id), ("campaign_id", self.campaign_id), ("episode_id", self.episode_id), ("attempt_id", self.attempt_id)):
            _id(value, label)
        if self.ordinal < 0:
            raise ContractError("transition ordinal must be non-negative")
        _hex(self.state_before_digest, "state_before_digest")
        if self.state_after_digest is not None:
            _hex(self.state_after_digest, "state_after_digest")
        for label, value in (("context_blob_digest", self.context_blob_digest), ("request_blob_digest", self.request_blob_digest), ("response_blob_digest", self.response_blob_digest)):
            _blob(value, label)
        for label, value in (("candidate_id", self.candidate_id), ("hypothesis_id", self.hypothesis_id)):
            if value is not None:
                _id(value, label)
        _ids(self.evaluation_ids, "evaluation_ids")
        if not self.terminal_status:
            raise ContractError("terminal_status is required")
        if self.backfill_status not in _BACKFILL:
            raise ContractError("invalid backfill status")
        if not 0.0 <= self.backfill_confidence <= 1.0:
            raise ContractError("backfill_confidence must be in [0, 1]")
        if any(not isinstance(value, (int, float)) or value < 0 for value in self.cost_vector.values()):
            raise ContractError("cost vector must contain non-negative numbers")
        to_json_value(self.action)
        to_json_value(self.behavior_policy)
        to_json_value(self.validation)

    def to_mapping(self) -> dict[str, Any]:
        return _record("agent_transition", self.transition_id, {
            "campaign_id": self.campaign_id,
            "episode_id": self.episode_id,
            "attempt_id": self.attempt_id,
            "ordinal": self.ordinal,
            "state_before_digest": self.state_before_digest,
            "state_after_digest": self.state_after_digest,
            "context_blob_digest": self.context_blob_digest,
            "request_blob_digest": self.request_blob_digest,
            "response_blob_digest": self.response_blob_digest,
            "action": self.action,
            "behavior_policy": self.behavior_policy,
            "validation": self.validation,
            "terminal_status": self.terminal_status,
            "candidate_id": self.candidate_id,
            "hypothesis_id": self.hypothesis_id,
            "evaluation_ids": list(self.evaluation_ids),
            "rejection_code": self.rejection_code,
            "cost_vector": self.cost_vector,
            "backfill_status": self.backfill_status,
            "backfill_confidence": self.backfill_confidence,
            "created_at": self.created_at,
        })

    @property
    def digest(self) -> str:
        return learning_digest(self.to_mapping())


@dataclass(frozen=True)
class ResearchAttempt:
    """Terminal summary of one proposal attempt, including failures."""

    attempt_id: str
    campaign_id: str
    episode_id: str
    attempt_number: int
    operator: str
    parent_ids: tuple[str, ...]
    state_before_digest: str
    state_after_digest: str | None
    transition_ids: tuple[str, ...]
    terminal_status: str
    candidate_id: str | None = None
    hypothesis_id: str | None = None
    evaluation_ids: tuple[str, ...] = ()
    reward: dict[str, Any] = field(default_factory=dict)
    cost_vector: dict[str, float | int] = field(default_factory=dict)
    backfill_status: str = "native_exact"
    backfill_confidence: float = 1.0
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for label, value in (("attempt_id", self.attempt_id), ("campaign_id", self.campaign_id), ("episode_id", self.episode_id)):
            _id(value, label)
        if self.attempt_number < 1 or not self.operator:
            raise ContractError("attempt_number and operator are required")
        _ids(self.parent_ids, "parent_ids")
        _ids(self.transition_ids, "transition_ids")
        if not self.transition_ids:
            raise ContractError("a finished research attempt requires at least one transition")
        _ids(self.evaluation_ids, "evaluation_ids")
        _hex(self.state_before_digest, "state_before_digest")
        if self.state_after_digest is not None:
            _hex(self.state_after_digest, "state_after_digest")
        for label, value in (("candidate_id", self.candidate_id), ("hypothesis_id", self.hypothesis_id)):
            if value is not None:
                _id(value, label)
        if not self.terminal_status or self.backfill_status not in _BACKFILL:
            raise ContractError("invalid attempt terminal status or backfill status")
        if not 0.0 <= self.backfill_confidence <= 1.0:
            raise ContractError("backfill_confidence must be in [0, 1]")
        if any(not isinstance(value, (int, float)) or value < 0 for value in self.cost_vector.values()):
            raise ContractError("cost vector must contain non-negative numbers")
        to_json_value(self.reward)

    def to_mapping(self) -> dict[str, Any]:
        return _record("research_attempt", self.attempt_id, {
            "campaign_id": self.campaign_id,
            "episode_id": self.episode_id,
            "attempt_number": self.attempt_number,
            "operator": self.operator,
            "parent_ids": list(self.parent_ids),
            "state_before_digest": self.state_before_digest,
            "state_after_digest": self.state_after_digest,
            "transition_ids": list(self.transition_ids),
            "terminal_status": self.terminal_status,
            "candidate_id": self.candidate_id,
            "hypothesis_id": self.hypothesis_id,
            "evaluation_ids": list(self.evaluation_ids),
            "reward": self.reward,
            "cost_vector": self.cost_vector,
            "backfill_status": self.backfill_status,
            "backfill_confidence": self.backfill_confidence,
            "created_at": self.created_at,
        })

    @property
    def digest(self) -> str:
        return learning_digest(self.to_mapping())


@dataclass(frozen=True)
class AgentPolicy:
    policy_id: str
    policy_kind: Literal["baseline", "learned"]
    policy_version: str
    representation: str
    feature_schema_digest: str
    normalizer_digest: str | None
    parameter_blob_digest: str | None
    memory_id: str | None
    memory_snapshot_digest: str | None
    training_dataset_digest: str | None
    evidence_role: str = "public_learning"
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _id(self.policy_id, "policy_id")
        if self.policy_kind not in {"baseline", "learned"} or not self.policy_version or not self.representation:
            raise ContractError("policy kind, version, and representation are required")
        _hex(self.feature_schema_digest, "feature_schema_digest")
        if self.memory_id is not None:
            _id(self.memory_id, "memory_id")
        for label, value in (("normalizer_digest", self.normalizer_digest), ("parameter_blob_digest", self.parameter_blob_digest), ("memory_snapshot_digest", self.memory_snapshot_digest), ("training_dataset_digest", self.training_dataset_digest)):
            _blob(value, label)
        if (self.memory_id is None) != (self.memory_snapshot_digest is None):
            raise ContractError("memory_id and memory_snapshot_digest must be set together")
        if self.policy_kind == "learned" and (self.parameter_blob_digest is None or self.training_dataset_digest is None):
            raise ContractError("learned policies require parameters and a training dataset")
        if self.evidence_role not in _EVIDENCE_ROLES:
            raise ContractError("invalid policy evidence role")

    def to_mapping(self) -> dict[str, Any]:
        return _record("agent_policy", self.policy_id, {
            "policy_kind": self.policy_kind,
            "policy_version": self.policy_version,
            "representation": self.representation,
            "feature_schema_digest": self.feature_schema_digest,
            "normalizer_digest": self.normalizer_digest,
            "parameter_blob_digest": self.parameter_blob_digest,
            "memory_id": self.memory_id,
            "memory_snapshot_digest": self.memory_snapshot_digest,
            "training_dataset_digest": self.training_dataset_digest,
            "evidence_role": self.evidence_role,
            "created_at": self.created_at,
        })

    @property
    def digest(self) -> str:
        return learning_digest(self.to_mapping())


@dataclass(frozen=True)
class DistributionProfile:
    profile_id: str
    family: str
    version: str
    parameter_schema: dict[str, Any]
    seed_set: tuple[int, ...]
    split_rule: str
    profile_blob_digest: str | None = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _id(self.profile_id, "profile_id")
        if not self.family or not self.version or not self.split_rule or not self.seed_set or any(seed < 0 for seed in self.seed_set):
            raise ContractError("distribution profile requires family, version, split rule, and non-negative seeds")
        if len(set(self.seed_set)) != len(self.seed_set):
            raise ContractError("distribution profile seeds must be unique")
        _blob(self.profile_blob_digest, "profile_blob_digest")
        to_json_value(self.parameter_schema)

    def to_mapping(self) -> dict[str, Any]:
        return _record("distribution_profile", self.profile_id, {
            "family": self.family,
            "version": self.version,
            "parameter_schema": self.parameter_schema,
            "seed_set": list(self.seed_set),
            "split_rule": self.split_rule,
            "profile_blob_digest": self.profile_blob_digest,
            "created_at": self.created_at,
        })

    @property
    def digest(self) -> str:
        return learning_digest(self.to_mapping())


@dataclass(frozen=True)
class LearningRun:
    learning_run_id: str
    command: str
    input_digests: dict[str, str]
    output_digests: dict[str, str]
    config: dict[str, Any]
    code_digest: str | None = None
    status: Literal["completed", "failed", "cancelled"] = "completed"
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _id(self.learning_run_id, "learning_run_id")
        if not self.command:
            raise ContractError("learning run command is required")
        for label, values in (("input", self.input_digests), ("output", self.output_digests)):
            if any(not isinstance(key, str) or not _HEX.fullmatch(value) for key, value in values.items()):
                raise ContractError(f"learning run {label} digests must be sha256 hex values")
        if self.code_digest is not None:
            _hex(self.code_digest, "code_digest")
        if self.status not in {"completed", "failed", "cancelled"}:
            raise ContractError("invalid learning run status")
        to_json_value(self.config)

    def to_mapping(self) -> dict[str, Any]:
        return _record("learning_run", self.learning_run_id, {
            "command": self.command,
            "input_digests": self.input_digests,
            "output_digests": self.output_digests,
            "config": self.config,
            "code_digest": self.code_digest,
            "status": self.status,
            "created_at": self.created_at,
        })

    @property
    def digest(self) -> str:
        return learning_digest(self.to_mapping())


@dataclass(frozen=True)
class AgentEvaluation:
    evaluation_id: str
    policy_id: str
    distribution_profile_id: str
    evidence_role: str
    metrics: dict[str, Any]
    paired_policy_id: str | None = None
    trajectory_blob_digest: str | None = None
    phase: str = "planned"
    policy_arms: tuple[str, ...] = ()
    model_arms: tuple[str, ...] = ()
    task_profile_ids: tuple[str, ...] = ()
    budgets: dict[str, Any] = field(default_factory=dict)
    integrity: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _id(self.evaluation_id, "evaluation_id")
        _id(self.policy_id, "policy_id")
        _id(self.distribution_profile_id, "distribution_profile_id")
        if self.paired_policy_id is not None:
            _id(self.paired_policy_id, "paired_policy_id")
        if self.evidence_role not in _EVIDENCE_ROLES:
            raise ContractError("invalid evaluation evidence role")
        if not self.phase:
            raise ContractError("agent evaluation phase is required")
        _ids(self.policy_arms, "policy_arms")
        _ids(self.task_profile_ids, "task_profile_ids")
        if any(not isinstance(item, str) or not item for item in self.model_arms):
            raise ContractError("model_arms must contain non-empty strings")
        _blob(self.trajectory_blob_digest, "trajectory_blob_digest")
        to_json_value(self.metrics)
        to_json_value(self.budgets)
        to_json_value(self.integrity)
        to_json_value(self.decision)

    def to_mapping(self) -> dict[str, Any]:
        return _record("agent_evaluation", self.evaluation_id, {
            "policy_id": self.policy_id,
            "distribution_profile_id": self.distribution_profile_id,
            "evidence_role": self.evidence_role,
            "metrics": self.metrics,
            "paired_policy_id": self.paired_policy_id,
            "trajectory_blob_digest": self.trajectory_blob_digest,
            "phase": self.phase,
            "policy_arms": list(self.policy_arms),
            "model_arms": list(self.model_arms),
            "task_profile_ids": list(self.task_profile_ids),
            "budgets": self.budgets,
            "integrity": self.integrity,
            "decision": self.decision,
            "created_at": self.created_at,
        })

    @property
    def digest(self) -> str:
        return learning_digest(self.to_mapping())


__all__ = [
    "AgentEpisode", "AgentEvaluation", "AgentPolicy", "AgentTransition",
    "DistributionProfile", "LearningRun", "ResearchAttempt", "learning_digest",
    "new_id",
]
