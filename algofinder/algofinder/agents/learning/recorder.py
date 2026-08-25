"""Crash-safe native provenance recording for research-agent episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import re
from typing import Any, Mapping

from algofinder.agents.contracts import AgentRun, new_id
from algofinder.agents.ledger import CampaignLedger
from algofinder.agents.learning.contracts import AgentEpisode, AgentTransition, ResearchAttempt
from algofinder.trace.serialize import canonical_dumps, to_json_value


class RecorderError(RuntimeError):
    """Raised when unsafe or inconsistent data is about to enter learning evidence."""


_SECRET_VALUE = re.compile(r"(?i)(?:\bsk-[a-z0-9_-]{12,}|\bghp_[a-z0-9]{12,}|\b(?:api[_-]?key|authorization|bearer)\s*[:=]\s*[^\s,;]+)")
_SECRET_KEY = re.compile(r"(?i)(?:api[_-]?key|authorization|password|secret|access[_-]?token|bearer)")
_FORBIDDEN_PATH = re.compile(r"(?i)(?:^|[/\\])(?:challenge|sealed|private)(?:[/\\]|$)")


def _safe_value(value: Any) -> Any:
    """Redact explicit secret fields and reject unstructured secret/path leaks."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            result[text_key] = "<redacted>" if _SECRET_KEY.search(text_key) else _safe_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if isinstance(value, str):
        if _SECRET_VALUE.search(value):
            raise RecorderError("refusing to store text that appears to contain a secret")
        if _FORBIDDEN_PATH.search(value):
            raise RecorderError("refusing to store a challenge, sealed, or private path in learning evidence")
    return value


def _state_digest(value: Mapping[str, Any]) -> str:
    return sha256(canonical_dumps(_safe_value(value)).encode("utf-8")).hexdigest()


@dataclass
class AttemptHandle:
    attempt_id: str
    episode_id: str
    attempt_number: int
    operator: str
    parent_ids: tuple[str, ...]
    state_before_digest: str
    current_state_digest: str
    transition_ids: list[str] = field(default_factory=list)
    finished: bool = False


class EpisodeRecorder:
    """Persist native data before each risky agent action.

    The recorder is intentionally independent of model and evaluator classes.
    An interrupted process therefore leaves a usable episode plus event-stream
    checkpoints, while only the final attempt/run summaries may be absent.
    """

    def __init__(self, ledger: CampaignLedger) -> None:
        self.ledger = ledger
        self.episode: AgentEpisode | None = None
        self._attempts: list[str] = []
        self._next_attempt_number = 1

    @property
    def episode_id(self) -> str:
        if self.episode is None:
            raise RecorderError("episode has not been started")
        return self.episode.episode_id

    def start(
        self,
        *,
        agent_spec_id: str,
        campaign_snapshot: Mapping[str, Any],
        model_identity: Mapping[str, Any],
        policy_id: str | None = None,
        policy_digest: str | None = None,
        distribution_profile_id: str | None = None,
        evidence_role: str = "public_learning",
        rng_seed: int | None = None,
    ) -> AgentEpisode:
        if self.episode is not None:
            raise RecorderError("a recorder can own exactly one episode")
        snapshot_digest = _state_digest(campaign_snapshot)
        episode = AgentEpisode(
            episode_id=new_id("episode"),
            campaign_id=self.ledger.campaign_id,
            agent_spec_id=agent_spec_id,
            policy_id=policy_id,
            policy_digest=policy_digest,
            campaign_snapshot_digest=snapshot_digest,
            model_identity=dict(_safe_value(model_identity)),
            distribution_profile_id=distribution_profile_id,
            evidence_role=evidence_role,  # validated by the immutable contract
            rng_seed=rng_seed,
        )
        self.ledger.write(episode)
        self.episode = episode
        self.ledger.append_episode_event(episode.episode_id, {
            "event": "episode_started",
            "episode_id": episode.episode_id,
            "campaign_snapshot_digest": snapshot_digest,
        })
        return episode

    def begin_attempt(
        self,
        *,
        attempt_number: int | None = None,
        operator: str,
        parent_ids: tuple[str, ...],
        state_before: Mapping[str, Any],
    ) -> AttemptHandle:
        selected_number = self._next_attempt_number if attempt_number is None else attempt_number
        if selected_number != self._next_attempt_number:
            raise RecorderError(
                f"attempt_number must be the next episode ordinal ({self._next_attempt_number})"
            )
        self._next_attempt_number += 1
        before_digest = _state_digest(state_before)
        handle = AttemptHandle(
            attempt_id=new_id("attempt"),
            episode_id=self.episode_id,
            attempt_number=selected_number,
            operator=operator,
            parent_ids=parent_ids,
            state_before_digest=before_digest,
            current_state_digest=before_digest,
        )
        self.ledger.append_episode_event(self.episode_id, {
            "event": "attempt_started",
            "attempt_id": handle.attempt_id,
            "attempt_number": selected_number,
            "operator": operator,
            "parent_ids": list(parent_ids),
            "state_before_digest": handle.state_before_digest,
        })
        return handle

    def record_transition(
        self,
        handle: AttemptHandle,
        *,
        context: Mapping[str, Any],
        action: Mapping[str, Any],
        behavior_policy: Mapping[str, Any],
        validation: Mapping[str, Any],
        terminal_status: str,
        exchange: Mapping[str, Any] | None = None,
        state_after: Mapping[str, Any] | None = None,
        candidate_id: str | None = None,
        hypothesis_id: str | None = None,
        evaluation_ids: tuple[str, ...] = (),
        rejection_code: str | None = None,
        cost_vector: Mapping[str, float | int] | None = None,
    ) -> AgentTransition:
        self._validate_handle(handle)
        after_digest = _state_digest(state_after) if state_after is not None else None
        safe_context = _safe_value(context)
        safe_exchange = _safe_value(exchange or {})
        request = safe_exchange.get("request") if isinstance(safe_exchange, Mapping) else None
        response = safe_exchange.get("response") if isinstance(safe_exchange, Mapping) else None
        transition = AgentTransition(
            transition_id=new_id("transition"),
            campaign_id=self.ledger.campaign_id,
            episode_id=self.episode_id,
            attempt_id=handle.attempt_id,
            ordinal=len(handle.transition_ids),
            state_before_digest=handle.current_state_digest,
            state_after_digest=after_digest,
            context_blob_digest=self._blob(safe_context),
            request_blob_digest=self._blob(request) if request is not None else None,
            response_blob_digest=self._blob(response) if response is not None else None,
            action=dict(_safe_value(action)),
            behavior_policy=dict(_safe_value(behavior_policy)),
            validation=dict(_safe_value(validation)),
            terminal_status=terminal_status,
            candidate_id=candidate_id,
            hypothesis_id=hypothesis_id,
            evaluation_ids=evaluation_ids,
            rejection_code=rejection_code,
            cost_vector=dict(cost_vector or {}),
        )
        self.ledger.write(transition)
        handle.transition_ids.append(transition.transition_id)
        if after_digest is not None:
            handle.current_state_digest = after_digest
        self.ledger.append_episode_event(self.episode_id, {
            "event": "transition_recorded",
            "attempt_id": handle.attempt_id,
            "transition_id": transition.transition_id,
            "terminal_status": terminal_status,
        })
        return transition

    def finish_attempt(
        self,
        handle: AttemptHandle,
        *,
        terminal_status: str,
        state_after: Mapping[str, Any] | None,
        candidate_id: str | None = None,
        hypothesis_id: str | None = None,
        evaluation_ids: tuple[str, ...] = (),
        reward: Mapping[str, Any] | None = None,
        cost_vector: Mapping[str, float | int] | None = None,
    ) -> ResearchAttempt:
        self._validate_handle(handle)
        if not handle.transition_ids:
            raise RecorderError("cannot finish an attempt with no recorded transition")
        attempt = ResearchAttempt(
            attempt_id=handle.attempt_id,
            campaign_id=self.ledger.campaign_id,
            episode_id=self.episode_id,
            attempt_number=handle.attempt_number,
            operator=handle.operator,
            parent_ids=handle.parent_ids,
            state_before_digest=handle.state_before_digest,
            state_after_digest=_state_digest(state_after) if state_after is not None else None,
            transition_ids=tuple(handle.transition_ids),
            terminal_status=terminal_status,
            candidate_id=candidate_id,
            hypothesis_id=hypothesis_id,
            evaluation_ids=evaluation_ids,
            reward=dict(_safe_value(reward or {})),
            cost_vector=dict(cost_vector or {}),
        )
        self.ledger.write(attempt)
        handle.finished = True
        self._attempts.append(attempt.attempt_id)
        self.ledger.append_episode_event(self.episode_id, {
            "event": "attempt_finished",
            "attempt_id": attempt.attempt_id,
            "terminal_status": terminal_status,
            "candidate_id": candidate_id,
        })
        return attempt

    def finish_run(
        self,
        *,
        inputs: Mapping[str, Any],
        produced_candidate_ids: tuple[str, ...],
        token_cost: int = 0,
        monetary_cost: float | None = None,
        terminal_outcome: str = "completed",
        error: str | None = None,
        completion_reason: str | None = None,
        cost_vector: Mapping[str, float | int] | None = None,
    ) -> AgentRun:
        if self.episode is None:
            raise RecorderError("episode has not been started")
        run = AgentRun(
            agent_run_id=new_id("agent-run"),
            campaign_id=self.ledger.campaign_id,
            agent_spec_id=self.episode.agent_spec_id,
            campaign_snapshot_digest=self.episode.campaign_snapshot_digest,
            inputs=dict(_safe_value(inputs)),
            tool_calls=(),
            produced_candidate_ids=produced_candidate_ids,
            token_cost=token_cost,
            monetary_cost=monetary_cost,
            terminal_outcome=terminal_outcome,  # validated by base contract
            error=error,
            episode_id=self.episode_id,
            policy_id=self.episode.policy_id,
            attempt_ids=tuple(self._attempts),
            cost_vector=dict(cost_vector or {}),
            completion_reason=completion_reason,
        )
        self.ledger.write(run)
        self.ledger.append_episode_event(self.episode_id, {
            "event": "episode_finished",
            "agent_run_id": run.agent_run_id,
            "terminal_outcome": terminal_outcome,
            "attempt_ids": list(self._attempts),
        })
        return run

    def _blob(self, value: Any) -> str:
        return self.ledger.put_blob(canonical_dumps(to_json_value(value)))

    def _validate_handle(self, handle: AttemptHandle) -> None:
        if handle.episode_id != self.episode_id:
            raise RecorderError("attempt handle belongs to another episode")
        if handle.finished:
            raise RecorderError("attempt has already been finished")


__all__ = ["AttemptHandle", "EpisodeRecorder", "RecorderError"]
