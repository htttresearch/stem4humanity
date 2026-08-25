"""Campaign lifecycle and budget accounting.

Campaign configuration is immutable in :mod:`algofinder.agents.ledger`; this
module maintains only a restartable state cache and an append-only budget log.
Every lifecycle change is also written as a typed ``Decision`` record so the
cache can be reconstructed after corruption or process interruption.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from algofinder.agents.contracts import (
    AgentSpec,
    BudgetLimits,
    CampaignLifecycle,
    CampaignSpec,
    ContractError,
    Decision,
    EvaluationStage,
    new_id,
    utc_now,
)
from algofinder.agents.ledger import CampaignLedger, LedgerError
from algofinder.trace.serialize import strict_dumps

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


class CampaignError(RuntimeError):
    """Raised when an operation violates campaign lifecycle or budget policy."""


_TRANSITIONS: dict[CampaignLifecycle, frozenset[CampaignLifecycle]] = {
    "draft": frozenset({"active", "stopped"}),
    "active": frozenset({"exhausted", "stopped", "review"}),
    "exhausted": frozenset({"review"}),
    "stopped": frozenset({"review"}),
    "review": frozenset({"closed"}),
    "closed": frozenset(),
}


@dataclass(frozen=True)
class BudgetStatus:
    limits: BudgetLimits
    used: dict[str, float | int]

    def remaining(self, key: str) -> float | int | None:
        limit_map: dict[str, float | int | None] = {
            "tokens": self.limits.tokens,
            "wall_seconds": self.limits.wall_seconds,
            "cpu_seconds": self.limits.cpu_seconds,
            "evaluation_seconds": self.limits.evaluation_seconds,
            "validation_submissions": self.limits.validation_submissions,
            "challenge_submissions": self.limits.challenge_submissions,
        }
        limit = limit_map[key]
        return None if limit is None else max(0, limit - self.used.get(key, 0))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "limits": self.limits.to_mapping(),
            "used": dict(self.used),
            "remaining": {
                key: self.remaining(key)
                for key in (
                    "tokens", "wall_seconds", "cpu_seconds", "evaluation_seconds",
                    "validation_submissions", "challenge_submissions",
                )
            },
        }


class Campaign:
    """A restartable campaign handle; it never mutates its specification."""

    def __init__(self, ledger: CampaignLedger) -> None:
        if not ledger.exists:
            raise CampaignError(f"campaign does not exist: {ledger.campaign_id}")
        self.ledger = ledger
        self._spec = ledger.campaign()

    @classmethod
    def create(
        cls,
        campaigns_root: str | Path,
        spec: CampaignSpec,
        agent_specs: Iterable[AgentSpec],
    ) -> "Campaign":
        ledger = CampaignLedger.create(campaigns_root, spec, agent_specs)
        campaign = cls(ledger)
        campaign._write_state("draft", reason="campaign created")
        return campaign

    @classmethod
    def open(cls, campaigns_root: str | Path, campaign_id: str) -> "Campaign":
        return cls(CampaignLedger(campaigns_root, campaign_id))

    @property
    def campaign_id(self) -> str:
        return self.ledger.campaign_id

    @property
    def root(self) -> Path:
        return self.ledger.root

    @property
    def spec(self) -> dict[str, Any]:
        """Verified immutable campaign specification."""
        return dict(self._spec)

    @property
    def state(self) -> CampaignLifecycle:
        state_path = self.root / "state.json"
        if not state_path.exists():
            return "draft"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CampaignError(f"invalid campaign state cache: {exc}") from exc
        value = state.get("lifecycle")
        if value not in _TRANSITIONS:
            raise CampaignError(f"invalid campaign lifecycle value {value!r}")
        return value

    def transition(self, target: CampaignLifecycle, *, reason: str, actor: str = "control-plane") -> None:
        current = self.state
        if target not in _TRANSITIONS[current]:
            raise CampaignError(f"cannot transition campaign from {current!r} to {target!r}")
        if not reason.strip():
            raise CampaignError("lifecycle transition needs a reason")
        decision = Decision(
            decision_id=new_id("decision"),
            campaign_id=self.campaign_id,
            kind="lifecycle",
            target_id=target,
            evidence_ids=(),
            policy_version="campaign-lifecycle@1",
            rationale=f"{actor}: {reason}; previous_state={current}",
        )
        self.ledger.decide(decision)
        self._write_state(target, reason=reason)

    def require_active(self) -> None:
        if self.state != "active":
            raise CampaignError(f"campaign must be active, currently {self.state!r}")

    def budget_status(self) -> BudgetStatus:
        used: dict[str, float | int] = {
            "tokens": 0,
            "wall_seconds": 0.0,
            "cpu_seconds": 0.0,
            "evaluation_seconds": 0.0,
            "validation_submissions": 0,
            "challenge_submissions": 0,
        }
        path = self.root / "budget.jsonl"
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                for line_no, raw in enumerate(handle, start=1):
                    if not raw.strip():
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise CampaignError(f"invalid budget event at line {line_no}") from exc
                    for key, amount in event.get("amounts", {}).items():
                        if key not in used or not isinstance(amount, (int, float)):
                            raise CampaignError(f"invalid budget amount {key!r} at line {line_no}")
                        used[key] += amount
        limits = self._limits()
        return BudgetStatus(limits=limits, used=used)

    def reserve_budget(self, *, purpose: str, amounts: dict[str, float | int]) -> BudgetStatus:
        """Atomically account for a cost before starting work.

        Reservations are deliberately never released.  A cancelled worker still
        consumed some scarce campaign capacity, and recording that fact makes
        later cost analyses honest.
        """
        self.require_active()
        if not purpose.strip():
            raise CampaignError("budget reservation needs a purpose")
        allowed = {
            "tokens", "wall_seconds", "cpu_seconds", "evaluation_seconds",
            "validation_submissions", "challenge_submissions",
        }
        if not amounts or set(amounts) - allowed:
            raise CampaignError(f"invalid budget dimensions {sorted(set(amounts) - allowed)}")
        if any(not isinstance(value, (int, float)) or value < 0 for value in amounts.values()):
            raise CampaignError("budget amounts must be non-negative numbers")

        lock_path = self.root / "budget.lock"
        lock_path.touch(exist_ok=True)
        with open(lock_path, "a+", encoding="utf-8") as lock:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                status = self.budget_status()
                for key, amount in amounts.items():
                    remaining = status.remaining(key)
                    if remaining is not None and amount > remaining:
                        self._exhaust_if_needed(
                            f"budget denied for {purpose}: {key} requires {amount}, remaining {remaining}"
                        )
                        raise CampaignError(
                            f"budget exceeded for {key}: requested {amount}, remaining {remaining}"
                        )
                self.ledger.append("budget.jsonl", {
                    "event": "reservation",
                    "purpose": purpose,
                    "amounts": amounts,
                    "created_at": utc_now(),
                })
                return self.budget_status()
            finally:
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def reserve_evaluation(self, stage: EvaluationStage, *, evaluation_seconds: float) -> BudgetStatus:
        """Reserve evaluator time and the appropriate holdout submission slot."""
        amounts: dict[str, float | int] = {"evaluation_seconds": evaluation_seconds}
        zone = self._zone_for_stage(stage)
        if zone == "validation":
            amounts["validation_submissions"] = 1
        elif zone == "challenge":
            amounts["challenge_submissions"] = 1
        return self.reserve_budget(purpose=f"evaluation:{stage}", amounts=amounts)

    def snapshot(self, *, agent_visible: bool = False) -> dict[str, Any]:
        """Stable observation for an agent episode; challenge paths are omitted."""
        campaign = dict(self._spec)
        if agent_visible:
            campaign["suites"] = [
                {key: value for key, value in suite.items() if key != "manifest_path" or suite.get("zone") != "challenge"}
                for suite in campaign.get("suites", [])
            ]
        return {
            "campaign": campaign,
            "campaign_state": self.state,
            "budget": self.budget_status().to_mapping(),
        }

    def _limits(self) -> BudgetLimits:
        data = self._spec["budgets"]
        return BudgetLimits(**data)

    def _zone_for_stage(self, stage: EvaluationStage) -> str:
        suites = [suite for suite in self._spec.get("suites", []) if suite["stage"] == stage]
        if not suites:
            raise CampaignError(f"campaign has no suite bound to {stage}")
        zones = {suite["zone"] for suite in suites}
        if len(zones) != 1:
            raise CampaignError(f"stage {stage} mixes feedback zones: {sorted(zones)}")
        return next(iter(zones))

    def _exhaust_if_needed(self, reason: str) -> None:
        if self.state == "active":
            self.transition("exhausted", reason=reason)

    def _write_state(self, lifecycle: CampaignLifecycle, *, reason: str) -> None:
        path = self.root / "state.json"
        data = {
            "schema_id": "algofinder.agents.state",
            "schema_version": "1",
            "campaign_id": self.campaign_id,
            "lifecycle": lifecycle,
            "reason": reason,
            "updated_at": utc_now(),
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(strict_dumps(data) + "\n", encoding="utf-8")
        os.replace(temporary, path)


__all__ = ["BudgetStatus", "Campaign", "CampaignError"]
