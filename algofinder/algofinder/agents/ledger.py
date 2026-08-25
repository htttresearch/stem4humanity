"""Append-only filesystem ledger for campaign evidence.

The ledger is the canonical research record.  JSON files hold immutable
objects; JSONL files hold append-only event streams.  Indexes and analytical
databases may be rebuilt from this directory, so no database is required to
recover a campaign after an interrupted process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Protocol

from algofinder.agents.contracts import (
    AgentRun,
    AgentSpec,
    Analysis,
    CampaignSpec,
    Candidate,
    ContractError,
    Decision,
    Evaluation,
    ExperimentSpec,
    Hypothesis,
    RECORD_KINDS,
    record_digest,
)
from algofinder.trace.serialize import strict_dumps

try:  # Unix is the supported evaluator environment; keep imports portable.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


class LedgerError(RuntimeError):
    """Raised when immutable campaign evidence is missing or inconsistent."""


class HasMapping(Protocol):
    def to_mapping(self) -> dict[str, Any]: ...


_DIRECTORIES = {
    "hypothesis": "hypotheses",
    "candidate": "candidates",
    "experiment": "experiments",
    "evaluation": "evaluations",
    "analysis": "analyses",
    "agent_run": "episodes",
}


class CampaignLedger:
    """Owns one campaign's immutable records and append-only event streams."""

    def __init__(self, campaigns_root: str | Path, campaign_id: str) -> None:
        self.campaigns_root = Path(campaigns_root).resolve()
        self.campaign_id = campaign_id
        self.root = self.campaigns_root / campaign_id

    @classmethod
    def create(
        cls,
        campaigns_root: str | Path,
        spec: CampaignSpec,
        agent_specs: Iterable[AgentSpec],
    ) -> "CampaignLedger":
        ledger = cls(campaigns_root, spec.campaign_id)
        if ledger.root.exists():
            raise LedgerError(f"campaign already exists: {spec.campaign_id}")
        supplied_list = list(agent_specs)
        supplied = {agent.agent_spec_id: agent for agent in supplied_list}
        if len(supplied) != len(supplied_list):
            raise LedgerError("agent specs contain duplicate ids")
        expected = set(spec.agent_spec_ids)
        if set(supplied) != expected:
            raise LedgerError(
                "agent specs must exactly match CampaignSpec.agent_spec_ids; "
                f"expected={sorted(expected)}, got={sorted(supplied)}"
            )
        ledger.root.mkdir(parents=True, exist_ok=False)
        for directory in ("agents", "hypotheses", "candidates", "experiments", "analyses", "evaluations", "episodes", "workspaces"):
            (ledger.root / directory).mkdir()
        ledger._write_immutable_path(ledger.root / "campaign.json", spec.to_mapping())
        for agent in supplied.values():
            ledger._write_immutable_path(
                ledger.root / "agents" / f"{agent.agent_spec_id}.json", agent.to_mapping()
            )
        return ledger

    @property
    def exists(self) -> bool:
        return (self.root / "campaign.json").is_file()

    def campaign(self) -> dict[str, Any]:
        return self.read_path(self.root / "campaign.json")

    def write(self, record: HasMapping) -> str:
        """Persist one immutable record and return its content digest.

        Replaying an identical object is idempotent; using an existing id for a
        different object fails loudly rather than silently replacing evidence.
        """
        mapping = record.to_mapping()
        if mapping.get("campaign_id") != self.campaign_id:
            raise LedgerError("record belongs to a different campaign")
        kind = mapping.get("kind")
        if kind not in RECORD_KINDS - {"campaign", "agent_spec", "decision"}:
            raise LedgerError(f"record kind cannot be stored here: {kind!r}")
        record_id = mapping.get("id")
        if not isinstance(record_id, str):
            raise LedgerError("record has no id")
        path = self._record_path(kind, record_id)
        digest = self._write_immutable_path(path, mapping)
        if kind == "agent_run":
            (path.parent / record_id / "events.jsonl").touch(exist_ok=True)
        return digest

    def decide(self, decision: Decision) -> str:
        """Persist an immutable decision and append it to the decision stream."""
        mapping = decision.to_mapping()
        if decision.campaign_id != self.campaign_id:
            raise LedgerError("decision belongs to a different campaign")
        path = self.root / "decisions" / f"{decision.decision_id}.json"
        path.parent.mkdir(exist_ok=True)
        digest = self._write_immutable_path(path, mapping)
        self.append("decisions.jsonl", {**mapping, "content_digest": digest})
        return digest

    def append_episode_event(self, agent_run_id: str, event: dict[str, Any]) -> None:
        """Append a JSON-safe tool or lifecycle event to an existing episode."""
        episode = self._record_path("agent_run", agent_run_id)
        if not episode.is_file():
            raise LedgerError(f"agent run does not exist: {agent_run_id}")
        self.append(f"episodes/{agent_run_id}/events.jsonl", event)

    def append(self, relative_path: str, record: dict[str, Any]) -> None:
        """Atomically append one strict-JSON record with a process lock."""
        target = self._safe_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (strict_dumps(record) + "\n").encode("utf-8")
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def read(self, kind: str, record_id: str) -> dict[str, Any]:
        return self.read_path(self._record_path(kind, record_id))

    def read_path(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        try:
            mapping = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise LedgerError(f"missing ledger record: {path}") from exc
        except json.JSONDecodeError as exc:
            raise LedgerError(f"invalid JSON in ledger record {path}: {exc}") from exc
        if not isinstance(mapping, dict):
            raise LedgerError(f"ledger record is not an object: {path}")
        recorded = mapping.pop("content_digest", None)
        try:
            actual = record_digest(mapping)
        except ContractError as exc:
            raise LedgerError(f"invalid ledger record {path}: {exc}") from exc
        if recorded is not None and recorded != actual:
            raise LedgerError(f"content digest mismatch for {path}")
        mapping["content_digest"] = actual
        return mapping

    def list(self, kind: str) -> list[dict[str, Any]]:
        """Return verified immutable objects of one kind ordered by record id."""
        if kind == "decision":
            directory = self.root / "decisions"
        elif kind in _DIRECTORIES:
            directory = self.root / _DIRECTORIES[kind]
        else:
            raise LedgerError(f"unsupported ledger kind {kind!r}")
        if not directory.exists():
            return []
        records: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.json")):
            records.append(self.read_path(path))
        return records

    def query(self, kind: str, **filters: Any) -> list[dict[str, Any]]:
        """Small deterministic query surface; DuckDB can index this later."""
        result: list[dict[str, Any]] = []
        for record in self.list(kind):
            if all(record.get(key) == value for key, value in filters.items()):
                result.append(record)
        return result

    def _record_path(self, kind: str, record_id: str) -> Path:
        if kind not in _DIRECTORIES:
            raise LedgerError(f"unsupported record kind {kind!r}")
        return self.root / _DIRECTORIES[kind] / f"{record_id}.json"

    def _safe_path(self, relative_path: str) -> Path:
        if Path(relative_path).is_absolute():
            raise LedgerError("ledger paths must be relative")
        target = (self.root / relative_path).resolve()
        try:
            target.relative_to(self.root.resolve())
        except ValueError as exc:
            raise LedgerError("ledger path escapes campaign root") from exc
        return target

    def _write_immutable_path(self, path: Path, mapping: dict[str, Any]) -> str:
        payload = dict(mapping)
        digest = record_digest(payload)
        payload["content_digest"] = digest
        encoded = (strict_dumps(payload) + "\n").encode("utf-8")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            current = self.read_path(path)
            if current["content_digest"] != digest:
                raise LedgerError(f"immutable record id collision: {path.name}")
            return digest
        try:
            os.write(fd, encoded)
            os.fsync(fd)
        finally:
            os.close(fd)
        return digest


Record = AgentRun | Analysis | Candidate | Evaluation | ExperimentSpec | Hypothesis

__all__ = ["CampaignLedger", "LedgerError", "Record"]
