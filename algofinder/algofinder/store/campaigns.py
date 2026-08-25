"""Derived registry entries for immutable agent-campaign ledgers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from algofinder.agents.contracts import ContractError, record_digest
from algofinder.store.registry import CAMPAIGN_KINDS, write_kind


class CampaignIndexError(RuntimeError):
    """Raised when campaign evidence cannot be safely indexed."""


_SOURCES = {
    "campaigns": ("campaign.json",),
    "agents": ("agents/*.json",),
    "hypotheses": ("hypotheses/*.json",),
    "candidates": ("candidates/*.json",),
    "experiments": ("experiments/*.json",),
    "evaluations": ("evaluations/*.json",),
    "analyses": ("analyses/*.json",),
    "decisions": ("decisions/*.json",),
    "episodes": ("episodes/*.json",),
    "attempts": ("attempts/*.json",),
    "transitions": ("transitions/*.json",),
}


def build_campaign_registry(
    *, campaigns_root: str | Path, registry_dir: str | Path
) -> dict[str, int]:
    """Rebuild campaign registry kinds without touching core benchmark kinds."""
    campaigns_root = Path(campaigns_root)
    entries: dict[str, list[dict[str, Any]]] = {kind: [] for kind in CAMPAIGN_KINDS}
    if campaigns_root.exists():
        for campaign_dir in sorted(path for path in campaigns_root.iterdir() if path.is_dir()):
            campaign_path = campaign_dir / "campaign.json"
            if not campaign_path.exists():
                continue
            for kind, patterns in _SOURCES.items():
                for pattern in patterns:
                    for path in sorted(campaign_dir.glob(pattern)):
                        record = _read_verified(path)
                        if kind == "campaigns" and record.get("kind") != "campaign":
                            raise CampaignIndexError(f"{path} is not a campaign record")
                        entries[kind].append(_summary(record, path))
    return {kind: len(write_kind(registry_dir, kind, entries[kind])) for kind in CAMPAIGN_KINDS}


def _read_verified(path: Path) -> dict[str, Any]:
    try:
        mapping = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CampaignIndexError(f"invalid campaign JSON {path}: {exc}") from exc
    if not isinstance(mapping, dict):
        raise CampaignIndexError(f"campaign record is not an object: {path}")
    recorded = mapping.pop("content_digest", None)
    if not isinstance(recorded, str):
        raise CampaignIndexError(f"campaign record has no content digest: {path}")
    try:
        actual = record_digest(mapping)
    except ContractError as exc:
        raise CampaignIndexError(f"invalid campaign record {path}: {exc}") from exc
    if recorded != actual:
        raise CampaignIndexError(f"campaign record digest mismatch: {path}")
    mapping["content_digest"] = actual
    return mapping


def _summary(record: dict[str, Any], path: Path) -> dict[str, Any]:
    result = {
        "id": record["id"],
        "campaign_id": record.get("campaign_id", record["id"]),
        "record_kind": record["kind"],
        "content_digest": record["content_digest"],
        "source": str(path),
        "schema_id": record.get("schema_id"),
        "schema_version": record.get("schema_version"),
    }
    for key in (
        "stage", "zone", "gate", "candidate_id", "experiment_id", "agent_spec_id",
        "episode_id", "attempt_id", "producing_episode_id", "producing_agent_run_id",
        "policy_id", "created_at",
    ):
        if key in record:
            result[key] = record[key]
    return result


__all__ = ["CampaignIndexError", "build_campaign_registry"]
