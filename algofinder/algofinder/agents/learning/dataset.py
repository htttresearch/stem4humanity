"""Deterministic dataset manifests, campaign-grouped splits, and Parquet projections."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

from algofinder.agents.ledger import CampaignLedger
from algofinder.agents.learning.features import FEATURE_SCHEMA_DIGEST, project_attempt
from algofinder.agents.learning.store import LearningArtifactStore
from algofinder.trace.serialize import canonical_dumps


class DatasetError(RuntimeError):
    """Raised when an experience projection cannot be safely materialized."""


def native_attempt_rows(campaigns_root: str | Path, campaign_ids: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for campaign_id in sorted(set(campaign_ids)):
        ledger = CampaignLedger(campaigns_root, campaign_id)
        campaign = ledger.campaign()
        episodes = {item["id"]: item for item in ledger.list("agent_episode")}
        agent_specs = {item["id"]: item for item in ledger.list("agent_spec")}
        transitions = {item["id"]: item for item in ledger.list("agent_transition")}
        for attempt in ledger.list("research_attempt"):
            transition_ids = attempt.get("transition_ids", ())
            attempt_transitions = [transitions[item] for item in transition_ids if item in transitions]
            proposal_transition = next(
                (
                    item for item in attempt_transitions
                    if isinstance(item.get("action"), dict)
                    and item.get("action", {}).get("kind") != "analysis"
                ),
                attempt_transitions[0] if attempt_transitions else None,
            )
            enriched = dict(attempt)
            episode = episodes.get(attempt.get("episode_id"))
            agent_spec = agent_specs.get(episode.get("agent_spec_id")) if isinstance(episode, dict) else None
            objective = campaign.get("objective", {}) if isinstance(campaign.get("objective"), dict) else {}
            seed = episode.get("rng_seed") if isinstance(episode, dict) else None
            source_digest = objective.get("experiment_source_digest")
            profile_id = episode.get("distribution_profile_id") if isinstance(episode, dict) else None
            enriched.update({
                "split_group_id": (
                    f"{source_digest or 'unknown-source'}:{profile_id or 'unknown-profile'}:seed-{seed}"
                    if isinstance(seed, int) else campaign_id
                ),
                "source_digest": source_digest,
                "prompt_digest": agent_spec.get("prompt_digest") if isinstance(agent_spec, dict) else None,
                "search_policy_version": campaign.get("search_policy_version"),
                "tool_api_version": agent_spec.get("tool_api_version") if isinstance(agent_spec, dict) else None,
            })
            if isinstance(proposal_transition, dict):
                enriched["policy_decision"] = proposal_transition.get("action", {})
                behavior = proposal_transition.get("behavior_policy", {})
                if isinstance(behavior, dict):
                    enriched["selection_probability"] = behavior.get("selection_probability")
            rows.append(project_attempt(enriched, episode=episode))
    return rows


def historical_backfill_rows(store: LearningArtifactStore) -> list[dict[str, Any]]:
    root = store.root / "backfills"
    result: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        record = store.read(path.relative_to(store.root))
        for attempt in record.get("attempts", []):
            if isinstance(attempt, dict):
                result.append(project_attempt(attempt))
    return result


def build_dataset(
    *,
    store: LearningArtifactStore,
    rows: Iterable[dict[str, Any]],
    dataset_id: str,
    split_seed: int = 75348,
    source_description: str = "native and/or append-only historical projections",
) -> dict[str, Any]:
    """Write deterministic Parquet rows grouped by campaign for split safety."""
    rows = sorted((dict(row) for row in rows), key=lambda row: (str(row.get("campaign_id")), str(row.get("attempt_id"))))
    if not rows:
        raise DatasetError("cannot build an empty learning dataset")
    forbidden_roles = {"sealed", "outer_test", "locked_test", "holdout", "shadow"}
    leaked = sorted({str(row.get("evidence_role")) for row in rows if row.get("evidence_role") in forbidden_roles})
    if leaked:
        raise DatasetError(f"outer-test or sealed evidence cannot enter a learning dataset: {leaked}")
    group_by_campaign: dict[str, str] = {}
    for row in rows:
        campaign_id = str(row.get("campaign_id"))
        group_id = str(row.get("split_group_id") or campaign_id)
        incumbent = group_by_campaign.setdefault(campaign_id, group_id)
        if incumbent != group_id:
            raise DatasetError(f"campaign {campaign_id} spans multiple split groups")
    group_assignments = _split_assignments(set(group_by_campaign.values()), split_seed)
    assignments = {
        campaign_id: group_assignments[group_id]
        for campaign_id, group_id in sorted(group_by_campaign.items())
    }
    for row in rows:
        row["split"] = assignments[str(row.get("campaign_id"))]
    root = store.root / "datasets" / dataset_id
    root.mkdir(parents=True, exist_ok=True)
    parquet = root / "attempts.parquet"
    if parquet.exists() or (root / "manifest.json").exists():
        raise DatasetError(f"dataset id already exists and is immutable: {dataset_id}")
    _write_parquet(rows, parquet)
    digest = sha256(parquet.read_bytes()).hexdigest()
    manifest = {
        "dataset_id": dataset_id,
        "source_description": source_description,
        "feature_schema_digest": FEATURE_SCHEMA_DIGEST,
        "split_plan": {
            "unit": "matched_campaign_group",
            "grouping_rule": "matched campaign group (source, distribution profile, and agent seed)",
            "seed": split_seed,
            "campaign_assignments": assignments,
            "group_assignments": group_assignments,
            "row_counts": {split: sum(row["split"] == split for row in rows) for split in ("train", "dev", "test")},
            "campaign_counts": {split: sum(value == split for value in assignments.values()) for split in ("train", "dev", "test")},
        },
        "tables": [{"name": "attempts", "path": "attempts.parquet", "sha256": digest, "rows": len(rows)}],
        "input_row_digest": sha256(canonical_dumps(rows).encode("utf-8")).hexdigest(),
        "sealed_data_used": False,
        "provenance": {
            key: sorted({str(row[key]) for row in rows if row.get(key) is not None})
            for key in ("source_digest", "prompt_digest", "search_policy_version", "tool_api_version")
        },
    }
    manifest_digest = store.write_dataset_manifest(dataset_id, manifest)
    return {"dataset_id": dataset_id, "manifest_digest": manifest_digest, **manifest}


def _split_assignments(group_ids: set[str], seed: int) -> dict[str, str]:
    ordered = sorted(
        group_ids,
        key=lambda item: (sha256(f"{seed}:{item}".encode("utf-8")).hexdigest(), item),
    )
    count = len(ordered)
    if count == 1:
        train_count, dev_count = 1, 0
    elif count == 2:
        train_count, dev_count = 1, 0
    else:
        dev_count = max(1, round(count * 0.15))
        test_count = max(1, round(count * 0.15))
        train_count = count - dev_count - test_count
        if train_count < 1:
            train_count, dev_count = 1, max(0, count - 2)
    return {
        group_id: ("train" if index < train_count else ("dev" if index < train_count + dev_count else "test"))
        for index, group_id in enumerate(ordered)
    }


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - project dependency supplies DuckDB.
        raise DatasetError("DuckDB is required to materialize Parquet datasets") from exc
    normalized = [
        {key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()}
        for row in rows
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", encoding="utf-8", delete=False) as handle:
        temp_path = Path(handle.name)
        for row in normalized:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
    try:
        escaped_source = str(temp_path).replace("'", "''")
        escaped_destination = str(path).replace("'", "''")
        with duckdb.connect() as connection:
            connection.execute(f"COPY (SELECT * FROM read_json_auto('{escaped_source}')) TO '{escaped_destination}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    finally:
        temp_path.unlink(missing_ok=True)


__all__ = ["DatasetError", "build_dataset", "historical_backfill_rows", "native_attempt_rows"]
