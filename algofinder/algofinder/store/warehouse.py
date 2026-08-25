"""Rebuildable provenance warehouse for AlgoFinder experiment artifacts.

The experiment directories and their immutable ledgers remain the source of
truth.  This module only inventories and projects those files into DuckDB; it
never rewrites an experiment.  Small, stable text evidence is copied into a
content-addressed corpus so a source tree can move without making the
provenance graph opaque.  Large/binary evidence remains in place and is
verified through ``catalog.artifact_location``.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Iterator, Sequence
from uuid import uuid4

import duckdb


WAREHOUSE_SCHEMA_VERSION = "1"
DEFAULT_DB_RELATIVE_PATH = Path("private/db/algofinder.duckdb")
DEFAULT_CORPUS_RELATIVE_PATH = Path("private/corpus/algofinder")
STABLE_TEXT_SUFFIXES = {
    ".json", ".jsonl", ".md", ".txt", ".patch", ".diff", ".py",
    ".toml", ".yaml", ".yml", ".csv", ".tsv",
}
MAX_CORPUS_TEXT_BYTES = 4 * 1024 * 1024
INGEST_TRANSACTION_SIZE = 250
PROHIBITED_PATH_PARTS = {"challenge", "sealed", "private_challenge"}
SECRET_PATTERNS = (
    re.compile(rb"(?:sk|rk|pk)-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
    re.compile(rb"(?i)(?:api[_-]?key|password|access[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{16,}"),
)


class WarehouseError(RuntimeError):
    """Raised when source evidence cannot be safely cataloged."""


class WarehouseInterrupted(WarehouseError):
    """Intentional interruption hook used to exercise resume semantics."""


@dataclass(frozen=True)
class SourceRoot:
    path: Path
    source_class: str
    lifecycle_state: str
    summary_path: Path | None = None

    @property
    def source_root_id(self) -> str:
        return sha256(str(self.path.resolve()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DiscoveredArtifact:
    source: SourceRoot
    path: Path
    relative_path: str
    kind: str
    campaign_id: str | None
    experiment_id: str | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _digest_file(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _json_object(path: Path) -> dict[str, Any] | None:
    if path.suffix.lower() != ".json":
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _source_class(path: Path, summary: dict[str, Any] | None) -> str:
    label = str(path).lower()
    if any(token in label for token in ("scratch", "_scratch")):
        return "scratch"
    if any(token in label for token in ("diagnostic", "failure", "failed", "debug", "ablation")):
        return "diagnostic"
    if "pilot" in label and "comparison" not in label:
        return "pilot"
    if summary and summary.get("schema_id") == "algofinder.agents.three-model-comparison":
        return "three_model_comparison"
    if summary and summary.get("schema_id") == "algofinder.agents.learned-policy-comparison":
        return "learned_policy_comparison"
    return "comparison"


def _lifecycle(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "recorded"
    if summary.get("complete") is True or str(summary.get("phase")) == "complete":
        return "complete"
    phase = str(summary.get("phase") or "incomplete")
    return "incomplete" if not phase else phase


def discover_source_roots(experiments_root: Path) -> list[SourceRoot]:
    """Find comparison roots plus standalone campaign roots beneath a path."""
    root = experiments_root.resolve()
    if not root.exists():
        raise WarehouseError(f"experiments root does not exist: {root}")
    found: list[SourceRoot] = []
    comparison_roots: set[Path] = set()
    campaign_paths: list[Path] = []
    # Stream directories rather than materializing and globally sorting a very
    # large worktree snapshot.  Per-directory sorting retains reproducibility
    # without the avoidable memory and latency spike.
    for directory, directories, files in _walk(root):
        if "comparison-summary.json" in files:
            summary_path = directory / "comparison-summary.json"
            summary = _json_object(summary_path)
            comparison_root = summary_path.parent.resolve()
            comparison_roots.add(comparison_root)
            found.append(SourceRoot(
                path=comparison_root,
                source_class=_source_class(comparison_root, summary),
                lifecycle_state=_lifecycle(summary),
                summary_path=summary_path.resolve(),
            ))
            # Its complete descendant tree will be scanned exactly once below
            # as this source root.  Do not walk it a second time merely to
            # discover nested campaign records.
            directories[:] = []
            continue
        if "campaign.json" in files:
            campaign_paths.append(directory / "campaign.json")

    for campaign_path in campaign_paths:
        campaign_root = campaign_path.parent.resolve()
        if any(campaign_root.is_relative_to(parent) for parent in comparison_roots):
            continue
        campaign = _json_object(campaign_path)
        found.append(SourceRoot(
            path=campaign_root,
            source_class=_source_class(campaign_root, campaign),
            lifecycle_state="recorded",
        ))

    # A caller may point directly at a source root that has not yet received a
    # comparison summary.  Its contents are still useful diagnostic evidence.
    if not found and root.is_dir() and any(root.rglob("*.json")):
        found.append(SourceRoot(root, _source_class(root, None), "recorded"))
    return found


def _campaign_id(path: Path) -> str | None:
    for parent in (path.parent, *path.parents):
        if parent.parent.name == "campaigns":
            return parent.name
    return None


def _artifact_kind(path: Path) -> str:
    name = path.name
    parent = path.parent.name
    if name == "comparison-summary.json":
        return "comparison_summary"
    if name == "campaign.json":
        return "campaign"
    if path.suffix == ".parquet":
        return "dataset_parquet" if "datasets" in path.parts else "parquet"
    mapping = {
        "agents": "agent_spec", "agent_specs": "agent_spec", "agent-runs": "agent_run",
        "agent_runs": "agent_run", "episodes": "episode", "attempts": "attempt",
        "transitions": "transition", "hypotheses": "hypothesis", "candidates": "candidate",
        "evaluations": "evaluation", "analyses": "analysis", "decisions": "decision",
        "backfills": "backfill", "distributions": "distribution", "policies": "policy",
        "memories": "memory", "runs": "learning_run", "manifests": "manifest",
    }
    if parent in mapping:
        return mapping[parent]
    if name in {"events.jsonl", "ledger.jsonl"}:
        return "lifecycle_event"
    if "blobs" in path.parts:
        return "blob"
    if "workspaces" in path.parts:
        return "source_snapshot"
    return "artifact"


def _walk(root: Path) -> Iterator[tuple[Path, list[str], list[str]]]:
    """Yield a deterministic stream of files without globally sorting paths."""
    for directory, directories, files in os.walk(root, followlinks=False):
        directories.sort()
        files.sort()
        yield Path(directory), directories, files


def discover(experiments_roots: Sequence[Path]) -> list[DiscoveredArtifact]:
    """Inventory all regular files under discovered experiment roots, no writes."""
    entries: dict[tuple[str, str], DiscoveredArtifact] = {}
    for experiments_root in experiments_roots:
        for source in discover_source_roots(experiments_root):
            for directory, _, filenames in _walk(source.path):
                for filename in filenames:
                    path = directory / filename
                    if path.is_symlink():
                        continue
                    try:
                        relative = str(path.relative_to(source.path))
                    except ValueError:  # defensive against a concurrently moved file
                        continue
                    item = DiscoveredArtifact(
                        source=source,
                        path=path,
                        relative_path=relative,
                        kind=_artifact_kind(path),
                        campaign_id=_campaign_id(path),
                        experiment_id=source.path.name,
                    )
                    entries[(source.source_root_id, relative)] = item
    return list(entries.values())


def discovery_report(experiments_roots: Sequence[Path]) -> dict[str, Any]:
    artifacts = discover(experiments_roots)
    return {
        "experiments_roots": [str(path.resolve()) for path in experiments_roots],
        "source_roots": len({item.source.source_root_id for item in artifacts}),
        "artifacts": len(artifacts),
        "by_source_class": dict(sorted(Counter(item.source.source_class for item in artifacts).items())),
        "by_lifecycle_state": dict(sorted(Counter(item.source.lifecycle_state for item in artifacts).items())),
        "by_kind": dict(sorted(Counter(item.kind for item in artifacts).items())),
        "campaigns": len({item.campaign_id for item in artifacts if item.campaign_id}),
        "experiments": len({item.experiment_id for item in artifacts}),
    }


def _schema(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute("CREATE SCHEMA IF NOT EXISTS catalog")
    connection.execute("CREATE SCHEMA IF NOT EXISTS experiment")
    connection.execute("CREATE SCHEMA IF NOT EXISTS campaign")
    connection.execute("CREATE SCHEMA IF NOT EXISTS learning")
    connection.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS catalog.warehouse_meta (
            key VARCHAR PRIMARY KEY, value VARCHAR NOT NULL, updated_at TIMESTAMP NOT NULL
        );
        CREATE TABLE IF NOT EXISTS catalog.source_root (
            source_root_id VARCHAR PRIMARY KEY, root_path VARCHAR NOT NULL,
            source_class VARCHAR NOT NULL, lifecycle_state VARCHAR NOT NULL,
            discovered_at TIMESTAMP NOT NULL, summary_path VARCHAR
        );
        CREATE TABLE IF NOT EXISTS catalog.ingestion_batch (
            batch_id VARCHAR PRIMARY KEY, started_at TIMESTAMP NOT NULL,
            finished_at TIMESTAMP, mode VARCHAR NOT NULL, status VARCHAR NOT NULL,
            source_roots_json VARCHAR NOT NULL, artifact_count BIGINT DEFAULT 0,
            message VARCHAR
        );
        CREATE TABLE IF NOT EXISTS catalog.artifact (
            artifact_digest VARCHAR PRIMARY KEY, byte_size BIGINT NOT NULL,
            mime_type VARCHAR, artifact_kind VARCHAR NOT NULL,
            source_class VARCHAR, lifecycle_state VARCHAR, schema_id VARCHAR,
            schema_version VARCHAR, provenance_digest VARCHAR, ingestion_status VARCHAR,
            canonical_json VARCHAR,
            corpus_path VARCHAR, corpus_status VARCHAR NOT NULL,
            first_seen_at TIMESTAMP NOT NULL, last_seen_at TIMESTAMP NOT NULL
        );
        CREATE TABLE IF NOT EXISTS catalog.artifact_location (
            source_root_id VARCHAR NOT NULL, relative_path VARCHAR NOT NULL,
            absolute_path VARCHAR NOT NULL, artifact_digest VARCHAR NOT NULL,
            source_class VARCHAR NOT NULL, lifecycle_state VARCHAR NOT NULL,
            ingestion_status VARCHAR NOT NULL, observed_at TIMESTAMP NOT NULL,
            verification_status VARCHAR NOT NULL,
            PRIMARY KEY (source_root_id, relative_path)
        );
        CREATE TABLE IF NOT EXISTS catalog.record (
            artifact_digest VARCHAR NOT NULL, record_ordinal INTEGER NOT NULL,
            record_kind VARCHAR NOT NULL, logical_id VARCHAR, campaign_id VARCHAR,
            experiment_id VARCHAR, source_class VARCHAR NOT NULL,
            lifecycle_state VARCHAR NOT NULL, schema_id VARCHAR, schema_version VARCHAR,
            provenance_digest VARCHAR, raw_json VARCHAR NOT NULL, extra_json VARCHAR NOT NULL,
            PRIMARY KEY (artifact_digest, record_ordinal)
        );
        CREATE TABLE IF NOT EXISTS catalog.logical_id (
            record_kind VARCHAR NOT NULL, logical_id VARCHAR NOT NULL,
            artifact_digest VARCHAR NOT NULL, record_ordinal INTEGER NOT NULL,
            first_seen_at TIMESTAMP NOT NULL,
            PRIMARY KEY (record_kind, logical_id, artifact_digest, record_ordinal)
        );
        CREATE TABLE IF NOT EXISTS catalog.schema_fingerprint (
            record_kind VARCHAR NOT NULL, schema_id VARCHAR NOT NULL, schema_version VARCHAR NOT NULL,
            fingerprint VARCHAR NOT NULL, artifact_digest VARCHAR NOT NULL,
            PRIMARY KEY (record_kind, schema_id, schema_version, fingerprint, artifact_digest)
        );
        CREATE TABLE IF NOT EXISTS catalog.integrity_issue (
            issue_id VARCHAR PRIMARY KEY, issue_type VARCHAR NOT NULL, severity VARCHAR NOT NULL,
            artifact_digest VARCHAR, source_root_id VARCHAR, logical_id VARCHAR,
            detail_json VARCHAR NOT NULL, observed_at TIMESTAMP NOT NULL, resolved_at TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS catalog.ingestion_seen (
            batch_id VARCHAR NOT NULL, artifact_digest VARCHAR NOT NULL, status VARCHAR NOT NULL,
            observed_at TIMESTAMP NOT NULL, PRIMARY KEY (batch_id, artifact_digest)
        );
    """)
    _ensure_catalog_columns(connection)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS experiment.comparison (
            summary_digest VARCHAR PRIMARY KEY, comparison_id VARCHAR NOT NULL,
            experiment_root VARCHAR NOT NULL, schema_id VARCHAR, schema_version VARCHAR,
            source_class VARCHAR NOT NULL, lifecycle_state VARCHAR NOT NULL,
            complete BOOLEAN, phase VARCHAR, started_at VARCHAR, completed_at VARCHAR,
            provenance_digest VARCHAR, plan_json VARCHAR, analysis_json VARCHAR, extra_json VARCHAR NOT NULL
        );
        CREATE TABLE IF NOT EXISTS experiment.plan (
            summary_digest VARCHAR PRIMARY KEY, comparison_id VARCHAR NOT NULL, model VARCHAR,
            representation VARCHAR, candidates_per_campaign INTEGER,
            per_cell_budget_seconds DOUBLE, plan_json VARCHAR NOT NULL
        );
        CREATE TABLE IF NOT EXISTS experiment.schedule (
            summary_digest VARCHAR NOT NULL, campaign_key VARCHAR NOT NULL, replicate_index INTEGER,
            arm VARCHAR, model VARCHAR, model_seed BIGINT, order_position INTEGER,
            policy_id VARCHAR, profile_id VARCHAR, schedule_json VARCHAR NOT NULL,
            PRIMARY KEY (summary_digest, campaign_key)
        );
        CREATE TABLE IF NOT EXISTS experiment.manifest (
            manifest_digest VARCHAR NOT NULL, source_digest VARCHAR NOT NULL, comparison_id VARCHAR,
            manifest_path VARCHAR, role VARCHAR, manifest_json VARCHAR, PRIMARY KEY (manifest_digest, source_digest)
        );
        CREATE TABLE IF NOT EXISTS experiment.model_identity (
            summary_digest VARCHAR NOT NULL, model VARCHAR NOT NULL, identity_json VARCHAR NOT NULL,
            PRIMARY KEY (summary_digest, model)
        );
        CREATE TABLE IF NOT EXISTS experiment.profile (
            summary_digest VARCHAR NOT NULL, profile_id VARCHAR NOT NULL, replicate_index INTEGER,
            profile_digest VARCHAR, profile_json VARCHAR NOT NULL, PRIMARY KEY (summary_digest, profile_id)
        );
        CREATE TABLE IF NOT EXISTS experiment.arm (
            summary_digest VARCHAR NOT NULL, campaign_key VARCHAR NOT NULL, campaign_id VARCHAR,
            replicate_index INTEGER, arm VARCHAR, model VARCHAR, model_seed BIGINT,
            policy_id VARCHAR, profile_id VARCHAR, campaign_state VARCHAR,
            evaluated_candidates INTEGER, proposal_attempts INTEGER, model_tokens DOUBLE,
            wall_seconds DOUBLE, holdout_finalized BOOLEAN, arm_json VARCHAR NOT NULL,
            PRIMARY KEY (summary_digest, campaign_key)
        );
        CREATE TABLE IF NOT EXISTS experiment.matched_pair (
            summary_digest VARCHAR NOT NULL, replicate_index INTEGER NOT NULL, profile_id VARCHAR,
            model_seed BIGINT, learned_minus_baseline_holdout_percent DOUBLE,
            pair_json VARCHAR NOT NULL, PRIMARY KEY (summary_digest, replicate_index)
        );
        CREATE TABLE IF NOT EXISTS experiment.model_pair (
            summary_digest VARCHAR NOT NULL, comparison_key VARCHAR NOT NULL,
            replicate_seed BIGINT NOT NULL, second_minus_first_percent DOUBLE,
            pair_json VARCHAR NOT NULL,
            PRIMARY KEY (summary_digest, comparison_key, replicate_seed)
        );
        CREATE TABLE IF NOT EXISTS experiment.checkpoint (
            summary_digest VARCHAR PRIMARY KEY, comparison_id VARCHAR NOT NULL, phase VARCHAR,
            complete BOOLEAN, checkpoint_at VARCHAR, analysis_json VARCHAR NOT NULL
        );
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS campaign.campaign (
            record_ref VARCHAR PRIMARY KEY, campaign_id VARCHAR, base_commit VARCHAR, state VARCHAR,
            search_policy_version VARCHAR, distribution_profile_id VARCHAR, provenance_digest VARCHAR,
            record_json VARCHAR NOT NULL, extra_json VARCHAR NOT NULL
        );
    """)
    for table in ("agent_spec", "agent_run", "episode", "attempt", "transition", "hypothesis", "candidate", "evaluation", "analysis", "decision", "budget_event", "lifecycle_event"):
        connection.execute(f"""
            CREATE TABLE IF NOT EXISTS campaign.{table} (
                record_ref VARCHAR PRIMARY KEY, entity_id VARCHAR, campaign_id VARCHAR,
                occurred_at VARCHAR, status VARCHAR, record_json VARCHAR NOT NULL,
                extra_json VARCHAR NOT NULL
            )
        """)
    for table in ("backfill", "dataset", "policy", "memory", "run", "distribution", "outer_evaluation"):
        connection.execute(f"""
            CREATE TABLE IF NOT EXISTS learning.{table} (
                record_ref VARCHAR PRIMARY KEY, entity_id VARCHAR, campaign_id VARCHAR,
                dataset_id VARCHAR, policy_id VARCHAR, created_at VARCHAR,
                record_json VARCHAR NOT NULL, extra_json VARCHAR NOT NULL
            )
        """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS learning.dataset_row (
            dataset_id VARCHAR NOT NULL, row_number BIGINT NOT NULL, source_digest VARCHAR NOT NULL,
            campaign_id VARCHAR, row_json VARCHAR NOT NULL,
            PRIMARY KEY (dataset_id, row_number, source_digest)
        );
    """)
    _views(connection)
    connection.execute("""
        INSERT INTO catalog.warehouse_meta VALUES ('schema_version', ?, CURRENT_TIMESTAMP)
        ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
    """, [WAREHOUSE_SCHEMA_VERSION])


def _ensure_catalog_columns(connection: duckdb.DuckDBPyConnection) -> None:
    """Migrate early warehouse files without touching authoritative evidence."""
    columns = {
        "catalog.artifact": (
            "source_class VARCHAR", "lifecycle_state VARCHAR", "schema_id VARCHAR",
            "schema_version VARCHAR", "provenance_digest VARCHAR", "ingestion_status VARCHAR",
        ),
        "catalog.artifact_location": (
            "source_class VARCHAR", "lifecycle_state VARCHAR", "ingestion_status VARCHAR",
        ),
    }
    for table, definitions in columns.items():
        for definition in definitions:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {definition}")


def _views(connection: duckdb.DuckDBPyConnection) -> None:
    # An artifact can be a historical checkpoint whose source path has since
    # been replaced.  The active location condition makes default analytics
    # use the actual source tree's current checkpoint, while the full catalog
    # retains earlier observed checkpoints for auditability.
    connection.execute("""
        CREATE OR REPLACE VIEW analytics.eligible_experiments AS
        SELECT comparison.*
        FROM experiment.comparison AS comparison
        WHERE comparison.complete
          AND comparison.source_class NOT IN ('diagnostic', 'scratch')
          AND EXISTS (
              SELECT 1 FROM catalog.artifact_location AS location
              WHERE location.artifact_digest = comparison.summary_digest
                AND location.verification_status = 'present'
          )
    """)
    connection.execute("""
        CREATE OR REPLACE VIEW analytics.matched_policy_model_comparisons AS
        SELECT 'policy' AS comparison_kind, eligible.comparison_id, eligible.summary_digest, eligible.source_class,
               pair.replicate_index, pair.profile_id, pair.model_seed,
               pair.learned_minus_baseline_holdout_percent, pair.pair_json
        FROM analytics.eligible_experiments AS eligible
        JOIN experiment.matched_pair AS pair USING (summary_digest)
        UNION ALL
        SELECT 'model' AS comparison_kind, eligible.comparison_id, eligible.summary_digest, eligible.source_class,
               NULL AS replicate_index, pair.comparison_key AS profile_id, pair.replicate_seed AS model_seed,
               pair.second_minus_first_percent AS learned_minus_baseline_holdout_percent, pair.pair_json
        FROM analytics.eligible_experiments AS eligible
        JOIN experiment.model_pair AS pair USING (summary_digest)
    """)
    connection.execute("""
        CREATE OR REPLACE VIEW analytics.learning_curves AS
        SELECT dataset_id, row_number, source_digest, campaign_id, row_json
        FROM learning.dataset_row
    """)
    connection.execute("""
        CREATE OR REPLACE VIEW analytics.candidate_outcomes AS
        SELECT candidate.campaign_id, candidate.entity_id AS candidate_id,
               candidate.status AS candidate_status, evaluation.entity_id AS evaluation_id,
               evaluation.status AS evaluation_status, evaluation.record_json AS evaluation_json
        FROM campaign.candidate AS candidate
        LEFT JOIN campaign.evaluation AS evaluation
          ON evaluation.campaign_id = candidate.campaign_id
    """)
    connection.execute("""
        CREATE OR REPLACE VIEW analytics.costs AS
        SELECT eligible.comparison_id, arm.campaign_id, arm.arm, arm.model,
               arm.model_tokens, arm.wall_seconds, arm.proposal_attempts,
               arm.evaluated_candidates
        FROM analytics.eligible_experiments AS eligible
        JOIN experiment.arm AS arm USING (summary_digest)
    """)
    connection.execute("""
        CREATE OR REPLACE VIEW analytics.quality_holdout_scorecards AS
        SELECT eligible.comparison_id, eligible.source_class, arm.arm, arm.model,
               arm.campaign_id, arm.evaluated_candidates, arm.holdout_finalized,
               arm.arm_json
        FROM analytics.eligible_experiments AS eligible
        JOIN experiment.arm AS arm USING (summary_digest)
    """)


@contextmanager
def _connect(db_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(db_path))
    try:
        _schema(connection)
        yield connection
    finally:
        connection.close()


def _provenance_digest(record: dict[str, Any], fallback: str) -> str:
    source = record.get("experiment_source_provenance")
    if isinstance(source, dict) and isinstance(source.get("source_digest"), str):
        return str(source["source_digest"])
    for key in ("content_digest", "projection_digest", "policy_digest", "digest"):
        if isinstance(record.get(key), str):
            return str(record[key])
    return fallback


def _entity_id(record: dict[str, Any], fallback: str) -> str | None:
    for key in ("id", "campaign_id", "candidate_id", "attempt_id", "analysis_id", "evaluation_id", "policy_id", "dataset_id", "backfill_id", "profile_id", "run_id", "memory_id", "agent_spec_id"):
        value = record.get(key)
        if isinstance(value, (str, int)):
            return str(value)
    return fallback if fallback else None


def _record_ref(digest: str, ordinal: int) -> str:
    return f"{digest}:{ordinal}"


def _normalized_record_kind(artifact_kind: str, record: dict[str, Any]) -> str:
    """Prefer the immutable artifact family over a producer-specific label."""
    if isinstance(record.get("amounts"), dict):
        return "budget_event"
    producer_kind = record.get("kind")
    aliases = {
        "agent_transition": "transition",
        "research_attempt": "attempt",
        "experience_backfill": "backfill",
        "distribution_profile": "distribution",
        "agent_memory": "memory",
        "agent_policy": "policy",
        "agent_episode": "episode",
        "agent_run": "agent_run",
        "learning_dataset": "dataset",
    }
    # Episode JSONL files may intentionally contain both episode and agent-run
    # records. In that case the producer kind is the stable entity identity.
    if isinstance(producer_kind, str) and producer_kind in aliases:
        return aliases[producer_kind]
    typed_artifact_kinds = {
        "comparison_summary", "campaign", "agent_spec", "agent_run", "episode",
        "attempt", "transition", "hypothesis", "candidate", "evaluation", "analysis",
        "decision", "backfill", "distribution", "policy", "memory", "learning_run",
    }
    if artifact_kind in typed_artifact_kinds:
        return artifact_kind
    if isinstance(producer_kind, str):
        return aliases.get(producer_kind, producer_kind)
    if artifact_kind == "lifecycle_event" and isinstance(record.get("amounts"), dict):
        return "budget_event"
    return artifact_kind


def _is_prohibited(path: Path, data: bytes) -> str | None:
    lower_parts = {part.lower() for part in path.parts}
    if lower_parts & PROHIBITED_PATH_PARTS:
        return "prohibited_path"
    if path.name.lower() in {".env", "credentials", "credentials.json"}:
        return "secret_filename"
    if any(pattern.search(data) for pattern in SECRET_PATTERNS):
        return "secret_payload"
    return None


def _copy_to_corpus(*, path: Path, data: bytes, digest: str, corpus_root: Path) -> tuple[str | None, str]:
    rejection = _is_prohibited(path, data)
    if rejection:
        return None, f"rejected_{rejection}"
    if path.suffix.lower() not in STABLE_TEXT_SUFFIXES or len(data) > MAX_CORPUS_TEXT_BYTES:
        return None, "source_only"
    target = corpus_root / "sha256" / digest[:2] / digest
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)
    return str(target), "corpus_copied"


def _safe_read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise WarehouseError(f"cannot read artifact {path}: {exc}") from exc


def _parse_records(data: bytes, path: Path, kind: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return []
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return []
        return [value] if isinstance(value, dict) else []
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows
    return []


def _as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _record_status(record: dict[str, Any]) -> str | None:
    for key in ("status", "state", "phase"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return None


def _record_time(record: dict[str, Any]) -> str | None:
    for key in ("occurred_at", "created_at", "updated_at", "timestamp", "completed_at"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return None


def _insert_issue(
    connection: duckdb.DuckDBPyConnection, *, issue_type: str, severity: str,
    detail: dict[str, Any], artifact_digest: str | None = None,
    source_root_id: str | None = None, logical_id: str | None = None,
) -> None:
    material = _canonical_json({"issue_type": issue_type, "artifact_digest": artifact_digest, "source_root_id": source_root_id, "logical_id": logical_id, "detail": detail})
    issue_id = _digest_bytes(material.encode("utf-8"))
    connection.execute("""
        INSERT INTO catalog.integrity_issue
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, NULL)
        ON CONFLICT (issue_id) DO NOTHING
    """, [issue_id, issue_type, severity, artifact_digest, source_root_id, logical_id, material])


def _project_comparison(
    connection: duckdb.DuckDBPyConnection, *, digest: str, source: SourceRoot,
    record: dict[str, Any], experiment_id: str,
) -> None:
    plan = record.get("plan") if isinstance(record.get("plan"), dict) else {}
    analysis = record.get("analysis") if isinstance(record.get("analysis"), dict) else {}
    comparison_id = str(record.get("comparison_id") or experiment_id)
    complete = record.get("complete") is True or record.get("phase") == "complete"
    connection.execute("DELETE FROM experiment.schedule WHERE summary_digest = ?", [digest])
    connection.execute("DELETE FROM experiment.model_identity WHERE summary_digest = ?", [digest])
    connection.execute("DELETE FROM experiment.profile WHERE summary_digest = ?", [digest])
    connection.execute("DELETE FROM experiment.arm WHERE summary_digest = ?", [digest])
    connection.execute("DELETE FROM experiment.matched_pair WHERE summary_digest = ?", [digest])
    connection.execute("DELETE FROM experiment.model_pair WHERE summary_digest = ?", [digest])
    connection.execute("""
        INSERT INTO experiment.comparison VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (summary_digest) DO UPDATE SET
            comparison_id = excluded.comparison_id, experiment_root = excluded.experiment_root,
            schema_id = excluded.schema_id, schema_version = excluded.schema_version,
            source_class = excluded.source_class, lifecycle_state = excluded.lifecycle_state,
            complete = excluded.complete, phase = excluded.phase, started_at = excluded.started_at,
            completed_at = excluded.completed_at, provenance_digest = excluded.provenance_digest,
            plan_json = excluded.plan_json, analysis_json = excluded.analysis_json,
            extra_json = excluded.extra_json
    """, [digest, comparison_id, str(source.path), record.get("schema_id"), str(record.get("schema_version") or ""), source.source_class, _lifecycle(record), complete, record.get("phase"), record.get("started_at"), record.get("completed_at"), _provenance_digest(record, digest), _canonical_json(plan), _canonical_json(analysis), _canonical_json(record)])
    connection.execute("""
        INSERT INTO experiment.plan VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (summary_digest) DO UPDATE SET comparison_id = excluded.comparison_id,
            model = excluded.model, representation = excluded.representation,
            candidates_per_campaign = excluded.candidates_per_campaign,
            per_cell_budget_seconds = excluded.per_cell_budget_seconds, plan_json = excluded.plan_json
    """, [digest, comparison_id, plan.get("model"), plan.get("representation"), _as_int(plan.get("candidates_per_campaign")), _as_float(plan.get("per_cell_budget_seconds")), _canonical_json(plan)])
    schedule = plan.get("schedule") if isinstance(plan.get("schedule"), list) else []
    for index, item in enumerate(schedule):
        if not isinstance(item, dict):
            continue
        key = str(item.get("campaign_key") or f"schedule-{index}")
        connection.execute("INSERT INTO experiment.schedule VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [digest, key, _as_int(item.get("replicate_index")), item.get("arm"), item.get("model") or plan.get("model"), _as_int(item.get("model_seed")), _as_int(item.get("order_position")), item.get("policy_id"), item.get("profile_id"), _canonical_json(item)])
    identities = record.get("model_identities")
    if not isinstance(identities, dict):
        identities = {str(plan["model"]): plan["model_identity"]} if isinstance(plan.get("model_identity"), dict) and isinstance(plan.get("model"), str) else {}
    for model, identity in identities.items():
        connection.execute("INSERT INTO experiment.model_identity VALUES (?, ?, ?)", [digest, str(model), _canonical_json(identity)])
    profiles = record.get("profiles") if isinstance(record.get("profiles"), list) else []
    for index, profile in enumerate(profiles):
        if not isinstance(profile, dict):
            continue
        profile_id = str(profile.get("profile_id") or f"profile-{index}")
        connection.execute("INSERT INTO experiment.profile VALUES (?, ?, ?, ?, ?)", [digest, profile_id, _as_int(profile.get("replicate_index")), profile.get("profile_digest"), _canonical_json(profile)])
    runs = record.get("runs") if isinstance(record.get("runs"), list) else []
    for index, arm in enumerate(runs):
        if not isinstance(arm, dict):
            continue
        key = str(arm.get("campaign_key") or arm.get("campaign_id") or f"run-{index}")
        usage = arm.get("model_usage_totals") if isinstance(arm.get("model_usage_totals"), dict) else arm.get("model_usage") if isinstance(arm.get("model_usage"), dict) else {}
        budget = arm.get("budget") if isinstance(arm.get("budget"), dict) else {}
        connection.execute("INSERT INTO experiment.arm VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [digest, key, arm.get("campaign_id"), _as_int(arm.get("replicate_index")), arm.get("arm"), arm.get("model") or plan.get("model"), _as_int(arm.get("model_seed")), arm.get("policy_id"), arm.get("profile_id"), arm.get("campaign_state"), _as_int(arm.get("evaluated_candidates")), _as_int(arm.get("proposal_attempts")), _as_float(usage.get("tokens")), _as_float(budget.get("wall_seconds")), arm.get("holdout_finalized") is True, _canonical_json(arm)])
    pairs = analysis.get("matched_pairs") if isinstance(analysis.get("matched_pairs"), list) else []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            continue
        replicate = _as_int(pair.get("replicate_index"))
        if replicate is None:
            replicate = index
        connection.execute("INSERT INTO experiment.matched_pair VALUES (?, ?, ?, ?, ?, ?)", [digest, replicate, pair.get("profile_id"), _as_int(pair.get("model_seed")), _as_float(pair.get("learned_minus_baseline_holdout_percent")), _canonical_json(pair)])
    pairwise = analysis.get("pairwise_matched") if isinstance(analysis.get("pairwise_matched"), dict) else {}
    for comparison_key, aggregate in pairwise.items():
        if not isinstance(aggregate, dict):
            continue
        replicates = aggregate.get("replicates") if isinstance(aggregate.get("replicates"), list) else []
        for pair in replicates:
            if not isinstance(pair, dict):
                continue
            seed = _as_int(pair.get("seed"))
            if seed is None:
                continue
            connection.execute("INSERT INTO experiment.model_pair VALUES (?, ?, ?, ?, ?)", [digest, str(comparison_key), seed, _as_float(pair.get("second_minus_first_percent")), _canonical_json(pair)])
    connection.execute("""
        INSERT INTO experiment.checkpoint VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (summary_digest) DO UPDATE SET comparison_id = excluded.comparison_id,
            phase = excluded.phase, complete = excluded.complete,
            checkpoint_at = excluded.checkpoint_at, analysis_json = excluded.analysis_json
    """, [digest, comparison_id, record.get("phase"), complete, record.get("updated_at") or record.get("completed_at") or record.get("started_at"), _canonical_json(analysis)])


def _project_campaign_entity(
    connection: duckdb.DuckDBPyConnection, *, digest: str, ordinal: int, kind: str,
    campaign_id: str | None, record: dict[str, Any], path: Path,
) -> None:
    record_ref = _record_ref(digest, ordinal)
    canonical = _canonical_json(record)
    if kind == "campaign":
        connection.execute("""
            INSERT INTO campaign.campaign VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (record_ref) DO NOTHING
        """, [record_ref, _entity_id(record, campaign_id or ""), record.get("base_commit"), record.get("state"), record.get("search_policy_version"), record.get("distribution_profile_id"), _provenance_digest(record, digest), canonical, canonical])
        return
    table = {
        "agent_spec": "agent_spec", "agent_run": "agent_run", "episode": "episode",
        "attempt": "attempt", "transition": "transition", "hypothesis": "hypothesis",
        "candidate": "candidate", "evaluation": "evaluation", "analysis": "analysis",
        "decision": "decision", "budget_event": "budget_event", "lifecycle_event": "lifecycle_event",
    }.get(kind)
    if table is None:
        return
    connection.execute(f"INSERT INTO campaign.{table} VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (record_ref) DO NOTHING", [record_ref, _entity_id(record, path.stem), campaign_id or record.get("campaign_id"), _record_time(record), _record_status(record), canonical, canonical])


def _project_learning(
    connection: duckdb.DuckDBPyConnection, *, digest: str, ordinal: int, kind: str,
    campaign_id: str | None, record: dict[str, Any], path: Path,
) -> None:
    table = {
        "backfill": "backfill", "distribution": "distribution", "policy": "policy",
        "memory": "memory", "learning_run": "run", "dataset": "dataset",
    }.get(kind)
    if table is None:
        return
    canonical = _canonical_json(record)
    connection.execute(f"INSERT INTO learning.{table} VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (record_ref) DO NOTHING", [_record_ref(digest, ordinal), _entity_id(record, path.stem), campaign_id or record.get("campaign_id"), record.get("dataset_id"), record.get("policy_id"), _record_time(record), canonical, canonical])


def _project_dataset_rows(connection: duckdb.DuckDBPyConnection, *, digest: str, path: Path, campaign_id: str | None) -> None:
    if path.suffix.lower() != ".parquet" or "datasets" not in path.parts:
        return
    try:
        dataset_id = path.parent.name
        rows = connection.execute("SELECT row_number() OVER (), to_json(row_value) FROM read_parquet(?) AS row_value", [str(path)]).fetchall()
    except duckdb.Error as exc:
        _insert_issue(connection, issue_type="parquet_read_error", severity="warning", artifact_digest=digest, detail={"path": str(path), "error": str(exc)})
        return
    for row_number, row_json in rows:
        connection.execute("INSERT INTO learning.dataset_row VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING", [dataset_id, row_number, digest, campaign_id, row_json])
    manifest = path.parent / "manifest.json"
    if manifest.is_file():
        data = _json_object(manifest) or {}
        canonical = _canonical_json(data)
        connection.execute("INSERT INTO learning.dataset VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (record_ref) DO NOTHING", [_record_ref(_digest_file(manifest), 0), _entity_id(data, dataset_id), campaign_id, dataset_id, data.get("policy_id"), _record_time(data), canonical, canonical])


def _register_record(
    connection: duckdb.DuckDBPyConnection, *, digest: str, ordinal: int,
    artifact: DiscoveredArtifact, record: dict[str, Any],
) -> None:
    kind = _normalized_record_kind(artifact.kind, record)
    logical_id = _entity_id(record, "")
    schema_id = record.get("schema_id") if isinstance(record.get("schema_id"), str) else None
    schema_version = str(record.get("schema_version")) if record.get("schema_version") is not None else None
    canonical = _canonical_json(record)
    connection.execute("""
        INSERT INTO catalog.record VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (artifact_digest, record_ordinal) DO NOTHING
    """, [digest, ordinal, kind, logical_id, artifact.campaign_id or record.get("campaign_id"), artifact.experiment_id, artifact.source.source_class, artifact.source.lifecycle_state, schema_id, schema_version, _provenance_digest(record, digest), canonical, canonical])
    if logical_id:
        existing = connection.execute("SELECT DISTINCT artifact_digest FROM catalog.logical_id WHERE record_kind = ? AND logical_id = ?", [kind, logical_id]).fetchall()
        if any(existing_digest != digest for (existing_digest,) in existing):
            _insert_issue(connection, issue_type="duplicate_logical_id_different_digest", severity="warning", artifact_digest=digest, source_root_id=artifact.source.source_root_id, logical_id=logical_id, detail={"record_kind": kind, "observed_digest": digest, "existing_digests": [value for (value,) in existing]})
        connection.execute("INSERT INTO catalog.logical_id VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP) ON CONFLICT DO NOTHING", [kind, logical_id, digest, ordinal])
    if schema_id and schema_version:
        fields = sorted(record)
        fingerprint = _digest_bytes(_canonical_json(fields).encode("utf-8"))
        existing_fingerprints = connection.execute("SELECT DISTINCT fingerprint FROM catalog.schema_fingerprint WHERE record_kind = ? AND schema_id = ? AND schema_version = ?", [kind, schema_id, schema_version]).fetchall()
        if any(existing != fingerprint for (existing,) in existing_fingerprints):
            _insert_issue(connection, issue_type="schema_field_conflict", severity="info", artifact_digest=digest, source_root_id=artifact.source.source_root_id, detail={"record_kind": kind, "schema_id": schema_id, "schema_version": schema_version, "fingerprint": fingerprint, "existing_fingerprints": [value for (value,) in existing_fingerprints]})
        connection.execute("INSERT INTO catalog.schema_fingerprint VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING", [kind, schema_id, schema_version, fingerprint, digest])
    if artifact.kind == "comparison_summary":
        _project_comparison(connection, digest=digest, source=artifact.source, record=record, experiment_id=artifact.experiment_id or artifact.source.path.name)
    _project_campaign_entity(connection, digest=digest, ordinal=ordinal, kind=kind, campaign_id=artifact.campaign_id, record=record, path=artifact.path)
    _project_learning(connection, digest=digest, ordinal=ordinal, kind=kind, campaign_id=artifact.campaign_id, record=record, path=artifact.path)


def _upsert_source_root(connection: duckdb.DuckDBPyConnection, source: SourceRoot) -> None:
    connection.execute("""
        INSERT INTO catalog.source_root VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
        ON CONFLICT (source_root_id) DO UPDATE SET root_path = excluded.root_path,
          source_class = excluded.source_class, lifecycle_state = excluded.lifecycle_state,
          discovered_at = excluded.discovered_at, summary_path = excluded.summary_path
    """, [source.source_root_id, str(source.path), source.source_class, source.lifecycle_state, str(source.summary_path) if source.summary_path else None])


def ingest(
    *, db_path: Path, corpus_root: Path, experiments_roots: Sequence[Path], mode: str = "dev",
    stop_after: int | None = None,
) -> dict[str, Any]:
    """Idempotently catalog sources and project their typed, rebuildable rows.

    ``mode='prod'`` intentionally imports only completed comparison roots.  It
    is the finalization path used after a costly search loop.  Development mode
    imports every currently recorded checkpoint, including incomplete evidence.
    """
    if mode not in {"dev", "prod"}:
        raise WarehouseError("mode must be 'dev' or 'prod'")
    artifacts = discover(experiments_roots)
    if mode == "prod":
        complete_ids = {item.source.source_root_id for item in artifacts if item.source.lifecycle_state == "complete"}
        skipped_incomplete = len({item.source.source_root_id for item in artifacts if item.source.source_root_id not in complete_ids})
        artifacts = [item for item in artifacts if item.source.source_root_id in complete_ids]
    else:
        skipped_incomplete = 0
    batch_id = f"ingest-{uuid4().hex}"
    counts: Counter[str] = Counter()
    with _connect(db_path.resolve()) as connection:
        for source in {item.source for item in artifacts}:
            _upsert_source_root(connection, source)
        connection.execute("INSERT INTO catalog.ingestion_batch VALUES (?, CURRENT_TIMESTAMP, NULL, ?, 'running', ?, 0, NULL)", [batch_id, mode, _canonical_json([str(path.resolve()) for path in experiments_roots])])
        transaction_open = False
        committed_artifacts = 0
        try:
            for index, artifact in enumerate(artifacts, start=1):
                if not transaction_open:
                    connection.execute("BEGIN TRANSACTION")
                    transaction_open = True
                data = _safe_read(artifact.path)
                digest = _digest_bytes(data)
                canonical_json = None
                records = _parse_records(data, artifact.path, artifact.kind)
                if artifact.path.suffix.lower() == ".json" and records:
                    canonical_json = _canonical_json(records[0])
                primary_record = records[0] if records else {}
                schema_id = primary_record.get("schema_id") if isinstance(primary_record.get("schema_id"), str) else None
                schema_version = str(primary_record.get("schema_version")) if primary_record.get("schema_version") is not None else None
                provenance_digest = _provenance_digest(primary_record, digest) if primary_record else digest
                corpus_path, corpus_status = _copy_to_corpus(path=artifact.path, data=data, digest=digest, corpus_root=corpus_root.resolve())
                mime_type = mimetypes.guess_type(artifact.path.name)[0] or "application/octet-stream"
                connection.execute("""
                    INSERT INTO catalog.artifact (
                        artifact_digest, byte_size, mime_type, artifact_kind, source_class,
                        lifecycle_state, schema_id, schema_version, provenance_digest,
                        ingestion_status, canonical_json, corpus_path, corpus_status,
                        first_seen_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ingested', ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (artifact_digest) DO UPDATE SET last_seen_at = excluded.last_seen_at
                """, [digest, len(data), mime_type, artifact.kind, artifact.source.source_class, artifact.source.lifecycle_state, schema_id, schema_version, provenance_digest, canonical_json, corpus_path, corpus_status])
                connection.execute("DELETE FROM catalog.artifact_location WHERE source_root_id = ? AND relative_path = ?", [artifact.source.source_root_id, artifact.relative_path])
                connection.execute("""
                    INSERT INTO catalog.artifact_location (
                        source_root_id, relative_path, absolute_path, artifact_digest,
                        source_class, lifecycle_state, ingestion_status, observed_at, verification_status
                    ) VALUES (?, ?, ?, ?, ?, ?, 'ingested', CURRENT_TIMESTAMP, 'present')
                """, [artifact.source.source_root_id, artifact.relative_path, str(artifact.path.resolve()), digest, artifact.source.source_class, artifact.source.lifecycle_state])
                for ordinal, record in enumerate(records):
                    _register_record(connection, digest=digest, ordinal=ordinal, artifact=artifact, record=record)
                _project_dataset_rows(connection, digest=digest, path=artifact.path, campaign_id=artifact.campaign_id)
                connection.execute("INSERT INTO catalog.ingestion_seen VALUES (?, ?, 'ingested', CURRENT_TIMESTAMP) ON CONFLICT DO NOTHING", [batch_id, digest])
                counts["artifacts"] += 1
                counts[f"kind:{artifact.kind}"] += 1
                if corpus_status.startswith("rejected"):
                    counts["rejected"] += 1
                    _insert_issue(connection, issue_type="rejected_artifact", severity="warning", artifact_digest=digest, source_root_id=artifact.source.source_root_id, detail={"path": str(artifact.path), "status": corpus_status})
                if stop_after is not None and index >= stop_after:
                    raise WarehouseInterrupted(f"intentional interruption after {stop_after} artifacts")
                if index % INGEST_TRANSACTION_SIZE == 0:
                    connection.execute("COMMIT")
                    transaction_open = False
                    committed_artifacts = counts["artifacts"]
                    connection.execute("UPDATE catalog.ingestion_batch SET artifact_count = ? WHERE batch_id = ?", [committed_artifacts, batch_id])
            if transaction_open:
                connection.execute("COMMIT")
                transaction_open = False
                committed_artifacts = counts["artifacts"]
            _validate_comparison_counts(connection)
        except BaseException as exc:
            if transaction_open:
                try:
                    connection.execute("ROLLBACK")
                except duckdb.Error:
                    pass
            connection.execute("UPDATE catalog.ingestion_batch SET finished_at = CURRENT_TIMESTAMP, status = 'failed', artifact_count = ?, message = ? WHERE batch_id = ?", [committed_artifacts, str(exc), batch_id])
            raise
        connection.execute("UPDATE catalog.ingestion_batch SET finished_at = CURRENT_TIMESTAMP, status = 'complete', artifact_count = ? WHERE batch_id = ?", [counts["artifacts"], batch_id])
    return {"batch_id": batch_id, "mode": mode, "artifacts_processed": counts["artifacts"], "rejected": counts["rejected"], "skipped_incomplete_source_roots": skipped_incomplete, "by_kind": dict(sorted((key[5:], value) for key, value in counts.items() if key.startswith("kind:")))}


def _validate_comparison_counts(connection: duckdb.DuckDBPyConnection) -> None:
    rows = connection.execute("SELECT summary_digest, extra_json FROM experiment.comparison").fetchall()
    for digest, raw in rows:
        try:
            summary = json.loads(raw)
        except json.JSONDecodeError:
            continue
        expected = len(summary.get("runs", [])) if isinstance(summary.get("runs"), list) else 0
        actual = connection.execute("SELECT count(*) FROM experiment.arm WHERE summary_digest = ?", [digest]).fetchone()[0]
        if expected != actual:
            _insert_issue(connection, issue_type="comparison_run_count_mismatch", severity="warning", artifact_digest=digest, detail={"expected_runs": expected, "projected_arms": actual})
        analysis = _mapping_or_empty(summary, "analysis")
        expected_pairs = _mapping_list(analysis, "matched_pairs")
        actual_pairs = connection.execute("SELECT count(*) FROM experiment.matched_pair WHERE summary_digest = ?", [digest]).fetchone()[0]
        if len(expected_pairs) != actual_pairs:
            _insert_issue(connection, issue_type="comparison_pair_count_mismatch", severity="warning", artifact_digest=digest, detail={"expected_pairs": len(expected_pairs), "projected_pairs": actual_pairs})
        arms = connection.execute("""
            SELECT campaign_key, evaluated_candidates, proposal_attempts, model_tokens, wall_seconds
            FROM experiment.arm WHERE summary_digest = ?
        """, [digest]).fetchall()
        projected_arms = {str(key): values for key, *values in arms}
        for index, run in enumerate(summary.get("runs", [])):
            if not isinstance(run, dict):
                continue
            key = str(run.get("campaign_key") or run.get("campaign_id") or f"run-{index}")
            projected = projected_arms.get(key)
            if projected is None:
                continue
            usage = run.get("model_usage_totals") if isinstance(run.get("model_usage_totals"), dict) else run.get("model_usage") if isinstance(run.get("model_usage"), dict) else {}
            budget = run.get("budget") if isinstance(run.get("budget"), dict) else {}
            expected_values = [
                _as_int(run.get("evaluated_candidates")), _as_int(run.get("proposal_attempts")),
                _as_float(usage.get("tokens")), _as_float(budget.get("wall_seconds")),
            ]
            if projected != expected_values:
                _insert_issue(connection, issue_type="comparison_arm_metric_mismatch", severity="warning", artifact_digest=digest, logical_id=key, detail={"expected": expected_values, "projected": projected})
        projected_pairs = {
            int(replicate): delta
            for replicate, delta in connection.execute("SELECT replicate_index, learned_minus_baseline_holdout_percent FROM experiment.matched_pair WHERE summary_digest = ?", [digest]).fetchall()
        }
        for index, pair in enumerate(expected_pairs):
            replicate = _as_int(pair.get("replicate_index"))
            if replicate is None:
                replicate = index
            expected_delta = _as_float(pair.get("learned_minus_baseline_holdout_percent"))
            if projected_pairs.get(replicate) != expected_delta:
                _insert_issue(connection, issue_type="comparison_pair_metric_mismatch", severity="warning", artifact_digest=digest, logical_id=str(replicate), detail={"expected": expected_delta, "projected": projected_pairs.get(replicate)})
        expected_model_pairs = sum(
            len(item.get("replicates", []))
            for item in _mapping_or_empty(analysis, "pairwise_matched").values()
            if isinstance(item, dict) and isinstance(item.get("replicates"), list)
        )
        actual_model_pairs = connection.execute("SELECT count(*) FROM experiment.model_pair WHERE summary_digest = ?", [digest]).fetchone()[0]
        if expected_model_pairs != actual_model_pairs:
            _insert_issue(connection, issue_type="comparison_model_pair_count_mismatch", severity="warning", artifact_digest=digest, detail={"expected_pairs": expected_model_pairs, "projected_pairs": actual_model_pairs})


def _mapping_or_empty(value: dict[str, Any], key: str) -> dict[str, Any]:
    candidate = value.get(key)
    return candidate if isinstance(candidate, dict) else {}


def _mapping_list(value: dict[str, Any], key: str) -> list[dict[str, Any]]:
    candidate = value.get(key)
    return [item for item in candidate if isinstance(item, dict)] if isinstance(candidate, list) else []


def reproject(*, db_path: Path) -> dict[str, int]:
    """Rebuild typed tables from the catalog without re-reading sources.

    This migration/repair operation is safe because `catalog.record` holds the
    canonical JSON and `catalog.artifact_location` holds the provenance paths.
    It is also the fast path for evolving a typed projection after a new
    producer schema is discovered.
    """
    if not db_path.exists():
        raise WarehouseError(f"warehouse does not exist: {db_path}")
    campaign_tables = (
        "campaign", "agent_spec", "agent_run", "episode", "attempt", "transition",
        "hypothesis", "candidate", "evaluation", "analysis", "decision", "budget_event", "lifecycle_event",
    )
    learning_tables = ("backfill", "dataset", "policy", "memory", "run", "distribution", "outer_evaluation")
    experiment_tables = ("comparison", "plan", "schedule", "manifest", "model_identity", "profile", "arm", "matched_pair", "model_pair", "checkpoint")
    counts: Counter[str] = Counter()
    with _connect(db_path.resolve()) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            for table in experiment_tables:
                connection.execute(f"DELETE FROM experiment.{table}")
            for table in campaign_tables:
                connection.execute(f"DELETE FROM campaign.{table}")
            for table in learning_tables:
                connection.execute(f"DELETE FROM learning.{table}")
            rows = connection.execute("""
                SELECT record.artifact_digest, record.record_ordinal, record.campaign_id,
                       record.experiment_id, record.source_class, record.lifecycle_state,
                       record.raw_json, artifact.artifact_kind,
                       coalesce(source_root.root_path, record.experiment_id) AS root_path,
                       coalesce(location.relative_path, '') AS relative_path
                FROM catalog.record AS record
                JOIN catalog.artifact AS artifact USING (artifact_digest)
                LEFT JOIN (
                    SELECT artifact_digest, min(source_root_id) AS source_root_id,
                           min(relative_path) AS relative_path
                    FROM catalog.artifact_location GROUP BY artifact_digest
                ) AS location USING (artifact_digest)
                LEFT JOIN catalog.source_root AS source_root USING (source_root_id)
                ORDER BY record.artifact_digest, record.record_ordinal
            """).fetchall()
            for digest, ordinal, campaign_id, experiment_id, source_class, lifecycle_state, raw_json, artifact_kind, root_path, relative_path in rows:
                try:
                    record = json.loads(raw_json)
                except json.JSONDecodeError:
                    continue
                source = SourceRoot(
                    path=Path(root_path), source_class=source_class,
                    lifecycle_state=lifecycle_state,
                )
                artifact = DiscoveredArtifact(
                    source=source, path=Path(root_path) / relative_path,
                    relative_path=relative_path, kind=artifact_kind,
                    campaign_id=campaign_id, experiment_id=experiment_id,
                )
                kind = _normalized_record_kind(artifact_kind, record)
                if artifact_kind == "comparison_summary":
                    _project_comparison(connection, digest=digest, source=source, record=record, experiment_id=experiment_id or source.path.name)
                _project_campaign_entity(connection, digest=digest, ordinal=ordinal, kind=kind, campaign_id=campaign_id, record=record, path=artifact.path)
                _project_learning(connection, digest=digest, ordinal=ordinal, kind=kind, campaign_id=campaign_id, record=record, path=artifact.path)
                counts["records"] += 1
            _validate_comparison_counts(connection)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        for table in campaign_tables:
            counts[f"campaign.{table}"] = connection.execute(f"SELECT count(*) FROM campaign.{table}").fetchone()[0]
        for table in learning_tables:
            counts[f"learning.{table}"] = connection.execute(f"SELECT count(*) FROM learning.{table}").fetchone()[0]
    return dict(counts)


def verify(*, db_path: Path) -> dict[str, Any]:
    """Re-hash recorded locations and report missing/moved/conflicting evidence."""
    if not db_path.exists():
        raise WarehouseError(f"warehouse does not exist: {db_path}")
    counts: Counter[str] = Counter()
    with _connect(db_path.resolve()) as connection:
        locations = connection.execute("SELECT source_root_id, relative_path, absolute_path, artifact_digest FROM catalog.artifact_location").fetchall()
        transaction_open = False
        try:
            for index, (source_root_id, relative_path, absolute_path, expected_digest) in enumerate(locations, start=1):
                if not transaction_open:
                    connection.execute("BEGIN TRANSACTION")
                    transaction_open = True
                path = Path(absolute_path)
                if not path.is_file():
                    connection.execute("UPDATE catalog.artifact_location SET verification_status = 'missing', observed_at = CURRENT_TIMESTAMP WHERE source_root_id = ? AND relative_path = ?", [source_root_id, relative_path])
                    _insert_issue(connection, issue_type="missing_source", severity="warning", artifact_digest=expected_digest, source_root_id=source_root_id, detail={"relative_path": relative_path, "absolute_path": absolute_path})
                    counts["missing"] += 1
                else:
                    actual_digest = _digest_file(path)
                    status = "present" if actual_digest == expected_digest else "changed"
                    connection.execute("UPDATE catalog.artifact_location SET verification_status = ?, observed_at = CURRENT_TIMESTAMP WHERE source_root_id = ? AND relative_path = ?", [status, source_root_id, relative_path])
                    if status == "changed":
                        _insert_issue(connection, issue_type="source_digest_mismatch", severity="warning", artifact_digest=expected_digest, source_root_id=source_root_id, detail={"relative_path": relative_path, "expected_digest": expected_digest, "actual_digest": actual_digest})
                        counts["changed"] += 1
                    else:
                        counts["present"] += 1
                if index % INGEST_TRANSACTION_SIZE == 0:
                    connection.execute("COMMIT")
                    transaction_open = False
            if transaction_open:
                connection.execute("COMMIT")
                transaction_open = False
        except BaseException:
            if transaction_open:
                try:
                    connection.execute("ROLLBACK")
                except duckdb.Error:
                    pass
            raise
        duplicate = connection.execute("""
            SELECT record_kind, logical_id, count(DISTINCT artifact_digest)
            FROM catalog.logical_id GROUP BY 1, 2 HAVING count(DISTINCT artifact_digest) > 1
        """).fetchall()
        counts["duplicate_logical_ids"] = len(duplicate)
        counts["schema_conflicts"] = connection.execute("SELECT count(*) FROM catalog.integrity_issue WHERE issue_type = 'schema_field_conflict'").fetchone()[0]
        _validate_comparison_counts(connection)
        counts["comparison_summary_mismatches"] = connection.execute("""
            SELECT count(*) FROM catalog.integrity_issue
            WHERE issue_type IN ('comparison_run_count_mismatch', 'comparison_pair_count_mismatch',
                                 'comparison_model_pair_count_mismatch', 'comparison_arm_metric_mismatch',
                                 'comparison_pair_metric_mismatch')
        """).fetchone()[0]
    return dict(counts)


def overview(*, db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        raise WarehouseError(f"warehouse does not exist: {db_path}")
    with _connect(db_path.resolve()) as connection:
        def count(table: str) -> int:
            return int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
        return {
            "db_path": str(db_path.resolve()),
            "source_roots": count("catalog.source_root"),
            "artifacts": count("catalog.artifact"),
            "locations": count("catalog.artifact_location"),
            "records": count("catalog.record"),
            "comparisons": count("experiment.comparison"),
            "eligible_experiments": count("analytics.eligible_experiments"),
            "campaigns": count("campaign.campaign"),
            "attempts": count("campaign.attempt"),
            "learning_backfills": count("learning.backfill"),
            "learning_dataset_rows": count("learning.dataset_row"),
            "integrity_issues": count("catalog.integrity_issue"),
            "batches": connection.execute("SELECT status, count(*) FROM catalog.ingestion_batch GROUP BY status ORDER BY status").fetchall(),
            "source_classes": connection.execute("SELECT source_class, lifecycle_state, count(*) FROM catalog.source_root GROUP BY 1, 2 ORDER BY 1, 2").fetchall(),
        }


def default_paths(repo_root: Path) -> tuple[Path, Path]:
    return repo_root / DEFAULT_DB_RELATIVE_PATH, repo_root / DEFAULT_CORPUS_RELATIVE_PATH


def incremental_ingest(*, repo_root: Path, experiment_root: Path, mode: str) -> dict[str, Any]:
    """Runtime hook: dev checkpoints ingest incrementally; prod only finalizes."""
    db_path, corpus_root = default_paths(repo_root)
    return ingest(db_path=db_path, corpus_root=corpus_root, experiments_roots=[experiment_root], mode=mode)


__all__ = [
    "DEFAULT_CORPUS_RELATIVE_PATH", "DEFAULT_DB_RELATIVE_PATH", "DiscoveredArtifact",
    "SourceRoot", "WAREHOUSE_SCHEMA_VERSION", "WarehouseError", "WarehouseInterrupted",
    "default_paths", "discover", "discover_source_roots", "discovery_report", "incremental_ingest",
    "ingest", "overview", "reproject", "verify",
]
