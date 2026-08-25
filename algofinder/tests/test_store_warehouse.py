from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from algofinder.store.warehouse import (
    WarehouseInterrupted,
    discovery_report,
    ingest,
    overview,
    verify,
)


def _comparison(root: Path, *, complete: bool = True, schema_version: str = "5") -> Path:
    campaign = root / "campaigns" / "campaign-one"
    (campaign / "attempts").mkdir(parents=True)
    summary = {
        "schema_id": "algofinder.agents.learned-policy-comparison",
        "schema_version": schema_version,
        "comparison_id": root.name,
        "phase": "complete" if complete else "development_search",
        "complete": complete,
        "started_at": "2026-08-24T00:00:00+00:00",
        "plan": {
            "model": "qwen3.5:9b",
            "candidates_per_campaign": 12,
            "schedule": [{
                "campaign_key": "r01-baseline",
                "replicate_index": 0,
                "arm": "baseline-qd",
                "model_seed": 1,
            }],
        },
        "runs": [{
            "campaign_key": "r01-baseline",
            "campaign_id": "campaign-one",
            "replicate_index": 0,
            "arm": "baseline-qd",
            "model": "qwen3.5:9b",
            "evaluated_candidates": 1,
            "proposal_attempts": 2,
            "holdout_finalized": complete,
            "budget": {"wall_seconds": 2.0},
            "model_usage_totals": {"tokens": 10},
        }] if complete else [],
        "analysis": {
            "matched_pairs": [{
                "replicate_index": 0,
                "profile_id": "profile-one",
                "model_seed": 1,
                "learned_minus_baseline_holdout_percent": -0.02,
            }]
        },
    }
    (root / "comparison-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (campaign / "campaign.json").write_text(json.dumps({
        "kind": "campaign", "id": "campaign-one", "schema_id": "algofinder.agents", "schema_version": "1",
    }), encoding="utf-8")
    (campaign / "attempts" / "attempt-a.json").write_text(json.dumps({"id": "attempt-a", "kind": "research_attempt", "status": "accepted"}), encoding="utf-8")
    (campaign / "events.jsonl").write_text(json.dumps({"event": "usage", "amounts": {"tokens": 5}}) + "\n", encoding="utf-8")
    return campaign


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    sources = tmp_path / "recorded-experiments"
    db_path = tmp_path / "private" / "db" / "algofinder.duckdb"
    corpus = tmp_path / "private" / "corpus" / "algofinder"
    return sources, db_path, corpus


def test_discover_is_read_only_and_ingest_is_idempotent(tmp_path: Path) -> None:
    sources, db_path, corpus = _paths(tmp_path)
    _comparison(sources / "completed-comparison")
    report = discovery_report([sources])
    assert report["artifacts"] == 4
    assert not db_path.exists()
    first = ingest(db_path=db_path, corpus_root=corpus, experiments_roots=[sources])
    second = ingest(db_path=db_path, corpus_root=corpus, experiments_roots=[sources])
    assert first["artifacts_processed"] == second["artifacts_processed"] == 4
    state = overview(db_path=db_path)
    assert state["artifacts"] == 4
    assert state["comparisons"] == 1
    assert state["eligible_experiments"] == 1
    assert state["attempts"] == 1
    connection = duckdb.connect(str(db_path))
    try:
        assert connection.execute("SELECT count(*) FROM campaign.budget_event").fetchone()[0] == 1
    finally:
        connection.close()
    assert len(list((corpus / "sha256").rglob("*"))) >= 3
    assert verify(db_path=db_path)["comparison_summary_mismatches"] == 0


def test_interrupted_ingest_resumes_from_source_evidence(tmp_path: Path) -> None:
    sources, db_path, corpus = _paths(tmp_path)
    campaign = _comparison(sources / "completed-comparison")
    for index in range(4):
        (campaign / "attempts" / f"attempt-{index}.json").write_text(json.dumps({"id": f"attempt-{index}"}), encoding="utf-8")
    with pytest.raises(WarehouseInterrupted):
        ingest(db_path=db_path, corpus_root=corpus, experiments_roots=[sources], stop_after=2)
    result = ingest(db_path=db_path, corpus_root=corpus, experiments_roots=[sources])
    assert result["artifacts_processed"] == 8
    state = overview(db_path=db_path)
    assert state["artifacts"] == 8
    assert ("complete", 1) in state["batches"]
    assert ("failed", 1) in state["batches"]


def test_conflicts_missing_sources_and_rejected_payloads_are_cataloged(tmp_path: Path) -> None:
    sources, db_path, corpus = _paths(tmp_path)
    campaign = _comparison(sources / "completed-comparison")
    attempts = campaign / "attempts"
    (attempts / "attempt-b.json").write_text(json.dumps({"id": "attempt-a", "status": "different"}), encoding="utf-8")
    secret = campaign / "notes.json"
    secret.write_text(json.dumps({"api_key": "sk-123456789012345678901234567890"}), encoding="utf-8")
    ingest(db_path=db_path, corpus_root=corpus, experiments_roots=[sources])
    secret.unlink()
    checked = verify(db_path=db_path)
    assert checked["missing"] == 1
    assert checked["duplicate_logical_ids"] == 1
    connection = duckdb.connect(str(db_path))
    try:
        statuses = {row[0] for row in connection.execute("SELECT corpus_status FROM catalog.artifact").fetchall()}
        assert "rejected_secret_payload" in statuses
        assert connection.execute("SELECT count(*) FROM catalog.integrity_issue WHERE issue_type = 'missing_source'").fetchone()[0] == 1
    finally:
        connection.close()


def test_prod_only_finalizes_completed_evidence_and_analytics_excludes_debug(tmp_path: Path) -> None:
    sources, db_path, corpus = _paths(tmp_path)
    _comparison(sources / "complete-comparison", complete=True)
    _comparison(sources / "incomplete-comparison", complete=False)
    _comparison(sources / "diagnostic-comparison", complete=True)
    _comparison(sources / "scratch-comparison", complete=True)
    production = ingest(db_path=db_path, corpus_root=corpus, experiments_roots=[sources], mode="prod")
    assert production["skipped_incomplete_source_roots"] == 1
    connection = duckdb.connect(str(db_path))
    try:
        assert connection.execute("SELECT count(*) FROM experiment.comparison").fetchone()[0] == 3
        assert connection.execute("SELECT count(*) FROM analytics.eligible_experiments").fetchone()[0] == 1
    finally:
        connection.close()
    dev = ingest(db_path=db_path, corpus_root=corpus, experiments_roots=[sources], mode="dev")
    assert dev["artifacts_processed"] == 16
    connection = duckdb.connect(str(db_path))
    try:
        assert connection.execute("SELECT count(*) FROM experiment.comparison").fetchone()[0] == 4
        assert connection.execute("SELECT count(*) FROM analytics.eligible_experiments").fetchone()[0] == 1
    finally:
        connection.close()


def test_fresh_rebuild_reproduces_typed_projection(tmp_path: Path) -> None:
    sources, db_path, corpus = _paths(tmp_path)
    _comparison(sources / "completed-comparison")
    ingest(db_path=db_path, corpus_root=corpus, experiments_roots=[sources])
    first = overview(db_path=db_path)
    rebuilt = tmp_path / "rebuilt.duckdb"
    ingest(db_path=rebuilt, corpus_root=corpus, experiments_roots=[sources])
    second = overview(db_path=rebuilt)
    for key in ("artifacts", "records", "comparisons", "campaigns", "attempts", "eligible_experiments"):
        assert second[key] == first[key]
