"""Structural trace tests: every dev-mode cell leaves a valid session.

Validates the universal envelope (contiguous seq, known categories,
state-reference chain), the invocation/outcome files, and that prod
mode touches no trace infrastructure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stem4humanity.harness.benchmark import run_benchmark
from stem4humanity.instances.base import load_manifest
from stem4humanity.trace.recorder import ENVELOPE_SCHEMA
from stem4humanity.trace.session import DevSession
from stem4humanity.trace.validate import TraceValidationError, read_events, validate_run

MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "instances"
    / "bin_packing_manifests.json"
)


@pytest.fixture(scope="module")
def small_instances():
    return [instance for instance in load_manifest(str(MANIFEST))
            if instance.split == "test"][:3]


def test_dev_session_structure_is_valid(small_instances, tmp_path):
    session = DevSession(root=tmp_path)
    session.__enter__()
    try:
        runs = run_benchmark(
            small_instances, timeout_seconds=10, session=session
        )
    finally:
        session.close()
    assert len(runs) > 0
    for run in runs:
        assert run.session_id == session.session_id
        assert run.run_id and run.trace_rel
        report = validate_run(session.dir / "runs" / run.run_id)
        assert report["valid"]
    assert session.dir / "session.json" in [p for p in session.dir.glob("*.json")]


def test_events_have_universal_envelope(small_instances, tmp_path):
    session = DevSession(root=tmp_path)
    session.__enter__()
    try:
        runs = run_benchmark(
            small_instances, timeout_seconds=10, session=session
        )
    finally:
        session.close()
    events = read_events(session.dir / "runs" / runs[0].run_id / "events.jsonl")
    for event in events:
        assert event["schema"] == ENVELOPE_SCHEMA
        assert event["run_id"] == runs[0].run_id
        assert isinstance(event["seq"], int)
        assert isinstance(event["elapsed_ns"], int)
        assert isinstance(event["category"], str)
        assert isinstance(event["name"], str)
    assert events[0]["name"] == "run.start"
    assert events[-1]["name"] == "run.result" or events[-1]["name"] == "run.skipped"


def test_invocation_and_outcome_written_by_parent(small_instances, tmp_path):
    session = DevSession(root=tmp_path)
    session.__enter__()
    try:
        runs = run_benchmark(
            small_instances, timeout_seconds=10, session=session
        )
    finally:
        session.close()
    run_dir = session.dir / "runs" / runs[0].run_id
    invocation = json.loads((run_dir / "invocation.json").read_text())
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert invocation["instance_name"] == runs[0].instance
    assert invocation["solver"] == runs[0].solver
    assert invocation["state_sha"]
    assert outcome["status"] == runs[0].status
    assert outcome["trace"]["complete"] is True
    assert session.dir / "index.jsonl" in list(session.dir.iterdir())


def test_prod_mode_creates_no_session(small_instances, tmp_path):
    before = {p.name for p in tmp_path.iterdir()}
    runs = run_benchmark(small_instances, timeout_seconds=10, session=None)
    assert len(runs) > 0
    assert all(run.session_id is None and run.run_id is None for run in runs)
    assert {p.name for p in tmp_path.iterdir()} == before


def test_state_fingerprint_is_stable(small_instances, tmp_path):
    from stem4humanity.problems.registry import get_problem

    fingerprints = {}
    for instance in small_instances:
        state = get_problem(instance.problem).build_state(instance)
        fingerprint = (
            __import__("stem4humanity.trace.codec", fromlist=["get_codec"])
            .get_codec(instance.problem)
            .fingerprint_state(state)
        )
        fingerprints[instance.name] = fingerprint
    assert len(set(fingerprints.values())) == len(fingerprints)
