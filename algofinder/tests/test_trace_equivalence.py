"""Behavioral equivalence: dev tracing must not change results.

For deterministic instances without time cutoffs, prod (null recorder)
and dev (full recorder) must return identical statuses and
harness-computed costs. Work counters like search nodes are
algorithm-owned and must not depend on tracing.
"""

from __future__ import annotations

from pathlib import Path

from stem4humanity.harness.benchmark import run_benchmark
from stem4humanity.instances.base import load_manifest
from stem4humanity.trace.session import DevSession

MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "instances"
    / "bin_packing_manifests.json"
)


def _run(instances, session=None):
    runs = run_benchmark(instances, timeout_seconds=10, session=session)
    return {
        (run.instance, run.solver): (run.status, run.cost)
        for run in runs
    }


def test_prod_and_dev_agree_on_cost_and_status(tmp_path):
    instances = [instance for instance in load_manifest(str(MANIFEST))
                 if instance.split == "test"][:3]
    prod = _run(instances)
    session = DevSession(root=tmp_path)
    session.__enter__()
    try:
        dev = _run(instances, session=session)
    finally:
        session.close()
    assert set(prod) == set(dev)
    for key in prod:
        assert prod[key] == dev[key], f"prod/dev differ on {key}"


def test_traced_and_untraced_solvers_agree(tmp_path):
    instances = [instance for instance in load_manifest(str(MANIFEST))
                 if instance.split == "test"][:2]
    session = DevSession(root=tmp_path)
    session.__enter__()
    try:
        dev = _run(instances, session=session)
    finally:
        session.close()
    for (instance, solver), (status, cost) in dev.items():
        if status == "ok":
            assert cost is not None and cost > 0
