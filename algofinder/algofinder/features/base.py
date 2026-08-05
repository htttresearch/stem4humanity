"""Per-instance feature sets and records (phase 6, spec section 6.7).

A :class:`FeatureSet` is a versioned, deterministic extractor of scalar
per-instance features (solver-independent, no wall clock / global RNG).
A feature record is the immutable spec 6.7 mapping: instance digest,
values, per-feature status, measured cost (timing + peak RSS), and the
extractor digest so a code change invalidates old records.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any, Callable

from algofinder.problems.base import Instance
from algofinder.trace.serialize import sha256_hex, to_json_value

FEATURE_RECORD_SCHEMA_VERSION = "algofinder.feature.v1"


@dataclass(frozen=True)
class FeatureSet:
    """One versioned feature extractor (one record per instance)."""

    id: str
    tier: int
    description: str
    extractor: Callable[[Instance], dict[str, Any]]
    applies_to: frozenset[str] = frozenset()

    def extractor_digest(self) -> str:
        source = inspect.getsource(self.extractor)
        payload = f"{FEATURE_RECORD_SCHEMA_VERSION}\n{self.id}\n{self.tier}\n{source}"
        return sha256_hex(payload)

    def applies(self, instance: Instance) -> bool:
        if not self.applies_to:
            return True
        return f"{instance.problem}:{instance.subproblem}" in self.applies_to


def instance_id(instance: Instance) -> str:
    """Content digest of the instance descriptor (registry-consistent)."""
    return sha256_hex(instance.to_mapping())


def _rss_worker(args: tuple[FeatureSet, Instance]) -> int:
    """Module-level spawn worker: peak RSS of one extraction."""
    import resource

    feature_set, instance = args
    feature_set.extractor(instance)
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _measure_rss_in_subprocess(feature_set: FeatureSet, instance: Instance) -> int:
    """Peak RSS of one extraction in a fresh process (imports included)."""
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=1) as pool:
        result = pool.apply(_rss_worker, ((feature_set, instance),))
    return int(result)


def compute_feature_record(
    feature_set: FeatureSet, instance: Instance
) -> dict[str, Any]:
    """One spec 6.7 record: values, per-feature status, measured cost."""
    record = {
        "instance_id": instance_id(instance),
        "feature_set_id": feature_set.id,
        "tier": feature_set.tier,
        "values": {},
        "status_by_feature": {},
        "timing_seconds": None,
        "peak_rss_bytes": None,
        "extractor_digest": feature_set.extractor_digest(),
        "charged_at_inference": False,
    }
    started = time.perf_counter()
    try:
        values = feature_set.extractor(instance)
        record["values"] = to_json_value(values)
        record["status_by_feature"] = {"ok": sorted(to_json_value(values))}
    except Exception as exc:
        record["status_by_feature"] = {
            "error": f"{type(exc).__name__}: {exc}"
        }
    record["timing_seconds"] = round(time.perf_counter() - started, 6)
    try:
        record["peak_rss_bytes"] = _measure_rss_in_subprocess(feature_set, instance)
    except Exception as exc:
        record["status_by_feature"] = {
            "rss_error": f"{type(exc).__name__}: {exc}"
        }
    return record


__all__ = [
    "FEATURE_RECORD_SCHEMA_VERSION",
    "FeatureSet",
    "compute_feature_record",
    "instance_id",
]
