"""Per-instance feature sets and canonical records (phase 6).

Tier-0 ``params@1`` applies to every instance; tier-1 ``etsp-geometry@1``
and ``etsp-geometry@2`` apply to Euclidean TSP instances only.  Feature records follow the
spec 6.7 schema and live under ``public/data/features/``.
"""

from __future__ import annotations

from algofinder.features.base import FeatureSet, compute_feature_record, instance_id
from algofinder.features.etsp_geometry import etsp_geometry_feature_set, etsp_geometry_v2_feature_set
from algofinder.features.params import params_feature_set

FEATURE_SETS: tuple[FeatureSet, ...] = (
    params_feature_set,
    etsp_geometry_feature_set,
    etsp_geometry_v2_feature_set,
)

FEATURE_SET_BY_ID = {feature_set.id: feature_set for feature_set in FEATURE_SETS}


def enabled_feature_sets(config: dict | None = None) -> tuple[FeatureSet, ...]:
    """Feature sets enabled by configuration (default: all declared)."""
    if not config or not config.get("enabled"):
        return FEATURE_SETS
    enabled = set(config["enabled"])
    return tuple(
        feature_set
        for feature_set in FEATURE_SETS
        if feature_set.id in enabled
    )


__all__ = [
    "FEATURE_SET_BY_ID",
    "FEATURE_SETS",
    "compute_feature_record",
    "enabled_feature_sets",
    "instance_id",
]
