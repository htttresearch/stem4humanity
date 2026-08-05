"""Tier-0 parameter features (``params@1``).

Cheap, deterministic descriptor features available for every instance
type: the seed, an integer size proxy, and per-key shape/count data
derived directly from the instance ``data`` mapping (lists measured by
length, scalars kept as-is).  No problem state is built and no solver
code runs, so the extractor never touches wall-clock-sensitive paths.
"""

from __future__ import annotations

from typing import Any

from algofinder.features.base import FeatureSet
from algofinder.problems.base import Instance

FEATURE_SET_ID = "params@1"
FEATURE_SET_TIER = 0
FEATURE_SET_DESCRIPTION = (
    "tier-0 descriptor features: seed, size proxy, and per-key data shapes"
)

_SIZE_SCALAR_KEYS = ("node_count", "hours", "periods")


def _instance_size(data: dict[str, Any]) -> int:
    """Deterministic integer size proxy: largest list length, else size scalar."""
    sizes = [len(value) for value in data.values() if isinstance(value, (list, tuple))]
    if sizes:
        return max(sizes)
    for key in _SIZE_SCALAR_KEYS:
        value = data.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return 0


def extract_params(instance: Instance) -> dict[str, Any]:
    """Parameter features for any instance (JSON-safe, deterministic)."""
    data = instance.data
    counts: dict[str, int] = {
        key: len(value)
        for key, value in sorted(data.items())
        if isinstance(value, (list, tuple))
    }
    scalars: dict[str, Any] = {
        key: value
        for key, value in sorted(data.items())
        if isinstance(value, (bool, int, float))
    }
    return {
        "seed": int(instance.seed),
        "n": _instance_size(data),
        "counts": counts,
        "scalars": scalars,
    }


params_feature_set = FeatureSet(
    id=FEATURE_SET_ID,
    tier=FEATURE_SET_TIER,
    description=FEATURE_SET_DESCRIPTION,
    extractor=extract_params,
)


__all__ = ["FEATURE_SET_ID", "extract_params", "params_feature_set"]
