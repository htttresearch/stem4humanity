"""Strict JSON normalization and canonical hashing for traces.

One normalization layer is used by problem codecs, event payloads,
solver metadata, and outcomes, so the same value has the same encoding
everywhere. Emitted JSON is strict JSON: no NaN/Infinity tokens, no
arbitrary mapping keys, numpy values converted explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

NONFINITE_TAGS = {
    "nan": {"_nonfinite": "nan"},
    "inf": {"_nonfinite": "+inf"},
    "-inf": {"_nonfinite": "-inf"},
}


def to_json_value(value: Any) -> Any:
    """Convert a value to JSON-native Python types, strictly.

    - numpy scalars become Python scalars
    - numpy arrays become nested lists (callers decide artifact splitting)
    - tuples become lists; sets become deterministically sorted lists
    - NaN/Inf become explicit tagged objects, never JSON numeric tokens
    - non-string mapping keys raise TypeError instead of silently
      stringifying into colliding keys
    """
    if value is None or isinstance(value, (bool, str, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            if math.isnan(value):
                return NONFINITE_TAGS["nan"]
            return NONFINITE_TAGS["+inf" if value > 0 else "-inf"]
        return value
    if isinstance(value, np.generic):
        return to_json_value(value.item())
    if isinstance(value, np.ndarray):
        return [to_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {key: to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    if isinstance(value, set):
        return [to_json_value(item) for item in sorted(value, key=repr)]
    if hasattr(value, "item") and hasattr(value, "shape") and value.shape == ():
        return to_json_value(value.item())
    if hasattr(value, "to_mapping"):
        return to_json_value(value.to_mapping())
    raise TypeError(
        f"value of type {type(value).__name__} is not JSON-serializable"
    )


def strict_dumps(value: Any) -> str:
    """Order-preserving strict JSON dump of a normalized value."""
    return json.dumps(
        to_json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def canonical_dumps(value: Any) -> str:
    """Deterministic key-sorted dump used for content hashes."""
    return json.dumps(
        to_json_value(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_hex(value: Any) -> str:
    """Canonical content hash of a value (dict/list/array/primitive)."""
    return hashlib.sha256(canonical_dumps(value).encode("utf-8")).hexdigest()


__all__ = ["canonical_dumps", "sha256_hex", "strict_dumps", "to_json_value"]
