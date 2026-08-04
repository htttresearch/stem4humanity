"""Content-addressed digests for instances, encodings, and references (spec section 4).

Canonical coordinate hashing is fully specified here:

- scalar encoding: one float64 per token, printed with 17 significant
  digits (round-trip-exact decimal form);
- endianness: N/A for decimal text, but the row/column order is fixed;
- ``-0.0`` is normalized to ``0.0``; NaN and infinity are rejected
  (benchmark instances must be finite);
- city order: normalized — rows are sorted lexicographically, so the same
  point set in a different city order has the same parent digest
  (duplicate cities keep their multiplicity);
- dimension and coordinate-unit metadata are embedded in the payload.

Original byte digests (``encoding_id``) are kept independently and never
replaced by canonical digests.
"""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

SCHEMA_VERSION = "etsp.digest.v1"


def canonical_points(points: Sequence[Sequence[float]]) -> NDArray[np.float64]:
    """Validate and normalize a coordinate array.

    Rejects NaN/inf, coerces ``-0.0`` to ``0.0``, and returns a float64 copy.
    """
    array = np.asarray(points, dtype=float)
    if array.ndim != 2:
        raise ValueError("points must be a 2-d array")
    if not np.isfinite(array).all():
        raise ValueError("coordinates must be finite (no NaN or infinity)")
    array = np.where(array == 0.0, 0.0, array)
    return np.asarray(array, dtype=np.float64)


def _payload_sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def coordinate_parent_id(points: Sequence[Sequence[float]]) -> str:
    """Digest of the canonical coordinate multiset plus dimension.

    Identifies the point set independent of city order and objective metric.
    """
    array = canonical_points(points)
    rows = [" ".join(f"{value:.17g}" for value in row) for row in array]
    rows.sort()
    payload = f"{SCHEMA_VERSION}\ncoordinate\n{array.shape[1]}\n" + "\n".join(rows)
    return _payload_sha256(payload)


def instance_id(
    points: Sequence[Sequence[float]], objective_full_id: str
) -> str:
    """Digest of (coordinate parent, objective spec): one ID per metric.

    The same coordinates under ``tsplib_euc_2d@1`` and ``raw_l2_f64@1`` must
    produce distinct instance IDs (phase 2 exit criterion).
    """
    payload = (
        f"{SCHEMA_VERSION}\ninstance\n{coordinate_parent_id(points)}\n"
        f"{objective_full_id}"
    )
    return _payload_sha256(payload)


def encoding_id(raw_bytes: bytes) -> str:
    """Digest of the original upstream file bytes."""
    return _payload_sha256(f"{SCHEMA_VERSION}\nencoding\n" + hashlib.sha256(raw_bytes).hexdigest())


def lineage_group_id(group_name: str) -> str:
    """Digest for a lineage root (transforms/morphs share the parent root)."""
    return _payload_sha256(f"{SCHEMA_VERSION}\nlineage\n{group_name}")


def objective_sibling_group_id(points: Sequence[Sequence[float]]) -> str:
    """Shared across metrics derived from the same coordinates."""
    return coordinate_parent_id(points)


def reference_id(
    instance_id_value: str,
    kind: str,
    value: float | int | str,
    tour_digest: str | None,
    provenance: dict[str, object],
) -> str:
    """Digest of (instance + kind + value/tour/bound + provenance)."""
    provenance_json = json.dumps(provenance, sort_keys=True, default=str)
    payload = (
        f"{SCHEMA_VERSION}\nreference\n{instance_id_value}\n{kind}\n{value}\n"
        f"{tour_digest if tour_digest else 'no-tour'}\n"
        f"{_payload_sha256(provenance_json)}"
    )
    return _payload_sha256(payload)


def sha256_hex(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "canonical_points",
    "coordinate_parent_id",
    "encoding_id",
    "instance_id",
    "lineage_group_id",
    "objective_sibling_group_id",
    "reference_id",
    "sha256_hex",
]
