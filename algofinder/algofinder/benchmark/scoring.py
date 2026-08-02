"""Independent tour scoring, normalization, digests, and metamorphic helpers.

The harness must never trust a solver-reported cost: every accepted tour is
scored here under a named objective spec, checked for feasibility, and stored
with a content digest of its canonical form. Metamorphic transforms
(catalog section 3.2) generate point sets whose optima must relate to the
parent in a known way, which the micro-corpus report verifies.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from algofinder.benchmark.objectives import ObjectiveSpec, get_objective

TOUR_ENCODING = "zero_based_city_permutation_without_repeated_start"


def verify_feasible(tour: Sequence[int], city_count: int) -> bool:
    """A tour is a permutation of every city index exactly once."""
    if len(tour) != city_count:
        return False
    try:
        values = [int(city) for city in tour]
    except (TypeError, ValueError):
        return False
    return set(values) == set(range(city_count))


def normalize_tour(tour: Sequence[int]) -> tuple[int, ...]:
    """Canonical tour form: start at the minimum index, lexicographic orientation.

    Rotations and reversals of the same geometric cycle map to one canonical
    encoding, so tours produced differently but representing the same cycle
    share a digest.
    """
    values = tuple(int(city) for city in tour)
    if not values:
        return values
    offset = values.index(min(values))
    rotated = values[offset:] + values[:offset]
    reversed_rotated = rotated[:1] + tuple(reversed(rotated[1:]))
    return min(rotated, reversed_rotated)


def tour_sha256(tour: Sequence[int]) -> str:
    """sha256 digest of the canonical tour encoding."""
    canonical = normalize_tour(tour)
    payload = ",".join(str(city) for city in canonical)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def points_sha256(points: NDArray[np.float64]) -> str:
    """Canonical point-set digest: scale-invariant? No — raw canonical bytes.

    Uses a fixed textual encoding of the float64 values so the digest is
    stable across runs and machines: each coordinate printed with 17
    significant digits, rows sorted lexicographically (point-set identity).
    """
    canonical = np.asarray(points, dtype=np.float64)
    rows = [" ".join(f"{value:.17g}" for value in row) for row in canonical]
    rows.sort()
    payload = "\n".join(rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def score_tour(
    points: NDArray[np.float64],
    tour: Sequence[int],
    objective_spec: str | ObjectiveSpec = "raw_l2_f64",
) -> float:
    """Feasibility-checked independent objective of a closed tour."""
    if isinstance(objective_spec, str):
        objective_spec = get_objective(objective_spec)
    coordinates = np.asarray(points, dtype=float)
    if not verify_feasible(tour, coordinates.shape[0]):
        raise ValueError("tour must be a permutation of every city index")
    return objective_spec.tour_objective(coordinates, np.asarray(tour, dtype=int))


# --- metamorphic transforms (catalog section 3.2) ---


def translate(points: NDArray[np.float64], vector: Sequence[float]) -> NDArray[np.float64]:
    return np.asarray(points, dtype=float) + np.asarray(vector, dtype=float)


def rotate(points: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    coordinates = np.asarray(points, dtype=float)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=float)
    return coordinates @ rotation.T


def reflect_x(points: NDArray[np.float64]) -> NDArray[np.float64]:
    coordinates = np.asarray(points, dtype=float).copy()
    coordinates[:, 0] *= -1.0
    return coordinates


def scale(points: NDArray[np.float64], factor: float) -> NDArray[np.float64]:
    return np.asarray(points, dtype=float) * factor


def permute_points(
    points: NDArray[np.float64], permutation: Sequence[int]
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Reorder cities by ``permutation``; returns new points and the tour remap."""
    permutation = np.asarray(permutation, dtype=int)
    if set(permutation.tolist()) != set(range(points.shape[0])):
        raise ValueError("permutation must map every index once")
    return np.asarray(points, dtype=float)[permutation], permutation


# --- expected metamorphic relations ---

PARENT_OPS = {
    "identity": lambda points: np.asarray(points, dtype=float),
    "translate": lambda points: translate(points, (0.37, -1.21)),
    "reflect_x": reflect_x,
    "rotate_0.7": lambda points: rotate(points, 0.7),
    "scale_2.5": lambda points: scale(points, 2.5),
}

SCALE_OPS = {"scale_2.5": 2.5}


def check_relation(
    parent_objective: float,
    sibling_objective: float,
    operation: str,
    *,
    atol: float = 1e-9,
) -> bool:
    """Expected relation between a parent and a sibling objective."""
    if operation == "scale_2.5":
        return abs(sibling_objective - SCALE_OPS[operation] * parent_objective) <= atol
    if operation == "identity":
        return abs(sibling_objective - parent_objective) <= atol
    return abs(sibling_objective - parent_objective) <= atol


def apply_operation(
    points: NDArray[np.float64], operation: str
) -> NDArray[np.float64]:
    try:
        return PARENT_OPS[operation](points)
    except KeyError:
        raise ValueError(f"unknown metamorphic operation: {operation}") from None


__all__ = [
    "TOUR_ENCODING",
    "PARENT_OPS",
    "apply_operation",
    "check_relation",
    "normalize_tour",
    "permute_points",
    "points_sha256",
    "reflect_x",
    "rotate",
    "scale",
    "score_tour",
    "tour_sha256",
    "translate",
    "verify_feasible",
]
