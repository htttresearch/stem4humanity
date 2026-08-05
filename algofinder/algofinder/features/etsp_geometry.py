"""Tier-1 Euclidean-TSP geometry features (``etsp-geometry@1``).

Deterministic geometry descriptors for Euclidean TSP instances: convex-hull
fraction, normalized MST weight, nearest-neighbour distance statistics,
candidate-graph degree statistics, and bounding-box / pairwise span stats.
All computation is pure (no global RNG, no wall clock), so records are
byte-reproducible for the same extractor digest.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from algofinder.features.base import FeatureSet
from algofinder.problems.base import Instance
from algofinder.problems.tsp import EuclideanTravellingSalespersonProblem
from algofinder.solvers.tsp.candidates import build_mixed_candidates, degree_stats

FEATURE_SET_ID = "etsp-geometry@1"
FEATURE_SET_TIER = 1
FEATURE_SET_DESCRIPTION = (
    "tier-1 Euclidean-TSP geometry: hull, MST, NN, candidate-degree, span"
)
APPLIES_TO = frozenset({"tsp:clustered", "tsp:euclidean"})


def _convex_hull(points: NDArray[np.float64]) -> list[int]:
    """Monotone-chain hull over 2-D points; returns hull vertex indices."""
    order = sorted(range(len(points)), key=lambda i: (points[i, 0], points[i, 1]))
    cross = (
        lambda o, a, b: (points[a, 0] - points[o, 0]) * (points[b, 1] - points[o, 1])
        - (points[a, 1] - points[o, 1]) * (points[b, 0] - points[o, 0])
    )

    def chain(indices: list[int]) -> list[int]:
        lower: list[int] = []
        for index in indices:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], index) <= 0:
                lower.pop()
            lower.append(index)
        return lower

    lower = chain(order)
    upper = chain(list(reversed(order)))
    return lower[:-1] + upper[:-1]


def _mst_weight(distances: NDArray[np.float64]) -> float:
    """Prim's MST over a complete distance matrix (deterministic)."""
    city_count = distances.shape[0]
    visited = np.zeros(city_count, dtype=bool)
    closest = np.full(city_count, np.inf)
    closest[0] = 0.0
    total = 0.0
    for _ in range(city_count):
        best = int(np.argmin(np.where(visited, np.inf, closest)))
        visited[best] = True
        total += float(closest[best])
        closest = np.minimum(closest, distances[best])
    return float(total)


def extract_etsp_geometry(instance: Instance) -> dict[str, Any]:
    """Geometry features for Euclidean TSP instances (deterministic)."""
    points = np.asarray(instance.data["points"], dtype=float)
    problem = EuclideanTravellingSalespersonProblem(
        name=f"features-{instance.name}", points=points
    )
    distances = problem.distances
    city_count = problem.city_count

    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    bbox_width = float(max_x - min_x)
    bbox_height = float(max_y - min_y)
    bbox_diag = float(math.hypot(bbox_width, bbox_height))

    hull = _convex_hull(points)
    hull_fraction = len(hull) / city_count if city_count else 0.0

    mst = _mst_weight(distances)
    scale = max(bbox_diag * math.sqrt(city_count), 1e-9)
    mst_normalized = mst / scale

    nn = np.sort(distances, axis=1)[:, 1]
    nn_stats = {
        "min": float(nn.min()),
        "mean": float(nn.mean()),
        "max": float(nn.max()),
        "std": float(nn.std()),
    }

    candidates, _ = build_mixed_candidates(problem)
    degrees = degree_stats(candidates)
    candidate_degree = {
        "min": float(degrees["min"]),
        "max": float(degrees["max"]),
        "mean": float(degrees["mean"]),
    }

    off_diag = distances[~np.eye(city_count, dtype=bool)]
    span = {
        "min": float(off_diag.min()),
        "max": float(off_diag.max()),
    }

    return {
        "n": city_count,
        "hull_fraction": round(hull_fraction, 12),
        "mst_weight_normalized": round(mst_normalized, 12),
        "nn": {key: round(value, 12) for key, value in nn_stats.items()},
        "candidate_degree": {key: round(value, 12) for key, value in candidate_degree.items()},
        "span": {key: round(value, 12) for key, value in span.items()},
        "bbox": {
            "width": round(bbox_width, 12),
            "height": round(bbox_height, 12),
            "diag": round(bbox_diag, 12),
        },
    }


etsp_geometry_feature_set = FeatureSet(
    id=FEATURE_SET_ID,
    tier=FEATURE_SET_TIER,
    description=FEATURE_SET_DESCRIPTION,
    extractor=extract_etsp_geometry,
    applies_to=APPLIES_TO,
)


__all__ = [
    "APPLIES_TO",
    "FEATURE_SET_ID",
    "etsp_geometry_feature_set",
    "extract_etsp_geometry",
]
