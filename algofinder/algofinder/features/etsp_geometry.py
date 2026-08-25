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

FEATURE_SET_V2_ID = "etsp-geometry@2"
FEATURE_SET_V2_DESCRIPTION = (
    "tier-1 Euclidean-TSP geometry v2: v1 plus orientation, anisotropy, "
    "angular concentration, radial structure, and lightweight cluster summaries"
)


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


def extract_etsp_geometry_v2(instance: Instance) -> dict[str, Any]:
    """Extend v1 with declared-orientation and distribution-profile features.

    The values are geometry-only and deliberately contain no solver outcome or
    benchmark information, so a profile can safely drive retrieval before an
    authority evaluation is run.
    """
    result = dict(extract_etsp_geometry(instance))
    points = np.asarray(instance.data["points"], dtype=float)
    count = len(points)
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = (centered.T @ centered) / max(count, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    principal = eigenvectors[:, order[0]]
    orientation = math.atan2(float(principal[1]), float(principal[0])) % (math.pi / 2.0)
    anisotropy = float(eigenvalues[0] / max(float(eigenvalues[-1]), 1e-12))

    if count >= 2:
        deltas = points[None, :, :] - points[:, None, :]
        distances = np.hypot(deltas[:, :, 0], deltas[:, :, 1])
        mask = ~np.eye(count, dtype=bool)
        pair_angles = np.arctan2(deltas[:, :, 1][mask], deltas[:, :, 0][mask])
        nearest = np.argmin(np.where(np.eye(count, dtype=bool), np.inf, distances), axis=1)
        nn_angles = np.arctan2(
            points[nearest, 1] - points[:, 1], points[nearest, 0] - points[:, 0]
        )
    else:  # pragma: no cover - TSP manifests normally have >= 2 cities.
        pair_angles = np.zeros(1)
        nn_angles = np.zeros(1)
    pair_hist = _angle_histogram(pair_angles)
    nn_hist = _angle_histogram(nn_angles)
    concentration_2 = _directional_concentration(pair_angles, 2)
    concentration_4 = _directional_concentration(pair_angles, 4)
    radial = np.hypot(centered[:, 0], centered[:, 1])
    cluster = _cluster_summary(points)

    result.update({
        "feature_set_version": FEATURE_SET_V2_ID,
        "centered_scale": round(float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))), 12),
        "aspect_ratio": round(max(result["bbox"]["width"], result["bbox"]["height"]) / max(min(result["bbox"]["width"], result["bbox"]["height"]), 1e-12), 12),
        "pca": {
            "eigenvalues": [round(float(value), 12) for value in eigenvalues],
            "anisotropy_ratio": round(anisotropy, 12),
            "principal_axis_orientation_mod_pi_over_2": round(orientation, 12),
        },
        "angles": {
            "pairwise_histogram": pair_hist,
            "nearest_neighbor_histogram": nn_hist,
            "twofold_concentration": round(concentration_2, 12),
            "fourfold_concentration": round(concentration_4, 12),
            "orientation_entropy": round(_entropy(pair_hist), 12),
            "grid_axis_alignment": round(concentration_4, 12),
        },
        "clusters": cluster,
        "radial": {
            "mean": round(float(radial.mean()), 12),
            "std": round(float(radial.std()), 12),
            "coefficient_of_variation": round(float(radial.std() / max(radial.mean(), 1e-12)), 12),
        },
    })
    depot = instance.data.get("depot") if isinstance(instance.data, dict) else None
    if isinstance(depot, (list, tuple)) and len(depot) == 2:
        depot_distances = np.hypot(points[:, 0] - float(depot[0]), points[:, 1] - float(depot[1]))
        result["depot_relative"] = {
            "mean_distance": round(float(depot_distances.mean()), 12),
            "std_distance": round(float(depot_distances.std()), 12),
        }
    return result


def _angle_histogram(angles: NDArray[np.float64], bins: int = 12) -> list[float]:
    counts, _ = np.histogram((angles + math.pi) % (2 * math.pi), bins=bins, range=(0.0, 2 * math.pi))
    total = max(int(counts.sum()), 1)
    return [round(float(value / total), 12) for value in counts]


def _directional_concentration(angles: NDArray[np.float64], harmonic: int) -> float:
    return float(abs(np.mean(np.exp(1j * harmonic * angles))))


def _entropy(histogram: list[float]) -> float:
    nonzero = [value for value in histogram if value > 0]
    if not nonzero:
        return 0.0
    return -sum(value * math.log(value) for value in nonzero) / math.log(len(histogram))


def _cluster_summary(points: NDArray[np.float64]) -> dict[str, float]:
    """Deterministic connected-components proxy; avoids a clustering dependency."""
    count = len(points)
    if count < 2:
        return {"count": float(count), "mean_density": 0.0, "separation_ratio": 0.0}
    deltas = points[None, :, :] - points[:, None, :]
    distances = np.hypot(deltas[:, :, 0], deltas[:, :, 1])
    nearest = np.min(np.where(np.eye(count, dtype=bool), np.inf, distances), axis=1)
    threshold = float(np.median(nearest) * 2.0)
    parent = list(range(count))
    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)
    for left in range(count):
        for right in range(left + 1, count):
            if distances[left, right] <= threshold:
                union(left, right)
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    centers = np.asarray([points[indices].mean(axis=0) for indices in groups.values()])
    if len(centers) > 1:
        center_delta = centers[None, :, :] - centers[:, None, :]
        separation = float(np.min(np.hypot(center_delta[:, :, 0] + np.eye(len(centers)) * 1e30, center_delta[:, :, 1])))
    else:
        separation = 0.0
    return {
        "count": float(len(groups)),
        "mean_density": round(float(np.mean([len(indices) for indices in groups.values()])), 12),
        "separation_ratio": round(separation / max(float(np.median(nearest)), 1e-12), 12),
    }


etsp_geometry_feature_set = FeatureSet(
    id=FEATURE_SET_ID,
    tier=FEATURE_SET_TIER,
    description=FEATURE_SET_DESCRIPTION,
    extractor=extract_etsp_geometry,
    applies_to=APPLIES_TO,
)

etsp_geometry_v2_feature_set = FeatureSet(
    id=FEATURE_SET_V2_ID,
    tier=FEATURE_SET_TIER,
    description=FEATURE_SET_V2_DESCRIPTION,
    extractor=extract_etsp_geometry_v2,
    applies_to=APPLIES_TO,
)


__all__ = [
    "APPLIES_TO",
    "FEATURE_SET_ID",
    "FEATURE_SET_V2_ID",
    "etsp_geometry_feature_set",
    "etsp_geometry_v2_feature_set",
    "extract_etsp_geometry",
    "extract_etsp_geometry_v2",
]
