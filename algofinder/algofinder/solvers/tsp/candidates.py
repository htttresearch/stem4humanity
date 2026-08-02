"""Mixed geometric candidate graphs for Euclidean TSP search.

The candidate layer follows the Stage-1 blueprint: a union of k-nearest-
neighbour edges and per-angular-sector nearest edges, symmetrized and
truncated to a per-node budget.  Elite and current-tour edges can be
added afterwards without eviction.  Provenance of every undirected edge
is tracked so ablations can attribute quality differences to a source.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from algofinder.problems.tsp import (
    EuclideanTravellingSalespersonProblem,
)

CandidateGraph = list[set[int]]
Provenance = dict[tuple[int, int], set[str]]


def edge_key(first: int, second: int) -> tuple[int, int]:
    """Canonical undirected edge key."""
    return (first, second) if first < second else (second, first)


def build_mixed_candidates(
    problem: EuclideanTravellingSalespersonProblem,
    *,
    nearest_neighbors: int = 12,
    sector_count: int = 8,
    per_sector: int = 1,
    max_degree: int = 30,
) -> tuple[CandidateGraph, Provenance]:
    """Build the union of kNN and per-sector candidate edges.

    The graph is symmetrized (``j in C(i)`` iff ``i in C(j)``) and each
    directed list is truncated to the ``max_degree`` shortest edges.
    Returns the candidate lists and an edge-to-source provenance map.
    """
    city_count = problem.city_count
    if nearest_neighbors < 1:
        raise ValueError("nearest_neighbors must be positive")
    if sector_count < 1:
        raise ValueError("sector_count must be positive")
    if per_sector < 1:
        raise ValueError("per_sector must be positive")
    if max_degree < 1:
        raise ValueError("max_degree must be positive")

    distances = problem.distances
    candidates: CandidateGraph = [set() for _ in problem.nodes]
    provenance: Provenance = {}

    if nearest_neighbors > 0:
        ordered = np.argsort(distances, axis=1, kind="stable")
        for city in range(city_count):
            for neighbor in ordered[city, 1 : min(city_count, nearest_neighbors + 1)]:
                neighbor = int(neighbor)
                candidates[city].add(neighbor)
                provenance.setdefault(edge_key(city, neighbor), set()).add("knn")

    offsets = problem.points[:, np.newaxis, :] - problem.points[np.newaxis, :, :]
    angles = np.mod(
        np.arctan2(offsets[:, :, 1], offsets[:, :, 0]),
        2.0 * np.pi,
    )
    sector_of = np.floor(angles / (2.0 * np.pi) * sector_count).astype(int)
    sector_of = np.minimum(sector_of, sector_count - 1)
    self_mask = np.arange(city_count)
    for city in range(city_count):
        for sector in range(sector_count):
            indices = np.flatnonzero(
                (sector_of[city] == sector) & (self_mask != city)
            )
            if indices.size == 0:
                continue
            keep = min(per_sector, indices.size)
            closest = indices[
                np.argsort(distances[city, indices], kind="stable")[:keep]
            ]
            for neighbor in closest:
                neighbor = int(neighbor)
                candidates[city].add(neighbor)
                provenance.setdefault(edge_key(city, neighbor), set()).add("sector")

    for city, neighbors in enumerate(candidates):
        for neighbor in tuple(neighbors):
            if city not in candidates[neighbor]:
                candidates[neighbor].add(city)

    if max_degree < city_count:
        for city in range(city_count):
            if len(candidates[city]) <= max_degree:
                continue
            neighbors = np.asarray(
                sorted(
                    candidates[city],
                    key=lambda neighbor: float(distances[city, neighbor]),
                ),
                dtype=int,
            )
            candidates[city] = set(int(neighbor) for neighbor in neighbors[:max_degree])

    return candidates, provenance


def add_tour_edges(
    candidates: CandidateGraph,
    tour: tuple[int, ...] | NDArray[np.int64],
    provenance: Provenance | None = None,
    source: str = "elite",
) -> int:
    """Add every edge of ``tour`` to the candidate graph; return edges added."""
    added = 0
    city_count = len(candidates)
    if provenance is not None:
        provenance.setdefault("__sources__", set())
    for position in range(city_count):
        first = int(tour[position])
        second = int(tour[(position + 1) % city_count])
        if second not in candidates[first]:
            candidates[first].add(second)
            candidates[second].add(first)
            added += 1
            if provenance is not None:
                provenance.setdefault(edge_key(first, second), set()).add(source)
    return added


def provenance_summary(provenance: Provenance) -> dict[str, int]:
    """Count edges per provenance source (excluding the sentinel)."""
    counts: dict[str, int] = {}
    for sources in provenance.values():
        for source in sources:
            if source == "__sources__":
                continue
            counts[source] = counts.get(source, 0) + 1
    return counts


def degree_stats(candidates: CandidateGraph) -> dict[str, float]:
    """Per-node directed degree statistics for reporting."""
    degrees = np.asarray([len(neighbors) for neighbors in candidates], dtype=float)
    if degrees.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(degrees.min()),
        "max": float(degrees.max()),
        "mean": float(degrees.mean()),
    }


__all__ = [
    "add_tour_edges",
    "build_mixed_candidates",
    "degree_stats",
    "edge_key",
    "provenance_summary",
]
