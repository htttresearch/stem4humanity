"""Reusable geometry-aware construction and candidate-edge routines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from algofinder.problems.tsp import (
    EuclideanTravellingSalespersonProblem,
)


def convex_hull_indices(points: NDArray[np.float64]) -> list[int]:
    """Return convex-hull vertices in cyclic order using the monotone chain."""
    indexed_points = sorted(
        ((float(point[0]), float(point[1]), index) for index, point in enumerate(points)),
        key=lambda item: (item[0], item[1], item[2]),
    )
    if len(indexed_points) <= 2:
        return [point[2] for point in indexed_points]

    def cross(
        origin: tuple[float, float, int],
        left: tuple[float, float, int],
        right: tuple[float, float, int],
    ) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (
            left[1] - origin[1]
        ) * (right[0] - origin[0])

    lower: list[tuple[float, float, int]] = []
    for point in indexed_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float, int]] = []
    for point in reversed(indexed_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    return [point[2] for point in hull]


def regret_insertion_tour(
    problem: EuclideanTravellingSalespersonProblem,
) -> tuple[int, ...]:
    """Construct a Euclidean tour from the hull using maximum-regret insertion."""
    tour = convex_hull_indices(problem.points)
    if len(tour) < 2:
        raise ValueError("at least two distinct points are needed for insertion")
    remaining = set(problem.nodes).difference(tour)

    while remaining:
        best_city = -1
        best_position = -1
        best_regret = -np.inf
        best_delta = np.inf
        for city in sorted(remaining):
            insertion_costs: list[tuple[float, int]] = []
            for position, left in enumerate(tour):
                right = tour[(position + 1) % len(tour)]
                delta = (
                    problem.distances[left, city]
                    + problem.distances[city, right]
                    - problem.distances[left, right]
                )
                insertion_costs.append((float(delta), position))
            insertion_costs.sort()
            candidate_delta, candidate_position = insertion_costs[0]
            next_delta = (
                insertion_costs[1][0] if len(insertion_costs) > 1 else candidate_delta
            )
            regret = next_delta - candidate_delta
            if (regret, -candidate_delta, -city) > (
                best_regret,
                -best_delta,
                -best_city,
            ):
                best_city = city
                best_position = candidate_position
                best_regret = regret
                best_delta = candidate_delta
        tour.insert(best_position + 1, best_city)
        remaining.remove(best_city)
    return tuple(tour)


def build_geometric_candidate_lists(
    problem: EuclideanTravellingSalespersonProblem,
    *,
    nearest_neighbors: int = 8,
    angular_sectors: int = 4,
) -> list[set[int]]:
    """Build kNN plus directional candidate edges for every city."""
    city_count = problem.city_count
    if nearest_neighbors < 1:
        raise ValueError("nearest_neighbors must be positive")
    if angular_sectors < 1:
        raise ValueError("angular_sectors must be positive")

    candidates = [set() for _ in problem.nodes]
    for city in problem.nodes:
        ordered = np.argsort(problem.distances[city], kind="stable")
        for neighbor in ordered[1 : min(city_count, nearest_neighbors + 1)]:
            candidates[city].add(int(neighbor))

        offsets = problem.points - problem.points[city]
        angles = np.mod(np.arctan2(offsets[:, 1], offsets[:, 0]), 2.0 * np.pi)
        sectors = np.floor(angles / (2.0 * np.pi) * angular_sectors).astype(int)
        sectors = np.minimum(sectors, angular_sectors - 1)
        for sector in range(angular_sectors):
            city_indices = np.flatnonzero(
                (sectors == sector) & (np.arange(city_count) != city)
            )
            if city_indices.size:
                nearest = city_indices[
                    np.argmin(problem.distances[city, city_indices])
                ]
                candidates[city].add(int(nearest))

    for city, neighbors in enumerate(candidates):
        for neighbor in tuple(neighbors):
            candidates[neighbor].add(city)
    return candidates


def candidate_edge_count(candidate_lists: Sequence[set[int]]) -> int:
    """Return the number of distinct undirected candidate edges."""
    return len(
        {
            tuple(sorted((city, neighbor)))
            for city, neighbors in enumerate(candidate_lists)
            for neighbor in neighbors
        }
    )


__all__ = [
    "build_geometric_candidate_lists",
    "candidate_edge_count",
    "convex_hull_indices",
    "regret_insertion_tour",
]
