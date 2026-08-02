"""Diversified Euclidean tour constructors.

Stage 1 provides two complementary seeds: a multi-fragment greedy over
candidate edges (with a global cheapest-edge fallback) and a heap-based
maximum-regret insertion from the convex hull.  Both consume the
candidate graph and degrade to exact scans when candidates are
insufficient, so they always return a feasible tour.
"""

from __future__ import annotations

import heapq

import numpy as np
from numpy.typing import NDArray

from algofinder.problems.tsp import (
    EuclideanTravellingSalespersonProblem,
)
from algofinder.solvers.tsp.candidates import CandidateGraph, edge_key
from algofinder.solvers.tsp.euclidean_geometry import convex_hull_indices


def _global_sorted_edges(distances: NDArray[np.float64]) -> tuple[list[float], list[int], list[int]]:
    """All strictly upper-triangle edges sorted ascending by cost."""
    city_count = distances.shape[0]
    row_indices, column_indices = np.triu_indices(city_count, 1)
    costs = distances[row_indices, column_indices]
    order = np.argsort(costs, kind="stable")
    return (
        [float(cost) for cost in costs[order]],
        [int(index) for index in row_indices[order]],
        [int(index) for index in column_indices[order]],
    )


def multi_fragment_tour(
    problem: EuclideanTravellingSalespersonProblem,
    candidate_lists: CandidateGraph,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Greedily add shortest compatible edges until every node has degree two.

    Uses a disjoint-set forest with degree constraints; candidate edges
    are consumed in cost order first, then the global cheapest-compatible
    edge list as a fallback so the construction cannot stall.
    """
    city_count = problem.city_count
    distances = problem.distances
    parent = np.arange(city_count, dtype=int)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    degree = np.zeros(city_count, dtype=int)
    candidate_edges: list[tuple[float, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for city, neighbors in enumerate(candidate_lists):
        for neighbor in neighbors:
            key = edge_key(city, neighbor)
            if key in seen:
                continue
            seen.add(key)
            candidate_edges.append((float(distances[city, neighbor]), key[0], key[1]))
    candidate_edges.sort(key=lambda edge: edge[0])

    chosen: list[tuple[int, int]] = []
    edge_index = 0
    global_costs: list[float] | None = None
    global_firsts: list[int] | None = None
    global_seconds: list[int] | None = None
    global_index = 0

    def try_edge(first: int, second: int) -> bool:
        nonlocal chosen
        if degree[first] >= 2 or degree[second] >= 2:
            return False
        root_first = find(first)
        root_second = find(second)
        if root_first == root_second:
            if len(chosen) != city_count - 1:
                return False
        else:
            parent[root_first] = root_second
        chosen.append((first, second))
        degree[first] += 1
        degree[second] += 1
        return True

    while len(chosen) < city_count:
        if edge_index < len(candidate_edges):
            _, first, second = candidate_edges[edge_index]
            edge_index += 1
            if try_edge(first, second):
                continue
            continue
        if global_costs is None:
            global_costs, global_firsts, global_seconds = _global_sorted_edges(
                distances
            )
        if global_index >= len(global_costs):
            raise RuntimeError("multi-fragment construction stalled on a complete graph")
        first = global_firsts[global_index]
        second = global_seconds[global_index]
        global_index += 1
        try_edge(first, second)

    successors: dict[int, list[int]] = {}
    for first, second in chosen:
        successors.setdefault(first, []).append(second)
        successors.setdefault(second, []).append(first)

    tour: list[int] = [int(chosen[0][0])]
    previous = -1
    current = tour[0]
    while len(tour) < city_count:
        neighbors = successors[current]
        nxt = neighbors[0] if neighbors[1] == previous else neighbors[1]
        tour.append(nxt)
        previous, current = current, nxt
    return tuple(tour)


def _best_position(
    distances: NDArray[np.float64],
    tour: list[int],
    candidate_lists: CandidateGraph,
    city: int,
) -> tuple[float, float, int, int]:
    """Best and second-best insertion deltas for ``city`` into ``tour``.

    Candidate positions are those whose tour edge touches a candidate of
    ``city``; the exact scan is the fallback when no candidate position
    exists.  Returns ``(best_delta, second_delta, best_pos, second_pos)``.
    """
    tour_array = np.asarray(tour, dtype=int)
    rights = np.roll(tour_array, -1)
    deltas = (
        distances[tour_array, city]
        + distances[city, rights]
        - distances[tour_array, rights]
    )
    candidates = np.asarray(list(candidate_lists[city]), dtype=int)
    mask = np.isin(tour_array, candidates) | np.isin(rights, candidates)
    if not mask.any():
        mask[:] = True
    masked = np.where(mask, deltas, np.inf)
    best = int(np.argmin(masked))
    best_delta = float(deltas[best])
    if mask.sum() > 1:
        masked[best] = np.inf
        second = int(np.argmin(masked))
        second_delta = float(deltas[second])
    else:
        second = best
        second_delta = best_delta
    return best_delta, second_delta, best, second


def hull_regret_tour(
    problem: EuclideanTravellingSalespersonProblem,
    candidate_lists: CandidateGraph,
) -> tuple[int, ...]:
    """Insert interior points by maximum regret from a convex-hull tour.

    Selection uses a max-heap keyed by regret with lazy invalidation:
    an entry carries the insertion version it was computed at, and is
    recomputed against the current tour when a later insertion made it
    stale.
    """
    city_count = problem.city_count
    distances = problem.distances
    tour = convex_hull_indices(problem.points)
    if len(tour) < 2:
        raise ValueError("at least two distinct points are needed for insertion")

    hull = set(tour)
    remaining = [city for city in range(city_count) if city not in hull]
    if not remaining:
        return tuple(tour)

    heap: list[tuple[float, float, int, int, int, int]] = []
    version = 0
    for city in remaining:
        best_delta, second_delta, best_pos, second_pos = _best_position(
            distances, tour, candidate_lists, city
        )
        heapq.heappush(
            heap,
            (-(second_delta - best_delta), -best_delta, best_pos, second_pos, version, city),
        )

    while heap:
        neg_regret, neg_delta, best_pos, second_pos, entry_version, city = heapq.heappop(heap)
        if entry_version != version:
            best_delta, second_delta, best_pos, second_pos = _best_position(
                distances, tour, candidate_lists, city
            )
            heapq.heappush(
                heap,
                (-(second_delta - best_delta), -best_delta, best_pos, second_pos, version, city),
            )
            continue
        tour.insert(best_pos + 1, city)
        version += 1
        remaining.remove(city)
    return tuple(tour)


def nearest_neighbor_tour(
    problem: EuclideanTravellingSalespersonProblem,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Greedy nearest-neighbour tour from a random start city."""
    city_count = problem.city_count
    distances = problem.distances
    start = int(rng.integers(city_count))
    tour = [start]
    remaining = {city for city in range(city_count) if city != start}
    current = start
    while remaining:
        nxt = min(remaining, key=lambda city: float(distances[current, city]))
        tour.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return tuple(tour)


__all__ = [
    "hull_regret_tour",
    "multi_fragment_tour",
    "nearest_neighbor_tour",
]
