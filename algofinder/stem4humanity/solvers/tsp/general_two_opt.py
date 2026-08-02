"""General 2-opt for arbitrary symmetric TSP instances (no geometry required)."""

from __future__ import annotations

import time

import numpy as np

from stem4humanity.problems.tsp import TravellingSalespersonProblem
from stem4humanity.solvers.base import Solver, SolverResult, register_solver


def _tour_cost(tour: tuple[int, ...], distances: np.ndarray) -> float:
    total = 0.0
    for i in range(len(tour)):
        total += distances[tour[i], tour[(i + 1) % len(tour)]]
    return float(total)


def _nearest_neighbor_tour(distances: np.ndarray, start: int) -> tuple[int, ...]:
    n = distances.shape[0]
    remaining = set(range(n))
    tour = [start]
    remaining.discard(start)
    current = start
    while remaining:
        nxt = min(remaining, key=lambda j: distances[current, j])
        tour.append(nxt)
        remaining.discard(nxt)
        current = nxt
    return tuple(tour)


def _cheapest_insertion_tour(distances: np.ndarray, start_a: int, start_b: int) -> tuple[int, ...]:
    n = distances.shape[0]
    tour = [start_a, start_b]
    remaining = set(range(n)) - {start_a, start_b}
    while remaining:
        best_position = -1
        best_node = -1
        best_delta = float("inf")
        for j in remaining:
            for i in range(len(tour)):
                a = tour[i]
                b = tour[(i + 1) % len(tour)]
                delta = distances[a, j] + distances[j, b] - distances[a, b]
                if delta < best_delta:
                    best_delta = delta
                    best_position = i
                    best_node = j
        tour.insert(best_position + 1, best_node)
        remaining.discard(best_node)
    return tuple(tour)


def _two_opt_improve(tour: tuple[int, ...], distances: np.ndarray) -> tuple[int, ...]:
    n = len(tour)
    current_cost = _tour_cost(tour, distances)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                a, b = tour[i], tour[(i + 1) % n]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (
                    distances[a, c]
                    + distances[b, d]
                    - distances[a, b]
                    - distances[c, d]
                )
                if delta < -1e-12:
                    tour = tour[: i + 1] + tour[i + 1 : j + 1][::-1] + tour[j + 1 :]
                    current_cost += delta
                    improved = True
                    break
            if improved:
                break
    return tour


@register_solver
class GeneralTwoOptSolver(Solver):
    id = "general-2opt"
    display = "NN/cheapest-insertion construction + 2-opt (any metric)"
    tags = frozenset({"heuristic"})
    applies_to = frozenset({"tsp:general", "tsp:euclidean", "tsp:clustered"})

    def __init__(self, starts: int = 8, seed: int = 0) -> None:
        self.starts = starts
        self.seed = seed

    def solve(
        self,
        problem: TravellingSalespersonProblem,
        *,
        budget_seconds: float | None = None,
    ) -> SolverResult:
        distances = np.asarray(problem.distances, dtype=float)
        n = distances.shape[0]
        started = time.perf_counter()
        if n <= 1:
            return SolverResult(
                solution=(0,),
                cost=0.0,
                exact=True,
                wall_seconds=time.perf_counter() - started,
                metadata={"algorithm": "general-2opt"},
            )

        rng = np.random.default_rng(self.seed)
        candidates: list[tuple[int, ...]] = []
        for k in range(self.starts):
            if k == 0:
                candidates.append(_nearest_neighbor_tour(distances, 0))
            elif k == 1:
                candidates.append(
                    _cheapest_insertion_tour(distances, 0, 1)
                )
            else:
                start_a, start_b = rng.choice(n, size=2, replace=False)
                candidates.append(_cheapest_insertion_tour(distances, int(start_a), int(start_b)))

        best_tour = min(candidates, key=lambda tour: _tour_cost(tour, distances))
        best_tour = _two_opt_improve(best_tour, distances)
        best_cost = _tour_cost(best_tour, distances)

        return SolverResult(
            solution=best_tour,
            cost=best_cost,
            exact=False,
            wall_seconds=time.perf_counter() - started,
            metadata={"algorithm": "general-2opt", "starts": self.starts},
        )
