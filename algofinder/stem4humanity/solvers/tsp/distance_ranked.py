"""Control solver: rank geometric candidate edges purely by Euclidean distance.

Serves as the non-ML baseline for the learned candidate ranker. Uses the same
chained 2-opt machinery, so any gap to ``learned-candidate-2opt`` is due to
the learned ranking.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from stem4humanity.problems.tsp import EuclideanTravellingSalespersonProblem
from stem4humanity.solvers.base import Solver, SolverResult, register_solver
from stem4humanity.solvers.tsp.euclidean_geometry import build_geometric_candidate_lists
from stem4humanity.solvers.tsp.euclidean_local_search import (
    candidate_edge_count,
    solve_with_candidate_lists,
)


@register_solver
class DistanceRankedTwoOptSolver(Solver):
    id = "distance-ranked-2opt"
    display = "Distance-ranked candidate chained 2-opt (control)"
    tags = frozenset({"control", "heuristic"})
    applies_to = frozenset({"tsp:euclidean", "tsp:clustered"})

    def __init__(
        self,
        *,
        nearest_neighbors: int = 12,
        angular_sectors: int = 8,
        top_k: int = 8,
        max_restarts: int = 8,
        max_2opt_iterations: int = 2_000,
        seed: int = 0,
    ) -> None:
        self.nearest_neighbors = nearest_neighbors
        self.angular_sectors = angular_sectors
        self.top_k = top_k
        self.max_restarts = max_restarts
        self.max_2opt_iterations = max_2opt_iterations
        self.seed = seed

    def solve(
        self,
        problem: EuclideanTravellingSalespersonProblem,
        *,
        budget_seconds: float | None = None,
    ) -> SolverResult:
        preprocessing_started = perf_counter()
        candidate_lists = build_geometric_candidate_lists(
            problem,
            nearest_neighbors=self.nearest_neighbors,
            angular_sectors=self.angular_sectors,
        )
        n = problem.city_count
        distances = np.asarray(problem.distances, dtype=float)
        ranked_lists: list[set[int]] = []
        for i in range(n):
            ranked_lists.append(
                set(
                    sorted(
                        candidate_lists[i],
                        key=lambda j: (distances[i, j], j),
                    )[: self.top_k]
                )
            )
        preprocessing_seconds = perf_counter() - preprocessing_started
        search_started = perf_counter()
        search = solve_with_candidate_lists(
            problem,
            ranked_lists,
            max_restarts=self.max_restarts,
            max_2opt_iterations=self.max_2opt_iterations,
            seed=self.seed,
        )
        search_seconds = perf_counter() - search_started
        return SolverResult(
            solution=search.solution,
            cost=search.cost,
            exact=False,
            wall_seconds=preprocessing_seconds + search_seconds,
            metadata={
                "algorithm": "distance-ranked-2opt",
                **search.metadata,
                "top_k": self.top_k,
                "preprocessing_seconds": preprocessing_seconds,
                "search_seconds": search_seconds,
            },
        )
