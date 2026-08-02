"""ML-guided Euclidean TSP search with classical feasibility operators."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from src.ml.candidate_ranker import CandidateEdgeRanker
from src.problems.travelling_salesperson_problem import (
    EuclideanTravellingSalespersonProblem,
)
from src.solvers.base_solver import BaseSolver, SolverResult
from src.solvers.euclidean_geometry import (
    build_geometric_candidate_lists,
    candidate_edge_count,
)
from src.solvers.euclidean_local_search import solve_with_candidate_lists


class LearnedCandidateTwoOptSolver(BaseSolver[EuclideanTravellingSalespersonProblem]):
    """Rank candidate edges with ML, then optimize using ordinary 2-opt."""

    def __init__(
        self,
        ranker: CandidateEdgeRanker,
        *,
        nearest_neighbors: int = 8,
        angular_sectors: int = 4,
        top_k: int = 5,
        max_restarts: int = 8,
        max_2opt_iterations: int = 2_000,
        seed: int = 0,
    ) -> None:
        if not ranker.is_fitted:
            raise ValueError("ranker must be fit or loaded before solving")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if max_restarts < 0:
            raise ValueError("max_restarts cannot be negative")
        self.ranker = ranker
        self.nearest_neighbors = nearest_neighbors
        self.angular_sectors = angular_sectors
        self.top_k = top_k
        self.max_restarts = max_restarts
        self.max_2opt_iterations = max_2opt_iterations
        self.seed = seed

    def solve(
        self,
        problem: EuclideanTravellingSalespersonProblem,
    ) -> SolverResult:
        preprocessing_started = perf_counter()
        geometric_candidates = build_geometric_candidate_lists(
            problem,
            nearest_neighbors=self.nearest_neighbors,
            angular_sectors=self.angular_sectors,
        )
        preprocessing_seconds = perf_counter() - preprocessing_started
        model_started = perf_counter()
        scores = self.ranker.score_candidate_edges(problem, geometric_candidates)
        candidate_lists = self.ranker.select_top_candidates(
            problem,
            geometric_candidates,
            scores,
            top_k=self.top_k,
        )
        model_seconds = perf_counter() - model_started
        search_started = perf_counter()
        search = solve_with_candidate_lists(
            problem,
            candidate_lists,
            max_restarts=self.max_restarts,
            max_2opt_iterations=self.max_2opt_iterations,
            seed=self.seed,
        )
        search_seconds = perf_counter() - search_started
        return SolverResult(
            tour=search.tour,
            cost=search.cost,
            metadata={
                "algorithm": "learned-candidate-chained-2opt",
                "exact": False,
                **search.metadata,
                "candidate_edges_before_ranking": candidate_edge_count(
                    geometric_candidates
                ),
                "candidate_edges_after_ranking": candidate_edge_count(candidate_lists),
                "mean_candidate_score": float(np.mean(list(scores.values()))),
                "top_k": self.top_k,
                "preprocessing_seconds": preprocessing_seconds,
                "model_seconds": model_seconds,
                "search_seconds": search_seconds,
            },
        )


__all__ = ["LearnedCandidateTwoOptSolver"]
