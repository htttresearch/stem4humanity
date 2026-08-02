"""ML-guided Euclidean TSP search with classical feasibility operators."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from stem4humanity.problems.tsp import (
    EuclideanTravellingSalespersonProblem,
)
from stem4humanity.solvers.base import Solver, SolverResult, register_solver, InapplicableError
from stem4humanity.solvers.tsp.euclidean_geometry import (
    build_geometric_candidate_lists,
    candidate_edge_count,
)
from stem4humanity.solvers.tsp.euclidean_local_search import solve_with_candidate_lists

if TYPE_CHECKING:
    from stem4humanity.ml.candidate_ranker import CandidateEdgeRanker

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "public" / "models" / "candidate_ranker.joblib"


@register_solver
class LearnedCandidateTwoOptSolver(Solver):
    id = "learned-candidate-2opt"
    display = "Learned candidate ranking + chained 2-opt"
    tags = frozenset({"ml", "heuristic"})
    applies_to = frozenset({"tsp:euclidean", "tsp:clustered"})
    """Rank candidate edges with ML, then optimize using ordinary 2-opt."""

    def __init__(
        self,
        ranker: CandidateEdgeRanker | None = None,
        *,
        nearest_neighbors: int = 8,
        angular_sectors: int = 4,
        top_k: int = 5,
        max_restarts: int = 8,
        max_2opt_iterations: int = 2_000,
        seed: int = 0,
    ) -> None:
        if ranker is None:
            from stem4humanity.ml.candidate_ranker import CandidateEdgeRanker

            if not DEFAULT_MODEL_PATH.exists():
                raise InapplicableError(
                    "no default ranker at "
                    f"{DEFAULT_MODEL_PATH}; train one with "
                    "'python -m stem4humanity.harness.runner train'"
                )
            ranker = CandidateEdgeRanker.load(DEFAULT_MODEL_PATH)
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
        *,
        budget_seconds: float | None = None,
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
            solution=search.solution,
            cost=search.cost,
            exact=False,
            wall_seconds=preprocessing_seconds + model_seconds + search_seconds,
            metadata={
                "algorithm": "learned-candidate-chained-2opt",
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
