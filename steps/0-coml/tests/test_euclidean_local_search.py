from __future__ import annotations

import unittest

import numpy as np

from src.data.euclidean_tsp import generate_euclidean_instance
from src.solvers.euclidean_geometry import (
    build_geometric_candidate_lists,
    regret_insertion_tour,
)
from src.solvers.euclidean_local_search import (
    EuclideanChainedTwoOptSolver,
    double_bridge_perturbation,
)
from src.solvers.held_karp import HeldKarpSolver


class EuclideanLocalSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = generate_euclidean_instance(12, "corridor", seed=41).problem

    def test_geometric_candidates_exclude_self_edges(self) -> None:
        candidates = build_geometric_candidate_lists(self.problem)
        self.assertEqual(len(candidates), self.problem.city_count)
        self.assertTrue(
            all(city not in neighbors for city, neighbors in enumerate(candidates))
        )

    def test_manual_solver_returns_feasible_tour_not_better_than_oracle(self) -> None:
        initial_cost = self.problem.objective_value(regret_insertion_tour(self.problem))
        result = EuclideanChainedTwoOptSolver(max_restarts=3, seed=5).solve(
            self.problem
        )
        oracle = HeldKarpSolver().solve(self.problem)

        self.assertTrue(self.problem.verify(result.tour))
        self.assertLessEqual(result.cost, initial_cost + 1e-9)
        self.assertGreaterEqual(result.cost, oracle.cost - 1e-9)
        self.assertGreater(result.metadata["move_evaluations"], 0)

    def test_double_bridge_preserves_tour_vertices(self) -> None:
        tour = tuple(range(12))
        perturbed = double_bridge_perturbation(tour, np.random.default_rng(7))
        self.assertEqual(set(perturbed), set(tour))
        self.assertEqual(len(perturbed), len(tour))


if __name__ == "__main__":
    unittest.main()
