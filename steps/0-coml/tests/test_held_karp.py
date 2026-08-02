from __future__ import annotations

from itertools import permutations
import unittest

import numpy as np

from src.problems.travelling_salesperson_problem import TravellingSalespersonProblem
from src.solvers.held_karp import HeldKarpSolver


class HeldKarpTests(unittest.TestCase):
    def test_returns_optimum_for_square(self) -> None:
        problem = TravellingSalespersonProblem(
            "square",
            [
                [0.0, 1.0, 2.0**0.5, 1.0],
                [1.0, 0.0, 1.0, 2.0**0.5],
                [2.0**0.5, 1.0, 0.0, 1.0],
                [1.0, 2.0**0.5, 1.0, 0.0],
            ],
        )
        result = HeldKarpSolver().solve(problem)

        self.assertTrue(problem.verify(result.tour))
        self.assertAlmostEqual(result.cost, 4.0)
        self.assertAlmostEqual(result.cost, problem.objective_value(result.tour))

    def test_matches_brute_force_on_small_symmetric_problem(self) -> None:
        rng = np.random.default_rng(22)
        raw = rng.uniform(0.1, 5.0, size=(6, 6))
        matrix = (raw + raw.T) / 2.0
        np.fill_diagonal(matrix, 0.0)
        problem = TravellingSalespersonProblem("random", matrix)

        exact = HeldKarpSolver().solve(problem)
        brute_force = min(
            problem.objective_value((0, *permutation))
            for permutation in permutations(range(1, problem.city_count))
        )
        self.assertAlmostEqual(exact.cost, brute_force)

    def test_city_limit_prevents_accidental_large_run(self) -> None:
        matrix = np.ones((5, 5), dtype=float)
        np.fill_diagonal(matrix, 0.0)
        problem = TravellingSalespersonProblem("five", matrix)
        with self.assertRaises(ValueError):
            HeldKarpSolver(max_cities=4).solve(problem)


if __name__ == "__main__":
    unittest.main()
