from __future__ import annotations

from itertools import permutations
import unittest

import numpy as np

from src.problems.travelling_salesperson_problem import TravellingSalespersonProblem
from src.solvers.exact_tsp_solver import AssignmentBranchAndBoundTSPSolver
from src.solvers.held_karp import HeldKarpSolver


class ExactTSPSolverTests(unittest.TestCase):
    def test_matches_held_karp_on_square(self) -> None:
        problem = TravellingSalespersonProblem(
            "square",
            [
                [0.0, 1.0, 2.0**0.5, 1.0],
                [1.0, 0.0, 1.0, 2.0**0.5],
                [2.0**0.5, 1.0, 0.0, 1.0],
                [1.0, 2.0**0.5, 1.0, 0.0],
            ],
        )
        exact = AssignmentBranchAndBoundTSPSolver().solve(problem)
        reference = HeldKarpSolver().solve(problem)

        self.assertTrue(problem.verify(exact.tour))
        self.assertAlmostEqual(exact.cost, reference.cost)
        self.assertAlmostEqual(exact.cost, 4.0)

    def test_matches_brute_force_on_small_symmetric_problem(self) -> None:
        rng = np.random.default_rng(22)
        raw = rng.uniform(0.1, 5.0, size=(6, 6))
        matrix = (raw + raw.T) / 2.0
        np.fill_diagonal(matrix, 0.0)
        problem = TravellingSalespersonProblem("random", matrix)

        exact = AssignmentBranchAndBoundTSPSolver().solve(problem)
        brute_force = min(
            problem.objective_value((0, *permutation))
            for permutation in permutations(range(1, problem.city_count))
        )
        self.assertAlmostEqual(exact.cost, brute_force)


if __name__ == "__main__":
    unittest.main()
