from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.euclidean_tsp import generate_euclidean_dataset, generate_euclidean_instance
from src.ml.candidate_ranker import (
    CandidateEdgeRanker,
    build_candidate_training_data,
)
from src.solvers.held_karp import HeldKarpSolver
from src.solvers.learned_candidate_tsp import LearnedCandidateTwoOptSolver


class LearnedCandidateTSPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        records = generate_euclidean_dataset(9, 8, seed=31)
        training_data = build_candidate_training_data(
            [record.problem for record in records],
            exact_solver=HeldKarpSolver(),
        )
        cls.ranker = CandidateEdgeRanker(n_estimators=30, random_state=31).fit(
            training_data
        )

    def test_ranker_round_trip(self) -> None:
        problem = generate_euclidean_instance(10, "uniform", seed=32).problem
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "ranker.joblib"
            self.ranker.save(model_path)
            restored = CandidateEdgeRanker.load(model_path)
        self.assertTrue(restored.is_fitted)
        self.assertEqual(
            self.ranker.metadata["feature_names"],
            restored.metadata["feature_names"],
        )

    def test_learned_solver_returns_feasible_tour(self) -> None:
        problem = generate_euclidean_instance(11, "clustered", seed=33).problem
        result = LearnedCandidateTwoOptSolver(
            self.ranker,
            top_k=5,
            max_restarts=2,
            seed=33,
        ).solve(problem)

        self.assertTrue(problem.verify(result.tour))
        self.assertAlmostEqual(result.cost, problem.objective_value(result.tour))
        self.assertGreater(result.metadata["candidate_edges_after_ranking"], 0)
        self.assertGreaterEqual(result.metadata["model_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
