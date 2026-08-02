from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.euclidean_tsp import (
    _augment_rigidly,
    generate_euclidean_dataset,
    generate_euclidean_instance,
    load_euclidean_dataset,
    save_euclidean_dataset,
)
from src.problems.travelling_salesperson_problem import (
    EuclideanTravellingSalespersonProblem,
)


class EuclideanTSPProblemTests(unittest.TestCase):
    def test_square_tour_has_expected_length(self) -> None:
        problem = EuclideanTravellingSalespersonProblem(
            "square",
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        )

        self.assertTrue(problem.verify((0, 1, 2, 3)))
        self.assertFalse(problem.verify((0, 1, 2)))
        self.assertFalse(problem.verify((0, 1, 1, 3)))
        self.assertAlmostEqual(problem.objective_value((0, 1, 2, 3)), 4.0)
        with self.assertRaises(ValueError):
            problem.objective_value((0, 1, 1, 3))

    def test_generation_is_reproducible_and_persisted(self) -> None:
        first = generate_euclidean_instance(10, "clustered", seed=9)
        second = generate_euclidean_instance(10, "clustered", seed=9)
        np.testing.assert_allclose(first.problem.points, second.problem.points)

        dataset = generate_euclidean_dataset(4, (8, 10), seed=13)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "instances.json"
            save_euclidean_dataset(dataset, path)
            restored = load_euclidean_dataset(path)
        self.assertEqual([record.family for record in dataset], [record.family for record in restored])
        for original, loaded in zip(dataset, restored):
            np.testing.assert_allclose(original.problem.points, loaded.problem.points)

    def test_augmentation_preserves_all_distance_ratios(self) -> None:
        points = np.array([[0.0, 0.0], [2.0, 0.0], [0.5, 1.0]])
        transformed = _augment_rigidly(points, np.random.default_rng(123))
        original_distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        transformed_distances = np.linalg.norm(
            transformed[:, None] - transformed[None, :],
            axis=2,
        )
        mask = original_distances > 0
        ratios = transformed_distances[mask] / original_distances[mask]
        np.testing.assert_allclose(ratios, ratios[0])


if __name__ == "__main__":
    unittest.main()
