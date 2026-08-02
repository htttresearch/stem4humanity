"""Symmetric TSP problem declarations.

The generic class accepts any complete symmetric distance matrix.  The
Euclidean subclass derives that matrix from two-dimensional city coordinates.
Tours are represented as permutations of city indices; the return edge to the
first city is implicit.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import ProblemState


class TravellingSalespersonProblem:
    """A complete, symmetric travelling-salesperson problem."""

    def __init__(self, name: str, distances: ArrayLike) -> None:
        matrix = np.asarray(distances, dtype=float)
        self._validate_distance_matrix(matrix)
        self.name = name
        self._distances: NDArray[np.float64] = matrix.copy()
        self._distances.setflags(write=False)
        self.nodes = tuple(range(matrix.shape[0]))

    @staticmethod
    def _validate_distance_matrix(matrix: NDArray[np.float64]) -> None:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("distances must be a square matrix")
        if matrix.shape[0] < 3:
            raise ValueError("a TSP instance needs at least three cities")
        if not np.isfinite(matrix).all():
            raise ValueError("distances must be finite")
        if np.any(matrix < 0):
            raise ValueError("distances cannot be negative")
        if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
            raise ValueError("this problem class requires symmetric distances")
        if not np.allclose(np.diag(matrix), 0.0, atol=1e-12):
            raise ValueError("distance-matrix diagonal must be zero")

    @property
    def distances(self) -> NDArray[np.float64]:
        """Read-only complete distance matrix."""
        return self._distances

    @property
    def city_count(self) -> int:
        return len(self.nodes)

    def verify(self, solution: Sequence[int]) -> bool:
        if len(solution) != self.city_count:
            return False
        try:
            tour = tuple(int(city) for city in solution)
        except (TypeError, ValueError):
            return False
        return set(tour) == set(self.nodes)

    def objective_value(self, solution: Sequence[int]) -> float:
        if not self.verify(solution):
            raise ValueError("solution must be a permutation of every city")
        tour = np.asarray(solution, dtype=int)
        return float(self.distances[tour, np.roll(tour, -1)].sum())


class EuclideanTravellingSalespersonProblem(TravellingSalespersonProblem):
    """A two-dimensional Euclidean complete-graph TSP instance."""

    def __init__(self, name: str, points: ArrayLike) -> None:
        coordinates = np.asarray(points, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("points must have shape (city_count, 2)")
        if coordinates.shape[0] < 3:
            raise ValueError("a TSP instance needs at least three cities")
        if not np.isfinite(coordinates).all():
            raise ValueError("point coordinates must be finite")

        immutable_points: NDArray[np.float64] = coordinates.copy()
        immutable_points.setflags(write=False)
        distances = np.linalg.norm(
            immutable_points[:, np.newaxis, :] - immutable_points[np.newaxis, :, :],
            axis=2,
        )
        super().__init__(name=name, distances=distances)
        self._points = immutable_points

    @property
    def points(self) -> NDArray[np.float64]:
        """Read-only `(city_count, 2)` coordinate array."""
        return self._points


EuclideanTSPProblem = EuclideanTravellingSalespersonProblem

__all__ = [
    "EuclideanTSPProblem",
    "EuclideanTravellingSalespersonProblem",
    "TravellingSalespersonProblem",
]
