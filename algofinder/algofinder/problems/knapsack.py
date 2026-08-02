"""Knapsack problem family: 0/1, multidimensional, and subset-sum."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


class KnapsackState:
    """A multi-dimensional knapsack instance (maximization).

    ``weights`` has shape ``(n, dims)``; ``capacities`` has length ``dims``.
    A solution is a tuple of item indices; every item is used at most once.
    """

    def __init__(
        self,
        name: str,
        weights: Any,
        values: Any,
        capacities: Any,
    ) -> None:
        weight_matrix = np.asarray(weights, dtype=float)
        value_vector = np.asarray(values, dtype=float)
        capacity_vector = np.asarray(capacities, dtype=float)
        if weight_matrix.ndim != 2 or weight_matrix.shape[0] != value_vector.size:
            raise ValueError("weights must have shape (item_count, dims)")
        if value_vector.size == 0:
            raise ValueError("at least one item is required")
        if capacity_vector.ndim != 1 or capacity_vector.size != weight_matrix.shape[1]:
            raise ValueError("capacities must match the weight dimensions")
        if not np.isfinite(weight_matrix).all() or not np.isfinite(value_vector).all():
            raise ValueError("weights and values must be finite")
        if np.any(weight_matrix < 0) or np.any(value_vector < 0):
            raise ValueError("weights and values cannot be negative")

        self.name = name
        self._weights: NDArray[np.float64] = weight_matrix.copy()
        self._weights.setflags(write=False)
        self._values: NDArray[np.float64] = value_vector.copy()
        self._values.setflags(write=False)
        self._capacities: NDArray[np.float64] = capacity_vector.copy()
        self._capacities.setflags(write=False)
        self.items = tuple(range(weight_matrix.shape[0]))

    @property
    def weights(self) -> NDArray[np.float64]:
        return self._weights

    @property
    def values(self) -> NDArray[np.float64]:
        return self._values

    @property
    def capacities(self) -> NDArray[np.float64]:
        return self._capacities

    @property
    def item_count(self) -> int:
        return len(self.items)

    @property
    def dims(self) -> int:
        return self._weights.shape[1]

    def verify(self, solution: Sequence[int]) -> bool:
        try:
            selected = tuple(int(item) for item in solution)
        except (TypeError, ValueError):
            return False
        if len(set(selected)) != len(selected):
            return False
        if any(item < 0 or item >= self.item_count for item in selected):
            return False
        total = self._weights[list(selected)].sum(axis=0) if selected else np.zeros(self.dims)
        return bool(np.all(total <= self._capacities + 1e-9))

    def objective_value(self, solution: Sequence[int]) -> float:
        if not self.verify(solution):
            raise ValueError("solution must be a feasible item selection")
        selected = tuple(int(item) for item in solution)
        return float(self._values[list(selected)].sum()) if selected else 0.0
