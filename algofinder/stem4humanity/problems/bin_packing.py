"""Bin-packing problem family: multi-dimensional (vector) packing.

The general class packs items with several resource dimensions into bins
of unit capacity per dimension (data-center VM/container placement).
The well-studied subclass is one-dimensional packing of items larger
than a third of the bin, where every bin holds at most two items and the
problem reduces to a maximum matching. Solutions are lists of bins,
each bin a list of item indices; the objective is the bin count.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


class BinPackingState:
    """A multi-dimensional bin-packing instance (minimize bin count)."""

    def __init__(
        self,
        name: str,
        items: Any,
        capacities: Any | None = None,
    ) -> None:
        item_matrix = np.asarray(items, dtype=float)
        if item_matrix.ndim != 2:
            raise ValueError("items must have shape (item_count, dims)")
        if not np.isfinite(item_matrix).all():
            raise ValueError("item sizes must be finite")
        if np.any(item_matrix <= 0):
            raise ValueError("item sizes must be positive")
        if capacities is None:
            capacities = np.ones(item_matrix.shape[1])
        capacity_vector = np.asarray(capacities, dtype=float)
        if (
            capacity_vector.ndim != 1
            or capacity_vector.size != item_matrix.shape[1]
        ):
            raise ValueError("capacities must match the item dimensions")
        if np.any(capacity_vector <= 0):
            raise ValueError("capacities must be positive")
        if np.any(item_matrix > capacity_vector + 1e-12):
            raise ValueError("every item must fit in a bin by itself")

        self.name = name
        self._items: NDArray[np.float64] = item_matrix.copy()
        self._items.setflags(write=False)
        self._capacities: NDArray[np.float64] = capacity_vector.copy()
        self._capacities.setflags(write=False)
        self.items = tuple(range(item_matrix.shape[0]))

    @property
    def sizes(self) -> NDArray[np.float64]:
        """Read-only ``(item_count, dims)`` size matrix."""
        return self._items

    @property
    def capacities(self) -> NDArray[np.float64]:
        """Read-only per-dimension bin capacities."""
        return self._capacities

    @property
    def item_count(self) -> int:
        return self._items.shape[0]

    @property
    def dims(self) -> int:
        return self._items.shape[1]

    def _bin_fits(self, bin_items: Sequence[int]) -> bool:
        if not bin_items:
            return True
        total = self._items[list(bin_items)].sum(axis=0)
        return bool(np.all(total <= self._capacities + 1e-9))

    def verify(self, solution: Any) -> bool:
        if not isinstance(solution, list):
            return False
        seen: set[int] = set()
        for bin_items in solution:
            try:
                indices = [int(item) for item in bin_items]
            except (TypeError, ValueError):
                return False
            if any(item < 0 or item >= self.item_count for item in indices):
                return False
            if len(set(indices)) != len(indices):
                return False
            if any(item in seen for item in indices):
                return False
            seen.update(indices)
            if not self._bin_fits(indices):
                return False
        return len(seen) == self.item_count

    def objective_value(self, solution: Any) -> float:
        if not self.verify(solution):
            raise ValueError("solution must partition every item into bins")
        return float(len(solution))


__all__ = ["BinPackingState"]
