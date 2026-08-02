"""Solver registry: imports every solver module so registration decorators run."""

from stem4humanity.solvers import bin_packing  # noqa: F401
from stem4humanity.solvers import knapsack  # noqa: F401
from stem4humanity.solvers import parallel_scheduling  # noqa: F401
from stem4humanity.solvers import scheduling  # noqa: F401
from stem4humanity.solvers import shortest_path  # noqa: F401
from stem4humanity.solvers import tsp  # noqa: F401
from stem4humanity.solvers import unit_commitment  # noqa: F401

__all__ = [
    "bin_packing",
    "knapsack",
    "parallel_scheduling",
    "scheduling",
    "shortest_path",
    "tsp",
    "unit_commitment",
]
