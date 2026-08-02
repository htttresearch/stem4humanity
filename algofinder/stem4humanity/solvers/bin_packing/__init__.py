"""Bin-packing solver package: importing this module registers all solvers."""

from stem4humanity.solvers.bin_packing import (  # noqa: F401
    exact_dfs,
    first_fit,
    large_items_exact,
    large_items_greedy,
    learned_order,
)

__all__ = [
    "exact_dfs",
    "first_fit",
    "large_items_exact",
    "large_items_greedy",
    "learned_order",
]
