"""Exact solver for one-dimensional large items (> capacity/3) via matching.

When every item exceeds a third of the bin capacity, no bin can hold
more than two items, so the bin count is ``n - maximum_matching`` where
an edge connects two items that fit together; a maximum cardinality
matching is found with Edmonds' blossom algorithm (networkx). Each
matched pair becomes a bin and every unmatched item a solo bin, which is
provably optimal for this item class. Items equal to capacity/3 are
declined (three of them share one bin, breaking the pairing argument).
"""

from __future__ import annotations

from time import perf_counter

import networkx as nx
import numpy as np

from algofinder.problems.bin_packing import BinPackingState
from algofinder.solvers.base import (
    InapplicableError,
    Solver,
    SolverResult,
    register_solver,
)


@register_solver
class BPLargeItemsExactSolver(Solver):
    id = "bp-large-items-exact"
    display = "Large-items exact (Edmonds blossom)"
    tags = frozenset({"exact"})
    applies_to = frozenset({"bin-packing:large-items"})

    def __init__(self, max_items: int = 80) -> None:
        self.max_items = max_items

    def solve(
        self,
        state: BinPackingState,
        *,
        budget_seconds: float | None = None,
    ) -> SolverResult:
        started = perf_counter()
        if state.dims != 1:
            raise InapplicableError("requires one-dimensional items")
        capacity = float(state.capacities[0])
        if not np.all(state.sizes[:, 0] > capacity / 3 + 1e-9):
            raise InapplicableError(
                "requires items strictly larger than a third of the capacity "
                "(items equal to capacity/3 can share a bin three at a time)"
            )
        if state.item_count > self.max_items:
            raise InapplicableError(
                f"blossom matching capped at {self.max_items} items "
                f"(instance has {state.item_count})"
            )
        graph = nx.Graph()
        graph.add_nodes_from(range(state.item_count))
        for i in range(state.item_count):
            for j in range(i + 1, state.item_count):
                if state.sizes[i, 0] + state.sizes[j, 0] <= capacity + 1e-9:
                    graph.add_edge(i, j)
        matching = nx.max_weight_matching(graph)
        paired: set[int] = set()
        packing: list[list[int]] = []
        for i, j in matching:
            paired.add(int(i))
            paired.add(int(j))
            packing.append([int(i), int(j)])
        for item in range(state.item_count):
            if item not in paired:
                packing.append([item])
        return SolverResult(
            solution=packing,
            cost=float(len(packing)),
            exact=True,
            wall_seconds=perf_counter() - started,
            metadata={
                "algorithm": "edmonds-blossom-matching",
                "items": state.item_count,
                "pairs": len(matching),
            },
        )


__all__ = ["BPLargeItemsExactSolver"]
