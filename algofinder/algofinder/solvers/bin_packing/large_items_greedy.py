"""Greedy pairwise packing for one-dimensional large items (> 1/3 bin).

Every bin holds at most two such items, so the packing reduces to a
pairing problem; this heuristic pairs each unplaced largest item with
the largest remaining item it still fits, aiming for near-maximum
matchings without the blossom machinery (see ``large_items_exact`` for
the optimal version).
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from algofinder.problems.bin_packing import BinPackingState
from algofinder.solvers.base import InapplicableError, Solver, SolverResult, register_solver


@register_solver
class BPLargeItemsGreedySolver(Solver):
    id = "bp-large-items-greedy"
    display = "Greedy pairing (large items)"
    tags = frozenset({"heuristic"})
    applies_to = frozenset({"bin-packing:large-items"})

    def solve(
        self,
        state: BinPackingState,
        *,
        budget_seconds: float | None = None,
    ) -> SolverResult:
        started = perf_counter()
        if state.dims != 1 or not np.all(state.sizes[:, 0] > 1 / 3 - 1e-9):
            raise InapplicableError("requires one-dimensional items larger than 1/3")
        order = sorted(range(state.item_count), key=lambda i: -float(state.sizes[i, 0]))
        capacity = float(state.capacities[0])
        unplaced = set(order)
        packing: list[list[int]] = []
        for item in order:
            if item not in unplaced:
                continue
            unplaced.remove(item)
            partner = -1
            for candidate in order:
                if (
                    candidate in unplaced
                    and state.sizes[item, 0] + state.sizes[candidate, 0]
                    <= capacity + 1e-9
                ):
                    partner = candidate
                    break
            if partner >= 0:
                unplaced.remove(partner)
                packing.append([item, partner])
            else:
                packing.append([item])
        return SolverResult(
            solution=packing,
            cost=float(len(packing)),
            exact=False,
            wall_seconds=perf_counter() - started,
            metadata={
                "algorithm": "greedy-pair-largest",
                "pairs": len(packing) - sum(1 for bin_items in packing if len(bin_items) == 1),
            },
        )


__all__ = ["BPLargeItemsGreedySolver"]
