"""First-fit-decreasing heuristic for vector bin packing."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from algofinder.problems.bin_packing import BinPackingState
from algofinder.solvers.base import Solver, SolverResult, register_solver
from algofinder.solvers.bin_packing.exact_dfs import first_fit_loads


@register_solver
class BPFirstFitDecreasingSolver(Solver):
    id = "bp-ffd"
    display = "First-fit decreasing"
    tags = frozenset({"heuristic"})
    applies_to = frozenset({"bin-packing:vector", "bin-packing:large-items"})

    def solve(
        self,
        state: BinPackingState,
        *,
        budget_seconds: float | None = None,
        context: object | None = None,
    ) -> SolverResult:
        started = perf_counter()
        order = sorted(
            range(state.item_count),
            key=lambda item: (-float(state.sizes[item].sum()), item),
        )
        packing = first_fit_loads(state, order, context=context)
        return SolverResult(
            solution=packing,
            cost=float(len(packing)),
            exact=False,
            wall_seconds=perf_counter() - started,
            metadata={"algorithm": "first-fit-decreasing"},
        )


__all__ = ["BPFirstFitDecreasingSolver"]
