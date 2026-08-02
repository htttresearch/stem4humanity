"""Exact DFS branch-and-bound solver for vector bin packing.

Items are placed in decreasing total-size order. Every placement into an
existing bin and into a new bin is enumerated; bins with identical load
vectors are explored once (symmetry), and the search is pruned with the
per-dimension volume lower bound.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from stem4humanity.problems.bin_packing import BinPackingState
from stem4humanity.solvers.base import (
    InapplicableError,
    Solver,
    SolverResult,
    register_solver,
)
from stem4humanity.trace.contract import trace_contract


def first_fit_loads(
    state: BinPackingState, order: list[int], context: object | None = None
) -> list[list[int]]:
    """First-fit packing for an item order (shared with FFD and the ML solver)."""
    from stem4humanity.trace.recorder import NullTraceRecorder

    trace = context.trace if context is not None else NullTraceRecorder()
    caps = state.capacities
    loads: list[np.ndarray] = []
    contents: list[list[int]] = []
    initial_ref = None
    if trace.enabled:
        initial_ref = trace.checkpoint({"order": order, "bins": []})
    for step, item in enumerate(order):
        size = state.sizes[item]
        for bin_index, load in enumerate(loads):
            if np.all(load + size <= caps + 1e-9):
                loads[bin_index] = load + size
                contents[bin_index].append(item)
                if trace.enabled:
                    trace.transition(
                        "bin-packing.ffd-place/v1",
                        before=initial_ref,
                        action={"step": step, "item": item},
                        outcome={"bin": bin_index, "opened": False},
                    )
                break
        else:
            loads.append(size.copy())
            contents.append([item])
            if trace.enabled:
                trace.transition(
                    "bin-packing.ffd-place/v1",
                    before=initial_ref,
                    action={"step": step, "item": item},
                    outcome={"bin": len(loads) - 1, "opened": True},
                )
    return contents


@register_solver
@trace_contract(
    version=1,
    state_schema="bin-packing.dfs-state/v1",
    atomic_steps={
        "bin-packing.dfs-node/v1": "enter one search-tree node (depth, bins, bound)",
        "bin-packing.prune/v1": "discard a node by volume bound",
        "bin-packing.incumbent/v1": "best packing improved",
    },
    derived_state={"best_packing": "contents of the best incumbent event"},
    coverage_scope="algorithm",
)
class BPDfsExactSolver(Solver):
    id = "bp-exact-dfs"
    display = "Exact DFS branch-and-bound"
    tags = frozenset({"exact"})
    applies_to = frozenset({"bin-packing:vector", "bin-packing:large-items"})

    def __init__(self, max_items: int = 16) -> None:
        self.max_items = max_items

    def solve(
        self,
        state: BinPackingState,
        *,
        budget_seconds: float | None = None,
        context: object | None = None,
    ) -> SolverResult:
        from stem4humanity.trace.recorder import NullTraceRecorder

        started = perf_counter()
        trace = context.trace if context is not None else NullTraceRecorder()
        if state.item_count > self.max_items:
            raise InapplicableError(
                f"exact DFS capped at {self.max_items} items "
                f"(instance has {state.item_count})"
            )
        order = sorted(
            range(state.item_count),
            key=lambda item: (-float(state.sizes[item].sum()), item),
        )
        caps = state.capacities
        total_volume = state.sizes.sum(axis=0)
        best = len(first_fit_loads(state, order))
        best_packing: list[list[int]] = []

        def volume_bound(loads: list[np.ndarray], start: int) -> int:
            """Per-dimension volume lower bound: every item must fit.

            The bound is ceil(total_volume[d] / caps[d]) for each
            dimension; existing bins' unused capacity is available to
            future items, so it must not be added on top.
            """
            bound = len(loads)
            for dim in range(state.dims):
                bound = max(
                    bound,
                    int(np.ceil((total_volume[dim] - 1e-9) / caps[dim])),
                )
            return bound

        initial_ref = None
        if trace.enabled:
            initial_ref = trace.checkpoint(
                {"order": order, "bins": [], "best": best}
            )
        node_count = 0

        def search(
            loads: list[np.ndarray],
            contents: list[list[int]],
            index: int,
        ) -> None:
            nonlocal best, node_count
            node_count += 1
            if trace.enabled:
                if trace.full:
                    trace.event(
                        "transition",
                        "bin-packing.dfs-node/v1",
                        before=initial_ref,
                        action={
                            "depth": index,
                            "item": order[index] if index < len(order) else None,
                        },
                        metrics={
                            "bins": len(loads),
                            "bound": (
                                volume_bound(loads, index)
                                if index < len(order)
                                else len(loads)
                            ),
                            "best": best,
                            "nodes": node_count,
                        },
                    )
            if index == len(order):
                if len(loads) < best:
                    best = len(loads)
                    best_packing[:] = [list(bin_items) for bin_items in contents]
                    if trace.enabled:
                        trace.event(
                            "incumbent",
                            "bin-packing.incumbent/v1",
                            outcome={"cost": len(loads), "nodes": node_count},
                        )
                return
            if volume_bound(loads, index) >= best:
                if trace.enabled and trace.decisions:
                    trace.event(
                        "prune",
                        "bin-packing.prune/v1",
                        before=initial_ref,
                        outcome={
                            "reason": "volume-bound",
                            "depth": index,
                            "bound": volume_bound(loads, index),
                            "best": best,
                        },
                    )
                return
            item = order[index]
            size = state.sizes[item]
            tried: set[tuple[float, ...]] = set()
            for bin_index, load in enumerate(loads):
                key = tuple(round(float(value), 9) for value in load)
                if key in tried:
                    continue
                tried.add(key)
                if np.all(load + size <= caps + 1e-9):
                    old_load = load.copy()
                    old_contents = list(contents[bin_index])
                    loads[bin_index] = load + size
                    contents[bin_index] = old_contents + [item]
                    search(loads, contents, index + 1)
                    loads[bin_index] = old_load
                    contents[bin_index] = old_contents
            loads.append(size.copy())
            contents.append([item])
            search(loads, contents, index + 1)
            loads.pop()
            contents.pop()

        search([], [], 0)
        if not best_packing:
            best_packing = first_fit_loads(state, order)
        if trace.enabled:
            trace.event(
                "lifecycle",
                "solve.result",
                outcome={
                    "cost": float(best),
                    "nodes": node_count,
                    "initial_ffd_bound": len(first_fit_loads(state, order)),
                    "wall_seconds": perf_counter() - started,
                },
            )
        return SolverResult(
            solution=best_packing,
            cost=float(best),
            exact=True,
            wall_seconds=perf_counter() - started,
            metadata={
                "algorithm": "dfs-branch-and-bound",
                "items": state.item_count,
                "initial_ffd_bound": best,
            },
        )


__all__ = ["BPDfsExactSolver", "first_fit_loads"]
