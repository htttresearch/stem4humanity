"""Exact branch-and-bound for multi-dimensional knapsack (also covers 0/1, subset-sum)."""

from __future__ import annotations

import time

import numpy as np

from stem4humanity.problems.knapsack import KnapsackState
from stem4humanity.solvers.base import Solver, SolverResult, register_solver


@register_solver
class BranchAndBoundExact(Solver):
    id = "knapsack-bnb-exact"
    display = "Branch and bound with fractional relaxation (exact)"
    tags = frozenset({"exact"})
    applies_to = frozenset(
        {"knapsack:0-1", "knapsack:multidimensional", "knapsack:subset-sum"}
    )

    def __init__(self, max_nodes: int = 5_000_000) -> None:
        self.max_nodes = max_nodes

    def solve(
        self, state: KnapsackState, *, budget_seconds: float | None = None
    ) -> SolverResult:
        n = state.item_count
        weights = state.weights
        values = state.values
        capacities = state.capacities
        dims = state.dims

        density = values / (weights + 1e-12).max(axis=1)
        order = np.argsort(density)[::-1]
        ordered_weights = weights[order]
        ordered_values = values[order]
        dfs_pos = np.empty(n, dtype=int)
        for pos, item in enumerate(order):
            dfs_pos[int(item)] = pos
        dim_orders = []
        for dim in range(dims):
            dim_density = values / (weights[:, dim] + 1e-12)
            dim_orders.append(np.argsort(dim_density)[::-1])

        def upper_bound(remaining: np.ndarray, start: int) -> float:
            """Per-dimension fractional relaxation (valid upper bound).

            Each dimension's relaxation orders items by that dimension's
            value/weight density and skips already-decided items; the
            maximum over dimensions is an upper bound on the optimum, so
            pruning against it never discards the optimum.
            """
            bound = 0.0
            for dim in range(dims):
                slack = float(remaining[dim])
                frac = 0.0
                for pos in range(len(dim_orders[dim])):
                    item = int(dim_orders[dim][pos])
                    if dfs_pos[item] < start:
                        continue
                    w = float(weights[item, dim])
                    if w <= 1e-12:
                        frac += float(values[item])
                        continue
                    if w >= slack:
                        frac += slack / w * float(values[item])
                        break
                    frac += float(values[item])
                    slack -= w
                bound = max(bound, frac)
            return bound

        deadline = None
        if budget_seconds is not None:
            deadline = time.perf_counter() + budget_seconds

        best_value = -1.0
        best_selection: tuple[int, ...] | None = None
        nodes = 0
        exhausted = ""

        # Frames: (item_index, value, remaining_capacities, node_upper, selection)
        stack: list[tuple[int, float, np.ndarray, float, tuple[int, ...]]] = [
            (0, 0.0, capacities.copy(), upper_bound(capacities, 0), ())
        ]
        while stack:
            nodes += 1
            if nodes > self.max_nodes:
                exhausted = "node_limit"
                break
            if deadline is not None and time.perf_counter() > deadline:
                exhausted = "budget"
                break

            i, value, remaining, node_upper, selection = stack.pop()
            if node_upper <= best_value + 1e-12:
                continue
            if i >= n:
                if value > best_value:
                    best_value = value
                    best_selection = selection
                continue

            item_weight = ordered_weights[i]
            item_value = float(ordered_values[i])

            included_remaining = remaining - item_weight
            if np.all(included_remaining >= -1e-9):
                included_value = value + item_value
                stack.append(
                    (
                        i + 1,
                        included_value,
                        included_remaining,
                        included_value + upper_bound(included_remaining, i + 1),
                        selection + (int(order[i]),),
                    )
                )

            stack.append(
                (
                    i + 1,
                    value,
                    remaining,
                    value + upper_bound(remaining, i + 1),
                    selection,
                )
            )

        if best_selection is None:
            best_selection = ()
        exact = not exhausted
        cost = float(state.objective_value(best_selection))
        return SolverResult(
            solution=best_selection,
            cost=cost,
            exact=exact,
            wall_seconds=0.0,
            metadata={"nodes": nodes, "exhausted": exhausted or None},
        )
