"""Exact job-shop solver: enumerate per-machine orderings with makespan.

For a fixed set of per-machine processing orders, the earliest-start
simulation yields the optimal schedule for those orders, and every
feasible schedule induces some per-machine ordering; enumerating all
orderings is therefore exact. The search space is factorial in the
number of operations per machine, so large instances are declined.
"""

from __future__ import annotations

import itertools
from time import perf_counter

from algofinder.problems.scheduling import JobShopState, earliest_start_makespan
from algofinder.solvers.base import (
    InapplicableError,
    Solver,
    SolverResult,
    register_solver,
)

MAX_ORDERINGS = 250_000


def simulate_makespan(state: JobShopState, orders: dict[int, list[tuple[int, int]]]) -> float:
    """Earliest-start makespan for the given per-machine processing orders."""
    return earliest_start_makespan(state.operations, orders)


@register_solver
class JobShopBnBExactSolver(Solver):
    id = "jobshop-bnb-exact"
    display = "Exact enumeration of per-machine orderings (small instances)"
    tags = frozenset({"exact"})
    applies_to = frozenset({"scheduling:job-shop"})

    def solve(
        self,
        state: JobShopState,
        *,
        budget_seconds: float | None = None,
    ) -> SolverResult:
        started = perf_counter()
        per_machine_ops: dict[int, list[tuple[int, int]]] = {}
        for machine in state.machines:
            per_machine_ops[machine] = [
                (job, step)
                for job in state.jobs
                for step, (op_machine, _) in enumerate(state.operations[job])
                if op_machine == machine
            ]
        ordering_count = 1
        for machine in state.machines:
            ordering_count *= _factorial(len(per_machine_ops[machine]))
        if ordering_count > MAX_ORDERINGS:
            raise InapplicableError(
                f"exact enumeration would need {ordering_count} orderings "
                f"(cap {MAX_ORDERINGS})"
            )

        best_cost = float("inf")
        best_orders: dict[int, list[tuple[int, int]]] | None = None
        evaluated = 0
        permutations = [
            itertools.permutations(per_machine_ops[machine])
            for machine in state.machines
        ]
        for combination in itertools.product(*permutations):
            orders = {
                machine: list(order)
                for machine, order in zip(state.machines, combination)
            }
            cost = simulate_makespan(state, orders)
            evaluated += 1
            if cost < best_cost:
                best_cost = cost
                best_orders = orders
        return SolverResult(
            solution=best_orders,
            cost=best_cost,
            exact=True,
            wall_seconds=perf_counter() - started,
            metadata={
                "algorithm": "jobshop-exact-enumeration",
                "orderings_evaluated": evaluated,
            },
        )


def _factorial(value: int) -> int:
    result = 1
    for factor in range(2, value + 1):
        result *= factor
    return result


__all__ = ["JobShopBnBExactSolver", "simulate_makespan"]
