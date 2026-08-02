"""Hu's algorithm: exact scheduling of unit-time out-forests.

For a forest of unit-time out-trees on identical machines, Hu's level
algorithm is optimal: every task gets a level equal to the length of the
longest chain of successors below it, and at each time step the up-to-``m``
available tasks with the highest levels are scheduled.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from algofinder.problems.parallel_scheduling import ParallelSchedulingState
from algofinder.solvers.base import (
    InapplicableError,
    Solver,
    SolverResult,
    register_solver,
)
from algofinder.solvers.parallel_scheduling.common import topological_order


def _levels(state: ParallelSchedulingState) -> np.ndarray:
    order = topological_order(state)
    level = np.zeros(state.task_count, dtype=float)
    for task in reversed(order):
        best = 0.0
        for successor in state.successors[task]:
            best = max(best, level[successor])
        level[task] = 1.0 + best
    return level


@register_solver
class PSHuSolver(Solver):
    id = "ps-hu"
    display = "Hu's level algorithm (unit-time out-forests)"
    tags = frozenset({"exact"})
    applies_to = frozenset({"parallel-scheduling:forest"})

    def solve(
        self,
        state: ParallelSchedulingState,
        *,
        budget_seconds: float | None = None,
    ) -> SolverResult:
        started = perf_counter()
        if not np.all(np.abs(state.times - 1.0) < 1e-9):
            raise InapplicableError("Hu's algorithm requires unit processing times")
        if any(len(preds) > 1 for preds in state.predecessors):
            raise InapplicableError("Hu's algorithm requires an out-forest")
        level = _levels(state)
        remaining = [len(preds) for preds in state.predecessors]
        start = [0.0] * state.task_count
        assignment = [-1] * state.task_count
        scheduled = 0
        time = 0
        machine_free = [0.0] * state.machines
        while scheduled < state.task_count:
            available = sorted(
                (
                    task
                    for task in range(state.task_count)
                    if remaining[task] == 0 and assignment[task] == -1
                ),
                key=lambda task: (-float(level[task]), task),
            )
            for task in available[: state.machines]:
                machine = min(
                    range(state.machines), key=lambda m: machine_free[m]
                )
                assignment[task] = machine
                start[task] = float(time)
                machine_free[machine] = float(time) + 1.0
                scheduled += 1
                for successor in state.successors[task]:
                    remaining[successor] -= 1
            time += 1
        solution = [
            [float(assignment[task]), start[task]] for task in range(state.task_count)
        ]
        return SolverResult(
            solution=solution,
            cost=state.objective_value(solution),
            exact=True,
            wall_seconds=perf_counter() - started,
            metadata={
                "algorithm": "hu-level-algorithm",
                "makespan": float(time),
            },
        )


__all__ = ["PSHuSolver"]
