"""Exact DFS branch-and-bound for DAG scheduling on identical machines.

At every node the search picks any task whose predecessors are all
scheduled and places it on a machine with a distinct free time (identical
machines are interchangeable) or on the single next unused machine
(symmetry). This enumerates every topological task order, so no schedule
is excluded. Pruning uses the incumbent, the task's predecessor-finish
time, and the average-completion volume bound: every completion has
makespan at least the average of the machine completion times.
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
from algofinder.solvers.parallel_scheduling.common import list_schedule


@register_solver
class PSExactSolver(Solver):
    id = "ps-exact"
    display = "Exact DFS branch-and-bound"
    tags = frozenset({"exact"})
    applies_to = frozenset(
        {"parallel-scheduling:general", "parallel-scheduling:forest"}
    )

    def __init__(self, max_tasks: int = 10, deadline_seconds: float = 20.0) -> None:
        self.max_tasks = max_tasks
        self.deadline_seconds = deadline_seconds

    def solve(
        self,
        state: ParallelSchedulingState,
        *,
        budget_seconds: float | None = None,
    ) -> SolverResult:
        started = perf_counter()
        if state.task_count > self.max_tasks:
            raise InapplicableError(
                f"exact DFS capped at {self.max_tasks} tasks "
                f"(instance has {state.task_count})"
            )
        upper = list_schedule(
            state,
            lambda task: (-float(state.times[task]), task),
        )
        best = state.objective_value(upper)
        machine_free = np.zeros(state.machines, dtype=float)
        assignment = [-1] * state.task_count
        start = [0.0] * state.task_count
        best_solution: list[list[float]] = []
        used_machines = 0
        completed = True
        node_count = 0
        ready = [task for task in state.tasks if not state.predecessors[task]]
        unscheduled_preds = [
            len(state.predecessors[task]) for task in state.tasks
        ]
        remaining_work = float(state.times.sum())
        deadline = perf_counter() + min(
            self.deadline_seconds, budget_seconds or self.deadline_seconds
        )

        def search(makespan: float) -> None:
            nonlocal best, used_machines, completed, node_count, remaining_work
            if not ready:
                if makespan < best:
                    best = makespan
                    best_solution[:] = [
                        [float(assignment[task]), start[task]]
                        for task in range(state.task_count)
                    ]
                return
            if perf_counter() > deadline:
                completed = False
                return
            avg_bound = (float(machine_free.sum()) + remaining_work) / state.machines
            if max(makespan, avg_bound) >= best - 1e-9:
                return
            node_count += 1
            for position in range(len(ready)):
                task = ready.pop(position)
                remaining_work -= float(state.times[task])
                pred_finish = 0.0
                for predecessor in state.predecessors[task]:
                    pred_finish = max(
                        pred_finish,
                        start[predecessor] + float(state.times[predecessor]),
                    )
                for successor in state.successors[task]:
                    unscheduled_preds[successor] -= 1
                    if unscheduled_preds[successor] == 0:
                        ready.append(successor)
                tried_free: set[float] = set()
                for machine in range(min(used_machines + 1, state.machines)):
                    free = float(machine_free[machine])
                    if free in tried_free:
                        continue
                    tried_free.add(free)
                    s = max(pred_finish, free)
                    if s >= best - 1e-9:
                        continue
                    finish = s + float(state.times[task])
                    old_free = free
                    old_used = used_machines
                    machine_free[machine] = finish
                    assignment[task] = machine
                    start[task] = s
                    used_machines = max(used_machines, machine + 1)
                    search(max(makespan, finish))
                    machine_free[machine] = old_free
                    used_machines = old_used
                for successor in state.successors[task]:
                    if unscheduled_preds[successor] == 0:
                        ready.pop()
                    unscheduled_preds[successor] += 1
                remaining_work += float(state.times[task])
                ready.insert(position, task)

        search(0.0)
        if best_solution:
            best = state.objective_value(best_solution)
        else:
            best_solution = upper
            best = state.objective_value(best_solution)
        return SolverResult(
            solution=best_solution,
            cost=best,
            exact=bool(completed),
            wall_seconds=perf_counter() - started,
            metadata={
                "algorithm": "dfs-branch-and-bound",
                "tasks": state.task_count,
                "machines": state.machines,
                "nodes": node_count,
                "exhausted": None if completed else "deadline",
            },
        )


__all__ = ["PSExactSolver"]
