"""Shared list-scheduling machinery for DAG-on-identical-machines problems.

A list schedule repeatedly picks the available task (predecessors
finished) with the best priority and assigns it to the earliest-free
machine at the earliest feasible start time. Priority is expressed as a
``priority_key`` callable ``(task) -> tuple`` compared with ``min``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from algofinder.problems.parallel_scheduling import ParallelSchedulingState


def critical_paths(state: ParallelSchedulingState) -> np.ndarray:
    """Longest path from each task to a sink (its critical-path tail weight)."""
    n = state.task_count
    order: list[int] = []
    remaining = [len(preds) for preds in state.predecessors]
    queue = [task for task in range(n) if remaining[task] == 0]
    while queue:
        task = queue.pop()
        order.append(task)
        for successor in state.successors[task]:
            remaining[successor] -= 1
            if remaining[successor] == 0:
                queue.append(successor)
    longest = np.zeros(n, dtype=float)
    for task in reversed(order):
        best_successor = 0.0
        for successor in state.successors[task]:
            best_successor = max(best_successor, longest[successor])
        longest[task] = state.times[task] + best_successor
    return longest


def topological_order(state: ParallelSchedulingState) -> list[int]:
    """Any topological order of the task DAG (Kahn's algorithm)."""
    n = state.task_count
    order: list[int] = []
    remaining = [len(preds) for preds in state.predecessors]
    queue = sorted(task for task in range(n) if remaining[task] == 0)
    while queue:
        task = queue.pop(0)
        order.append(task)
        for successor in state.successors[task]:
            remaining[successor] -= 1
            if remaining[successor] == 0:
                queue.append(successor)
    return order


def list_schedule(
    state: ParallelSchedulingState,
    priority_key: Callable[[int], tuple[float, ...]],
    context: object | None = None,
) -> list[list[float]]:
    """Build a ``[[machine, start], ...]`` schedule with the given priority."""
    from algofinder.trace.recorder import NullTraceRecorder

    trace = context.trace if context is not None else NullTraceRecorder()
    n = state.task_count
    machines = state.machines
    remaining = [len(preds) for preds in state.predecessors]
    machine_assignment = [-1] * n
    start = [0.0] * n
    machine_free = [0.0] * machines
    available: list[int] = sorted(
        (task for task in range(n) if remaining[task] == 0), key=priority_key
    )
    initial_ref = None
    if trace.enabled:
        initial_ref = trace.checkpoint(
            {
                "available": available,
                "machine_free": machine_free,
                "remaining": remaining,
            }
        )
    step = 0
    while available:
        task = available.pop(0)
        pred_finish = 0.0
        for predecessor in state.predecessors[task]:
            pred_finish = max(pred_finish, start[predecessor] + state.times[predecessor])
        machine = min(range(machines), key=lambda m: machine_free[m])
        s = max(pred_finish, machine_free[machine])
        machine_assignment[task] = machine
        start[task] = s
        machine_free[machine] = s + state.times[task]
        if trace.enabled:
            trace.transition(
                "parallel-scheduling.list-step/v1",
                before=initial_ref,
                action={"step": step, "task": task, "machine": machine},
                outcome={"start": s},
                metrics={"available_count": len(available)},
            )
        step += 1
        for successor in state.successors[task]:
            remaining[successor] -= 1
            if remaining[successor] == 0:
                available.append(successor)
        available.sort(key=priority_key)
    return [
        [float(machine_assignment[task]), start[task]] for task in range(n)
    ]


__all__ = ["critical_paths", "list_schedule", "topological_order"]
