"""Parallel scheduling problem family: DAG tasks on identical machines.

The general class is P|prec|Cmax: a DAG of tasks with processing times
scheduled on ``m`` identical machines to minimize the makespan
(NP-hard). The well-studied subclass restricts the DAG to a forest of
unit-time out-trees, where Hu's level algorithm is optimal. Solutions
are per-task ``[machine, start]`` pairs; the objective is the makespan.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


class ParallelSchedulingState:
    """A precedence-constrained scheduling instance on identical machines."""

    def __init__(
        self,
        name: str,
        task_count: int,
        machines: int,
        times: Any,
        predecessors: Any,
        successors: Any = None,
    ) -> None:
        task_count = int(task_count)
        machines = int(machines)
        if task_count < 1 or machines < 1:
            raise ValueError("at least one task and one machine are required")
        time_vector = np.asarray(times, dtype=float)
        if time_vector.shape != (task_count,):
            raise ValueError("times must have one entry per task")
        if not np.isfinite(time_vector).all():
            raise ValueError("task times must be finite")
        if np.any(time_vector <= 0):
            raise ValueError("task times must be positive")
        normalized_predecessors: list[list[int]] = []
        for task, task_predecessors in enumerate(predecessors):
            preds = [int(item) for item in task_predecessors]
            if any(pred < 0 or pred >= task_count or pred == task for pred in preds):
                raise ValueError("predecessors must be valid task indices")
            if len(set(preds)) != len(preds):
                raise ValueError("predecessor lists must be duplicate-free")
            normalized_predecessors.append(preds)
        normalized_successors: list[list[int]] = [[] for _ in range(task_count)]
        for task, task_predecessors in enumerate(normalized_predecessors):
            for predecessor in task_predecessors:
                normalized_successors[predecessor].append(task)

        self.name = name
        self._task_count = task_count
        self._machines = machines
        self._times: NDArray[np.float64] = time_vector.copy()
        self._times.setflags(write=False)
        self._predecessors: tuple[tuple[int, ...], ...] = tuple(
            tuple(preds) for preds in normalized_predecessors
        )
        self._successors: tuple[tuple[int, ...], ...] = tuple(
            tuple(succs) for succs in normalized_successors
        )
        self.tasks = tuple(range(task_count))
        self.machine_ids = tuple(range(machines))

    @property
    def task_count(self) -> int:
        return self._task_count

    @property
    def machines(self) -> int:
        return self._machines

    @property
    def times(self) -> NDArray[np.float64]:
        """Read-only per-task processing times."""
        return self._times

    @property
    def predecessors(self) -> tuple[tuple[int, ...], ...]:
        """Per-task predecessor index tuples."""
        return self._predecessors

    @property
    def successors(self) -> tuple[tuple[int, ...], ...]:
        """Per-task successor index tuples."""
        return self._successors

    def verify(self, solution: Any) -> bool:
        if not isinstance(solution, list) or len(solution) != self._task_count:
            return False
        assignments: list[tuple[int, float]] = []
        for entry in solution:
            try:
                machine, start = int(entry[0]), float(entry[1])
            except (TypeError, ValueError, IndexError):
                return False
            if not 0 <= machine < self._machines:
                return False
            if not np.isfinite(start) or start < 0:
                return False
            assignments.append((machine, start))
        for task in self.tasks:
            start = assignments[task][1]
            for predecessor in self._predecessors[task]:
                if (
                    start
                    < assignments[predecessor][1] + self._times[predecessor] - 1e-9
                ):
                    return False
        for machine in self.machine_ids:
            scheduled = [
                (assignments[task][1], self._times[task], task)
                for task in self.tasks
                if assignments[task][0] == machine
            ]
            scheduled.sort()
            for (start_first, duration_first, _), (start_second, _, _) in zip(
                scheduled, scheduled[1:]
            ):
                if start_second < start_first + duration_first - 1e-9:
                    return False
        return True

    def objective_value(self, solution: Any) -> float:
        if not self.verify(solution):
            raise ValueError("solution must be a feasible parallel schedule")
        makespan = 0.0
        for task in self.tasks:
            machine, start = int(solution[task][0]), float(solution[task][1])
            makespan = max(makespan, start + self._times[task])
        return float(makespan)

    def earliest_start(self, assignments: list[tuple[int, float]]) -> float:
        """Earliest feasible start of ``task`` given prior assignments."""
        task = len(assignments)
        pred_finish = 0.0
        for predecessor in self._predecessors[task]:
            pred_finish = max(
                pred_finish,
                assignments[predecessor][1] + self._times[predecessor],
            )
        return float(pred_finish)


__all__ = ["ParallelSchedulingState"]
