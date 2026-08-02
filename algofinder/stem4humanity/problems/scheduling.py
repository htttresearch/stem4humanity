"""Scheduling problem family: job shop and two-machine flow shop.

The general class is the (NP-hard) job shop: each job visits machines in
its own order, and a solution fixes the processing order on every
machine. The well-studied subclass is the two-machine permutation flow
shop, where Johnson's rule solves the makespan in O(n log n); solutions
are job permutations.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


def earliest_start_makespan(
    operations: Sequence[Sequence[tuple[int, float]]],
    solution: dict[int, Sequence[Sequence[int]]],
) -> float:
    """Earliest-start makespan for per-machine processing orders.

    Operations are evaluated in dependency order (machine-queue edges and
    job-precedence edges), so the result does not depend on machine
    numbering. Infeasible orders -- cycles between machine and job
    precedence -- yield infinity.
    """
    position: dict[tuple[int, int], tuple[int, int]] = {}
    for machine, entries in solution.items():
        for idx, (job, step) in enumerate(entries):
            position[(int(job), int(step))] = (machine, idx)
    n_nodes = sum(len(entries) for entries in solution.values())
    if n_nodes == 0:
        return 0.0
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    indeg: dict[tuple[int, int], int] = {}
    for machine, entries in solution.items():
        for idx in range(len(entries) - 1):
            a = (machine, idx)
            b = (machine, idx + 1)
            adj.setdefault(a, []).append(b)
            indeg[b] = indeg.get(b, 0) + 1
        for idx, (job, step) in enumerate(entries):
            if step + 1 < len(operations[job]):
                nxt = position[(job, step + 1)]
                a = (machine, idx)
                adj.setdefault(a, []).append(nxt)
                indeg[nxt] = indeg.get(nxt, 0) + 1
    ready = [
        (machine, idx)
        for machine, entries in solution.items()
        for idx in range(len(entries))
        if indeg.get((machine, idx), 0) == 0
    ]
    machine_free = [0.0] * len(solution)
    job_finish = [[0.0] * (len(ops) + 1) for ops in operations]
    processed = 0
    makespan = 0.0
    while ready:
        machine, idx = ready.pop()
        processed += 1
        job, step = solution[machine][idx]
        duration = operations[job][step][1]
        start = max(machine_free[machine], job_finish[job][step])
        finish = start + duration
        job_finish[job][step + 1] = finish
        machine_free[machine] = finish
        makespan = max(makespan, finish)
        for nxt in adj.get((machine, idx), []):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
    if processed < n_nodes:
        return float("inf")
    return float(makespan)


class FlowShopState:
    """A two-machine permutation flow-shop instance (makespan)."""

    def __init__(self, name: str, processing_times: Any) -> None:
        matrix = np.asarray(processing_times, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != 2:
            raise ValueError("processing_times must have shape (job_count, 2)")
        if not np.isfinite(matrix).all():
            raise ValueError("processing times must be finite")
        if np.any(matrix <= 0):
            raise ValueError("processing times must be positive")
        self.name = name
        self._times: NDArray[np.float64] = matrix.copy()
        self._times.setflags(write=False)
        self.jobs = tuple(range(matrix.shape[0]))

    @property
    def times(self) -> NDArray[np.float64]:
        """Read-only ``(job_count, 2)`` processing-time matrix."""
        return self._times

    @property
    def job_count(self) -> int:
        return len(self.jobs)

    def verify(self, solution: Any) -> bool:
        try:
            order = [int(job) for job in solution]
        except (TypeError, ValueError):
            return False
        return len(order) == self.job_count and set(order) == set(self.jobs)

    def objective_value(self, solution: Any) -> float:
        if not self.verify(solution):
            raise ValueError("solution must be a permutation of every job")
        first_machine = 0.0
        second_machine = 0.0
        for job in (int(job) for job in solution):
            first_machine += self._times[job, 0]
            second_machine = max(second_machine, first_machine) + self._times[job, 1]
        return float(second_machine)


class JobShopState:
    """A job-shop instance: makespan over per-machine processing orders.

    ``operations`` has one entry per job: a list of ``(machine, duration)``
    steps in the job's required order. A solution maps each machine to the
    ordered list of ``(job, step)`` pairs it processes.
    """

    def __init__(
        self,
        name: str,
        n_jobs: int,
        n_machines: int,
        operations: Any,
    ) -> None:
        n_jobs = int(n_jobs)
        n_machines = int(n_machines)
        if n_jobs < 1 or n_machines < 1:
            raise ValueError("at least one job and one machine are required")
        normalized = []
        for job_index, job_operations in enumerate(operations):
            steps = []
            for machine, duration in job_operations:
                machine = int(machine)
                duration = float(duration)
                if not 0 <= machine < n_machines:
                    raise ValueError("operation machine is out of range")
                if not np.isfinite(duration) or duration <= 0:
                    raise ValueError("operation durations must be positive")
                steps.append((machine, duration))
            if not steps:
                raise ValueError("every job needs at least one operation")
            normalized.append(steps)
        if len(normalized) != n_jobs:
            raise ValueError("operations must have one entry per job")

        self.name = name
        self._n_jobs = n_jobs
        self._n_machines = n_machines
        self._operations: tuple[tuple[tuple[int, float], ...], ...] = tuple(
            tuple(steps) for steps in normalized
        )
        self.jobs = tuple(range(n_jobs))
        self.machines = tuple(range(n_machines))

    @property
    def n_jobs(self) -> int:
        return self._n_jobs

    @property
    def n_machines(self) -> int:
        return self._n_machines

    @property
    def operations(self) -> tuple[tuple[tuple[int, float], ...], ...]:
        """Per-job operation lists as ``(machine, duration)`` steps."""
        return self._operations

    def verify(self, solution: Any) -> bool:
        if not isinstance(solution, dict):
            return False
        seen: set[tuple[int, int]] = set()
        for machine in self.machines:
            if machine not in solution:
                return False
            for entry in solution[machine]:
                try:
                    job, step = (int(entry[0]), int(entry[1]))
                except (TypeError, ValueError, IndexError):
                    return False
                if not 0 <= job < self._n_jobs:
                    return False
                if step < 0 or step >= len(self._operations[job]):
                    return False
                if self._operations[job][step][0] != machine:
                    return False
                if (job, step) in seen:
                    return False
                seen.add((job, step))
        if len(seen) != sum(len(ops) for ops in self._operations):
            return False
        return earliest_start_makespan(self._operations, solution) < float("inf")

    def objective_value(self, solution: Any) -> float:
        if not self.verify(solution):
            raise ValueError("solution must schedule every operation exactly once")
        return earliest_start_makespan(self._operations, solution)


__all__ = ["FlowShopState", "JobShopState", "earliest_start_makespan"]
