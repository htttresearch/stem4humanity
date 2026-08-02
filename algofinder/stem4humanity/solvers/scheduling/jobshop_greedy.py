"""Greedy job-shop dispatching heuristic (Giffler-Thompson style).

Repeatedly schedules the available operation with the earliest possible
completion time on its machine; a single deterministic pass, no
backtracking.
"""

from __future__ import annotations

from time import perf_counter

from stem4humanity.problems.scheduling import JobShopState
from stem4humanity.solvers.base import Solver, SolverResult, register_solver
from stem4humanity.solvers.scheduling.jobshop_bnb import simulate_makespan


@register_solver
class JobShopGreedySolver(Solver):
    id = "jobshop-greedy"
    display = "Earliest-completion dispatching (greedy, no backtracking)"
    tags = frozenset({"heuristic"})
    applies_to = frozenset({"scheduling:job-shop"})

    def solve(
        self,
        state: JobShopState,
        *,
        budget_seconds: float | None = None,
    ) -> SolverResult:
        started = perf_counter()
        machine_free = [0.0] * state.n_machines
        job_finish = [
            [0.0] * (len(state.operations[job]) + 1) for job in state.jobs
        ]
        next_step = [0] * state.n_jobs
        orders: dict[int, list[tuple[int, int]]] = {
            machine: [] for machine in state.machines
        }
        remaining = sum(len(ops) for ops in state.operations)
        while remaining:
            best_key: tuple[float, int, int] | None = None
            best_entry: tuple[int, int] | None = None
            for job in state.jobs:
                step = next_step[job]
                if step >= len(state.operations[job]):
                    continue
                machine, duration = state.operations[job][step]
                start = max(
                    machine_free[machine], job_finish[job][step]
                )
                completion = start + duration
                key = (completion, job, step)
                if best_key is None or key < best_key:
                    best_key = key
                    best_entry = (job, step)
            job, step = best_entry
            machine, duration = state.operations[job][step]
            start = max(machine_free[machine], job_finish[job][step])
            finish = start + duration
            job_finish[job][step + 1] = finish
            machine_free[machine] = finish
            orders[machine].append((job, step))
            next_step[job] = step + 1
            remaining -= 1
        cost = simulate_makespan(state, orders)
        return SolverResult(
            solution=orders,
            cost=cost,
            exact=False,
            wall_seconds=perf_counter() - started,
            metadata={"algorithm": "greedy-earliest-completion-dispatching"},
        )


__all__ = ["JobShopGreedySolver"]
