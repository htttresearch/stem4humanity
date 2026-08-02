"""Priority-list heuristic for thermal unit commitment.

Units are committed greedily: off units are started cheapest-first (by
variable cost) until the reserve requirement of the hour is met, while
units that are free to stop (past their minimum-up time) are
decommitted most-expensive-first whenever their capacity is not needed
now and will not be needed during their minimum-down lockout window.
Minimum-up/down times are honored by construction.
"""

from __future__ import annotations

from time import perf_counter

from stem4humanity.problems.unit_commitment import UnitCommitmentState
from stem4humanity.solvers.base import (
    InapplicableError,
    Solver,
    SolverResult,
    register_solver,
)
from stem4humanity.solvers.unit_commitment.exact_dp import _dispatch
from stem4humanity.trace.contract import trace_contract
from stem4humanity.trace.recorder import NullTraceRecorder


def priority_list_schedule(
    state: UnitCommitmentState,
    priority: list[float],
    context: object | None = None,
) -> list[list[list[float]]]:
    """Commit units by descending ``priority`` with reserve-triggered starts."""
    generators = list(range(state.generator_count))
    order = sorted(generators, key=lambda g: (-float(priority[g]), g))
    hours = state.hours
    p_max = state.p_max
    p_min = state.p_min
    min_up = state.min_up
    min_down = state.min_down
    demand = state.demand
    reserve = state.reserve
    ages = [int(age) for age in state.initial_age]
    status: list[bool] = [age > 0 for age in ages]
    solution: list[list[list[float]]] = []
    trace = context.trace if context is not None else NullTraceRecorder()
    initial_ref = None
    if trace.enabled:
        initial_ref = trace.checkpoint(
            {
                "ages": ages,
                "status": status,
                "priority": [float(x) for x in priority],
            }
        )
    for hour in range(hours):
        on_before = list(status)
        target = float(demand[hour] + reserve[hour])
        changed = True
        while changed:
            changed = False
            for g in reversed(order):
                if not status[g] or ages[g] < int(min_up[g]):
                    continue
                remaining = [gg for gg in generators if status[gg] and gg != g]
                if not remaining:
                    continue
                new_committed = float(p_max[remaining].sum())
                if new_committed < target - 1e-9:
                    continue
                if float(p_min[remaining].sum()) > demand[hour] + 1e-9:
                    continue
                if any(
                    new_committed < demand[h] + reserve[h] - 1e-9
                    for h in range(hour + 1, min(hours, hour + int(min_down[g]) + 1))
                ):
                    continue
                status[g] = False
                changed = True
        on_units: list[int] = []
        for g in generators:
            if status[g]:
                on_units.append(g)
        committed_power = float(p_max[on_units].sum())
        for g in order:
            if committed_power >= target - 1e-9:
                break
            if status[g] or -ages[g] < int(min_down[g]):
                continue
            status[g] = True
            on_units.append(g)
            committed_power += float(p_max[g])
        if committed_power < target - 1e-9:
            raise InapplicableError("no feasible commitment exists for this instance")
        mask = 0
        for g in on_units:
            mask |= 1 << g
        dispatch = _dispatch(state, mask, hour)
        if dispatch is None:
            raise InapplicableError("no feasible commitment exists for this instance")
        _, order_by_cost, generation = dispatch
        if trace.enabled:
            trace.transition(
                "uc.priority-hour/v1",
                before=initial_ref,
                action={
                    "hour": hour,
                    "mask": mask,
                    "target": target,
                    "committed_power": committed_power,
                },
                outcome={
                    "started": [
                        g for g in generators if not on_before[g] and status[g]
                    ],
                    "decommitted": [
                        g for g in generators if on_before[g] and not status[g]
                    ],
                    "dispatch_cost": dispatch[0]
                    + sum(
                        float(state.c_start[g])
                        for g in generators
                        if not on_before[g] and status[g]
                    ),
                },
                metrics={"on_units": len(on_units)},
            )
        solution.append(
            [[float(unit), float(generation[order_by_cost.index(unit)])]
             for unit in order_by_cost]
        )
        for g in generators:
            if on_before[g]:
                if status[g]:
                    ages[g] = min(ages[g] + 1, int(min_up[g]))
                else:
                    ages[g] = -1
            elif status[g]:
                ages[g] = 1
            else:
                ages[g] = -min(-ages[g] + 1, int(min_down[g]))
    return solution


@register_solver
@trace_contract(
    version=1,
    state_schema="uc.priority-state/v1",
    atomic_steps={
        "uc.priority-hour/v1": "one hourly commit/decommit cycle and dispatch",
    },
    derived_state={"ages": "ages and status after the recorded hour"},
    coverage_scope="algorithm",
)
class UCPriorityListSolver(Solver):
    id = "uc-priority-list"
    display = "Priority-list commitment (cheapest first)"
    tags = frozenset({"heuristic"})
    applies_to = frozenset({"unit-commitment:classic"})

    def solve(
        self,
        state: UnitCommitmentState,
        *,
        budget_seconds: float | None = None,
        context: object | None = None,
    ) -> SolverResult:
        started = perf_counter()
        priority = [
            -float(state.c_var[g]) for g in range(state.generator_count)
        ]
        solution = priority_list_schedule(state, priority, context=context)
        return SolverResult(
            solution=solution,
            cost=state.objective_value(solution),
            exact=False,
            wall_seconds=perf_counter() - started,
            metadata={"algorithm": "priority-list"},
        )


__all__ = ["UCPriorityListSolver", "priority_list_schedule"]
