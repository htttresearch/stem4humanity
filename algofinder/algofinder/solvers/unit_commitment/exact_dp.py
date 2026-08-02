"""Exact dynamic-programming solver for thermal unit commitment.

The state is ``(commitment mask, per-unit on/off ages)``: for each unit
the age is capped at its minimum up (positive) or minimum down
(negative) time, which makes the transition rules local. Per-hour
dispatch cost for a mask is the cheapest generation mix meeting demand
(greedy by variable cost, respecting min/max outputs); reserve
requirements are checked before dispatching.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from algofinder.problems.unit_commitment import UnitCommitmentState
from algofinder.solvers.base import (
    InapplicableError,
    Solver,
    SolverResult,
    register_solver,
)
from algofinder.trace.contract import trace_contract


def _dispatch(
    state: UnitCommitmentState, mask: int, hour: int
) -> tuple[float, list[int], list[float]] | None:
    """(cost, units, generations) for the cheapest feasible dispatch, or None."""
    generators = [g for g in range(state.generator_count) if (mask >> g) & 1]
    if not generators:
        return None
    if state.p_min[generators].sum() > state.demand[hour] + 1e-9:
        return None
    if state.p_max[generators].sum() < (
        state.demand[hour] + state.reserve[hour] - 1e-9
    ):
        return None
    order = sorted(generators, key=lambda g: float(state.c_var[g]))
    generation = [float(state.p_min[g]) for g in order]
    remaining = float(state.demand[hour]) - state.p_min[generators].sum()
    for index, g in enumerate(order):
        add = min(remaining, float(state.p_max[g] - state.p_min[g]))
        generation[index] += add
        remaining -= add
        if remaining <= 1e-9:
            break
    cost = sum(float(state.c_var[g]) * generation[index] for index, g in enumerate(order))
    return cost, order, generation


@register_solver
@trace_contract(
    version=1,
    state_schema="uc.dp-state/v1",
    atomic_steps={
        "uc.dp-hour/v1": "one hourly DP frontier transition (mask x ages)",
    },
    derived_state={"frontier": "reconstructable only from checkpoints; "
                              "per-hour events are frontier summaries"},
    coverage_scope="boundary",
)
class UCExactDPSolver(Solver):
    id = "uc-exact-dp"
    display = "Exact DP over commitments (min-up/min-down)"
    tags = frozenset({"exact"})
    applies_to = frozenset({"unit-commitment:classic"})

    def __init__(self, max_generators: int = 6, max_hours: int = 24) -> None:
        self.max_generators = max_generators
        self.max_hours = max_hours

    def solve(
        self,
        state: UnitCommitmentState,
        *,
        budget_seconds: float | None = None,
        context: object | None = None,
    ) -> SolverResult:
        from algofinder.trace.recorder import NullTraceRecorder

        started = perf_counter()
        trace = context.trace if context is not None else NullTraceRecorder()
        generator_count = state.generator_count
        hours = state.hours
        if generator_count > self.max_generators:
            raise InapplicableError(
                f"exact DP capped at {self.max_generators} generators "
                f"(instance has {generator_count})"
            )
        if hours > self.max_hours:
            raise InapplicableError(
                f"exact DP capped at {self.max_hours} hours "
                f"(instance has {hours})"
            )
        mask_count = 1 << generator_count
        dispatch_cache: dict[tuple[int, int], tuple[float, list[int], list[float]]] = {}
        for mask in range(mask_count):
            for hour in range(hours):
                dispatch_cache[(mask, hour)] = _dispatch(state, mask, hour)

        initial_mask = 0
        initial_ages: list[int] = []
        for g in range(generator_count):
            initial_age = int(state.initial_age[g])
            if initial_age > 0:
                initial_mask |= 1 << g
                initial_ages.append(min(initial_age, int(state.min_up[g])))
            else:
                initial_ages.append(-min(-initial_age, int(state.min_down[g])))

        infinity = float("inf")
        dp: dict[tuple[int, tuple[int, ...]], float] = {
            (initial_mask, tuple(initial_ages)): 0.0
        }
        predecessors: dict[
            tuple[int, tuple[int, ...], tuple[int, ...]],
            tuple[int, tuple[int, ...]],
        ] = {}
        initial_ref = None
        if trace.enabled:
            initial_ref = trace.checkpoint(
                {
                    "initial_mask": initial_mask,
                    "initial_ages": initial_ages,
                    "dispatch_cache_cells": mask_count * hours,
                }
            )
        for hour in range(hours):
            if trace.enabled and dp:
                best_key_so_far = min(dp, key=dp.get)
                trace.transition(
                    "uc.dp-hour/v1",
                    before=initial_ref,
                    action={"hour": hour},
                    outcome={
                        "states": len(dp),
                        "best_partial_cost": float(dp[best_key_so_far]),
                        "best_mask": best_key_so_far[0],
                        "best_ages": list(best_key_so_far[1]),
                    },
                    metrics={"frontier_size": len(dp)},
                )
            next_dp: dict[tuple[int, tuple[int, ...]], float] = {}
            for (mask, ages), cost in dp.items():
                for next_mask in range(mask_count):
                    next_ages: list[int] = []
                    startup = 0.0
                    valid = True
                    for g in range(generator_count):
                        on = (mask >> g) & 1
                        want_on = (next_mask >> g) & 1
                        age = ages[g]
                        if on:
                            if age < int(state.min_up[g]):
                                if not want_on:
                                    valid = False
                                    break
                                next_ages.append(age + 1)
                            elif want_on:
                                next_ages.append(int(state.min_up[g]))
                            else:
                                next_ages.append(-1)
                        else:
                            down_age = -age
                            if down_age < int(state.min_down[g]):
                                if want_on:
                                    valid = False
                                    break
                                next_ages.append(-(down_age + 1))
                            elif want_on:
                                startup += float(state.c_start[g])
                                next_ages.append(1)
                            else:
                                next_ages.append(-int(state.min_down[g]))
                    if not valid:
                        continue
                    dispatch = dispatch_cache[(next_mask, hour)]
                    if dispatch is None:
                        continue
                    key = (next_mask, tuple(next_ages))
                    candidate = cost + startup + dispatch[0]
                    if candidate < next_dp.get(key, infinity):
                        next_dp[key] = candidate
                        predecessors[(hour, key)] = (mask, ages)
            dp = next_dp
            if not dp:
                raise InapplicableError("no feasible commitment exists for this instance")

        best_key = min(dp, key=dp.get)
        best_cost = dp[best_key]
        masks_by_hour: list[int] = [best_key[0]] * hours
        key = best_key
        for hour in range(hours - 1, -1, -1):
            key = predecessors[(hour, key)]
            if hour > 0:
                masks_by_hour[hour - 1] = key[0]

        solution: list[list[list[float]]] = []
        for hour in range(hours):
            dispatch = dispatch_cache[(masks_by_hour[hour], hour)]
            if dispatch is None:  # pragma: no cover - reachable states are feasible
                raise InapplicableError("internal dispatch inconsistency")
            cost_units, order, generation = dispatch
            rows = [[float(unit), float(generation[order.index(unit)])] for unit in order]
            solution.append(rows)
        if trace.enabled:
            trace.event(
                "lifecycle",
                "solve.result",
                outcome={
                    "cost": float(best_cost),
                    "states_reached": len(dp),
                    "wall_seconds": perf_counter() - started,
                },
            )
        return SolverResult(
            solution=solution,
            cost=best_cost,
            exact=True,
            wall_seconds=perf_counter() - started,
            metadata={
                "algorithm": "commitment-dp",
                "generators": generator_count,
                "hours": hours,
                "states_reached": len(dp),
            },
        )


__all__ = ["UCExactDPSolver"]
