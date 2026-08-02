"""Exact DP solver for storage arbitrage.

Energy is discretized on a fine grid (default 0.01 MWh). Charging moves
energy up by ``c * efficiency`` on the grid, so exactness requires the
rate to be a multiple of 0.1 MW and the efficiency to keep charge gains
on-grid (e.g. 0.8 or 0.9); instances are generated to satisfy this.
The per-period transition is a max-convolution over the discharge range,
evaluated with a sliding-window maximum, keeping the DP fast.
"""

from __future__ import annotations

from collections import deque
from time import perf_counter

import numpy as np

from algofinder.problems.unit_commitment import StorageArbitrageState
from algofinder.solvers.base import (
    InapplicableError,
    Solver,
    SolverResult,
    register_solver,
)


@register_solver
class UCStorageDPSolver(Solver):
    id = "uc-storage-dp"
    display = "Exact DP over the energy grid"
    tags = frozenset({"exact"})
    applies_to = frozenset({"unit-commitment:storage"})

    def __init__(
        self,
        scale: int = 100,
        charge_step_grid: int = 10,
        max_capacity_grid: int = 3000,
        max_periods: int = 96,
    ) -> None:
        self.scale = scale
        self.charge_step_grid = charge_step_grid
        self.max_capacity_grid = max_capacity_grid
        self.max_periods = max_periods

    def solve(
        self,
        state: StorageArbitrageState,
        *,
        budget_seconds: float | None = None,
    ) -> SolverResult:
        started = perf_counter()
        scale = self.scale
        charge_step = self.charge_step_grid
        capacity_grid = int(round(state.capacity * scale))
        rate_grid = int(round(state.rate * scale))
        initial_grid = int(round(state.initial_energy * scale))
        target_grid = int(round(state.target_energy * scale))
        efficiency = state.efficiency
        periods = state.periods
        if capacity_grid <= 0 or capacity_grid > self.max_capacity_grid:
            raise InapplicableError("storage capacity out of the DP grid range")
        if periods > self.max_periods:
            raise InapplicableError(f"DP capped at {self.max_periods} periods")
        if rate_grid % charge_step != 0:
            raise InapplicableError(
                "charge rate must be a multiple of 0.1 MW for the DP grid"
            )
        if abs(efficiency * charge_step - round(efficiency * charge_step)) > 1e-9:
            raise InapplicableError(
                "efficiency must keep charge gains on the energy grid"
            )
        if not (0 <= initial_grid <= capacity_grid and 0 <= target_grid <= capacity_grid):
            raise InapplicableError("energy levels must lie in the storage range")
        if (
            abs(state.capacity - capacity_grid / scale) > 1e-6
            or abs(state.initial_energy - initial_grid / scale) > 1e-6
            or abs(state.target_energy - target_grid / scale) > 1e-6
        ):
            raise InapplicableError(
                "capacity and energy levels must lie exactly on the DP grid"
            )

        charge_options = list(range(0, rate_grid + 1, charge_step))
        gains = [round(option * efficiency) for option in charge_options]
        infinity = -float("inf")
        value: list[float] = [infinity] * (capacity_grid + 1)
        for energy in range(target_grid, capacity_grid + 1):
            value[energy] = 0.0
        decisions: list[tuple[list[int], list[int]]] = []
        prices = state.prices

        for period in range(periods - 1, -1, -1):
            price = float(prices[period])
            shifted = [
                value[energy] - price * energy / scale
                for energy in range(capacity_grid + 1)
            ]
            best_discharge: list[float] = [infinity] * (capacity_grid + 1)
            discharge_arg: list[int] = [-1] * (capacity_grid + 1)
            window: deque[int] = deque()
            for mid in range(capacity_grid + 1):
                while window and shifted[window[-1]] <= shifted[mid]:
                    window.pop()
                window.append(mid)
                while window and window[0] < mid - rate_grid:
                    window.popleft()
                best_discharge[mid] = shifted[window[0]] + price * mid / scale
                discharge_arg[mid] = window[0]
            current: list[float] = [infinity] * (capacity_grid + 1)
            charge_arg: list[int] = [0] * (capacity_grid + 1)
            for energy in range(capacity_grid + 1):
                best = infinity
                best_option = 0
                for option, gain in zip(charge_options, gains):
                    mid = energy + gain
                    if mid <= capacity_grid and best_discharge[mid] > infinity / 2:
                        candidate = (
                            best_discharge[mid] - price * option / scale
                        )
                        if candidate > best:
                            best = candidate
                            best_option = option
                current[energy] = best
                charge_arg[energy] = best_option
            decisions.append((charge_arg, discharge_arg))
            value = current

        if value[initial_grid] <= infinity / 2:
            raise InapplicableError("no feasible arbitrage schedule exists")

        solution: list[list[float]] = []
        energy = initial_grid
        for period in range(periods):
            charge_arg, discharge_arg = decisions[periods - 1 - period]
            charge = charge_arg[energy]
            mid = energy + round(charge * efficiency)
            discharge = mid - discharge_arg[mid]
            solution.append([charge / scale, discharge / scale])
            energy = mid - discharge
        return SolverResult(
            solution=solution,
            cost=state.objective_value(solution),
            exact=True,
            wall_seconds=perf_counter() - started,
            metadata={
                "algorithm": "energy-grid-dp",
                "scale": scale,
                "grid_levels": capacity_grid + 1,
                "optimal_profit": float(
                    state.rate * state.prices.sum() - state.objective_value(solution)
                ),
            },
        )


__all__ = ["UCStorageDPSolver"]
