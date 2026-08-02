"""Shared storage-schedule helpers: action simulation and terminal repair.

Actions are kept on the same grid the exact DP uses: charges in steps of
0.1 MW and discharges in steps of 0.01 MWh. Heuristics therefore never
outperform the DP by exploiting sub-grid power levels.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from algofinder.problems.unit_commitment import StorageArbitrageState
from algofinder.solvers.base import InapplicableError


def _floor_charge(value: float) -> float:
    """Round a charge down to a multiple of 0.1 MW (the DP charge grid)."""
    return math.floor(value * 10 + 1e-9) / 10.0


def simulate_actions(
    state: StorageArbitrageState, actions: Sequence[int]
) -> list[list[float]]:
    """Simulate per-period actions (0=charge, 1=hold, 2=discharge).

    Charge fills at max rate up to capacity; discharge empties at max
    rate down to zero; feasibility is preserved by construction.
    """
    solution: list[list[float]] = []
    energy = float(state.initial_energy)
    for action in actions:
        if action == 0:
            charge = _floor_charge(
                min(state.rate, (state.capacity - energy) / state.efficiency)
            )
            discharge = 0.0
        elif action == 2:
            discharge = min(state.rate, energy)
            charge = 0.0
        else:
            charge = discharge = 0.0
        energy += state.efficiency * charge - discharge
        solution.append([charge, discharge])
    return solution


def repair_terminal(
    state: StorageArbitrageState, solution: list[list[float]]
) -> list[list[float]]:
    """Force the terminal energy to the target with minimum value loss.

    If energy is short, buy at the cheapest remaining periods (top up
    within the rate headroom, then cancel discharge there); if it is
    high, sell at the priciest periods (cancel charge, then discharge
    within the rate headroom). Every edit is clipped against the whole
    remaining energy profile, so capacity bounds and power limits hold
    at every period.
    """
    repaired = [list(entry) for entry in solution]
    periods = state.periods
    energy = float(state.initial_energy)
    profile: list[float] = []
    for entry in repaired:
        energy += state.efficiency * entry[0] - entry[1]
        profile.append(energy)
    target = state.target_energy
    if energy < target - 1e-9:
        needed = target - energy
        order = sorted(range(periods), key=lambda t: float(state.prices[t]))
        for period in order:
            if needed <= 1e-9:
                break
            peak = max(profile[s] for s in range(period, periods))
            headroom = state.rate - repaired[period][0]
            charge = _floor_charge(
                min(
                    headroom,
                    needed / state.efficiency,
                    (state.capacity - peak) / state.efficiency,
                )
            )
            if charge > 1e-9:
                repaired[period][0] += charge
                for s in range(period, periods):
                    profile[s] += state.efficiency * charge
                needed = target - profile[-1]
        for period in order:
            if needed <= 1e-9:
                break
            peak = max(profile[s] for s in range(period, periods))
            drop = min(repaired[period][1], needed, state.capacity - peak)
            if drop > 1e-9:
                repaired[period][1] -= drop
                for s in range(period, periods):
                    profile[s] += drop
                needed = target - profile[-1]
        if needed > 1e-9:
            raise InapplicableError("terminal energy target unreachable")
    elif energy > target + 1e-9:
        excess = energy - target
        order = sorted(range(periods), key=lambda t: -float(state.prices[t]))
        for period in order:
            if excess <= 1e-9:
                break
            floor = min(profile[s] for s in range(period, periods))
            drop = _floor_charge(
                min(
                    repaired[period][0],
                    excess / state.efficiency,
                    floor / state.efficiency,
                )
            )
            if drop > 1e-9:
                repaired[period][0] -= drop
                for s in range(period, periods):
                    profile[s] -= state.efficiency * drop
                excess = profile[-1] - target
        for period in order:
            if excess <= 1e-9:
                break
            floor = min(profile[s] for s in range(period, periods))
            headroom = state.rate - repaired[period][1]
            discharge = min(headroom, excess, floor)
            if discharge > 1e-9:
                repaired[period][1] += discharge
                for s in range(period, periods):
                    profile[s] -= discharge
                excess = profile[-1] - target
        if excess > 1e-9:
            raise InapplicableError("terminal energy target unreachable")
    return repaired


__all__ = ["repair_terminal", "simulate_actions"]
