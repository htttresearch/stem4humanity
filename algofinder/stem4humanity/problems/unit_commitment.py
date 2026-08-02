"""Unit-commitment problem family: thermal commitment and storage arbitrage.

The general class is the classic thermal unit commitment (UC): given a
demand curve and generators with min/max outputs, linear variable costs,
start-up costs and minimum up/down times, choose per-hour commitments and
dispatches minimizing production plus start-up cost while meeting demand
and a reserve margin (NP-hard, solved daily by every utility). The
well-studied subclass is storage arbitrage: when to charge and discharge
a battery (or hydro reservoir) against a known price curve to maximize
profit.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

Generator = dict[str, float | int]


class UnitCommitmentState:
    """A thermal unit-commitment instance (minimize operating cost)."""

    def __init__(
        self,
        name: str,
        hours: int,
        generators: Sequence[Generator],
        demand: Any,
        reserve: Any,
    ) -> None:
        hours = int(hours)
        if hours < 1:
            raise ValueError("at least one hour is required")
        if not generators:
            raise ValueError("at least one generator is required")
        demand_vector = np.asarray(demand, dtype=float)
        reserve_vector = np.asarray(reserve, dtype=float)
        if demand_vector.shape != (hours,) or reserve_vector.shape != (hours,):
            raise ValueError("demand and reserve must have one entry per hour")
        if not np.isfinite(demand_vector).all() or not np.isfinite(reserve_vector).all():
            raise ValueError("demand and reserve must be finite")
        if np.any(demand_vector <= 0) or np.any(reserve_vector < 0):
            raise ValueError("demand must be positive and reserve non-negative")

        p_min: list[float] = []
        p_max: list[float] = []
        c_var: list[float] = []
        c_start: list[float] = []
        min_up: list[int] = []
        min_down: list[int] = []
        initial_age: list[int] = []
        for generator in generators:
            p_min.append(float(generator["p_min"]))
            p_max.append(float(generator["p_max"]))
            c_var.append(float(generator["c_var"]))
            c_start.append(float(generator["c_start"]))
            min_up.append(int(generator["min_up"]))
            min_down.append(int(generator["min_down"]))
            initial_age.append(int(generator.get("initial_age", -min_down[-1])))
        p_min_vector = np.asarray(p_min, dtype=float)
        p_max_vector = np.asarray(p_max, dtype=float)
        c_var_vector = np.asarray(c_var, dtype=float)
        c_start_vector = np.asarray(c_start, dtype=float)
        min_up_vector = np.asarray(min_up, dtype=int)
        min_down_vector = np.asarray(min_down, dtype=int)
        initial_age_vector = np.asarray(initial_age, dtype=int)
        if not (
            np.all(p_min_vector > 0)
            and np.all(p_max_vector >= p_min_vector)
            and np.all(c_var_vector > 0)
            and np.all(c_start_vector >= 0)
            and np.all(min_up_vector >= 1)
            and np.all(min_down_vector >= 1)
        ):
            raise ValueError("invalid generator parameters")
        if not np.all(np.abs(initial_age_vector) >= 1):
            raise ValueError("initial ages must be nonzero (sign = on/off)")

        self.name = name
        self.hours = hours
        self._generator_count = len(generators)
        self._p_min: NDArray[np.float64] = p_min_vector.copy()
        self._p_max: NDArray[np.float64] = p_max_vector.copy()
        self._c_var: NDArray[np.float64] = c_var_vector.copy()
        self._c_start: NDArray[np.float64] = c_start_vector.copy()
        self._min_up: NDArray[np.int64] = min_up_vector.copy()
        self._min_down: NDArray[np.int64] = min_down_vector.copy()
        self._initial_age: NDArray[np.int64] = initial_age_vector.copy()
        self._p_min.setflags(write=False)
        self._p_max.setflags(write=False)
        self._c_var.setflags(write=False)
        self._c_start.setflags(write=False)
        self._min_up.setflags(write=False)
        self._min_down.setflags(write=False)
        self._initial_age.setflags(write=False)
        self._demand: NDArray[np.float64] = demand_vector.copy()
        self._reserve: NDArray[np.float64] = reserve_vector.copy()
        self._demand.setflags(write=False)
        self._reserve.setflags(write=False)
        self.units = tuple(range(self._generator_count))

    @property
    def generator_count(self) -> int:
        return self._generator_count

    @property
    def p_min(self) -> NDArray[np.float64]:
        return self._p_min

    @property
    def p_max(self) -> NDArray[np.float64]:
        return self._p_max

    @property
    def c_var(self) -> NDArray[np.float64]:
        return self._c_var

    @property
    def c_start(self) -> NDArray[np.float64]:
        return self._c_start

    @property
    def min_up(self) -> NDArray[np.int64]:
        return self._min_up

    @property
    def min_down(self) -> NDArray[np.int64]:
        return self._min_down

    @property
    def initial_age(self) -> NDArray[np.int64]:
        """Positive = on for that many hours; negative = off."""
        return self._initial_age

    @property
    def demand(self) -> NDArray[np.float64]:
        return self._demand

    @property
    def reserve(self) -> NDArray[np.float64]:
        return self._reserve

    def _commitment_sequence(
        self, solution: Any
    ) -> list[list[int]] | None:
        """Per-hour on/off unit lists, or None if structurally malformed."""
        if not isinstance(solution, list) or len(solution) != self.hours:
            return None
        commitments: list[list[int]] = []
        for hour, row in enumerate(solution):
            if not isinstance(row, list):
                return None
            unit_ids: list[int] = []
            for entry in row:
                try:
                    unit, generation = int(entry[0]), float(entry[1])
                except (TypeError, ValueError, IndexError):
                    return None
                if unit < 0 or unit >= self._generator_count:
                    return None
                if unit in unit_ids:
                    return None
                if generation < -1e-9:
                    return None
                if generation > 1e-9:
                    if generation < self._p_min[unit] - 1e-9:
                        return None
                    if generation > self._p_max[unit] + 1e-9:
                        return None
                    unit_ids.append(unit)
                elif abs(generation) > 1e-9:
                    return None
            commitments.append(unit_ids)
        return commitments

    def verify(self, solution: Any) -> bool:
        commitments = self._commitment_sequence(solution)
        if commitments is None:
            return False
        for hour, unit_ids in enumerate(commitments):
            total = sum(float(row[1]) for row in solution[hour])
            if total < self._demand[hour] - 1e-9:
                return False
            if self._p_max[list(unit_ids)].sum() < (
                self._demand[hour] + self._reserve[hour] - 1e-9
            ):
                return False
        for unit in self.units:
            age = int(self._initial_age[unit])
            for hour in range(self.hours):
                is_on = unit in commitments[hour]
                if is_on:
                    if age < 0:
                        if -age < self._min_down[unit]:
                            return False
                        age = 1
                    else:
                        age += 1
                else:
                    if age > 0:
                        if age < self._min_up[unit]:
                            return False
                        age = -1
                    else:
                        age -= 1
        return True

    def objective_value(self, solution: Any) -> float:
        if not self.verify(solution):
            raise ValueError("solution must be a feasible commitment schedule")
        cost = 0.0
        for unit in self.units:
            previous_on = int(self._initial_age[unit]) > 0
            for hour in range(self.hours):
                is_on = unit in [int(row[0]) for row in solution[hour]]
                if is_on:
                    cost += self._c_var[unit] * float(
                        next(row[1] for row in solution[hour] if int(row[0]) == unit)
                    )
                    if not previous_on:
                        cost += self._c_start[unit]
                previous_on = is_on
        return float(cost)


class StorageArbitrageState:
    """A single storage unit arbitraging a known price curve (max profit).

    The objective is reported as ``cost = K - profit`` where
    ``K = sum(C_max * price_t)`` is the maximum gross revenue, so costs
    stay non-negative and the usual "lower is better" gap math holds.
    """

    def __init__(
        self,
        name: str,
        periods: int,
        prices: Any,
        capacity: float,
        rate: float,
        efficiency: float,
        initial_energy: float,
        target_energy: float | None = None,
    ) -> None:
        periods = int(periods)
        if periods < 1:
            raise ValueError("at least one period is required")
        price_vector = np.asarray(prices, dtype=float)
        if price_vector.shape != (periods,):
            raise ValueError("prices must have one entry per period")
        if not np.isfinite(price_vector).all() or np.any(price_vector <= 0):
            raise ValueError("prices must be positive and finite")
        if not (capacity > 0 and rate > 0 and 0 < efficiency <= 1):
            raise ValueError("capacity, rate and efficiency must be positive")
        if not 0 <= initial_energy <= capacity + 1e-9:
            raise ValueError("initial energy must lie in the storage range")
        target = (
            float(target_energy) if target_energy is not None else initial_energy
        )
        if not 0 <= target <= capacity + 1e-9:
            raise ValueError("target energy must lie in the storage range")

        self.name = name
        self.periods = periods
        self._prices: NDArray[np.float64] = price_vector.copy()
        self._prices.setflags(write=False)
        self.capacity = float(capacity)
        self.rate = float(rate)
        self.efficiency = float(efficiency)
        self.initial_energy = float(initial_energy)
        self.target_energy = target

    @property
    def prices(self) -> NDArray[np.float64]:
        return self._prices

    def _energy_profile(self, solution: Any) -> list[float] | None:
        """Per-period energy trajectory, or None if structurally malformed."""
        if not isinstance(solution, list) or len(solution) != self.periods:
            return None
        energy = float(self.initial_energy)
        profile = [energy]
        for entry in solution:
            try:
                charge, discharge = float(entry[0]), float(entry[1])
            except (TypeError, ValueError, IndexError):
                return None
            if charge < -1e-9 or discharge < -1e-9:
                return None
            if charge > self.rate + 1e-9 or discharge > self.rate + 1e-9:
                return None
            energy += self.efficiency * charge - discharge
            if energy < -1e-9 or energy > self.capacity + 1e-9:
                return None
            profile.append(energy)
        return profile

    def verify(self, solution: Any) -> bool:
        profile = self._energy_profile(solution)
        if profile is None:
            return False
        return profile[-1] >= self.target_energy - 1e-9

    def objective_value(self, solution: Any) -> float:
        if not self.verify(solution):
            raise ValueError("solution must be a feasible arbitrage schedule")
        profit = 0.0
        for hour, entry in enumerate(solution):
            charge, discharge = float(entry[0]), float(entry[1])
            profit += (discharge - charge) * self._prices[hour]
        gross = float(self.rate * self._prices.sum())
        return gross - profit


__all__ = ["Generator", "StorageArbitrageState", "UnitCommitmentState"]
