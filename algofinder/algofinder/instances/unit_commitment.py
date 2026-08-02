"""Unit-commitment instance generators: thermal classic and storage arbitrage.

``classic`` instances are thermal unit commitment problems: a demand
curve with a fixed reserve requirement and a small fleet of generators
with min-up/min-down times and startup costs (free-start initial states,
``initial_age = -min_down``). ``storage`` instances are single-unit
arbitrage problems over a lognormal price curve with grid-friendly
parameters (rate multiple of 0.1 MW, 0.9 efficiency, energy levels on
the 0.01 MWh grid). Best-known values come from the exact DP solvers.
"""

from __future__ import annotations

import numpy as np

from algofinder.instances.base import annotate_best_known, save_manifest
from algofinder.problems.base import Instance
from algofinder.problems.registry import get_problem
from algofinder.solvers.base import all_solvers
import algofinder.solvers.registry  # noqa: F401  (run solver registration)


def generate_classic_instances(
    units: int | tuple[int, int],
    hours: int | tuple[int, int],
    *,
    count: int,
    seed: int,
    split: str = "test",
) -> list[Instance]:
    """Thermal unit-commitment instances with a sinusoidal demand curve."""
    if isinstance(units, tuple):
        unit_lo, unit_hi = units
    else:
        unit_lo = unit_hi = units
    if isinstance(hours, tuple):
        hour_lo, hour_hi = hours
    else:
        hour_lo = hour_hi = hours

    rng = np.random.default_rng(seed)
    records: list[Instance] = []
    for index in range(count):
        n = int(rng.integers(unit_lo, unit_hi + 1)) if unit_lo != unit_hi else unit_lo
        t = int(rng.integers(hour_lo, hour_hi + 1)) if hour_lo != hour_hi else hour_lo
        p_max = rng.uniform(50.0, 150.0, size=n)
        generators = []
        for g in range(n):
            min_up = int(rng.integers(1, 4))
            min_down = int(rng.integers(1, 4))
            generators.append(
                {
                    "p_min": float(p_max[g] * rng.uniform(0.2, 0.45)),
                    "p_max": float(p_max[g]),
                    "c_var": float(rng.uniform(15.0, 60.0)),
                    "c_start": float(rng.uniform(200.0, 900.0)),
                    "min_up": min_up,
                    "min_down": min_down,
                    "initial_age": -min_down,
                }
            )
        base = float(0.45 * p_max.sum())
        hours_index = np.arange(t)
        demand = base * (
            0.75 + 0.25 * np.sin(2 * np.pi * hours_index / t + rng.uniform(0, 6.28))
        )
        demand = demand + rng.normal(0.0, base * 0.03, size=t)
        demand = np.clip(demand, base * 0.55, None)
        reserve = 0.1 * demand
        name = f"unit-commitment:classic:g{n}h{t}-s{seed}-i{index}"
        records.append(
            Instance(
                name=name,
                problem="unit-commitment",
                subproblem="classic",
                family="random",
                seed=seed,
                data={
                    "hours": t,
                    "generators": generators,
                    "demand": demand.tolist(),
                    "reserve": reserve.tolist(),
                },
                best_known=None,
                split=split,
            )
        )
    return records


def generate_storage_instances(
    periods: int | tuple[int, int],
    *,
    count: int,
    seed: int,
    split: str = "test",
) -> list[Instance]:
    """Single-battery arbitrage over lognormal (GBM) price curves."""
    if isinstance(periods, tuple):
        period_lo, period_hi = periods
    else:
        period_lo = period_hi = periods

    rng = np.random.default_rng(seed)
    records: list[Instance] = []
    for index in range(count):
        t = (
            int(rng.integers(period_lo, period_hi + 1))
            if period_lo != period_hi
            else period_lo
        )
        log_steps = rng.normal(0.0, 0.08, size=t)
        log_steps[0] = 0.0
        prices = 50.0 * np.exp(np.cumsum(log_steps))
        name = f"unit-commitment:storage:t{t}-s{seed}-i{index}"
        records.append(
            Instance(
                name=name,
                problem="unit-commitment",
                subproblem="storage",
                family="random",
                seed=seed,
                data={
                    "periods": t,
                    "prices": prices.tolist(),
                    "capacity": 5.0,
                    "rate": 1.0,
                    "efficiency": 0.9,
                    "initial_energy": 2.5,
                    "target_energy": 2.5,
                },
                best_known=None,
                split=split,
            )
        )
    return records


def generate_unit_commitment_manifests(
    output_dir: str = "public/data",
    *,
    best_known_budget_seconds: float | None = None,
) -> None:
    classic_instances: list[Instance] = []
    for sizes in ((4, 4), (5, 5)):
        classic_instances.extend(
            generate_classic_instances(
                sizes, (12, 18), count=4, seed=7801, split="test"
            )
        )
    classic_instances.extend(
        generate_classic_instances((6, 6), (24, 24), count=6, seed=7802, split="train")
    )

    storage_instances: list[Instance] = []
    storage_instances.extend(
        generate_storage_instances((30, 48), count=8, seed=7901, split="test")
    )
    storage_instances.extend(
        generate_storage_instances((24, 60), count=6, seed=7902, split="train")
    )

    problem = get_problem("unit-commitment")
    classic_exact = next(
        solver for solver in all_solvers().values() if solver.id == "uc-exact-dp"
    )
    classic_instances = list(
        annotate_best_known(classic_instances, classic_exact, problem, budget_seconds=best_known_budget_seconds)
    )
    storage_exact = next(
        solver for solver in all_solvers().values() if solver.id == "uc-storage-dp"
    )
    storage_instances = list(
        annotate_best_known(storage_instances, storage_exact, problem, budget_seconds=best_known_budget_seconds)
    )

    save_manifest(
        classic_instances, f"{output_dir}/unit_commitment_classic_manifests.json"
    )
    save_manifest(
        storage_instances, f"{output_dir}/unit_commitment_storage_manifests.json"
    )


__all__ = [
    "generate_classic_instances",
    "generate_storage_instances",
    "generate_unit_commitment_manifests",
]
