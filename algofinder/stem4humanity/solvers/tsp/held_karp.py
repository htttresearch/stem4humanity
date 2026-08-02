"""Exact Held--Karp dynamic programming for symmetric complete TSP."""

from __future__ import annotations

from typing import Any, Callable, TypeVar, cast

import numpy as np

from stem4humanity.problems.tsp import (
    TravellingSalespersonProblem,
)
import time
from stem4humanity.solvers.base import Solver, SolverResult, register_solver, InapplicableError


Function = TypeVar("Function", bound=Callable[..., Any])

try:
    from numba import njit as _njit
except ImportError:

    def _compile(function: Function) -> Function:
        """Keep an exact pure-Python fallback when Numba is unavailable."""
        return function

else:

    def _compile(function: Function) -> Function:
        # Do not enable fastmath: it may alter floating-point comparisons.
        return cast(Function, _njit(cache=True, fastmath=False)(function))


@_compile
def _held_karp_kernel(
    distances: np.ndarray,
) -> tuple[float, int, np.ndarray]:
    """Compute an exact Held--Karp solution.

    City zero is fixed as the start city and excluded from subset masks.
    Local city index ``j`` represents original city ``j + 1``.

    Returns:
        The optimal tour cost, the final local city index, and the parent
        table used to reconstruct the tour.
    """
    city_count = distances.shape[0]
    non_start_count = city_count - 1
    subset_count = 1 << non_start_count

    # Only states whose endpoint belongs to the subset will be accessed.
    # Avoiding np.full prevents initialization of unused entries.
    costs = np.empty(
        (subset_count, non_start_count),
        dtype=np.float64,
    )

    # Any computationally feasible Held--Karp instance has far fewer than
    # 256 cities, so one byte per predecessor is sufficient.
    parents = np.empty(
        (subset_count, non_start_count),
        dtype=np.uint8,
    )

    # Maps an isolated subset bit to its corresponding local city index.
    # Only power-of-two entries are read.
    bit_to_index = np.empty(subset_count, dtype=np.uint8)

    # Base cases: travel directly from city zero to each other city.
    for city in range(non_start_count):
        bit = 1 << city
        bit_to_index[bit] = city
        costs[bit, city] = distances[0, city + 1]

    # DP recurrence:
    #
    # C[S, j] = min(C[S - {j}, i] + d[i, j])
    #             i in S - {j}
    for subset in range(1, subset_count):
        endpoint_bits = subset

        while endpoint_bits:
            endpoint_bit = endpoint_bits & -endpoint_bits
            endpoint = int(bit_to_index[endpoint_bit])
            previous_subset = subset ^ endpoint_bit

            # Singleton subsets were initialized above.
            if previous_subset:
                best_cost = np.inf
                best_parent = 0
                predecessor_bits = previous_subset

                while predecessor_bits:
                    predecessor_bit = (
                        predecessor_bits & -predecessor_bits
                    )
                    predecessor = int(bit_to_index[predecessor_bit])

                    candidate = (
                        costs[previous_subset, predecessor]
                        + distances[
                            predecessor + 1,
                            endpoint + 1,
                        ]
                    )

                    # Strict comparison gives deterministic tie handling.
                    if candidate < best_cost:
                        best_cost = candidate
                        best_parent = predecessor

                    predecessor_bits ^= predecessor_bit

                costs[subset, endpoint] = best_cost
                parents[subset, endpoint] = best_parent

            endpoint_bits ^= endpoint_bit

    # Close the cycle by returning to city zero.
    full_subset = subset_count - 1
    best_tour_cost = np.inf
    final_city = -1

    for city in range(non_start_count):
        candidate = (
            costs[full_subset, city]
            + distances[city + 1, 0]
        )

        if candidate < best_tour_cost:
            best_tour_cost = candidate
            final_city = city

    return best_tour_cost, final_city, parents


@register_solver
class HeldKarpSolver(Solver):
    id = "held-karp"
    display = "Held-Karp exact dynamic programming"
    tags = frozenset({"exact"})
    applies_to = frozenset({"tsp:general", "tsp:euclidean", "tsp:clustered"})
    """Solve small complete symmetric TSP instances exactly.

    The algorithm uses O(n² 2ⁿ) time and O(n 2ⁿ) memory. The explicit
    limit prevents accidental use as a large-instance benchmark solver.

    Numba is used when installed. Without Numba, the same exact algorithm
    runs as a pure-Python fallback.
    """

    def __init__(self, max_cities: int = 18) -> None:
        if max_cities < 3:
            raise ValueError("max_cities must be at least three")
        self.max_cities = max_cities

    def solve(
        self,
        problem: TravellingSalespersonProblem,
        *,
        budget_seconds: float | None = None,
    ) -> SolverResult:
        started = time.perf_counter()
        city_count = problem.city_count

        if city_count < 1:
            raise ValueError(
                "Held--Karp requires at least one city"
            )

        if city_count > self.max_cities:
            raise InapplicableError(
                f"Held--Karp is limited to {self.max_cities} cities; "
                f"received {city_count}"
            )

        distances = np.ascontiguousarray(
            problem.distances,
            dtype=np.float64,
        )

        expected_shape = (city_count, city_count)
        if distances.shape != expected_shape:
            raise ValueError(
                "Distance matrix shape does not match city_count: "
                f"expected {expected_shape}, received {distances.shape}"
            )

        if not np.isfinite(distances).all():
            raise ValueError(
                "A complete TSP distance matrix must contain only "
                "finite values"
            )

        if city_count == 1:
            return SolverResult(
                solution=(0,),
                cost=0.0,
                exact=True,
                wall_seconds=time.perf_counter() - started,
                metadata={
                    "algorithm": "held-karp",
                    "states": 1,
                },
            )

        if city_count == 2:
            return SolverResult(
                solution=(0, 1),
                cost=float(
                    distances[0, 1] + distances[1, 0]
                ),
                exact=True,
                wall_seconds=time.perf_counter() - started,
                metadata={
                    "algorithm": "held-karp",
                    "states": 2,
                },
            )

        tour_cost, final_city, parents = _held_karp_kernel(
            distances
        )

        if final_city < 0 or not np.isfinite(tour_cost):
            raise ValueError(
                "The distance matrix does not contain a finite tour"
            )

        non_start_count = city_count - 1
        full_subset = (1 << non_start_count) - 1

        # Reconstruct the tour backwards. Parent entries use local indices,
        # so one is added when writing original city indices.
        tour_array = np.empty(city_count, dtype=np.intp)
        tour_array[0] = 0

        subset = full_subset
        current = final_city

        for position in range(city_count - 1, 0, -1):
            tour_array[position] = current + 1

            # No predecessor is needed after writing the first visited city.
            if position > 1:
                previous = int(parents[subset, current])
                subset ^= 1 << current
                current = previous

        tour = tuple(int(city) for city in tour_array)

        # There is one start state plus one state for every pair (S, j)
        # where j belongs to the nonempty subset S.
        states = 1 + non_start_count * (
            1 << (non_start_count - 1)
        )

        return SolverResult(
            solution=tour,
            cost=float(tour_cost),
            exact=True,
            wall_seconds=time.perf_counter() - started,
            metadata={
                "algorithm": "held-karp",
                "states": states,
            },
        )








