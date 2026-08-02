"""Hand-designed geometry-aware local search for Euclidean TSP."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from src.problems.travelling_salesperson_problem import (
    EuclideanTravellingSalespersonProblem,
)
from src.solvers.base_solver import BaseSolver, SolverResult
from src.solvers.euclidean_geometry import (
    build_geometric_candidate_lists,
    candidate_edge_count,
    regret_insertion_tour,
)


@dataclass(frozen=True)
class TwoOptSearchResult:
    """The output and work counters from candidate-restricted 2-opt."""

    tour: tuple[int, ...]
    cost: float
    accepted_moves: int
    move_evaluations: int


def improve_with_candidate_two_opt(
    problem: EuclideanTravellingSalespersonProblem,
    initial_tour: tuple[int, ...],
    candidate_lists: list[set[int]],
    *,
    max_iterations: int,
) -> TwoOptSearchResult:
    """Use first-improvement 2-opt while inspecting only candidate additions."""
    if not problem.verify(initial_tour):
        raise ValueError("initial_tour is not feasible for the problem")
    if len(candidate_lists) != problem.city_count:
        raise ValueError("candidate_lists must contain one set per city")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    tour = list(initial_tour)
    accepted_moves = 0
    move_evaluations = 0
    city_count = problem.city_count

    while accepted_moves < max_iterations:
        positions = {city: position for position, city in enumerate(tour)}
        applied = False
        for first_edge in range(city_count):
            first_city = tour[first_edge]
            for candidate_city in sorted(candidate_lists[first_city]):
                second_edge = positions[candidate_city]
                left, right = sorted((first_edge, second_edge))
                if right - left <= 1 or (left == 0 and right == city_count - 1):
                    continue

                left_city = tour[left]
                after_left = tour[left + 1]
                right_city = tour[right]
                after_right = tour[(right + 1) % city_count]
                delta = (
                    problem.distances[left_city, right_city]
                    + problem.distances[after_left, after_right]
                    - problem.distances[left_city, after_left]
                    - problem.distances[right_city, after_right]
                )
                move_evaluations += 1
                if delta < -1e-12:
                    tour[left + 1 : right + 1] = reversed(tour[left + 1 : right + 1])
                    accepted_moves += 1
                    applied = True
                    break
            if applied:
                break
        if not applied:
            break

    result_tour = tuple(tour)
    return TwoOptSearchResult(
        tour=result_tour,
        cost=problem.objective_value(result_tour),
        accepted_moves=accepted_moves,
        move_evaluations=move_evaluations,
    )


def double_bridge_perturbation(
    tour: tuple[int, ...],
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Apply a non-local four-edge double-bridge perturbation."""
    city_count = len(tour)
    if city_count < 8:
        return tour
    cuts = sorted(int(index) for index in rng.choice(city_count, size=4, replace=False))
    rotated = list(tour[cuts[0] + 1 :]) + list(tour[: cuts[0] + 1])
    lengths = (
        cuts[1] - cuts[0],
        cuts[2] - cuts[1],
        cuts[3] - cuts[2],
        city_count - (cuts[3] - cuts[0]),
    )
    segments: list[list[int]] = []
    cursor = 0
    for length in lengths:
        segments.append(rotated[cursor : cursor + length])
        cursor += length
    return tuple(segments[0] + segments[2] + segments[1] + segments[3])


def solve_with_candidate_lists(
    problem: EuclideanTravellingSalespersonProblem,
    candidate_lists: list[set[int]],
    *,
    max_restarts: int,
    max_2opt_iterations: int,
    seed: int,
) -> SolverResult:
    """Construct, improve, and perturb a tour using supplied candidate edges."""
    initial_tour = regret_insertion_tour(problem)
    initial_cost = problem.objective_value(initial_tour)
    current = improve_with_candidate_two_opt(
        problem,
        initial_tour,
        candidate_lists,
        max_iterations=max_2opt_iterations,
    )
    best = current
    total_move_evaluations = current.move_evaluations
    total_accepted_moves = current.accepted_moves
    rng = np.random.default_rng(seed)

    for _ in range(max_restarts):
        perturbed_tour = double_bridge_perturbation(current.tour, rng)
        candidate = improve_with_candidate_two_opt(
            problem,
            perturbed_tour,
            candidate_lists,
            max_iterations=max_2opt_iterations,
        )
        total_move_evaluations += candidate.move_evaluations
        total_accepted_moves += candidate.accepted_moves
        if candidate.cost <= current.cost:
            current = candidate
        if candidate.cost < best.cost:
            best = candidate

    return SolverResult(
        tour=best.tour,
        cost=best.cost,
        metadata={
            "initial_cost": initial_cost,
            "candidate_edges": candidate_edge_count(candidate_lists),
            "move_evaluations": total_move_evaluations,
            "accepted_moves": total_accepted_moves,
            "restarts": max_restarts,
        },
    )


class EuclideanChainedTwoOptSolver(
    BaseSolver[EuclideanTravellingSalespersonProblem]
):
    """Hull-regret construction with geometric candidates, 2-opt, and kicks."""

    def __init__(
        self,
        *,
        nearest_neighbors: int = 12,
        angular_sectors: int = 8,
        max_restarts: int = 8,
        max_2opt_iterations: int = 2_000,
        seed: int = 0,
    ) -> None:
        if max_restarts < 0:
            raise ValueError("max_restarts cannot be negative")
        self.nearest_neighbors = nearest_neighbors
        self.angular_sectors = angular_sectors
        self.max_restarts = max_restarts
        self.max_2opt_iterations = max_2opt_iterations
        self.seed = seed

    def solve(
        self,
        problem: EuclideanTravellingSalespersonProblem,
    ) -> SolverResult:
        preprocessing_started = perf_counter()
        candidate_lists = build_geometric_candidate_lists(
            problem,
            nearest_neighbors=self.nearest_neighbors,
            angular_sectors=self.angular_sectors,
        )
        preprocessing_seconds = perf_counter() - preprocessing_started
        search_started = perf_counter()
        search = solve_with_candidate_lists(
            problem,
            candidate_lists,
            max_restarts=self.max_restarts,
            max_2opt_iterations=self.max_2opt_iterations,
            seed=self.seed,
        )
        search_seconds = perf_counter() - search_started
        return SolverResult(
            tour=search.tour,
            cost=search.cost,
            metadata={
                "algorithm": "euclidean-chained-2opt",
                "exact": False,
                **search.metadata,
                "preprocessing_seconds": preprocessing_seconds,
                "search_seconds": search_seconds,
            },
        )


__all__ = [
    "EuclideanChainedTwoOptSolver",
    "TwoOptSearchResult",
    "double_bridge_perturbation",
    "improve_with_candidate_two_opt",
    "solve_with_candidate_lists",
]
