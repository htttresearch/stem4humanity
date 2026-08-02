"""Stage 1 strong non-learned Euclidean TSP solver.

Portfolio structure: mixed geometric candidates, diversified seeds
(multi-fragment, hull-regret, nearest neighbour), a candidate-restricted
2-opt / Or-opt / 3-opt local search, double-bridge iterated local
search, an elite pool whose edges are added back to the candidate graph,
and candidate widening on stagnation.  The search starts with a cheap
2-opt sweep over every seed, then cycles one ILS chain per seed ordered
by sweep quality, with 3-opt passes every ``deep_every`` iterations and
as each chain's final pass.  Every search loop is deadline-aware so the
solver can be compared at matched wall time.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from algofinder.problems.tsp import (
    EuclideanTravellingSalespersonProblem,
)
from algofinder.solvers.base import Solver, SolverResult, register_solver
from algofinder.solvers.tsp.candidates import (
    add_tour_edges,
    build_mixed_candidates,
    degree_stats,
    provenance_summary,
)
from algofinder.solvers.tsp.constructors import (
    hull_regret_tour,
    multi_fragment_tour,
    nearest_neighbor_tour,
)
from algofinder.solvers.tsp.euclidean_geometry import candidate_edge_count
from algofinder.solvers.tsp.move_engine import (
    Tour,
    double_bridge,
    improve_tour,
    two_opt_improve,
)

CONSTRUCTORS = {
    "multi-fragment": lambda problem, candidate_lists, rng: multi_fragment_tour(
        problem, candidate_lists, rng
    ),
    "hull-regret": lambda problem, candidate_lists, rng: hull_regret_tour(
        problem, candidate_lists
    ),
    "nn": lambda problem, candidate_lists, rng: nearest_neighbor_tour(problem, rng),
}


def _pool_insert(pool: list[Tour], tour: Tour, pool_size: int) -> None:
    """Insert ``tour`` into the elite pool, keeping the best distinct costs."""
    for candidate in pool:
        if candidate.cost == tour.cost:
            return
    if len(pool) < pool_size:
        pool.append(tour)
    elif tour.cost < pool[-1].cost:
        pool[-1] = tour
    pool.sort(key=lambda candidate: candidate.cost)


def _deadline_hit(deadline: float | None) -> bool:
    return deadline is not None and perf_counter() > deadline


@register_solver
class StrongEuclideanIlsSolver(Solver):
    id = "strong-ils-euclidean"
    display = "Mixed candidates + seeds + 2/3-opt/Or-opt ILS + elite union"
    tags = frozenset({"heuristic"})
    applies_to = frozenset({"tsp:euclidean", "tsp:clustered"})

    def __init__(
        self,
        *,
        nearest_neighbors: int = 12,
        sector_count: int = 8,
        per_sector: int = 1,
        max_degree: int = 30,
        seeds: list[str] | None = None,
        max_ils_iterations: int = 0,
        stagnation_exit: int = 12,
        deep_every: int = 8,
        kick_block: int = 6,
        min_handoff_budget: float = 0.5,
        max_2opt_moves: int = 0,
        max_segment_length: int = 3,
        three_opt_passes: int = 2,
        three_opt_slack: int = 1,
        elite_pool_size: int = 8,
        widen_factor: int = 2,
        widen_max_degree: int = 60,
        widen_after: int = 4,
        seed: int = 0,
    ) -> None:
        self.nearest_neighbors = nearest_neighbors
        self.sector_count = sector_count
        self.per_sector = per_sector
        self.max_degree = max_degree
        self.seeds = list(seeds) if seeds else ["multi-fragment", "hull-regret", "nn"]
        if not self.seeds or not all(name in CONSTRUCTORS for name in self.seeds):
            raise ValueError(f"unknown seed constructors: {self.seeds}")
        self.max_ils_iterations = max_ils_iterations
        self.stagnation_exit = stagnation_exit
        self.deep_every = deep_every
        self.kick_block = kick_block
        self.min_handoff_budget = min_handoff_budget
        self.max_2opt_moves = max_2opt_moves
        self.max_segment_length = max_segment_length
        self.three_opt_passes = three_opt_passes
        self.three_opt_slack = three_opt_slack
        self.elite_pool_size = elite_pool_size
        self.widen_factor = widen_factor
        self.widen_max_degree = widen_max_degree
        self.widen_after = widen_after
        self.seed = seed

    def solve(
        self,
        problem: EuclideanTravellingSalespersonProblem,
        *,
        budget_seconds: float | None = None,
        context: object | None = None,
    ) -> SolverResult:
        started = perf_counter()
        deadline = (
            started + budget_seconds
            if budget_seconds is not None and budget_seconds > 0
            else None
        )
        city_count = problem.city_count
        distances = problem.distances
        rng = np.random.default_rng(self.seed)

        preprocessing_started = perf_counter()
        candidate_lists, provenance = build_mixed_candidates(
            problem,
            nearest_neighbors=self.nearest_neighbors,
            sector_count=self.sector_count,
            per_sector=self.per_sector,
            max_degree=self.max_degree,
        )
        current_degree = self.max_degree
        preprocessing_seconds = perf_counter() - preprocessing_started

        search_started = perf_counter()
        pool: list[Tour] = []
        total_counters = {
            "two_opt": {"moves": 0, "evaluations": 0},
            "or_opt": {"moves": 0, "evaluations": 0},
            "three_opt": {"moves": 0, "evaluations": 0},
        }
        il_iterations = 0
        widening_events = 0
        seed_costs: dict[str, float] = {}
        cycles = 0

        # Phase 1: cheap multi-start sweep.  Construct every seed and
        # 2-opt-improve it so the incumbent is already competitive at any
        # deadline, then order the seeds by improved quality.
        prepared: list[tuple[float, Tour]] = []
        for seed_name in self.seeds:
            if _deadline_hit(deadline):
                break
            constructor = CONSTRUCTORS[seed_name]
            seed_tour = constructor(problem, candidate_lists, rng)
            seed_costs[seed_name] = float(problem.objective_value(seed_tour))
            tour = Tour(seed_tour, distances)
            counters = two_opt_improve(
                problem,
                tour,
                candidate_lists,
                max_moves=self.max_2opt_moves,
                deadline=deadline,
            )
            _accumulate(total_counters, {"two_opt": counters})
            _pool_insert(pool, tour, self.elite_pool_size)
            prepared.append((tour.cost, tour))
        prepared.sort(key=lambda item: item[0])
        best_tour: Tour | None = prepared[0][1] if prepared else None
        trial_count = len(prepared)

        # Phase 2: one ILS chain per seed in sweep-quality order, single
        # pass.  A chain leaves its stagnation exit only when enough
        # budget remains for the next chain to be meaningful
        # (``min_handoff_budget`` of the original budget); otherwise it
        # keeps grinding until the deadline, so tight budgets are spent
        # in the strongest basin instead of cold-starting weaker ones.
        # Each chain reseeds its kick rng every ``kick_block`` iterations
        # so one unlucky kick sequence cannot stall the whole chain
        # (chained-restart style diversity), and the sweep draws never
        # shift a chain's kick sequence.  Kicks use cheap 2-opt/Or-opt
        # passes; 3-opt runs every ``deep_every`` iterations and once
        # more as each chain's final pass.
        budget_span = (
            budget_seconds if budget_seconds is not None and budget_seconds > 0 else None
        )
        for chain_index, (_, tour) in enumerate(prepared):
            if _deadline_hit(deadline):
                break
            chain_best = tour
            stagnation = 0
            iterations = 0
            while not _deadline_hit(deadline):
                if self.max_ils_iterations and iterations >= self.max_ils_iterations:
                    break
                kick_block = iterations // self.kick_block
                chain_rng = np.random.default_rng(
                    self.seed + 4096 * (chain_index + 1) + 7919 * kick_block
                )
                kicked = Tour(double_bridge(tour.order, chain_rng), distances)
                deep = (iterations % self.deep_every) == (self.deep_every - 1)
                counters = improve_tour(
                    problem,
                    kicked,
                    candidate_lists,
                    max_2opt_moves=self.max_2opt_moves,
                    max_segment_length=self.max_segment_length,
                    three_opt_passes=self.three_opt_passes,
                    three_opt_slack=self.three_opt_slack,
                    deep=deep,
                    deadline=deadline,
                )
                _accumulate(total_counters, counters)
                tour = kicked
                il_iterations += 1
                iterations += 1
                _pool_insert(pool, tour, self.elite_pool_size)
                if tour.cost < chain_best.cost:
                    chain_best = tour
                if tour.cost < best_tour.cost:
                    best_tour = tour
                    stagnation = 0
                else:
                    stagnation += 1
                if stagnation >= self.stagnation_exit:
                    if budget_span is not None:
                        remaining = deadline - perf_counter()
                        if remaining < self.min_handoff_budget * budget_span:
                            stagnation = 0
                            continue
                    break
                if stagnation >= self.widen_after and current_degree < self.widen_max_degree:
                    current_degree = min(
                        current_degree * self.widen_factor, self.widen_max_degree
                    )
                    candidate_lists, provenance = build_mixed_candidates(
                        problem,
                        nearest_neighbors=self.nearest_neighbors,
                        sector_count=self.sector_count,
                        per_sector=self.per_sector,
                        max_degree=current_degree,
                    )
                    if best_tour is not None:
                        add_tour_edges(candidate_lists, best_tour.order, provenance)
                    for elite in pool:
                        add_tour_edges(candidate_lists, elite.order, provenance)
                    widening_events += 1
                    stagnation = 0
            if not _deadline_hit(deadline):
                counters = improve_tour(
                    problem,
                    chain_best,
                    candidate_lists,
                    max_2opt_moves=self.max_2opt_moves,
                    max_segment_length=self.max_segment_length,
                    three_opt_passes=self.three_opt_passes,
                    three_opt_slack=self.three_opt_slack,
                    deep=True,
                    deadline=deadline,
                )
                _accumulate(total_counters, counters)
                _pool_insert(pool, chain_best, self.elite_pool_size)
                if chain_best.cost < best_tour.cost:
                    best_tour = chain_best
            add_tour_edges(candidate_lists, chain_best.order, provenance)
            for elite in pool:
                add_tour_edges(candidate_lists, elite.order, provenance)
        search_seconds = perf_counter() - search_started

        if best_tour is None:
            raise RuntimeError("strong ILS produced no tour")

        return SolverResult(
            solution=tuple(int(city) for city in best_tour.order),
            cost=float(best_tour.cost),
            exact=False,
            wall_seconds=perf_counter() - started,
            metadata={
                "algorithm": "strong-ils-euclidean",
                "honored_budget_seconds": (
                    budget_seconds if budget_seconds is not None and budget_seconds > 0 else 0
                ),
                "preprocessing_seconds": preprocessing_seconds,
                "search_seconds": search_seconds,
                "candidate_edges": candidate_edge_count(candidate_lists),
                "candidate_degree": degree_stats(candidate_lists),
                "provenance": provenance_summary(provenance),
                "seed_costs": seed_costs,
                "trial_count": trial_count,
                "ils_iterations": il_iterations,
                "ils_cycles": cycles,
                "widening_events": widening_events,
                "widened_degree": current_degree,
                "elite_pool_size": len(pool),
                **flatten_counters(total_counters),
            },
        )


def _accumulate(total: dict, stats: dict) -> None:
    """Add per-phase move/evaluation counters into the running total."""
    for phase, counters in stats.items():
        for key, value in counters.items():
            total[phase][key] += value


def flatten_counters(total: dict) -> dict:
    """Flatten the per-phase counters into metadata keys."""
    flat: dict = {}
    for phase, counters in total.items():
        flat[f"{phase}_moves"] = counters["moves"]
        flat[f"{phase}_evaluations"] = counters["evaluations"]
    return flat


__all__ = ["StrongEuclideanIlsSolver"]
