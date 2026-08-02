"""Benchmark exact, manual, and learned 2D Euclidean TSP solvers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter
from typing import Any

from src.data.euclidean_tsp import generate_euclidean_dataset
from src.ml.candidate_ranker import CandidateEdgeRanker, tour_edge_set
from src.problems.travelling_salesperson_problem import (
    EuclideanTravellingSalespersonProblem,
)
from src.progress import ProgressTracker, log, log_phase
from src.solvers.base_solver import SolverResult
from src.solvers.euclidean_geometry import build_geometric_candidate_lists
from src.solvers.euclidean_local_search import (
    EuclideanChainedTwoOptSolver,
    solve_with_candidate_lists,
)
from src.solvers.held_karp import HeldKarpSolver
from src.solvers.learned_candidate_tsp import LearnedCandidateTwoOptSolver


def _edge_recall(candidate_lists: list[set[int]], reference_tour: tuple[int, ...]) -> float:
    candidates = {
        tuple(sorted((city, neighbor)))
        for city, neighbors in enumerate(candidate_lists)
        for neighbor in neighbors
    }
    reference_edges = tour_edge_set(reference_tour)
    return len(candidates & reference_edges) / len(reference_edges)


def _distance_ranked_solver(
    problem: EuclideanTravellingSalespersonProblem,
    *,
    nearest_neighbors: int,
    angular_sectors: int,
    top_k: int,
    max_restarts: int,
    max_2opt_iterations: int,
    seed: int,
) -> tuple[SolverResult, list[set[int]]]:
    candidates = build_geometric_candidate_lists(
        problem,
        nearest_neighbors=nearest_neighbors,
        angular_sectors=angular_sectors,
    )
    scores = {
        tuple(sorted((city, neighbor))): -float(problem.distances[city, neighbor])
        for city, neighbors in enumerate(candidates)
        for neighbor in neighbors
    }
    selected = CandidateEdgeRanker.select_top_candidates(
        problem,
        candidates,
        scores,
        top_k=top_k,
    )
    result = solve_with_candidate_lists(
        problem,
        selected,
        max_restarts=max_restarts,
        max_2opt_iterations=max_2opt_iterations,
        seed=seed,
    )
    return (
        SolverResult(
            tour=result.tour,
            cost=result.cost,
            metadata={"algorithm": "distance-ranked-chained-2opt", **result.metadata},
        ),
        selected,
    )


def _row(
    *,
    problem: EuclideanTravellingSalespersonProblem,
    family: str,
    solver: str,
    result: SolverResult | None,
    elapsed_seconds: float | None,
    reference_cost: float | None,
    candidate_recall: float | None,
) -> dict[str, Any]:
    if result is None:
        return {
            "instance": problem.name,
            "family": family,
            "city_count": problem.city_count,
            "solver": solver,
            "status": "skipped",
        }
    gap_percent = (
        100.0 * (result.cost - reference_cost) / reference_cost
        if reference_cost is not None
        else None
    )
    return {
        "instance": problem.name,
        "family": family,
        "city_count": problem.city_count,
        "solver": solver,
        "status": "ok",
        "cost": result.cost,
        "gap_percent": gap_percent,
        "elapsed_seconds": elapsed_seconds,
        "candidate_recall": candidate_recall,
        **result.metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--city-counts", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--instances", type=int, default=6)
    parser.add_argument("--exact-limit", type=int, default=14)
    parser.add_argument("--nearest-neighbors", type=int, default=8)
    parser.add_argument("--angular-sectors", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--max-2opt-iterations", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet
    pipeline_started = perf_counter()
    total_instances = len(args.city_counts) * args.instances

    if verbose:
        log_phase("Euclidean TSP benchmark")
        log(
            "config: "
            f"model={args.model} "
            f"city_counts={args.city_counts} "
            f"instances_per_size={args.instances} "
            f"exact_limit={args.exact_limit} "
            f"restarts={args.restarts} "
            f"max_2opt_iterations={args.max_2opt_iterations} "
            f"seed={args.seed}"
        )

    if verbose:
        log_phase("Loading model and solvers")
    load_started = perf_counter()
    ranker = CandidateEdgeRanker.load(args.model)
    exact_solver = HeldKarpSolver(max_cities=args.exact_limit)
    manual_solver = EuclideanChainedTwoOptSolver(
        nearest_neighbors=args.nearest_neighbors,
        angular_sectors=args.angular_sectors,
        max_restarts=args.restarts,
        max_2opt_iterations=args.max_2opt_iterations,
        seed=args.seed,
    )
    learned_solver = LearnedCandidateTwoOptSolver(
        ranker,
        nearest_neighbors=args.nearest_neighbors,
        angular_sectors=args.angular_sectors,
        top_k=args.top_k,
        max_restarts=args.restarts,
        max_2opt_iterations=args.max_2opt_iterations,
        seed=args.seed,
    )
    if verbose:
        log(f"loaded model and solvers in {perf_counter() - load_started:.1f}s")

    rows: list[dict[str, Any]] = []
    progress = (
        ProgressTracker(
            total_instances,
            "benchmark",
            report_every=max(1, total_instances // 20),
        )
        if verbose
        else None
    )

    for city_count in args.city_counts:
        if verbose:
            log_phase(f"City count n={city_count}")
        group_started = perf_counter()
        records = generate_euclidean_dataset(
            args.instances,
            city_count,
            seed=args.seed + city_count,
        )
        if verbose:
            log(f"generated {len(records)} instances for n={city_count}")

        for record in records:
            problem = record.problem
            instance_started = perf_counter()
            if verbose:
                log(
                    f"instance {record.problem.name} "
                    f"(family={record.family}, n={city_count})"
                )

            exact: SolverResult | None = None
            exact_elapsed: float | None = None
            if city_count <= args.exact_limit:
                started = perf_counter()
                exact = exact_solver.solve(problem)
                exact_elapsed = perf_counter() - started
                if verbose:
                    log(
                        f"  held-karp: cost={exact.cost:.6f} "
                        f"time={exact_elapsed:.3f}s"
                    )
            elif verbose:
                log("  held-karp: skipped (above exact limit)")

            started = perf_counter()
            manual = manual_solver.solve(problem)
            manual_elapsed = perf_counter() - started
            if verbose:
                log(
                    f"  manual-euclidean: cost={manual.cost:.6f} "
                    f"time={manual_elapsed:.3f}s "
                    f"moves={manual.metadata.get('move_evaluations')}"
                )

            started = perf_counter()
            distance_ranked, distance_candidates = _distance_ranked_solver(
                problem,
                nearest_neighbors=args.nearest_neighbors,
                angular_sectors=args.angular_sectors,
                top_k=args.top_k,
                max_restarts=args.restarts,
                max_2opt_iterations=args.max_2opt_iterations,
                seed=args.seed,
            )
            distance_elapsed = perf_counter() - started
            if verbose:
                log(
                    f"  distance-ranked-control: cost={distance_ranked.cost:.6f} "
                    f"time={distance_elapsed:.3f}s "
                    f"moves={distance_ranked.metadata.get('move_evaluations')}"
                )

            started = perf_counter()
            learned = learned_solver.solve(problem)
            learned_elapsed = perf_counter() - started
            if verbose:
                log(
                    f"  learned-candidate: cost={learned.cost:.6f} "
                    f"time={learned_elapsed:.3f}s "
                    f"moves={learned.metadata.get('move_evaluations')} "
                    f"model_time={learned.metadata.get('model_seconds', 0.0):.3f}s"
                )

            manual_candidates = build_geometric_candidate_lists(
                problem,
                nearest_neighbors=args.nearest_neighbors,
                angular_sectors=args.angular_sectors,
            )
            learned_candidates = build_geometric_candidate_lists(
                problem,
                nearest_neighbors=args.nearest_neighbors,
                angular_sectors=args.angular_sectors,
            )
            learned_scores = ranker.score_candidate_edges(problem, learned_candidates)
            learned_candidates = ranker.select_top_candidates(
                problem,
                learned_candidates,
                learned_scores,
                top_k=args.top_k,
            )

            reference_cost = exact.cost if exact is not None else min(
                manual.cost, distance_ranked.cost
            )
            reference_tour = (
                exact.tour
                if exact is not None
                else manual.tour
                if manual.cost <= distance_ranked.cost
                else distance_ranked.tour
            )
            if verbose:
                log(
                    f"  reference_cost={reference_cost:.6f} "
                    f"instance_total_time={perf_counter() - instance_started:.3f}s"
                )

            rows.append(
                _row(
                    problem=problem,
                    family=record.family,
                    solver="held-karp",
                    result=exact,
                    elapsed_seconds=exact_elapsed,
                    reference_cost=reference_cost if exact is not None else None,
                    candidate_recall=None,
                )
            )
            rows.extend(
                [
                    _row(
                        problem=problem,
                        family=record.family,
                        solver="manual-euclidean",
                        result=manual,
                        elapsed_seconds=manual_elapsed,
                        reference_cost=reference_cost,
                        candidate_recall=_edge_recall(
                            manual_candidates, reference_tour
                        ),
                    ),
                    _row(
                        problem=problem,
                        family=record.family,
                        solver="distance-ranked-control",
                        result=distance_ranked,
                        elapsed_seconds=distance_elapsed,
                        reference_cost=reference_cost,
                        candidate_recall=_edge_recall(
                            distance_candidates, reference_tour
                        ),
                    ),
                    _row(
                        problem=problem,
                        family=record.family,
                        solver="learned-candidate",
                        result=learned,
                        elapsed_seconds=learned_elapsed,
                        reference_cost=reference_cost,
                        candidate_recall=_edge_recall(
                            learned_candidates, reference_tour
                        ),
                    ),
                ]
            )
            if progress is not None:
                progress.step(f"n={city_count} family={record.family}")

        if verbose:
            log(
                f"finished n={city_count} group in "
                f"{perf_counter() - group_started:.1f}s"
            )

    if progress is not None:
        progress.finish()

    if verbose:
        log_phase("Writing results")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        log(f"saved {len(rows)} benchmark rows to {args.output}")
        log(f"total benchmark time: {perf_counter() - pipeline_started:.1f}s")
    print(f"wrote {len(rows)} benchmark rows to {args.output}")


if __name__ == "__main__":
    main()
