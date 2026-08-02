"""Train and evaluate the supervised Euclidean TSP candidate-edge ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from src.data.euclidean_tsp import generate_euclidean_dataset
from src.ml.candidate_ranker import (
    CandidateEdgeRanker,
    build_candidate_training_data,
    tour_edge_set,
)
from src.progress import ProgressTracker, log, log_phase
from src.solvers.euclidean_geometry import build_geometric_candidate_lists
from src.solvers.held_karp import HeldKarpSolver


def _validation_recall(
    ranker: CandidateEdgeRanker,
    problems: list,
    exact_solver: HeldKarpSolver,
    *,
    nearest_neighbors: int,
    angular_sectors: int,
    top_k: int,
    verbose: bool,
) -> float:
    retained = 0
    total = 0
    progress = (
        ProgressTracker(
            len(problems),
            "validation",
            report_every=max(1, len(problems) // 10),
        )
        if verbose
        else None
    )
    for problem in problems:
        candidates = build_geometric_candidate_lists(
            problem,
            nearest_neighbors=nearest_neighbors,
            angular_sectors=angular_sectors,
        )
        scores = ranker.score_candidate_edges(problem, candidates)
        selected = ranker.select_top_candidates(
            problem,
            candidates,
            scores,
            top_k=top_k,
        )
        selected_edges = {
            tuple(sorted((city, neighbor)))
            for city, neighbors in enumerate(selected)
            for neighbor in neighbors
        }
        optimum = tour_edge_set(exact_solver.solve(problem).tour)
        instance_retained = len(selected_edges & optimum)
        retained += instance_retained
        total += len(optimum)
        if progress is not None:
            progress.step(
                f"{problem.name} recall={instance_retained}/{len(optimum)}"
            )
    if progress is not None:
        progress.finish(f"aggregate_recall={retained / total:.3f}")
    return retained / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a learned candidate-edge ranker for 2D Euclidean TSP."
    )
    parser.add_argument("--output", type=Path, required=True, help="Model .joblib path")
    parser.add_argument("--instances", type=int, default=60)
    parser.add_argument("--validation-instances", type=int, default=18)
    parser.add_argument("--city-count", type=int, default=14)
    parser.add_argument("--nearest-neighbors", type=int, default=8)
    parser.add_argument("--angular-sectors", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--trees", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress logging (sklearn tree progress still prints).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet
    pipeline_started = perf_counter()

    if verbose:
        log_phase("Euclidean TSP ranker training")
        log(
            "config: "
            f"train_instances={args.instances} "
            f"validation_instances={args.validation_instances} "
            f"city_count={args.city_count} "
            f"trees={args.trees} "
            f"top_k={args.top_k} "
            f"seed={args.seed}"
        )

    exact_solver = HeldKarpSolver(max_cities=args.city_count)

    if verbose:
        log_phase("Generating training instances")
    train_started = perf_counter()
    train_records = generate_euclidean_dataset(
        args.instances,
        args.city_count,
        seed=args.seed,
    )
    if verbose:
        log(
            f"generated {len(train_records)} training instances "
            f"in {perf_counter() - train_started:.1f}s"
        )

    if verbose:
        log_phase("Generating validation instances")
    validation_started = perf_counter()
    validation_records = generate_euclidean_dataset(
        args.validation_instances,
        args.city_count,
        seed=args.seed + 1,
    )
    if verbose:
        log(
            f"generated {len(validation_records)} validation instances "
            f"in {perf_counter() - validation_started:.1f}s"
        )

    if verbose:
        log_phase("Building exact labels and candidate features")
    training_data = build_candidate_training_data(
        [record.problem for record in train_records],
        exact_solver=exact_solver,
        nearest_neighbors=args.nearest_neighbors,
        angular_sectors=args.angular_sectors,
        verbose=verbose,
    )

    if verbose:
        log_phase("Training ranker")
    ranker = CandidateEdgeRanker(
        n_estimators=args.trees,
        random_state=args.seed,
    ).fit(
        training_data,
        metadata={
            "city_count": args.city_count,
            "nearest_neighbors": args.nearest_neighbors,
            "angular_sectors": args.angular_sectors,
            "seed": args.seed,
        },
        verbose=verbose,
    )

    if verbose:
        log_phase("Evaluating on validation set")
    validation_recall = _validation_recall(
        ranker,
        [record.problem for record in validation_records],
        exact_solver,
        nearest_neighbors=args.nearest_neighbors,
        angular_sectors=args.angular_sectors,
        top_k=args.top_k,
        verbose=verbose,
    )

    if verbose:
        log_phase("Saving artifacts")
    ranker.save(args.output)
    report = {
        **ranker.metadata,
        "validation_instances": args.validation_instances,
        "validation_optimal_edge_recall_at_k": validation_recall,
        "top_k": args.top_k,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if verbose:
        log(f"saved model to {args.output}")
        log(f"saved report to {report_path}")
        log(f"total pipeline time: {perf_counter() - pipeline_started:.1f}s")

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
