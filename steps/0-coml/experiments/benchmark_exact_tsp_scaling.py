"""Measure assignment branch-and-bound exact TSP wall-clock time."""

from __future__ import annotations

import argparse
import csv
import gc
import resource
import sys
from pathlib import Path
from time import perf_counter

from src.data.euclidean_tsp import EUCLIDEAN_FAMILIES, generate_euclidean_instance
from src.progress import log, log_phase
from src.solvers.exact_tsp_solver import AssignmentBranchAndBoundTSPSolver


def _peak_rss_mib() -> float:
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak_kib / (1024 * 1024)
    return peak_kib / 1024


def _format_memory(mib: float) -> str:
    if mib < 1.0:
        return f"{mib * 1024:.1f} KiB"
    return f"{mib:.2f} MiB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/exact_tsp_scaling.csv"),
    )
    parser.add_argument("--min-cities", type=int, default=3)
    parser.add_argument("--max-cities", type=int, default=20)
    parser.add_argument(
        "--family",
        choices=[*EUCLIDEAN_FAMILIES, "mixed"],
        default="mixed",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--nearest-neighbour-starts",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--node-two-opt-passes",
        type=int,
        default=4,
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet
    if args.min_cities < 3:
        raise SystemExit("--min-cities must be at least 3")
    if args.max_cities < args.min_cities:
        raise SystemExit("--max-cities must be >= --min-cities")

    solver = AssignmentBranchAndBoundTSPSolver(
        nearest_neighbour_starts=args.nearest_neighbour_starts,
        node_two_opt_passes=args.node_two_opt_passes,
    )
    rows: list[dict[str, object]] = []

    if verbose:
        log_phase("Assignment branch-and-bound TSP scaling benchmark")
        log(
            "config: "
            f"sizes={args.min_cities}..{args.max_cities} "
            f"family={args.family} "
            f"seed={args.seed} "
            f"nn_starts={args.nearest_neighbour_starts} "
            f"output={args.output}"
        )

    for city_count in range(args.min_cities, args.max_cities + 1):
        instance = generate_euclidean_instance(
            city_count,
            args.family,
            seed=args.seed + city_count,
            name=f"exact-tsp-scaling-{city_count}",
        )
        gc.collect()
        rss_before_mib = _peak_rss_mib()
        started = perf_counter()
        result = solver.solve(instance.problem)
        elapsed = perf_counter() - started
        peak_memory_mib = _peak_rss_mib() - rss_before_mib
        metadata = result.metadata

        row = {
            "city_count": city_count,
            "instance_name": instance.problem.name,
            "family": instance.family,
            "seed": args.seed + city_count,
            "cost": result.cost,
            "wall_time_seconds": elapsed,
            "solver_runtime_seconds": metadata.get("runtime_seconds"),
            "peak_memory_mib": peak_memory_mib,
            "nodes": metadata.get("nodes"),
            "nodes_generated": metadata.get("nodes_generated"),
            "assignment_solves": metadata.get("assignment_solves"),
            "root_lower_bound": metadata.get("root_lower_bound"),
            "exact": metadata.get("exact", True),
        }
        rows.append(row)

        if verbose:
            log(
                f"n={city_count:2d}  "
                f"time={elapsed:.6f}s  "
                f"memory={_format_memory(peak_memory_mib)}  "
                f"cost={result.cost:.6f}  "
                f"nodes={row['nodes']}  "
                f"assignments={row['assignment_solves']}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log_phase("Summary")
    for row in rows:
        city_count = int(row["city_count"])
        elapsed = float(row["wall_time_seconds"])
        peak_memory_mib = float(row["peak_memory_mib"])
        log(
            f"n={city_count:2d}  "
            f"time={elapsed:.6f}s  "
            f"memory={_format_memory(peak_memory_mib)}  "
            f"nodes={row['nodes']}"
        )

    total_time = sum(float(row["wall_time_seconds"]) for row in rows)
    log(f"saved {len(rows)} rows to {args.output}")
    log(f"total solve time: {total_time:.3f}s")


if __name__ == "__main__":
    main()
