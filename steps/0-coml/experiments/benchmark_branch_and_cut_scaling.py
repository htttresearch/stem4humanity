"""Measure branch-and-cut TSP wall-clock time from small n up to a limit."""

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
from src.solvers.branch_and_cut_tsp import BranchAndCutTSPSolver


def _peak_rss_mib() -> float:
    """Return the process peak RSS so far, in MiB."""
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
        default=Path("artifacts/results/branch_and_cut_scaling.csv"),
        help="CSV output path (default: artifacts/results/branch_and_cut_scaling.csv)",
    )
    parser.add_argument(
        "--min-cities",
        type=int,
        default=3,
        help="smallest instance size to solve (default: 3)",
    )
    parser.add_argument(
        "--max-cities",
        type=int,
        default=20,
        help="largest instance size to solve (default: 20)",
    )
    parser.add_argument(
        "--family",
        choices=[*EUCLIDEAN_FAMILIES, "mixed"],
        default="mixed",
        help="instance generator family (default: mixed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="base seed; each size n uses seed + n (default: 7)",
    )
    parser.add_argument(
        "--solver-seed",
        type=int,
        default=0,
        help="Gurobi seed passed to BranchAndCutTSPSolver (default: 0)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Gurobi thread count (default: Gurobi default)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="optional per-instance Gurobi time limit in seconds",
    )
    parser.add_argument(
        "--log-gurobi",
        action="store_true",
        help="enable Gurobi console logging",
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

    solver = BranchAndCutTSPSolver(
        seed=args.solver_seed,
        threads=args.threads,
        time_limit=args.time_limit,
        log_output=args.log_gurobi,
    )
    city_counts = range(args.min_cities, args.max_cities + 1)
    rows: list[dict[str, object]] = []

    if verbose:
        log_phase("Branch-and-cut TSP scaling benchmark")
        log(
            "config: "
            f"sizes={args.min_cities}..{args.max_cities} "
            f"family={args.family} "
            f"seed={args.seed} "
            f"solver_seed={args.solver_seed} "
            f"threads={args.threads} "
            f"time_limit={args.time_limit} "
            f"output={args.output}"
        )

    for city_count in city_counts:
        instance = generate_euclidean_instance(
            city_count,
            args.family,
            seed=args.seed + city_count,
            name=f"branch-and-cut-scaling-{city_count}",
        )
        gc.collect()
        rss_before_mib = _peak_rss_mib()
        started = perf_counter()
        try:
            result = solver.solve(instance.problem)
        except ImportError as error:
            raise SystemExit(str(error)) from error
        except RuntimeError as error:
            elapsed = perf_counter() - started
            peak_memory_mib = _peak_rss_mib() - rss_before_mib
            row = {
                "city_count": city_count,
                "instance_name": instance.problem.name,
                "family": instance.family,
                "seed": args.seed + city_count,
                "status": "error",
                "error": str(error),
                "wall_time_seconds": elapsed,
                "peak_memory_mib": peak_memory_mib,
            }
            rows.append(row)
            if verbose:
                log(
                    f"n={city_count:2d}  FAILED after {elapsed:.3f}s  "
                    f"memory={_format_memory(peak_memory_mib)}  "
                    f"error={error}"
                )
            continue

        elapsed = perf_counter() - started
        peak_memory_mib = _peak_rss_mib() - rss_before_mib
        metadata = result.metadata

        row = {
            "city_count": city_count,
            "instance_name": instance.problem.name,
            "family": instance.family,
            "seed": args.seed + city_count,
            "status": "ok",
            "cost": result.cost,
            "wall_time_seconds": elapsed,
            "peak_memory_mib": peak_memory_mib,
            "gurobi_runtime_seconds": metadata.get("runtime_seconds"),
            "nodes": metadata.get("nodes"),
            "lazy_subtour_cuts": metadata.get("lazy_subtour_cuts"),
            "fractional_subtour_cuts": metadata.get("fractional_subtour_cuts"),
            "initial_tour_cost": metadata.get("initial_tour_cost"),
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
                f"lazy={row['lazy_subtour_cuts']}  "
                f"fractional={row['fractional_subtour_cuts']}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "city_count",
        "instance_name",
        "family",
        "seed",
        "status",
        "cost",
        "wall_time_seconds",
        "peak_memory_mib",
        "gurobi_runtime_seconds",
        "nodes",
        "lazy_subtour_cuts",
        "fractional_subtour_cuts",
        "initial_tour_cost",
        "exact",
        "error",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log_phase("Summary")
    for row in rows:
        city_count = int(row["city_count"])
        elapsed = float(row["wall_time_seconds"])
        peak_memory_mib = float(row["peak_memory_mib"])
        status = row.get("status", "ok")
        if status == "ok":
            log(
                f"n={city_count:2d}  "
                f"time={elapsed:.6f}s  "
                f"memory={_format_memory(peak_memory_mib)}  "
                f"nodes={row.get('nodes')}"
            )
        else:
            log(
                f"n={city_count:2d}  FAILED  "
                f"time={elapsed:.6f}s  "
                f"memory={_format_memory(peak_memory_mib)}"
            )

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    total_time = sum(float(row["wall_time_seconds"]) for row in rows)
    log(f"saved {len(rows)} rows to {args.output}")
    log(
        f"total wall time: {total_time:.3f}s "
        f"({len(ok_rows)} ok, {len(rows) - len(ok_rows)} failed)"
    )


if __name__ == "__main__":
    main()
