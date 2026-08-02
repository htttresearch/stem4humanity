"""Benchmark the native C++ Held--Karp solver against Python instances."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import subprocess
from pathlib import Path
from time import perf_counter

from src.data.euclidean_tsp import EUCLIDEAN_FAMILIES, generate_euclidean_instance
from src.progress import log, log_phase
from src.solvers.held_karp import HeldKarpSolver


def _format_memory(mib: float) -> str:
    if mib < 1.0:
        return f"{mib * 1024:.1f} KiB"
    return f"{mib:.2f} MiB"


def _native_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "native" / "held_karp"


def _binary_path() -> Path:
    return _native_dir() / "held_karp_solver"


def _build_binary() -> None:
    native_dir = _native_dir()
    log(f"building {_binary_path()} ...")
    subprocess.run(
        ["make", "-C", str(native_dir), "clean", "all"],
        check=True,
    )


def _write_instance(path: Path, distances) -> None:
    city_count = distances.shape[0]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{city_count}\n")
        handle.write(
            " ".join(f"{value:.17g}" for value in distances.reshape(-1))
        )
        handle.write("\n")


def _parse_solver_output(output: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == "tour":
            parsed[key] = tuple(int(city) for city in value.split(",") if city)
        elif key == "states":
            parsed[key] = int(value)
        elif key in {"cost", "solve_seconds", "peak_rss_mib"}:
            parsed[key] = float(value)
        else:
            parsed[key] = value
    required = {"cost", "states", "solve_seconds", "peak_rss_mib", "tour"}
    missing = required - parsed.keys()
    if missing:
        raise RuntimeError(
            f"native solver output missing fields: {', '.join(sorted(missing))}"
        )
    return parsed


def _run_native_solver(instance_path: Path, binary: Path) -> dict[str, object]:
    completed = subprocess.run(
        [str(binary), "--input", str(instance_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return _parse_solver_output(completed.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/held_karp_cpp_scaling.csv"),
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
        "--build",
        action="store_true",
        help="force rebuild of the native binary before benchmarking",
    )
    parser.add_argument(
        "--verify-python",
        action="store_true",
        help="compare each native cost against the Python Held--Karp solver",
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

    binary = _binary_path()
    if args.build or not binary.exists():
        _build_binary()
    if not binary.exists():
        raise SystemExit(f"native binary not found: {binary}")

    python_solver = (
        HeldKarpSolver(max_cities=args.max_cities) if args.verify_python else None
    )
    rows: list[dict[str, object]] = []
    instance_dir = Path("artifacts/tmp/held_karp_cpp")
    instance_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        log_phase("C++ Held--Karp scaling benchmark")
        log(
            "config: "
            f"sizes={args.min_cities}..{args.max_cities} "
            f"family={args.family} "
            f"seed={args.seed} "
            f"binary={binary} "
            f"verify_python={args.verify_python} "
            f"output={args.output}"
        )

    for city_count in range(args.min_cities, args.max_cities + 1):
        instance = generate_euclidean_instance(
            city_count,
            args.family,
            seed=args.seed + city_count,
            name=f"held-karp-cpp-scaling-{city_count}",
        )
        instance_path = instance_dir / f"{city_count}.txt"
        _write_instance(instance_path, instance.problem.distances)

        gc.collect()
        started = perf_counter()
        native = _run_native_solver(instance_path, binary)
        wall_time = perf_counter() - started

        row: dict[str, object] = {
            "city_count": city_count,
            "instance_name": instance.problem.name,
            "family": instance.family,
            "seed": args.seed + city_count,
            "cost": native["cost"],
            "native_solve_seconds": native["solve_seconds"],
            "wall_time_seconds": wall_time,
            "peak_memory_mib": native["peak_rss_mib"],
            "states": native["states"],
            "exact": True,
        }

        if python_solver is not None:
            reference = python_solver.solve(instance.problem)
            row["python_cost"] = reference.cost
            row["cost_delta"] = float(native["cost"]) - reference.cost
            if not math.isclose(
                float(native["cost"]),
                reference.cost,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    f"native/Python cost mismatch at n={city_count}: "
                    f"native={native['cost']} python={reference.cost}"
                )

        rows.append(row)

        if verbose:
            suffix = ""
            if python_solver is not None:
                suffix = "  verified=ok"
            log(
                f"n={city_count:2d}  "
                f"native={float(native['solve_seconds']):.6f}s  "
                f"wall={wall_time:.6f}s  "
                f"memory={_format_memory(float(native['peak_rss_mib']))}  "
                f"cost={float(native['cost']):.6f}  "
                f"states={native['states']}{suffix}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "city_count",
        "instance_name",
        "family",
        "seed",
        "cost",
        "native_solve_seconds",
        "wall_time_seconds",
        "peak_memory_mib",
        "states",
        "exact",
        "python_cost",
        "cost_delta",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    log_phase("Summary")
    for row in rows:
        city_count = int(row["city_count"])
        native_seconds = float(row["native_solve_seconds"])
        peak_memory_mib = float(row["peak_memory_mib"])
        log(
            f"n={city_count:2d}  "
            f"native={native_seconds:.6f}s  "
            f"memory={_format_memory(peak_memory_mib)}"
        )

    total_native = sum(float(row["native_solve_seconds"]) for row in rows)
    total_wall = sum(float(row["wall_time_seconds"]) for row in rows)
    log(f"saved {len(rows)} rows to {args.output}")
    log(f"total native solve time: {total_native:.3f}s")
    log(f"total wall time (incl. IO): {total_wall:.3f}s")


if __name__ == "__main__":
    main()
