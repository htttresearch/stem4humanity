"""Generate a reproducible JSON dataset of Euclidean TSP instances."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from src.data.euclidean_tsp import (
    EUCLIDEAN_FAMILIES,
    generate_euclidean_dataset,
    save_euclidean_dataset,
)
from src.progress import log, log_phase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=30)
    parser.add_argument("--city-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=EUCLIDEAN_FAMILIES,
        default=list(EUCLIDEAN_FAMILIES),
    )
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet
    started = perf_counter()

    if verbose:
        log_phase("Generating Euclidean TSP dataset")
        log(
            "config: "
            f"instances={args.instances} "
            f"city_count={args.city_count} "
            f"families={','.join(args.families)} "
            f"augment={not args.no_augment} "
            f"seed={args.seed}"
        )

    instances = generate_euclidean_dataset(
        args.instances,
        args.city_count,
        seed=args.seed,
        families=tuple(args.families),
        augment=not args.no_augment,
    )

    if verbose:
        families = sorted({instance.family for instance in instances})
        log(
            f"generated {len(instances)} instances "
            f"({', '.join(families)}) in {perf_counter() - started:.1f}s"
        )
        log_phase("Writing dataset")
    save_euclidean_dataset(instances, args.output)

    if verbose:
        log(f"saved {len(instances)} instances to {args.output}")
        log(f"total time: {perf_counter() - started:.1f}s")
    print(f"wrote {len(instances)} instances to {args.output}")


if __name__ == "__main__":
    main()
