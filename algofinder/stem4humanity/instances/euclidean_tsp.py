"""Deterministic synthetic 2D Euclidean TSP instance generators.

Port of the coml distribution code, emitting manifest ``Instance`` records.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from stem4humanity.instances.base import annotate_best_known, save_manifest
from stem4humanity.problems.base import Instance
from stem4humanity.problems.registry import get_problem
from stem4humanity.solvers.base import all_solvers
import stem4humanity.solvers.registry  # noqa: F401  (run solver registration)

EuclideanFamily = Literal["uniform", "clustered", "corridor", "mixed"]
EUCLIDEAN_FAMILIES: tuple[str, ...] = ("uniform", "clustered", "corridor")


def _uniform_points(city_count: int, rng: np.random.Generator) -> NDArray[np.float64]:
    return rng.random((city_count, 2))


def _clustered_points(city_count: int, rng: np.random.Generator) -> NDArray[np.float64]:
    cluster_count = min(city_count, int(rng.integers(2, min(5, city_count) + 1)))
    centers: list[NDArray[np.float64]] = []
    while len(centers) < cluster_count:
        candidate = rng.uniform(0.15, 0.85, size=2)
        if all(np.linalg.norm(candidate - center) >= 0.22 for center in centers):
            centers.append(candidate)

    assignments = rng.integers(0, cluster_count, size=city_count)
    spreads = rng.uniform(0.025, 0.085, size=cluster_count)
    points = np.empty((city_count, 2), dtype=float)
    for cluster_index, center in enumerate(centers):
        mask = assignments == cluster_index
        points[mask] = center + rng.normal(
            scale=spreads[cluster_index],
            size=(int(mask.sum()), 2),
        )
    return points


def _corridor_points(city_count: int, rng: np.random.Generator) -> NDArray[np.float64]:
    longitudinal = rng.uniform(-1.0, 1.0, size=city_count)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    centerline = 0.15 * np.sin(2.5 * longitudinal + phase)
    transverse_noise = rng.normal(scale=rng.uniform(0.025, 0.09), size=city_count)
    return np.column_stack((longitudinal, centerline + transverse_noise))


def _augment_rigidly(
    points: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    centered = points - points.mean(axis=0, keepdims=True)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=float,
    )
    transformed = centered @ rotation.T
    if rng.random() < 0.5:
        transformed[:, 0] *= -1.0
    scale = rng.uniform(0.75, 1.5)
    translation = rng.uniform(-1.0, 1.0, size=2)
    return transformed * scale + translation


def generate_euclidean_instances(
    city_count: int | tuple[int, int],
    family: EuclideanFamily = "mixed",
    *,
    count: int,
    seed: int,
    split: str = "test",
    problem_id: str = "tsp",
) -> list[Instance]:
    """Generate ``count`` reproducible Euclidean TSP manifest instances."""
    if isinstance(city_count, tuple):
        lo, hi = city_count
        if lo > hi:
            raise ValueError("city_count tuple must be ascending")
    else:
        lo = hi = city_count

    rng = np.random.default_rng(seed)
    records: list[Instance] = []
    for index in range(count):
        n = int(rng.integers(lo, hi + 1)) if lo != hi else lo
        if n < 3:
            raise ValueError("city_count must be at least three")
        if family == "mixed":
            selected_family = EUCLIDEAN_FAMILIES[
                int(rng.integers(len(EUCLIDEAN_FAMILIES)))
            ]
        elif family in EUCLIDEAN_FAMILIES:
            selected_family = family
        else:
            raise ValueError(f"unknown Euclidean family: {family}")

        generators = {
            "uniform": _uniform_points,
            "clustered": _clustered_points,
            "corridor": _corridor_points,
        }
        points = generators[selected_family](n, rng)
        points = _augment_rigidly(points, rng)

        subproblem = "clustered" if selected_family == "clustered" else "euclidean"
        name = f"{problem_id}:{subproblem}:{selected_family}-n{n}-s{seed}-i{index}"
        records.append(
            Instance(
                name=name,
                problem=problem_id,
                subproblem=subproblem,
                family=selected_family,
                seed=seed,
                data={"points": points.tolist()},
                best_known=None,
                split=split,
            )
        )
    return records


def generate_tsp_manifests(
    output_dir: str = "data/instances",
    *,
    opt_limit: int = 16,
) -> None:
    """Generate and save the full TSP instance set with exact best-known values."""
    test_sizes: list[tuple[int, int]] = [
        (9, 13),
        (18, 24),
        (38, 46),
        (95, 105),
    ]
    train_sizes: list[tuple[int, int]] = [(11, 15)]

    all_instances: list[Instance] = []
    for sizes in test_sizes:
        all_instances.extend(
            generate_euclidean_instances(
                sizes, "mixed", count=6, seed=1101, split="test"
            )
        )
    for sizes in train_sizes:
        all_instances.extend(
            generate_euclidean_instances(
                sizes, "mixed", count=12, seed=2101, split="train"
            )
        )

    small = [i for i in all_instances if len(i.data["points"]) <= opt_limit]
    problem = get_problem("tsp")
    exact = next(s for s in all_solvers().values() if s.id == "held-karp")
    small = annotate_best_known(small, exact, problem)
    annotated = {instance.name: instance for instance in small}
    all_instances = [
        annotated.get(instance.name, instance) for instance in all_instances
    ]

    save_manifest(
        all_instances,
        f"{output_dir}/tsp_manifests.json",
    )
