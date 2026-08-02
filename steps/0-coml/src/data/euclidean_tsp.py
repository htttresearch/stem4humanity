"""Deterministic synthetic distributions for two-dimensional Euclidean TSP."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from src.problems.travelling_salesperson_problem import (
    EuclideanTravellingSalespersonProblem,
)

EuclideanFamily = Literal["uniform", "clustered", "corridor"]
EUCLIDEAN_FAMILIES: tuple[EuclideanFamily, ...] = (
    "uniform",
    "clustered",
    "corridor",
)


@dataclass(frozen=True)
class GeneratedEuclideanInstance:
    """An instance paired with the distribution that generated it."""

    problem: EuclideanTravellingSalespersonProblem
    family: EuclideanFamily
    seed: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.problem.name,
            "family": self.family,
            "seed": self.seed,
            "points": self.problem.points.tolist(),
        }

    @classmethod
    def from_mapping(cls, mapping: dict[str, object]) -> "GeneratedEuclideanInstance":
        family = str(mapping["family"])
        if family not in EUCLIDEAN_FAMILIES:
            raise ValueError(f"unknown Euclidean family: {family}")
        return cls(
            problem=EuclideanTravellingSalespersonProblem(
                name=str(mapping["name"]),
                points=np.asarray(mapping["points"], dtype=float),
            ),
            family=family,  # type: ignore[arg-type]
            seed=int(mapping["seed"]),
        )


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
    """Generate a long, noisy, gently curved spatial corridor."""
    longitudinal = rng.uniform(-1.0, 1.0, size=city_count)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    centerline = 0.15 * np.sin(2.5 * longitudinal + phase)
    transverse_noise = rng.normal(scale=rng.uniform(0.025, 0.09), size=city_count)
    return np.column_stack((longitudinal, centerline + transverse_noise))


def _augment_rigidly(
    points: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Apply a Euclidean-symmetry-preserving transform plus global scaling."""
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


def generate_euclidean_instance(
    city_count: int,
    family: EuclideanFamily | Literal["mixed"] = "mixed",
    *,
    seed: int,
    augment: bool = True,
    name: str | None = None,
) -> GeneratedEuclideanInstance:
    """Generate one reproducible 2D Euclidean TSP instance.

    ``mixed`` samples uniformly from uniform-square, clustered, and corridor
    distributions.  Augmentation uses only transformations that preserve tour
    ordering and relative tour quality.
    """
    if city_count < 3:
        raise ValueError("city_count must be at least three")
    rng = np.random.default_rng(seed)
    if family == "mixed":
        selected_family = EUCLIDEAN_FAMILIES[int(rng.integers(len(EUCLIDEAN_FAMILIES)))]
    elif family in EUCLIDEAN_FAMILIES:
        selected_family = family
    else:
        raise ValueError(f"unknown Euclidean family: {family}")

    generators = {
        "uniform": _uniform_points,
        "clustered": _clustered_points,
        "corridor": _corridor_points,
    }
    points = generators[selected_family](city_count, rng)
    if augment:
        points = _augment_rigidly(points, rng)
    problem_name = name or f"{selected_family}-{city_count}-{seed}"
    return GeneratedEuclideanInstance(
        problem=EuclideanTravellingSalespersonProblem(problem_name, points),
        family=selected_family,
        seed=seed,
    )


def generate_euclidean_dataset(
    instance_count: int,
    city_count: int | tuple[int, int],
    *,
    seed: int,
    families: Sequence[EuclideanFamily] = EUCLIDEAN_FAMILIES,
    augment: bool = True,
) -> list[GeneratedEuclideanInstance]:
    """Create a deterministic distribution-balanced collection of instances."""
    if instance_count < 1:
        raise ValueError("instance_count must be positive")
    if not families:
        raise ValueError("families cannot be empty")
    if any(family not in EUCLIDEAN_FAMILIES for family in families):
        raise ValueError("families contains an unknown distribution")

    if isinstance(city_count, tuple):
        lower, upper = city_count
        if lower < 3 or upper < lower:
            raise ValueError("city_count bounds must satisfy 3 <= lower <= upper")
    elif city_count < 3:
        raise ValueError("city_count must be at least three")

    rng = np.random.default_rng(seed)
    records: list[GeneratedEuclideanInstance] = []
    for index in range(instance_count):
        count = (
            int(rng.integers(lower, upper + 1))
            if isinstance(city_count, tuple)
            else city_count
        )
        family = families[index % len(families)]
        instance_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        records.append(
            generate_euclidean_instance(
                count,
                family,
                seed=instance_seed,
                augment=augment,
                name=f"{family}-{count}-{index:04d}",
            )
        )
    return records


def save_euclidean_dataset(
    instances: Iterable[GeneratedEuclideanInstance],
    path: str | Path,
) -> None:
    """Save a human-readable, deterministic dataset manifest."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [instance.to_mapping() for instance in instances]
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_euclidean_dataset(path: str | Path) -> list[GeneratedEuclideanInstance]:
    """Load a dataset written by :func:`save_euclidean_dataset`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("dataset payload must be a list of instances")
    return [GeneratedEuclideanInstance.from_mapping(item) for item in payload]


__all__ = [
    "EUCLIDEAN_FAMILIES",
    "EuclideanFamily",
    "GeneratedEuclideanInstance",
    "generate_euclidean_dataset",
    "generate_euclidean_instance",
    "load_euclidean_dataset",
    "save_euclidean_dataset",
]
