"""Synthetic data generation and persistence."""

from .euclidean_tsp import (
    GeneratedEuclideanInstance,
    generate_euclidean_dataset,
    generate_euclidean_instance,
    load_euclidean_dataset,
    save_euclidean_dataset,
)

__all__ = [
    "GeneratedEuclideanInstance",
    "generate_euclidean_dataset",
    "generate_euclidean_instance",
    "load_euclidean_dataset",
    "save_euclidean_dataset",
]
