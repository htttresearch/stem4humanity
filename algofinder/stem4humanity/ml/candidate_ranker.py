"""Supervised ranking of geometrically plausible Euclidean TSP edges."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier

from stem4humanity.util.progress import ProgressTracker, log
from stem4humanity.problems.tsp import (
    EuclideanTravellingSalespersonProblem,
)
from stem4humanity.solvers.tsp.euclidean_geometry import (
    build_geometric_candidate_lists,
    convex_hull_indices,
)
from stem4humanity.solvers.tsp.held_karp import HeldKarpSolver

FEATURE_NAMES: tuple[str, ...] = (
    "relative_length",
    "min_neighbor_rank",
    "max_neighbor_rank",
    "mutual_candidate",
    "density_ratio",
    "min_density_relative_to_global",
    "max_density_relative_to_global",
    "both_on_hull",
    "one_on_hull",
    "midpoint_radius",
    "min_angular_gap",
    "max_angular_gap",
)


@dataclass(frozen=True)
class CandidateTrainingData:
    """Feature rows and labels derived from whole training instances."""

    features: NDArray[np.float64]
    labels: NDArray[np.int_]
    instance_count: int
    candidate_recall: float


def _canonical_edge(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first < second else (second, first)


def tour_edge_set(tour: Sequence[int]) -> set[tuple[int, int]]:
    """Return the undirected edge set of an implicit-cycle TSP tour."""
    return {
        _canonical_edge(int(city), int(next_city))
        for city, next_city in zip(tour, (*tour[1:], tour[0]))
    }


def _candidate_edges(candidate_lists: Sequence[set[int]]) -> list[tuple[int, int]]:
    return sorted(
        {
            _canonical_edge(city, neighbor)
            for city, neighbors in enumerate(candidate_lists)
            for neighbor in neighbors
            if city != neighbor
        }
    )


def _angular_gaps(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-directed-edge local angular isolation, normalized to [0, 1]."""
    city_count = len(points)
    gaps = np.zeros((city_count, city_count), dtype=float)
    if city_count <= 2:
        return gaps
    for city in range(city_count):
        others = np.array([other for other in range(city_count) if other != city])
        offsets = points[others] - points[city]
        angles = np.mod(np.arctan2(offsets[:, 1], offsets[:, 0]), 2.0 * np.pi)
        order = np.argsort(angles)
        ordered_others = others[order]
        ordered_angles = angles[order]
        wrapped = np.concatenate(
            ([ordered_angles[-1] - 2.0 * np.pi], ordered_angles, [ordered_angles[0] + 2.0 * np.pi])
        )
        for position, neighbor in enumerate(ordered_others, start=1):
            gaps[city, neighbor] = min(
                wrapped[position] - wrapped[position - 1],
                wrapped[position + 1] - wrapped[position],
            ) / (2.0 * np.pi)
    return gaps


def extract_candidate_features(
    problem: EuclideanTravellingSalespersonProblem,
    candidate_lists: Sequence[set[int]],
) -> tuple[list[tuple[int, int]], NDArray[np.float64]]:
    """Create scale-, translation-, and rotation-invariant candidate features."""
    if len(candidate_lists) != problem.city_count:
        raise ValueError("candidate_lists must contain one set per city")
    edges = _candidate_edges(candidate_lists)
    if not edges:
        raise ValueError("candidate_lists produced no candidate edges")

    city_count = problem.city_count
    distances = problem.distances
    neighbor_ranks = np.empty((city_count, city_count), dtype=float)
    for city in problem.nodes:
        order = np.argsort(distances[city], kind="stable")
        neighbor_ranks[city, order] = np.arange(city_count, dtype=float)
    local_density = np.empty(city_count, dtype=float)
    density_count = min(5, city_count - 1)
    for city in problem.nodes:
        local_density[city] = distances[city, np.argsort(distances[city])[1 : density_count + 1]].mean()

    global_density = max(float(local_density.mean()), 1e-12)
    centroid = problem.points.mean(axis=0)
    global_scale = max(
        float(np.linalg.norm(problem.points - centroid, axis=1).mean()),
        1e-12,
    )
    hull = set(convex_hull_indices(problem.points))
    angular_gaps = _angular_gaps(problem.points)

    rows: list[list[float]] = []
    for first, second in edges:
        length = float(distances[first, second])
        density_mean = max((local_density[first] + local_density[second]) / 2.0, 1e-12)
        first_rank = neighbor_ranks[first, second] / (city_count - 1)
        second_rank = neighbor_ranks[second, first] / (city_count - 1)
        midpoint = (problem.points[first] + problem.points[second]) / 2.0
        hull_count = int(first in hull) + int(second in hull)
        rows.append(
            [
                length / density_mean,
                min(first_rank, second_rank),
                max(first_rank, second_rank),
                float(second in candidate_lists[first] and first in candidate_lists[second]),
                min(local_density[first], local_density[second])
                / max(local_density[first], local_density[second], 1e-12),
                min(local_density[first], local_density[second]) / global_density,
                max(local_density[first], local_density[second]) / global_density,
                float(hull_count == 2),
                float(hull_count == 1),
                float(np.linalg.norm(midpoint - centroid) / global_scale),
                min(angular_gaps[first, second], angular_gaps[second, first]),
                max(angular_gaps[first, second], angular_gaps[second, first]),
            ]
        )
    return edges, np.asarray(rows, dtype=float)


def build_candidate_training_data(
    problems: Iterable[EuclideanTravellingSalespersonProblem],
    *,
    exact_solver: HeldKarpSolver,
    nearest_neighbors: int = 8,
    angular_sectors: int = 4,
    verbose: bool = True,
) -> CandidateTrainingData:
    """Use exact reference tours to label candidate edges for supervised learning."""
    problem_list = list(problems)
    if not problem_list:
        raise ValueError("at least one training problem is required")

    feature_batches: list[NDArray[np.float64]] = []
    label_batches: list[NDArray[np.int_]] = []
    optimum_edges_available = 0
    optimum_edges_retained = 0
    progress = (
        ProgressTracker(
            len(problem_list),
            "labeling",
            report_every=max(1, len(problem_list) // 20),
        )
        if verbose
        else None
    )

    for problem in problem_list:
        candidate_lists = build_geometric_candidate_lists(
            problem,
            nearest_neighbors=nearest_neighbors,
            angular_sectors=angular_sectors,
        )
        edges, features = extract_candidate_features(problem, candidate_lists)
        optimum_tour = exact_solver.solve(problem).solution
        optimum_edges = tour_edge_set(optimum_tour)
        labels = np.asarray([int(edge in optimum_edges) for edge in edges], dtype=int)
        feature_batches.append(features)
        label_batches.append(labels)
        optimum_edges_available += len(optimum_edges)
        optimum_edges_retained += int(labels.sum())
        if progress is not None:
            progress.step(
                f"{problem.name} n={problem.city_count} "
                f"candidates={len(edges)} positives={int(labels.sum())}"
            )

    if progress is not None:
        progress.finish(
            f"recall={optimum_edges_retained / optimum_edges_available:.3f}"
        )

    feature_matrix = np.vstack(feature_batches)
    labels = np.concatenate(label_batches)
    if labels.min() == labels.max():
        raise ValueError(
            "training labels need both retained and non-retained candidate edges"
        )
    if verbose:
        log(
            f"assembled training matrix shape={feature_matrix.shape} "
            f"positive_rate={labels.mean():.3f}"
        )
    return CandidateTrainingData(
        features=feature_matrix,
        labels=labels,
        instance_count=len(problem_list),
        candidate_recall=optimum_edges_retained / optimum_edges_available,
    )


class CandidateEdgeRanker:
    """A persisted scikit-learn ranker for geometric TSP candidate edges."""

    def __init__(
        self,
        *,
        n_estimators: int = 250,
        max_depth: int | None = 10,
        min_samples_leaf: int = 2,
        random_state: int = 0,
    ) -> None:
        if n_estimators < 1 or min_samples_leaf < 1:
            raise ValueError("n_estimators and min_samples_leaf must be positive")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model: RandomForestClassifier | None = None
        self.metadata: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def fit(
        self,
        training_data: CandidateTrainingData,
        *,
        metadata: dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> "CandidateEdgeRanker":
        if training_data.features.shape[1] != len(FEATURE_NAMES):
            raise ValueError("feature schema does not match CandidateEdgeRanker")
        if verbose:
            log(
                f"training random forest: trees={self.n_estimators} "
                f"rows={training_data.features.shape[0]} "
                f"features={training_data.features.shape[1]}"
            )
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1 if verbose else 0,
        )
        model.fit(training_data.features, training_data.labels)
        if verbose:
            log("random forest training finished")
        self.model = model
        self.metadata = {
            "feature_names": FEATURE_NAMES,
            "training_instances": training_data.instance_count,
            "training_rows": int(training_data.features.shape[0]),
            "positive_rate": float(training_data.labels.mean()),
            "candidate_recall": training_data.candidate_recall,
            **(metadata or {}),
        }
        return self

    def score_candidate_edges(
        self,
        problem: EuclideanTravellingSalespersonProblem,
        candidate_lists: Sequence[set[int]],
    ) -> dict[tuple[int, int], float]:
        """Return estimated reference-tour probabilities for candidate edges."""
        if self.model is None:
            raise RuntimeError("CandidateEdgeRanker must be fit or loaded before scoring")
        edges, features = extract_candidate_features(problem, candidate_lists)
        positive_index = int(np.flatnonzero(self.model.classes_ == 1)[0])
        probabilities = self.model.predict_proba(features)[:, positive_index]
        return {edge: float(probability) for edge, probability in zip(edges, probabilities)}

    @staticmethod
    def select_top_candidates(
        problem: EuclideanTravellingSalespersonProblem,
        candidate_lists: Sequence[set[int]],
        scores: dict[tuple[int, int], float],
        *,
        top_k: int,
    ) -> list[set[int]]:
        """Keep exactly the highest-scoring directed choices per city."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        selected: list[set[int]] = []
        for city, neighbors in enumerate(candidate_lists):
            ordered = sorted(
                neighbors,
                key=lambda neighbor: (
                    -scores[_canonical_edge(city, neighbor)],
                    float(problem.distances[city, neighbor]),
                    neighbor,
                ),
            )
            selected.append(set(ordered[:top_k]))
        return selected

    def save(self, path: str | Path) -> None:
        """Persist the fitted model and feature schema."""
        if self.model is None:
            raise RuntimeError("cannot save an unfitted CandidateEdgeRanker")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "metadata": self.metadata,
                "parameters": {
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "min_samples_leaf": self.min_samples_leaf,
                    "random_state": self.random_state,
                },
                "feature_names": FEATURE_NAMES,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CandidateEdgeRanker":
        """Load a model checkpoint produced by :meth:`save`."""
        payload = joblib.load(Path(path))
        if tuple(payload["feature_names"]) != FEATURE_NAMES:
            raise ValueError("model feature schema is incompatible with this version")
        ranker = cls(**payload["parameters"])
        ranker.model = payload["model"]
        ranker.metadata = dict(payload["metadata"])
        return ranker


__all__ = [
    "FEATURE_NAMES",
    "CandidateEdgeRanker",
    "CandidateTrainingData",
    "build_candidate_training_data",
    "extract_candidate_features",
    "tour_edge_set",
]
