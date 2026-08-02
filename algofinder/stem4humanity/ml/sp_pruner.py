"""Supervised pruning of DAG arcs that cannot lie on any shortest path.

For a single-source instance, an arc ``(u, v, w)`` is *useful* iff
``dist[u] + w == dist[v]`` for the exact distances from the source; the
model learns this from structural features, and the learned solver prunes
low-scoring arcs before relaxing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier

from stem4humanity.problems.shortest_path import ShortestPathState
from stem4humanity.util.progress import ProgressTracker, log

SP_FEATURE_NAMES: tuple[str, ...] = (
    "weight_over_mean_out",
    "weight_over_global_mean",
    "tail_out_degree",
    "head_in_degree",
    "tail_is_source",
    "weight_rank_out",
    "weight_over_max_out",
    "head_is_source",
)


def _exact_distances(state: ShortestPathState) -> NDArray[np.float64]:
    """Array Dijkstra over the full graph (label generation only)."""
    node_count = state.node_count
    distances = np.full(node_count, np.inf)
    distances[state.source] = 0.0
    settled = [False] * node_count
    for _ in range(node_count):
        unsettled = np.flatnonzero(~np.asarray(settled))
        if not unsettled.size:
            break
        pivot = int(unsettled[np.argmin(distances[unsettled])])
        if not np.isfinite(distances[pivot]):
            break
        settled[pivot] = True
        for head, weight in state.out_arcs(pivot):
            distances[head] = min(distances[head], distances[pivot] + weight)
    return distances


def extract_arc_features(
    state: ShortestPathState,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """One feature row per arc of ``state`` (aligned with ``state.arcs``)."""
    arcs = np.asarray(state.arcs, dtype=float)
    tails = arcs[:, 0].astype(int)
    heads = arcs[:, 1].astype(int)
    weights = arcs[:, 2]
    global_mean = max(float(weights.mean()), 1e-12)
    out_degree = np.asarray([len(state.out_edges(node)) for node in state.nodes])
    in_degree = np.zeros(state.node_count, dtype=int)
    for head in heads:
        in_degree[head] += 1
    rank_fraction = np.empty(len(arcs))
    over_max = np.empty(len(arcs))
    for node in state.nodes:
        out_weights = weights[tails == node]
        if not out_weights.size:
            continue
        denominator = max(float(out_weights.max()), 1e-12)
        for index in np.flatnonzero(tails == node):
            fraction = float((out_weights < weights[index]).mean())
            rank_fraction[index] = fraction
            over_max[index] = weights[index] / denominator
    mean_out = np.empty(len(arcs))
    for index, (tail, weight) in enumerate(zip(tails, weights)):
        tail_weights = weights[tails == tail]
        mean_out[index] = weight / max(float(tail_weights.mean()), 1e-12)
    source = state.source
    rows = np.column_stack(
        [
            mean_out,
            weights / global_mean,
            out_degree[tails].astype(float),
            in_degree[heads].astype(float),
            (tails == source).astype(float),
            rank_fraction,
            over_max,
            (heads == source).astype(float),
        ]
    )
    return arcs[:, :2].astype(np.int64), rows


def build_arc_training_data(
    states: Iterable[ShortestPathState],
    *,
    verbose: bool = True,
) -> tuple[list[tuple[int, int, int]], NDArray[np.float64], NDArray[np.int_]]:
    """Label arcs by whether they lie on some shortest path from the source."""
    state_list = list(states)
    if not state_list:
        raise ValueError("at least one training state is required")
    edge_list: list[tuple[int, int, int]] = []
    feature_batches: list[NDArray[np.float64]] = []
    label_batches: list[NDArray[np.int_]] = []
    progress = (
        ProgressTracker(len(state_list), "labeling-sp-arcs", report_every=5)
        if verbose
        else None
    )
    for state in state_list:
        distances = _exact_distances(state)
        arcs, features = extract_arc_features(state)
        labels = np.zeros(len(arcs), dtype=int)
        for index, (tail, head) in enumerate(arcs):
            tail = int(tail)
            head = int(head)
            if not np.isfinite(distances[tail]):
                continue
            weight = float(state.arcs[index, 2])
            if abs(distances[tail] + weight - distances[head]) <= 1e-9:
                labels[index] = 1
        edge_list.extend(
            (int(tail), int(head), int(state.arcs[index, 2]))
            for index, (tail, head) in enumerate(arcs)
        )
        feature_batches.append(features)
        label_batches.append(labels)
        if progress is not None:
            progress.step(f"{state.name} arcs={len(arcs)} useful={int(labels.sum())}")
    if progress is not None:
        progress.finish()
    feature_matrix = np.vstack(feature_batches)
    labels = np.concatenate(label_batches)
    if labels.min() == labels.max():
        raise ValueError("training labels need both useful and prunable arcs")
    return edge_list, feature_matrix, labels


class ShortestPathArcPruner:
    """A persisted scikit-learn pruner for DAG shortest-path arcs."""

    def __init__(
        self,
        *,
        n_estimators: int = 150,
        max_depth: int | None = 8,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model: RandomForestClassifier | None = None
        self.metadata: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def fit(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.int_],
        *,
        metadata: dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> "ShortestPathArcPruner":
        if features.shape[1] != len(SP_FEATURE_NAMES):
            raise ValueError("feature schema does not match ShortestPathArcPruner")
        if verbose:
            log(
                f"training arc pruner: trees={self.n_estimators} "
                f"rows={features.shape[0]} features={features.shape[1]}"
            )
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=self.random_state,
        )
        model.fit(features, labels)
        self.model = model
        self.metadata = {
            "feature_names": SP_FEATURE_NAMES,
            "training_rows": int(features.shape[0]),
            "positive_rate": float(labels.mean()),
            **(metadata or {}),
        }
        return self

    def score_arcs(self, state: ShortestPathState) -> NDArray[np.float64]:
        """Estimated useful-arc probabilities aligned with ``state.arcs``."""
        if self.model is None:
            raise RuntimeError("pruner must be fit or loaded before scoring")
        _, features = extract_arc_features(state)
        positive_index = int(np.flatnonzero(self.model.classes_ == 1)[0])
        return self.model.predict_proba(features)[:, positive_index]

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("cannot save an unfitted ShortestPathArcPruner")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "metadata": self.metadata,
                "parameters": {
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "random_state": self.random_state,
                },
                "feature_names": SP_FEATURE_NAMES,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ShortestPathArcPruner":
        payload = joblib.load(Path(path))
        if tuple(payload["feature_names"]) != SP_FEATURE_NAMES:
            raise ValueError("model feature schema is incompatible with this version")
        pruner = cls(**payload["parameters"])
        pruner.model = payload["model"]
        pruner.metadata = dict(payload["metadata"])
        return pruner


__all__ = [
    "SP_FEATURE_NAMES",
    "ShortestPathArcPruner",
    "build_arc_training_data",
    "extract_arc_features",
]
