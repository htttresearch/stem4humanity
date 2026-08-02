"""Supervised item-ordering model for bin packing.

For large-item instances every bin holds at most two items, so the
optimal packing pairs items via a maximum matching; an item is labeled
1 iff it shares its bin with another item in that optimum. The model
learns this from dimension-agnostic per-item features, and the learned
solver sorts items by descending predicted pair probability before
packing with first-fit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier

from stem4humanity.problems.bin_packing import BinPackingState
from stem4humanity.util.progress import ProgressTracker, log

BP_FEATURE_NAMES: tuple[str, ...] = (
    "size_over_capacity_max",
    "size_over_capacity_sum",
    "size_over_capacity_mean",
    "size_over_capacity_min",
    "cofit_count",
    "cofit_count_normalized",
    "size_rank",
    "larger_than_half",
    "larger_than_third",
)


def extract_item_features(
    state: BinPackingState,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """One feature row per item (aligned with ``state.items``)."""
    sizes = state.sizes
    caps = state.capacities
    normalized = sizes / caps
    counts = np.zeros(state.item_count, dtype=int)
    for i in range(state.item_count):
        counts[i] = int(
            np.all(sizes[i] + sizes <= caps + 1e-9, axis=1).sum() - 1
        )
    size_sums = sizes.sum(axis=1)
    ranks = np.empty(state.item_count)
    for i in range(state.item_count):
        ranks[i] = float((size_sums < size_sums[i]).mean())
    rows = np.column_stack(
        [
            normalized.max(axis=1),
            normalized.sum(axis=1),
            normalized.mean(axis=1),
            normalized.min(axis=1),
            counts.astype(float),
            counts.astype(float) / max(state.item_count - 1, 1),
            ranks,
            (normalized.max(axis=1) > 0.5).astype(float),
            (normalized.max(axis=1) > 1 / 3).astype(float),
        ]
    )
    return np.arange(state.item_count, dtype=np.int64), rows


def build_pair_training_data(
    states: Iterable[BinPackingState],
    *,
    verbose: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Label items by whether they pair up in an optimal packing."""
    from stem4humanity.solvers.bin_packing.large_items_exact import BPLargeItemsExactSolver

    state_list = list(states)
    if not state_list:
        raise ValueError("at least one training state is required")
    exact = BPLargeItemsExactSolver()
    feature_batches: list[NDArray[np.float64]] = []
    label_batches: list[NDArray[np.int_]] = []
    progress = (
        ProgressTracker(len(state_list), "labeling-bp-items", report_every=5)
        if verbose
        else None
    )
    for state in state_list:
        _, features = extract_item_features(state)
        result = exact.solve(state)
        paired = {
            item for bin_items in result.solution if len(bin_items) > 1
            for item in bin_items
        }
        labels = np.asarray(
            [1 if item in paired else 0 for item in state.items], dtype=int
        )
        feature_batches.append(features)
        label_batches.append(labels)
        if progress is not None:
            progress.step(f"{state.name} items={state.item_count} paired={int(labels.sum())}")
    if progress is not None:
        progress.finish()
    features = np.vstack(feature_batches)
    labels = np.concatenate(label_batches)
    if labels.min() == labels.max():
        raise ValueError("training labels need both paired and solo items")
    return features, labels


class BinPackingPacker:
    """A persisted scikit-learn item-order packer for bin packing."""

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
    ) -> "BinPackingPacker":
        if features.shape[1] != len(BP_FEATURE_NAMES):
            raise ValueError("feature schema does not match BinPackingPacker")
        if verbose:
            log(
                f"training item packer: trees={self.n_estimators} "
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
            "feature_names": BP_FEATURE_NAMES,
            "training_rows": int(features.shape[0]),
            "positive_rate": float(labels.mean()),
            **(metadata or {}),
        }
        return self

    def score_items(self, state: BinPackingState) -> NDArray[np.float64]:
        """Estimated pair probabilities aligned with ``state.items``."""
        if self.model is None:
            raise RuntimeError("packer must be fit or loaded before scoring")
        _, features = extract_item_features(state)
        positive_index = int(np.flatnonzero(self.model.classes_ == 1)[0])
        return self.model.predict_proba(features)[:, positive_index]

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("cannot save an unfitted BinPackingPacker")
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
                "feature_names": BP_FEATURE_NAMES,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> "BinPackingPacker":
        payload = joblib.load(Path(path))
        if tuple(payload["feature_names"]) != BP_FEATURE_NAMES:
            raise ValueError("model feature schema is incompatible with this version")
        packer = cls(**payload["parameters"])
        packer.model = payload["model"]
        packer.metadata = dict(payload["metadata"])
        return packer


__all__ = [
    "BP_FEATURE_NAMES",
    "BinPackingPacker",
    "build_pair_training_data",
    "extract_item_features",
]
