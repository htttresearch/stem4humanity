"""Supervised learning of the Johnson pairwise precedence relation.

For two jobs ``i`` and ``j`` in a two-machine flow shop, Johnson's rule
puts ``i`` before ``j`` when ``min(p1[i], p2[j]) <= min(p1[j], p2[i])``.
The model learns this relation from job-pair features so the solver can
order an entire job set with a learned comparator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from algofinder.problems.scheduling import FlowShopState
from algofinder.util.progress import log

SCHED_FEATURE_NAMES: tuple[str, ...] = (
    "p1_first",
    "p2_first",
    "p1_second",
    "p2_second",
    "p1_first_share",
    "p2_second_share",
    "min_first_key",
    "min_second_key",
    "p1_minus_p2_first",
    "p1_minus_p2_second",
)


def johnson_precedes(times: NDArray[np.float64], first: int, second: int) -> bool:
    """Whether Johnson's rule orders ``first`` before ``second``."""
    return min(times[first, 0], times[second, 1]) <= min(
        times[second, 0], times[first, 1]
    )


def extract_pair_features(
    times: NDArray[np.float64], first: int, second: int
) -> list[float]:
    """Features for the ordered job pair ``(first, second)``."""
    p1f, p2f = times[first]
    p1s, p2s = times[second]
    return [
        p1f,
        p2f,
        p1s,
        p2s,
        p1f / (p1f + p1s),
        p2s / (p2f + p2s),
        min(p1f, p2s),
        min(p1s, p2f),
        p1f - p2f,
        p1s - p2s,
    ]


def build_pair_training_data(
    states: Iterable[FlowShopState],
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """One row per ordered job pair, labeled by the Johnson rule."""
    state_list = list(states)
    if not state_list:
        raise ValueError("at least one training state is required")
    feature_rows: list[list[float]] = []
    label_rows: list[int] = []
    for state in state_list:
        for first in state.jobs:
            for second in state.jobs:
                if first == second:
                    continue
                feature_rows.append(
                    extract_pair_features(state.times, first, second)
                )
                label_rows.append(
                    int(johnson_precedes(state.times, first, second))
                )
    return np.asarray(feature_rows, dtype=float), np.asarray(label_rows, dtype=int)


class FlowShopComparator:
    """A persisted scikit-learn pairwise comparator for flow-shop jobs."""

    def __init__(
        self,
        *,
        c_regularization: float = 1.0,
        random_state: int = 0,
    ) -> None:
        self.c_regularization = c_regularization
        self.random_state = random_state
        self.model: LogisticRegression | None = None
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
    ) -> "FlowShopComparator":
        if features.shape[1] != len(SCHED_FEATURE_NAMES):
            raise ValueError("feature schema does not match FlowShopComparator")
        if verbose:
            log(
                f"training flow-shop comparator: rows={features.shape[0]} "
                f"features={features.shape[1]} positive_rate={labels.mean():.3f}"
            )
        model = LogisticRegression(
            C=self.c_regularization,
            max_iter=2_000,
            random_state=self.random_state,
        )
        model.fit(features, labels)
        self.model = model
        self.metadata = {
            "feature_names": SCHED_FEATURE_NAMES,
            "training_rows": int(features.shape[0]),
            "train_accuracy": float(model.score(features, labels)),
            **(metadata or {}),
        }
        return self

    def score_pair(self, state: FlowShopState, first: int, second: int) -> float:
        """Estimated probability that ``first`` precedes ``second``."""
        if self.model is None:
            raise RuntimeError(
                "comparator must be fit or loaded before scoring"
            )
        features = np.asarray(
            [extract_pair_features(state.times, first, second)], dtype=float
        )
        return float(self.model.predict_proba(features)[0, 1])

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("cannot save an unfitted FlowShopComparator")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "metadata": self.metadata,
                "parameters": {
                    "c_regularization": self.c_regularization,
                    "random_state": self.random_state,
                },
                "feature_names": SCHED_FEATURE_NAMES,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FlowShopComparator":
        payload = joblib.load(Path(path))
        if tuple(payload["feature_names"]) != SCHED_FEATURE_NAMES:
            raise ValueError("model feature schema is incompatible with this version")
        comparator = cls(**payload["parameters"])
        comparator.model = payload["model"]
        comparator.metadata = dict(payload["metadata"])
        return comparator


__all__ = [
    "SCHED_FEATURE_NAMES",
    "FlowShopComparator",
    "build_pair_training_data",
    "johnson_precedes",
]
