"""Supervised per-period action model for storage arbitrage.

For each period the model learns whether the optimal schedule charges
(0), holds (1) or discharges (2); labels come from the exact energy-grid
DP on train instances. All features are normalized price statistics, so
the model generalizes across horizons. The learned solver simulates the
predicted actions and repairs the terminal energy, keeping the result
feasible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier

from stem4humanity.problems.unit_commitment import StorageArbitrageState
from stem4humanity.util.progress import ProgressTracker, log

UC_STORAGE_FEATURE_NAMES: tuple[str, ...] = (
    "price_over_mean",
    "price_over_max",
    "price_over_min",
    "price_return",
    "price_over_rolling_mean",
    "price_rank",
    "price_slope_ahead",
    "period_fraction",
    "price_over_future_max",
    "future_min_over_price",
    "future_mean_over_price",
)


def extract_period_features(
    state: StorageArbitrageState,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """One feature row per period."""
    prices = state.prices
    periods = state.periods
    mean = float(prices.mean())
    maximum = float(prices.max())
    minimum = float(prices.min())
    rows = np.empty((periods, len(UC_STORAGE_FEATURE_NAMES)), dtype=float)
    for t in range(periods):
        window = max(1, min(periods, max(3, periods // 4)))
        lower = max(0, t - window)
        rolling = float(prices[lower : t + 1].mean())
        future_max = float(prices[t:].max())
        future_min = float(prices[t:].min())
        future_mean = float(prices[t:].mean())
        rows[t] = [
            float(prices[t]) / mean,
            float(prices[t]) / maximum,
            float(prices[t]) / minimum,
            float(prices[t]) / float(prices[t - 1]) if t > 0 else 1.0,
            float(prices[t]) / rolling,
            float((prices < prices[t]).mean()),
            float(prices[t + 1] - prices[t]) / float(prices[t])
            if t + 1 < periods
            else 0.0,
            t / periods,
            float(prices[t]) / future_max,
            future_min / float(prices[t]),
            future_mean / float(prices[t]),
        ]
    return np.arange(periods, dtype=np.int64), rows


def build_action_training_data(
    states: Iterable[StorageArbitrageState],
    *,
    verbose: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Label periods by the optimal action (0=charge, 1=hold, 2=discharge)."""
    from stem4humanity.solvers.unit_commitment.storage_dp import UCStorageDPSolver

    state_list = list(states)
    if not state_list:
        raise ValueError("at least one training state is required")
    exact = UCStorageDPSolver()
    feature_batches: list[NDArray[np.float64]] = []
    label_batches: list[NDArray[np.int_]] = []
    progress = (
        ProgressTracker(len(state_list), "labeling-uc-actions", report_every=2)
        if verbose
        else None
    )
    for state in state_list:
        _, features = extract_period_features(state)
        result = exact.solve(state)
        labels = np.zeros(state.periods, dtype=int)
        for period, entry in enumerate(result.solution):
            charge, discharge = float(entry[0]), float(entry[1])
            if charge > 1e-9:
                labels[period] = 0
            elif discharge > 1e-9:
                labels[period] = 2
            else:
                labels[period] = 1
        feature_batches.append(features)
        label_batches.append(labels)
        if progress is not None:
            progress.step(
                f"{state.name} periods={state.periods} "
                f"labels={labels.tolist().count(0)}/{labels.tolist().count(1)}/"
                f"{labels.tolist().count(2)}"
            )
    if progress is not None:
        progress.finish()
    features = np.vstack(feature_batches)
    labels = np.concatenate(label_batches)
    if labels.min() == labels.max():
        raise ValueError("training labels need more than one action class")
    return features, labels


class StorageActionPredictor:
    """A persisted scikit-learn per-period storage action classifier."""

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
    ) -> "StorageActionPredictor":
        if features.shape[1] != len(UC_STORAGE_FEATURE_NAMES):
            raise ValueError("feature schema does not match StorageActionPredictor")
        if verbose:
            log(
                f"training action predictor: trees={self.n_estimators} "
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
            "feature_names": UC_STORAGE_FEATURE_NAMES,
            "training_rows": int(features.shape[0]),
            "positive_rate": float((labels > 0).mean()),
            **(metadata or {}),
        }
        return self

    def predict_actions(self, state: StorageArbitrageState) -> NDArray[np.int_]:
        """Predicted per-period actions: 0=charge, 1=hold, 2=discharge."""
        if self.model is None:
            raise RuntimeError("predictor must be fit or loaded before predicting")
        _, features = extract_period_features(state)
        return self.model.predict(features)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("cannot save an unfitted StorageActionPredictor")
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
                "feature_names": UC_STORAGE_FEATURE_NAMES,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> "StorageActionPredictor":
        payload = joblib.load(Path(path))
        if tuple(payload["feature_names"]) != UC_STORAGE_FEATURE_NAMES:
            raise ValueError("model feature schema is incompatible with this version")
        predictor = cls(**payload["parameters"])
        predictor.model = payload["model"]
        predictor.metadata = dict(payload["metadata"])
        return predictor


__all__ = [
    "StorageActionPredictor",
    "UC_STORAGE_FEATURE_NAMES",
    "build_action_training_data",
    "extract_period_features",
]
