"""Supervised commitment model for thermal unit commitment.

For each (unit, hour) pair the model learns the probability that the
unit is online in an optimal commitment (labels from the exact DP on
train instances). All features are normalized to be instance-agnostic,
so a model trained on 6-unit / 24-hour instances scores smaller test
fleets. The learned solver turns the scores into commitment priorities
for the priority-list scheduler, which keeps the output feasible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier

from algofinder.problems.unit_commitment import UnitCommitmentState
from algofinder.util.progress import ProgressTracker, log

UC_FEATURE_NAMES: tuple[str, ...] = (
    "c_var_normalized",
    "c_start_normalized",
    "p_min_over_demand",
    "p_max_over_demand",
    "p_min_over_p_max",
    "p_max_share",
    "c_var_rank",
    "demand_level",
    "reserve_fraction",
    "demand_slope_ahead",
    "demand_slope_behind",
    "demand_ahead_mean",
    "hour_fraction",
)


def extract_commitment_features(
    state: UnitCommitmentState,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """One feature row per (unit, hour) pair, flattened unit-major."""
    generators = state.generator_count
    hours = state.hours
    c_var = state.c_var
    c_start = state.c_start
    p_min = state.p_min
    p_max = state.p_max
    demand = state.demand
    reserve = state.reserve
    max_demand = float(demand.max()) if hours else 1.0
    max_c_start = float(c_start.max()) if generators else 1.0
    rows = np.empty((generators, hours, len(UC_FEATURE_NAMES)), dtype=float)
    for g in range(generators):
        c_var_rank = float((c_var < c_var[g]).mean())
        for h in range(hours):
            ahead = (
                float(demand[h + 1] - demand[h]) / max_demand
                if h + 1 < hours
                else 0.0
            )
            behind = float(demand[h] - demand[h - 1]) / max_demand if h > 0 else 0.0
            ahead_mean = (
                float(demand[h : min(h + 3, hours)].mean()) / max_demand
                if hours
                else 0.0
            )
            rows[g, h] = [
                float(c_var[g]) / float(c_var.max()) if generators else 0.0,
                float(c_start[g]) / max_c_start,
                float(p_min[g]) / float(demand[h]) if demand[h] > 0 else 0.0,
                float(p_max[g]) / float(demand[h]) if demand[h] > 0 else 0.0,
                float(p_min[g]) / float(p_max[g]),
                float(p_max[g]) / float(p_max.sum()),
                c_var_rank,
                float(demand[h]) / max_demand,
                float(reserve[h]) / float(demand[h]) if demand[h] > 0 else 0.0,
                ahead,
                behind,
                ahead_mean,
                h / hours,
            ]
    flat = rows.reshape(generators * hours, len(UC_FEATURE_NAMES))
    ids = np.arange(generators * hours, dtype=np.int64)
    return ids, flat


def build_commitment_training_data(
    states: Iterable[UnitCommitmentState],
    *,
    verbose: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Label (unit, hour) pairs by whether the unit is on in the optimum."""
    from algofinder.solvers.unit_commitment.exact_dp import UCExactDPSolver

    state_list = list(states)
    if not state_list:
        raise ValueError("at least one training state is required")
    exact = UCExactDPSolver()
    feature_batches: list[NDArray[np.float64]] = []
    label_batches: list[NDArray[np.int_]] = []
    progress = (
        ProgressTracker(len(state_list), "labeling-uc-commitments", report_every=2)
        if verbose
        else None
    )
    for state in state_list:
        _, features = extract_commitment_features(state)
        result = exact.solve(state)
        online: set[int] = set()
        for hour in range(state.hours):
            for row in result.solution[hour]:
                online.add(int(row[0]) * state.hours + hour)
        labels = np.zeros(state.generator_count * state.hours, dtype=int)
        for index in online:
            labels[index] = 1
        feature_batches.append(features)
        label_batches.append(labels)
        if progress is not None:
            progress.step(
                f"{state.name} pairs={len(labels)} online={int(labels.sum())}"
            )
    if progress is not None:
        progress.finish()
    features = np.vstack(feature_batches)
    labels = np.concatenate(label_batches)
    if labels.min() == labels.max():
        raise ValueError("training labels need both online and offline pairs")
    return features, labels


class UnitCommitmentLearner:
    """A persisted scikit-learn (unit, hour) commitment classifier."""

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
    ) -> "UnitCommitmentLearner":
        if features.shape[1] != len(UC_FEATURE_NAMES):
            raise ValueError("feature schema does not match UnitCommitmentLearner")
        if verbose:
            log(
                f"training commitment learner: trees={self.n_estimators} "
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
            "feature_names": UC_FEATURE_NAMES,
            "training_rows": int(features.shape[0]),
            "positive_rate": float(labels.mean()),
            **(metadata or {}),
        }
        return self

    def score_commitments(self, state: UnitCommitmentState) -> NDArray[np.float64]:
        """Estimated online probabilities, shape (generators, hours)."""
        if self.model is None:
            raise RuntimeError("learner must be fit or loaded before scoring")
        _, features = extract_commitment_features(state)
        positive_index = int(np.flatnonzero(self.model.classes_ == 1)[0])
        probabilities = self.model.predict_proba(features)[:, positive_index]
        return probabilities.reshape(state.generator_count, state.hours)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("cannot save an unfitted UnitCommitmentLearner")
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
                "feature_names": UC_FEATURE_NAMES,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> "UnitCommitmentLearner":
        payload = joblib.load(Path(path))
        if tuple(payload["feature_names"]) != UC_FEATURE_NAMES:
            raise ValueError("model feature schema is incompatible with this version")
        learner = cls(**payload["parameters"])
        learner.model = payload["model"]
        learner.metadata = dict(payload["metadata"])
        return learner


__all__ = [
    "UC_FEATURE_NAMES",
    "UnitCommitmentLearner",
    "build_commitment_training_data",
    "extract_commitment_features",
]
