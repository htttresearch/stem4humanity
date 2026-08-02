"""Supervised task-priority model for DAG scheduling on identical machines.

For unit-time out-forests, Hu's level algorithm is optimal; a task that
finishes exactly at the optimal makespan is a "last wave" task that the
algorithm schedules first. The model learns this signal from structural
task features, and the learned solver ranks available tasks by
descending predicted probability before list scheduling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier

from algofinder.problems.parallel_scheduling import ParallelSchedulingState
from algofinder.util.progress import ProgressTracker, log

PS_FEATURE_NAMES: tuple[str, ...] = (
    "processing_time_normalized",
    "depth_from_root",
    "level_to_leaf",
    "in_degree",
    "out_degree",
    "ancestor_count",
    "descendant_count",
    "machines_normalized",
)


def _depth_and_descendants(
    state: ParallelSchedulingState,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Longest path from a root into each task, plus subtree sizes."""
    n = state.task_count
    depth = np.zeros(n, dtype=int)
    descendants = np.zeros(n, dtype=int)
    for task in range(n):
        for predecessor in state.predecessors[task]:
            depth[task] = max(depth[task], depth[predecessor] + 1)
    for task in reversed(range(n)):
        descendants[task] += 1
        for predecessor in state.predecessors[task]:
            descendants[predecessor] += descendants[task]
    return depth, descendants


def extract_task_features(
    state: ParallelSchedulingState,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """One feature row per task (aligned with ``state.tasks``)."""
    depth, descendants = _depth_and_descendants(state)
    level = np.zeros(state.task_count, dtype=float)
    for task in reversed(range(state.task_count)):
        best = 0.0
        for successor in state.successors[task]:
            best = max(best, level[successor])
        level[task] = state.times[task] + best
    ancestors = np.zeros(state.task_count, dtype=int)
    for task in range(state.task_count):
        for predecessor in state.predecessors[task]:
            ancestors[task] = max(ancestors[task], ancestors[predecessor] + 1)
    max_time = max(float(state.times.max()), 1e-12)
    in_degree = np.asarray([len(preds) for preds in state.predecessors], dtype=int)
    out_degree = np.asarray([len(succs) for succs in state.successors], dtype=int)
    rows = np.column_stack(
        [
            state.times / max_time,
            depth.astype(float),
            level,
            in_degree.astype(float),
            out_degree.astype(float),
            ancestors.astype(float),
            descendants.astype(float),
            np.full(state.task_count, state.machines / 4.0),
        ]
    )
    return np.arange(state.task_count, dtype=np.int64), rows


def build_priority_training_data(
    states: Iterable[ParallelSchedulingState],
    *,
    verbose: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Label tasks by whether they finish at the optimal (Hu) makespan."""
    from algofinder.solvers.parallel_scheduling.hu import PSHuSolver

    state_list = list(states)
    if not state_list:
        raise ValueError("at least one training state is required")
    exact = PSHuSolver()
    feature_batches: list[NDArray[np.float64]] = []
    label_batches: list[NDArray[np.int_]] = []
    progress = (
        ProgressTracker(len(state_list), "labeling-ps-tasks", report_every=5)
        if verbose
        else None
    )
    for state in state_list:
        _, features = extract_task_features(state)
        result = exact.solve(state)
        makespan = result.cost
        labels = np.asarray(
            [
                1
                if abs(state.times[task] + result.solution[task][1] - makespan) < 1e-9
                else 0
                for task in state.tasks
            ],
            dtype=int,
        )
        feature_batches.append(features)
        label_batches.append(labels)
        if progress is not None:
            progress.step(
                f"{state.name} tasks={state.task_count} "
                f"critical={int(labels.sum())} makespan={makespan:.0f}"
            )
    if progress is not None:
        progress.finish()
    features = np.vstack(feature_batches)
    labels = np.concatenate(label_batches)
    if labels.min() == labels.max():
        raise ValueError("training labels need both critical and non-critical tasks")
    return features, labels


class ParallelPriorityPredictor:
    """A persisted scikit-learn priority predictor for parallel scheduling."""

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
    ) -> "ParallelPriorityPredictor":
        if features.shape[1] != len(PS_FEATURE_NAMES):
            raise ValueError(
                "feature schema does not match ParallelPriorityPredictor"
            )
        if verbose:
            log(
                f"training priority predictor: trees={self.n_estimators} "
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
            "feature_names": PS_FEATURE_NAMES,
            "training_rows": int(features.shape[0]),
            "positive_rate": float(labels.mean()),
            **(metadata or {}),
        }
        return self

    def score_tasks(self, state: ParallelSchedulingState) -> NDArray[np.float64]:
        """Estimated critical-task probabilities aligned with ``state.tasks``."""
        if self.model is None:
            raise RuntimeError("predictor must be fit or loaded before scoring")
        _, features = extract_task_features(state)
        positive_index = int(np.flatnonzero(self.model.classes_ == 1)[0])
        return self.model.predict_proba(features)[:, positive_index]

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("cannot save an unfitted ParallelPriorityPredictor")
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
                "feature_names": PS_FEATURE_NAMES,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ParallelPriorityPredictor":
        payload = joblib.load(Path(path))
        if tuple(payload["feature_names"]) != PS_FEATURE_NAMES:
            raise ValueError("model feature schema is incompatible with this version")
        predictor = cls(**payload["parameters"])
        predictor.model = payload["model"]
        predictor.metadata = dict(payload["metadata"])
        return predictor


__all__ = [
    "PS_FEATURE_NAMES",
    "ParallelPriorityPredictor",
    "build_priority_training_data",
    "extract_task_features",
]
