"""ML commitment solver: learn which generators run, then priority-list.

A trained model scores every (unit, hour) pair with the estimated
probability that the unit is on in an optimal commitment (labels from
the exact DP on train instances). The learned solver runs the
priority-list scheduler with units ranked by descending predicted
on-probability, so the result is always feasible.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from stem4humanity.problems.unit_commitment import UnitCommitmentState
from stem4humanity.solvers.base import (
    InapplicableError,
    Solver,
    SolverResult,
    register_solver,
)
from stem4humanity.solvers.unit_commitment.priority_list import priority_list_schedule
from stem4humanity.trace.contract import trace_contract

if TYPE_CHECKING:
    from stem4humanity.ml.uc_committer import UnitCommitmentLearner

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "models" / "uc_committer.joblib"
)


@register_solver
@trace_contract(
    version=1,
    state_schema="uc.learned-state/v1",
    atomic_steps={
        "uc.score-commitments/v1": "ML model scores every (unit, hour) pair",
        "uc.priority-hour/v1": "one hourly commit/decommit cycle and dispatch",
    },
    coverage_scope="algorithm",
)
class UCLearnedCommitmentSolver(Solver):
    id = "uc-learned-commitment"
    display = "Learned commitment priorities + dispatch"
    tags = frozenset({"ml", "heuristic"})
    applies_to = frozenset({"unit-commitment:classic"})

    def __init__(self, learner: UnitCommitmentLearner | None = None) -> None:
        if learner is None:
            from stem4humanity.ml.uc_committer import UnitCommitmentLearner

            if not DEFAULT_MODEL_PATH.exists():
                raise InapplicableError(
                    "no default commitment learner at "
                    f"{DEFAULT_MODEL_PATH}; train one with "
                    "'python -m stem4humanity.harness.runner train'"
                )
            learner = UnitCommitmentLearner.load(DEFAULT_MODEL_PATH)
        if not learner.is_fitted:
            raise ValueError("learner must be fit or loaded before solving")
        self.learner = learner

    def solve(
        self,
        state: UnitCommitmentState,
        *,
        budget_seconds: float | None = None,
        context: object | None = None,
    ) -> SolverResult:
        from stem4humanity.trace.recorder import NullTraceRecorder

        started = perf_counter()
        trace = context.trace if context is not None else NullTraceRecorder()
        if trace.enabled:
            trace.span("prediction.start", model="uc_committer.joblib")
        probabilities = self.learner.score_commitments(state)
        mean_probability = float(np.mean(probabilities))
        if trace.enabled:
            trace.event(
                "prediction",
                "uc.score-commitments/v1",
                outcome={"mean_on_probability": mean_probability},
                metrics={"cells": int(probabilities.size)},
            )
            trace.span("prediction.end")
        priority = [
            float(probabilities[g].mean()) for g in range(state.generator_count)
        ]
        solution = priority_list_schedule(state, priority, context=context)
        return SolverResult(
            solution=solution,
            cost=state.objective_value(solution),
            exact=False,
            wall_seconds=perf_counter() - started,
            metadata={
                "algorithm": "learned-commitment-priority",
                "mean_on_probability": mean_probability,
            },
        )


__all__ = ["UCLearnedCommitmentSolver"]
