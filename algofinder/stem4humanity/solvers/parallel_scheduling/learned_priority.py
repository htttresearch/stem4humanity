"""ML priority solver: predict which tasks lie on the critical path.

A trained model scores every task with the estimated probability that it
ends at the makespan of an optimal schedule (labels from Hu's exact
algorithm on unit-time out-forests). The learned solver schedules
available tasks by descending predicted probability, with the true
critical path as tie-break.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from stem4humanity.problems.parallel_scheduling import ParallelSchedulingState
from stem4humanity.solvers.base import (
    InapplicableError,
    Solver,
    SolverResult,
    register_solver,
)
from stem4humanity.solvers.parallel_scheduling.common import critical_paths, list_schedule
from stem4humanity.trace.contract import trace_contract

if TYPE_CHECKING:
    from stem4humanity.ml.ps_prioritizer import ParallelPriorityPredictor

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "models" / "ps_prioritizer.joblib"
)


@register_solver
@trace_contract(
    version=1,
    state_schema="ps.learned-state/v1",
    atomic_steps={
        "ps.score-tasks/v1": "ML model scores every task for criticality",
        "parallel-scheduling.list-step/v1": "assign one available task to a machine",
    },
    derived_state={"priority": "-score, -critical-tail, task"},
    coverage_scope="algorithm",
)
class PSLearnedPrioritySolver(Solver):
    id = "ps-learned-priority"
    display = "Learned priority list scheduling"
    tags = frozenset({"ml", "heuristic"})
    applies_to = frozenset(
        {"parallel-scheduling:general", "parallel-scheduling:forest"}
    )

    def __init__(self, predictor: ParallelPriorityPredictor | None = None) -> None:
        if predictor is None:
            from stem4humanity.ml.ps_prioritizer import ParallelPriorityPredictor

            if not DEFAULT_MODEL_PATH.exists():
                raise InapplicableError(
                    "no default priority predictor at "
                    f"{DEFAULT_MODEL_PATH}; train one with "
                    "'python -m stem4humanity.harness.runner train'"
                )
            predictor = ParallelPriorityPredictor.load(DEFAULT_MODEL_PATH)
        if not predictor.is_fitted:
            raise ValueError("predictor must be fit or loaded before solving")
        self.predictor = predictor

    def solve(
        self,
        state: ParallelSchedulingState,
        *,
        budget_seconds: float | None = None,
        context: object | None = None,
    ) -> SolverResult:
        from stem4humanity.trace.recorder import NullTraceRecorder

        started = perf_counter()
        trace = context.trace if context is not None else NullTraceRecorder()
        if trace.enabled:
            trace.span("prediction.start", model="ps_prioritizer.joblib")
        scores = self.predictor.score_tasks(state)
        longest = critical_paths(state)
        if trace.enabled:
            trace.event(
                "prediction",
                "ps.score-tasks/v1",
                outcome={
                    "learned": [float(x) for x in scores],
                    "critical_path_tail": [float(x) for x in longest],
                },
                metrics={
                    "mean_critical_probability": float(np.mean(scores)),
                    "max_critical_probability": float(np.max(scores)),
                },
            )
            trace.span("prediction.end")
        priority = lambda task: (
            -float(scores[task]),
            -float(longest[task]),
            task,
        )
        solution = list_schedule(state, priority, context=context)
        if trace.enabled:
            trace.event(
                "lifecycle",
                "solve.result",
                outcome={"cost": float(state.objective_value(solution))},
            )
        return SolverResult(
            solution=solution,
            cost=state.objective_value(solution),
            exact=False,
            wall_seconds=perf_counter() - started,
            metadata={
                "algorithm": "learned-priority-list-scheduling",
                "mean_critical_probability": float(np.mean(scores)),
                "max_critical_probability": float(np.max(scores)),
            },
        )


__all__ = ["PSLearnedPrioritySolver"]
