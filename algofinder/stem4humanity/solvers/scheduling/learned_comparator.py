"""Learned pairwise-comparison flow-shop heuristic.

A model learns the Johnson pairwise precedence relation between jobs
from training instances; at solve time the jobs are sorted with the
learned comparator. Imperfect comparisons yield suboptimal (or
inconsistent) orders, giving a measurable makespan gap.
"""

from __future__ import annotations

from functools import cmp_to_key
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from stem4humanity.problems.scheduling import FlowShopState
from stem4humanity.solvers.base import (
    InapplicableError,
    Solver,
    SolverResult,
    register_solver,
)

if TYPE_CHECKING:
    from stem4humanity.ml.sched_comparator import FlowShopComparator

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "models"
    / "sched_comparator.joblib"
)


@register_solver
class LearnedFlowShopSolver(Solver):
    id = "learned-flow-shop"
    display = "Learned Johnson-style pairwise comparator"
    tags = frozenset({"ml", "heuristic"})
    applies_to = frozenset({"scheduling:flow-shop-2"})

    def __init__(
        self,
        comparator: FlowShopComparator | None = None,
    ) -> None:
        if comparator is None:
            from stem4humanity.ml.sched_comparator import FlowShopComparator

            if not DEFAULT_MODEL_PATH.exists():
                raise InapplicableError(
                    "no default flow-shop comparator at "
                    f"{DEFAULT_MODEL_PATH}; train one with "
                    "'python -m stem4humanity.harness.runner train'"
                )
            comparator = FlowShopComparator.load(DEFAULT_MODEL_PATH)
        if not comparator.is_fitted:
            raise ValueError("comparator must be fit or loaded before solving")
        self.comparator = comparator

    def solve(
        self,
        state: FlowShopState,
        *,
        budget_seconds: float | None = None,
    ) -> SolverResult:
        started = perf_counter()
        def compare(first: int, second: int) -> int:
            if self.comparator.score_pair(state, first, second) > 0.5:
                return -1
            return 1

        order = sorted(state.jobs, key=cmp_to_key(compare))
        cost = state.objective_value(order)
        return SolverResult(
            solution=order,
            cost=cost,
            exact=False,
            wall_seconds=perf_counter() - started,
            metadata={
                "algorithm": "learned-pairwise-comparator",
                "comparisons": len(state.jobs) * (len(state.jobs) - 1) // 2,
            },
        )


__all__ = ["LearnedFlowShopSolver"]
