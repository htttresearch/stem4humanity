"""ML item-ordering solver: predict which items end up paired, then FFD.

A trained model scores every item with the estimated probability that it
shares its bin with another item in an optimal packing (labels from the
exact blossom matching on large-item instances). The learned solver
orders items by decreasing predicted pair probability and packs them
with first-fit, which concentrates pairable items early so they get
paired.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from stem4humanity.problems.bin_packing import BinPackingState
from stem4humanity.solvers.base import (
    InapplicableError,
    Solver,
    SolverResult,
    register_solver,
)
from stem4humanity.solvers.bin_packing.exact_dfs import first_fit_loads
from stem4humanity.trace.contract import trace_contract

if TYPE_CHECKING:
    from stem4humanity.ml.bp_packer import BinPackingPacker

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "public" / "models" / "bp_packer.joblib"
)


@register_solver
class BPLearnedOrderSolver(Solver):
    id = "bp-learned-order"
    display = "Learned item ordering + first-fit"
    tags = frozenset({"ml", "heuristic"})
    applies_to = frozenset({"bin-packing:vector", "bin-packing:large-items"})

    def __init__(self, packer: BinPackingPacker | None = None) -> None:
        if packer is None:
            from stem4humanity.ml.bp_packer import BinPackingPacker

            if not DEFAULT_MODEL_PATH.exists():
                raise InapplicableError(
                    "no default packer at "
                    f"{DEFAULT_MODEL_PATH}; train one with "
                    "'python -m stem4humanity.harness.runner train'"
                )
            packer = BinPackingPacker.load(DEFAULT_MODEL_PATH)
        if not packer.is_fitted:
            raise ValueError("packer must be fit or loaded before solving")
        self.packer = packer

    def solve(
        self,
        state: BinPackingState,
        *,
        budget_seconds: float | None = None,
        context: object | None = None,
    ) -> SolverResult:
        from stem4humanity.trace.recorder import NullTraceRecorder

        started = perf_counter()
        trace = context.trace if context is not None else NullTraceRecorder()
        if trace.enabled:
            trace.span("prediction.start", model="bp_packer.joblib")
        scores = self.packer.score_items(state)
        if trace.enabled:
            trace.event(
                "prediction",
                "bp.score-items/v1",
                outcome={"scores": [float(x) for x in scores]},
            )
            trace.span("prediction.end")
        order = sorted(
            range(state.item_count),
            key=lambda item: (-float(scores[item]), item),
        )
        packing = first_fit_loads(state, order, context=context)
        return SolverResult(
            solution=packing,
            cost=float(len(packing)),
            exact=False,
            wall_seconds=perf_counter() - started,
            metadata={
                "algorithm": "learned-order-first-fit",
                "mean_pair_probability": float(np.mean(scores)),
                "max_pair_probability": float(np.max(scores)),
            },
        )


__all__ = ["BPLearnedOrderSolver"]
