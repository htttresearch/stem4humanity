"""ML storage solver: predict the optimal action per period.

A trained model picks the best of charge / hold / discharge for every
period (labels from the exact energy-grid DP on train instances). The
learned solver simulates the predicted actions and repairs the terminal
energy, so the result is always feasible.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from algofinder.problems.unit_commitment import StorageArbitrageState
from algofinder.solvers.base import (
    InapplicableError,
    Solver,
    SolverResult,
    register_solver,
)
from algofinder.solvers.unit_commitment.common import repair_terminal, simulate_actions

if TYPE_CHECKING:
    from algofinder.ml.uc_storage import StorageActionPredictor

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "public" / "models" / "uc_storage.joblib"
)


@register_solver
class UCStorageLearnedSolver(Solver):
    id = "uc-storage-learned"
    display = "Learned per-period actions"
    tags = frozenset({"ml", "heuristic"})
    applies_to = frozenset({"unit-commitment:storage"})

    def __init__(self, predictor: StorageActionPredictor | None = None) -> None:
        if predictor is None:
            from algofinder.ml.uc_storage import StorageActionPredictor

            if not DEFAULT_MODEL_PATH.exists():
                raise InapplicableError(
                    "no default storage predictor at "
                    f"{DEFAULT_MODEL_PATH}; train one with "
                    "'python -m algofinder.harness.runner train'"
                )
            predictor = StorageActionPredictor.load(DEFAULT_MODEL_PATH)
        if not predictor.is_fitted:
            raise ValueError("predictor must be fit or loaded before solving")
        self.predictor = predictor

    def solve(
        self,
        state: StorageArbitrageState,
        *,
        budget_seconds: float | None = None,
    ) -> SolverResult:
        started = perf_counter()
        actions = self.predictor.predict_actions(state)
        solution = repair_terminal(state, simulate_actions(state, actions))
        return SolverResult(
            solution=solution,
            cost=state.objective_value(solution),
            exact=False,
            wall_seconds=perf_counter() - started,
            metadata={"algorithm": "learned-actions"},
        )


__all__ = ["UCStorageLearnedSolver"]
