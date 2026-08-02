"""ML arc-pruning solver: learn which arcs matter, relax on the rest.

A trained model scores every arc of a DAG instance with the estimated
probability that it lies on some shortest path from the source. Arcs
below a threshold are pruned, and topological relaxation runs on the
pruned graph. The pruned distances are optimal only if the Bellman
conditions hold on the full arc set (every arc obeys the triangle
inequality with the computed distances); when they fail, the solver
falls back to exact Dijkstra on the full graph, so the reported result
is always optimal. The pruning success rate and kept-arc ratio are
reported in the metadata.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from stem4humanity.problems.shortest_path import ShortestPathState
from stem4humanity.solvers.base import (
    InapplicableError,
    Solver,
    SolverResult,
    register_solver,
)
from stem4humanity.solvers.shortest_path.dag_topological import topological_order
from stem4humanity.solvers.shortest_path.dijkstra import DijkstraSolver

if TYPE_CHECKING:
    from stem4humanity.ml.sp_pruner import ShortestPathArcPruner

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "models" / "sp_pruner.joblib"
)


@register_solver
class LearnedArcPruningSolver(Solver):
    id = "learned-sp-pruning"
    display = "Learned arc pruning + topological relaxation"
    tags = frozenset({"ml", "heuristic"})
    applies_to = frozenset({"shortest-path:dag"})

    def __init__(
        self,
        pruner: ShortestPathArcPruner | None = None,
        *,
        threshold: float = 0.35,
    ) -> None:
        if pruner is None:
            from stem4humanity.ml.sp_pruner import ShortestPathArcPruner

            if not DEFAULT_MODEL_PATH.exists():
                raise InapplicableError(
                    "no default arc pruner at "
                    f"{DEFAULT_MODEL_PATH}; train one with "
                    "'python -m stem4humanity.harness.runner train'"
                )
            pruner = ShortestPathArcPruner.load(DEFAULT_MODEL_PATH)
        if not pruner.is_fitted:
            raise ValueError("pruner must be fit or loaded before solving")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must lie in (0, 1)")
        self.pruner = pruner
        self.threshold = threshold

    def solve(
        self,
        state: ShortestPathState,
        *,
        budget_seconds: float | None = None,
    ) -> SolverResult:
        started = perf_counter()
        arcs = np.asarray(state.arcs, dtype=float)
        scores = self.pruner.score_arcs(state)
        kept = scores >= self.threshold
        pruned_arcs = arcs[kept]
        pruned_state = ShortestPathState(
            state.name, state.node_count, state.source, pruned_arcs
        )
        order = topological_order(pruned_state)
        solution: list[int] | None = None
        pruning_safe = False
        if order:
            node_count = state.node_count
            source = state.source
            distances = np.full(node_count, np.inf)
            distances[source] = 0.0
            parents = [-1] * node_count
            for node in order:
                if not np.isfinite(distances[node]):
                    continue
                for tail, head, weight in pruned_arcs:
                    if int(tail) != node:
                        continue
                    improved = distances[node] + weight
                    if improved < distances[int(head)]:
                        distances[int(head)] = improved
                        parents[int(head)] = int(tail)
            solution = [
                parents[node] if np.isfinite(distances[node]) else -1
                for node in range(node_count)
            ]
            bellman_holds = all(
                distances[int(head)] <= distances[int(tail)] + weight + 1e-9
                for tail, head, weight in arcs
                if np.isfinite(distances[int(tail)])
            )
            pruning_safe = state.verify(solution) and bellman_holds
        if not pruning_safe:
            fallback = DijkstraSolver().solve(state)
            solution = fallback.solution
        cost = state.objective_value(solution)
        return SolverResult(
            solution=solution,
            cost=cost,
            exact=True,
            wall_seconds=perf_counter() - started,
            metadata={
                "algorithm": "learned-arc-pruning",
                "pruning_safe": pruning_safe,
                "fallback_used": not pruning_safe,
                "arcs_total": len(arcs),
                "arcs_kept": int(kept.sum()),
                "arcs_pruned": int((~kept).sum()),
                "mean_score": float(np.mean(scores)),
                "threshold": self.threshold,
            },
        )


__all__ = ["LearnedArcPruningSolver"]
