"""Train and persist the supervised candidate-edge ranker.

The ranker learns which geometrically plausible Euclidean TSP edges belong
to an optimal tour; the learned-candidate solver then keeps only the top
scoring edges for its chained 2-opt search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from algofinder.ml.candidate_ranker import (
    CandidateEdgeRanker,
    CandidateTrainingData,
    build_candidate_training_data,
    tour_edge_set,
)
from algofinder.problems.base import Instance
from algofinder.problems.registry import get_problem
from algofinder.solvers.base import all_solvers
from algofinder.solvers.tsp.euclidean_geometry import build_geometric_candidate_lists
from algofinder.util.progress import log, log_phase

import numpy as np


def _exact_solver() -> object:
    import algofinder.solvers.registry  # noqa: F401  (run solver registration)

    return next(
        solver for solver in all_solvers().values() if solver.id == "held-karp"
    )()


def train_ranker(
    instances: Iterable[Instance],
    *,
    output_path: str | Path,
    nearest_neighbors: int = 8,
    angular_sectors: int = 4,
    top_k: int = 5,
    n_estimators: int = 250,
    max_depth: int | None = 10,
    random_state: int = 0,
) -> CandidateEdgeRanker:
    """Label candidates with exact tours, fit the forest, and persist it."""
    exact = _exact_solver()
    problem = get_problem("tsp")
    states = []
    for instance in instances:
        if instance.problem != "tsp" or instance.subproblem not in (
            "euclidean",
            "clustered",
        ):
            continue
        states.append(problem.build_state(instance))
    if not states:
        log("no Euclidean TSP train instances; skipping ranker training")
        return None

    log_phase("labeling candidate edges with exact tours")
    training_data: CandidateTrainingData = build_candidate_training_data(
        states,
        exact_solver=exact,
        nearest_neighbors=nearest_neighbors,
        angular_sectors=angular_sectors,
    )

    log_phase("fitting the random forest")
    ranker = CandidateEdgeRanker(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    ).fit(
        training_data,
        metadata={
            "nearest_neighbors": nearest_neighbors,
            "angular_sectors": angular_sectors,
        },
    )

    log_phase("validating top-k candidate recall")
    recall = _validation_recall(
        ranker,
        states,
        exact,
        nearest_neighbors=nearest_neighbors,
        angular_sectors=angular_sectors,
        top_k=top_k,
    )
    log(f"validation top-{top_k} recall: {recall:.3f}")

    ranker.save(output_path)
    log(f"saved ranker to {output_path}")
    return ranker


def _validation_recall(
    ranker: CandidateEdgeRanker,
    states: list[object],
    exact: object,
    *,
    nearest_neighbors: int,
    angular_sectors: int,
    top_k: int,
) -> float:
    retained = 0
    total = 0
    for state in states:
        candidates = build_geometric_candidate_lists(
            state,
            nearest_neighbors=nearest_neighbors,
            angular_sectors=angular_sectors,
        )
        scores = ranker.score_candidate_edges(state, candidates)
        selected = ranker.select_top_candidates(state, candidates, scores, top_k=top_k)
        selected_edges = {
            tuple(sorted((city, neighbor)))
            for city, neighbors in enumerate(selected)
            for neighbor in neighbors
        }
        optimum = tour_edge_set(exact.solve(state).solution)
        retained += len(selected_edges & optimum)
        total += len(optimum)
    return retained / total if total else 0.0


def train_sp_pruner(
    instances: Iterable[Instance],
    *,
    output_path: str | Path,
    n_estimators: int = 150,
    max_depth: int | None = 8,
    random_state: int = 0,
    threshold: float = 0.35,
) -> None:
    """Label DAG arcs with exact distances and persist an arc pruner."""
    from algofinder.ml.sp_pruner import (
        ShortestPathArcPruner,
        build_arc_training_data,
    )
    from algofinder.problems.registry import get_problem

    problem = get_problem("shortest-path")
    states = [
        problem.build_state(instance)
        for instance in instances
        if instance.problem == "shortest-path"
        and instance.subproblem == "dag"
    ]
    if not states:
        log("no shortest-path DAG train instances; skipping arc-pruner training")
        return

    log_phase("labeling DAG arcs with exact distances")
    _, features, labels = build_arc_training_data(states)
    log_phase("fitting the arc pruner")
    pruner = ShortestPathArcPruner(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    ).fit(
        features,
        labels,
        metadata={"positive_rate": float(labels.mean())},
    )
    kept_rates = [
        float((pruner.score_arcs(state) >= threshold).mean())
        for state in states
    ]
    log(
        f"train arc-kept rate at threshold {threshold}: "
        f"{np.mean(kept_rates):.3f} (mean over instances)"
    )
    pruner.save(output_path)
    log(f"saved arc pruner to {output_path}")


def train_sched_comparator(
    instances: Iterable[Instance],
    *,
    output_path: str | Path,
    c_regularization: float = 1.0,
    random_state: int = 0,
) -> None:
    """Label job pairs with the Johnson rule and persist a comparator."""
    from algofinder.ml.sched_comparator import (
        FlowShopComparator,
        build_pair_training_data,
    )
    from algofinder.problems.registry import get_problem

    problem = get_problem("scheduling")
    states = [
        problem.build_state(instance)
        for instance in instances
        if instance.problem == "scheduling"
        and instance.subproblem == "flow-shop-2"
    ]
    if not states:
        log("no flow-shop train instances; skipping comparator training")
        return

    log_phase("labeling job pairs with Johnson's rule")
    features, labels = build_pair_training_data(states)
    log_phase("fitting the flow-shop comparator")
    comparator = FlowShopComparator(
        c_regularization=c_regularization,
        random_state=random_state,
    ).fit(features, labels)
    log(
        f"train pairwise accuracy vs Johnson's rule: "
        f"{comparator.metadata['train_accuracy']:.3f}"
    )
    comparator.save(output_path)
    log(f"saved flow-shop comparator to {output_path}")


def train_bp_packer(
    instances: Iterable[Instance],
    *,
    output_path: str | Path,
    n_estimators: int = 150,
    max_depth: int | None = 8,
    random_state: int = 0,
) -> None:
    """Label items with blossom-optimal packings and persist an item packer."""
    from algofinder.ml.bp_packer import BinPackingPacker, build_pair_training_data
    from algofinder.problems.registry import get_problem

    problem = get_problem("bin-packing")
    states = [
        problem.build_state(instance)
        for instance in instances
        if instance.problem == "bin-packing"
        and instance.subproblem == "large-items"
    ]
    if not states:
        log("no large-items train instances; skipping packer training")
        return

    log_phase("labeling items with blossom-optimal packings")
    features, labels = build_pair_training_data(states)
    log_phase("fitting the item packer")
    packer = BinPackingPacker(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    ).fit(features, labels)
    log(f"train paired-item rate: {packer.metadata['positive_rate']:.3f}")
    packer.save(output_path)
    log(f"saved item packer to {output_path}")


def train_ps_prioritizer(
    instances: Iterable[Instance],
    *,
    output_path: str | Path,
    n_estimators: int = 150,
    max_depth: int | None = 8,
    random_state: int = 0,
) -> None:
    """Label tasks with Hu-optimal schedules and persist a priority predictor."""
    from algofinder.ml.ps_prioritizer import (
        ParallelPriorityPredictor,
        build_priority_training_data,
    )
    from algofinder.problems.registry import get_problem

    problem = get_problem("parallel-scheduling")
    states = [
        problem.build_state(instance)
        for instance in instances
        if instance.problem == "parallel-scheduling"
        and instance.subproblem == "forest"
    ]
    if not states:
        log("no out-forest train instances; skipping priority-predictor training")
        return

    log_phase("labeling tasks with Hu-optimal schedules")
    features, labels = build_priority_training_data(states)
    log_phase("fitting the priority predictor")
    predictor = ParallelPriorityPredictor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    ).fit(features, labels)
    log(
        f"train critical-task rate: "
        f"{predictor.metadata['positive_rate']:.3f}"
    )
    predictor.save(output_path)
    log(f"saved priority predictor to {output_path}")


def train_uc_committer(
    instances: Iterable[Instance],
    *,
    output_path: str | Path,
    n_estimators: int = 150,
    max_depth: int | None = 8,
    random_state: int = 0,
) -> None:
    """Label (unit, hour) pairs with exact commitments and persist a learner."""
    from algofinder.ml.uc_committer import (
        UnitCommitmentLearner,
        build_commitment_training_data,
    )
    from algofinder.problems.registry import get_problem

    problem = get_problem("unit-commitment")
    states = [
        problem.build_state(instance)
        for instance in instances
        if instance.problem == "unit-commitment" and instance.subproblem == "classic"
    ]
    if not states:
        log("no classic unit-commitment train instances; skipping learner training")
        return

    log_phase("labeling (unit, hour) pairs with exact commitments")
    features, labels = build_commitment_training_data(states)
    log_phase("fitting the commitment learner")
    learner = UnitCommitmentLearner(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    ).fit(features, labels)
    log(f"train online-pair rate: {learner.metadata['positive_rate']:.3f}")
    learner.save(output_path)
    log(f"saved commitment learner to {output_path}")


def train_uc_storage(
    instances: Iterable[Instance],
    *,
    output_path: str | Path,
    n_estimators: int = 150,
    max_depth: int | None = 8,
    random_state: int = 0,
) -> None:
    """Label periods with exact arbitrage actions and persist a predictor."""
    from algofinder.ml.uc_storage import (
        StorageActionPredictor,
        build_action_training_data,
    )
    from algofinder.problems.registry import get_problem

    problem = get_problem("unit-commitment")
    states = [
        problem.build_state(instance)
        for instance in instances
        if instance.problem == "unit-commitment" and instance.subproblem == "storage"
    ]
    if not states:
        log("no storage train instances; skipping action-predictor training")
        return

    log_phase("labeling periods with exact arbitrage actions")
    features, labels = build_action_training_data(states)
    log_phase("fitting the action predictor")
    predictor = StorageActionPredictor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    ).fit(features, labels)
    log(f"train non-hold rate: {predictor.metadata['positive_rate']:.3f}")
    predictor.save(output_path)
    log(f"saved action predictor to {output_path}")


__all__ = [
    "train_bp_packer",
    "train_ps_prioritizer",
    "train_ranker",
    "train_sched_comparator",
    "train_sp_pruner",
    "train_uc_committer",
    "train_uc_storage",
]
