"""Evaluator-selected evolutionary proposal backend for the typed TSP HIR lane."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Iterable

from algofinder.agents.research import (
    AnalysisDraft,
    CandidateProposal,
    ResearchContext,
    SearchDirective,
)
from algofinder.agents.tsp_hir import (
    TspGenome,
    default_genome,
    edit_between_genomes,
    mutate_genome,
)


@dataclass
class EvolutionaryHIRModel:
    """Matched non-LLM variation whose parents come from evaluator fitness.

    Population selection belongs to the shared MAP-Elites/UCB policy. This
    backend is intentionally stateless apart from its seeded proposal stream:
    it mutates the exact selected parent from ``ResearchContext`` and emits the
    component that actually changed. Rejected or inferior candidates therefore
    cannot silently enter a private, unevaluated population.
    """

    seed: int = 75348

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._turn = 0

    def propose(self, context: ResearchContext, directive: SearchDirective) -> CandidateProposal:
        self._turn += 1
        parents = tuple(
            genome
            for parent_id in directive.parent_ids
            if (genome := _context_genome(context, parent_id)) is not None
        )
        base = parents[0] if parents else default_genome()
        if directive.operator == "recombine" and len(parents) >= 2:
            child, donor_component = _one_component_crossover(base, parents[1], self._rng)
            mechanism = f"copy evaluator-selected donor component {donor_component} into the first elite"
        elif directive.operator == "tune":
            child, tuned = _tune_one_parameter(base, self._rng)
            mechanism = f"tune evaluator-selected HIR parameter {tuned}"
        elif directive.operator == "distill":
            child, simplified = _distill_one_component(base)
            mechanism = f"distill evaluator-selected HIR component {simplified}"
        else:
            child = mutate_genome(base, seed=self._rng.randrange(2**31))
            mechanism = "type-safe mutation of the evaluator-selected HIR elite"
        edit = edit_between_genomes(base, child, hypothesis=mechanism)
        return CandidateProposal(
            mechanism=mechanism,
            affected_strata=("tsp:euclidean", "tsp:clustered"),
            predicted_tradeoffs="one structural change under the shared fixed HIR compiler and evaluator budget",
            falsification_test="reject when authority-measured public fitness does not improve its archive niche",
            expected_result="an evaluator-verified feasible TSP tour",
            template_values={"edit_json": edit.to_json()},
            descriptors={
                "algorithm_family": "typed-hir",
                "runtime_class": "compiled-local-search",
                "generation_backend": "evolution",
                "hir_operator": edit.operator,
                "hir_target": edit.target,
            },
            confidence=0.5,
        )

    def analyze(
        self,
        context: ResearchContext,
        directive: SearchDirective,
        proposal: CandidateProposal,
        feedback: tuple[dict[str, Any], ...],
    ) -> AnalysisDraft:
        del context, directive, proposal
        final = feedback[-1] if feedback else {}
        gate = str(final.get("gate", "missing"))
        metrics = final.get("metric_vector", {})
        return AnalysisDraft(
            observation=f"authority gate={gate}; public metrics={metrics}",
            interpretation="the shared QD policy will retain this genotype only if evaluator fitness warrants it",
            confidence=1.0,
            next_test="mutate or recombine an evaluator-selected archive elite under the same budget",
        )


def _candidate_records(context: ResearchContext) -> Iterable[dict[str, Any]]:
    yield from context.candidates
    for lineage in context.selected_lineages:
        yield from lineage


def _context_genome(context: ResearchContext, candidate_id: str) -> TspGenome | None:
    for candidate in _candidate_records(context):
        if candidate.get("id") != candidate_id:
            continue
        descriptors = candidate.get("descriptors", {})
        state = descriptors.get("template_state", {}) if isinstance(descriptors, dict) else {}
        genome = state.get("genome") if isinstance(state, dict) else None
        if isinstance(genome, dict):
            return TspGenome.from_mapping(genome)
    return None


def _one_component_crossover(
    left: TspGenome,
    right: TspGenome,
    rng: random.Random,
) -> tuple[TspGenome, str]:
    left_map = left.to_mapping()
    right_map = right.to_mapping()
    differing = [
        component
        for component in (
            "constructor", "edge_generator", "edge_score", "move_schedule",
            "acceptance", "diversification", "adaptation",
        )
        if left_map[component] != right_map[component]
    ]
    if not differing:
        child = mutate_genome(left, seed=rng.randrange(2**31))
        changed = next(
            component
            for component in (
                "constructor", "edge_generator", "edge_score", "move_schedule",
                "acceptance", "diversification", "adaptation",
            )
            if left_map[component] != child.to_mapping()[component]
        )
        return child, changed
    component = rng.choice(sorted(differing))
    left_map[component] = right_map[component]
    return TspGenome.from_mapping(left_map), component


def _tune_one_parameter(
    genome: TspGenome, rng: random.Random
) -> tuple[TspGenome, str]:
    mapping = genome.to_mapping()
    choices = ["edge_generator.top_k", "move_schedule.max_iterations", "adaptation.stagnation_limit"]
    if mapping["diversification"]["kind"] != "none":
        choices.append("diversification.restarts")
    target = rng.choice(choices)
    component, _, field = target.partition(".")
    values = dict(mapping[component])
    current = int(values[field])
    bounds = {
        "edge_generator.top_k": (2, 32, 2),
        "move_schedule.max_iterations": (20, 20_000, 200),
        "adaptation.stagnation_limit": (1, 1_000, 4),
        "diversification.restarts": (0, 48, 1),
    }
    low, high, step = bounds[target]
    direction = -1 if current >= high else (1 if current <= low else rng.choice((-1, 1)))
    values[field] = max(low, min(high, current + direction * step))
    mapping[component] = values
    return TspGenome.from_mapping(mapping), target


def _distill_one_component(genome: TspGenome) -> tuple[TspGenome, str]:
    mapping = genome.to_mapping()
    distance = {"kind": "feature", "name": "distance"}
    if mapping["edge_score"] != distance:
        mapping["edge_score"] = distance
        return TspGenome.from_mapping(mapping), "edge_score"
    if mapping["move_schedule"]["kind"] != "two_opt":
        mapping["move_schedule"] = {
            **mapping["move_schedule"],
            "kind": "two_opt",
            "or_opt_segment": 1,
            "three_opt_trials": 0,
        }
        return TspGenome.from_mapping(mapping), "move_schedule"
    diversification = dict(mapping["diversification"])
    if diversification["kind"] != "none":
        diversification["restarts"] = max(0, int(diversification["restarts"]) // 2)
        if diversification["restarts"] == 0:
            diversification["kind"] = "none"
        mapping["diversification"] = diversification
        return TspGenome.from_mapping(mapping), "diversification"
    edge = dict(mapping["edge_generator"])
    if int(edge["top_k"]) > 2:
        edge["top_k"] = int(edge["top_k"]) - 1
    elif edge["kind"] != "k_nearest":
        edge["kind"] = "k_nearest"
        edge["angular_sectors"] = 1
    else:
        moves = dict(mapping["move_schedule"])
        moves["max_iterations"] = max(20, int(moves["max_iterations"]) - 200)
        if moves["max_iterations"] == mapping["move_schedule"]["max_iterations"]:
            moves["max_iterations"] = 21
        mapping["move_schedule"] = moves
        return TspGenome.from_mapping(mapping), "move_schedule"
    mapping["edge_generator"] = edge
    return TspGenome.from_mapping(mapping), "edge_generator"


__all__ = ["EvolutionaryHIRModel"]
