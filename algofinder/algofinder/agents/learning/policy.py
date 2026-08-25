"""Smoothed contextual operator advisor; it never writes evaluator outcomes."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Any, Iterable, Mapping

from algofinder.agents.learning.memory import CaseMemory
from algofinder.agents.learning.models import BetaBinomialTable, ContinuousStatsTable, LinearProbabilityModel
from algofinder.agents.learning.features import FEATURE_SCHEMA_DIGEST
from algofinder.agents.learning.normalization import normalize_formula
from algofinder.agents.learning.store import LearningArtifactStore, LearningStoreError
from algofinder.agents.research import QualityDiversityPolicy, ResearchContext, SearchDirective


@dataclass(frozen=True)
class PolicyDecision:
    operator: str
    probabilities: dict[str, float]
    selection_probability: float
    rng_seed: int
    context_key: str


@dataclass(frozen=True)
class ProgramRecommendation:
    priority_expression: str
    formula_behavior_digest: str
    predicted_gap_percent: float
    uncertainty_percent: float
    conservative_gap_percent: float
    generalization_observations: int
    development_observations: int
    matched_families: tuple[str, ...]
    predicted_runtime_p95_seconds: float | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "kind": "learned_program_recommendation",
            "priority_expression": self.priority_expression,
            "formula_behavior_digest": self.formula_behavior_digest,
            "predicted_gap_percent": self.predicted_gap_percent,
            "uncertainty_percent": self.uncertainty_percent,
            "conservative_gap_percent": self.conservative_gap_percent,
            "generalization_observations": self.generalization_observations,
            "development_observations": self.development_observations,
            "matched_families": list(self.matched_families),
            "predicted_runtime_p95_seconds": self.predicted_runtime_p95_seconds,
        }


class ProgramValueAdvisor:
    """Rank canonical formulas by retrospective generalization and profile fit."""

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        self.development = ContinuousStatsTable.from_mapping(_mapping(mapping, "development"))
        self.generalization = ContinuousStatsTable.from_mapping(_mapping(mapping, "generalization"))
        self.family = ContinuousStatsTable.from_mapping(_mapping(mapping, "family"))
        self.runtime = ContinuousStatsTable.from_mapping(_mapping(mapping, "runtime"))
        catalog = mapping.get("formula_catalog", ())
        self.catalog = tuple(dict(item) for item in catalog if isinstance(item, Mapping)) if isinstance(catalog, list) else ()

    def recommend(
        self,
        profile: Mapping[str, Any] | None,
        *,
        limit: int = 4,
        exclude_behavior_digests: Iterable[str] = (),
    ) -> tuple[ProgramRecommendation, ...]:
        parameters = profile.get("parameter_schema", {}) if isinstance(profile, Mapping) else {}
        generators = parameters.get("generators", {}) if isinstance(parameters, Mapping) else {}
        raw_families = generators.get("families", ()) if isinstance(generators, Mapping) else ()
        families = tuple(sorted(str(item) for item in raw_families if isinstance(item, str)))
        excluded = frozenset(str(item) for item in exclude_behavior_digests)
        ranked: list[tuple[float, str, ProgramRecommendation]] = []
        for item in self.catalog:
            behavior = item.get("formula_behavior_digest")
            expression = item.get("priority_expression")
            if not isinstance(behavior, str) or not isinstance(expression, str):
                continue
            if behavior in excluded:
                continue
            global_mean, global_uncertainty, global_trials = self.generalization.estimate((f"formula|{behavior}",))
            development_mean, development_uncertainty, development_trials = self.development.estimate((f"formula|{behavior}",))
            family_estimates = [
                (family, *self.family.estimate((f"formula|{behavior}|family|{family}",)))
                for family in families
            ]
            observed_families = [row for row in family_estimates if row[3] > 0]
            if global_trials:
                predicted, uncertainty = global_mean, global_uncertainty
            else:
                predicted, uncertainty = development_mean, development_uncertainty
            if observed_families:
                family_mean = sum(row[1] for row in observed_families) / len(observed_families)
                family_uncertainty = sum(row[2] for row in observed_families) / len(observed_families)
                weight = 0.40 if global_trials else 0.65
                predicted = (1.0 - weight) * predicted + weight * family_mean
                uncertainty = (1.0 - weight) * uncertainty + weight * family_uncertainty
            runtime_mean, _, runtime_trials = self.runtime.estimate((f"formula|{behavior}",))
            runtime = runtime_mean if runtime_trials else None
            runtime_penalty = max(0.0, runtime_mean - 0.12) * 0.25 if runtime_trials else 0.0
            conservative = predicted + uncertainty + runtime_penalty
            recommendation = ProgramRecommendation(
                priority_expression=expression,
                formula_behavior_digest=behavior,
                predicted_gap_percent=predicted,
                uncertainty_percent=uncertainty,
                conservative_gap_percent=conservative,
                generalization_observations=global_trials,
                development_observations=development_trials,
                matched_families=tuple(row[0] for row in observed_families),
                predicted_runtime_p95_seconds=runtime,
            )
            ranked.append((conservative, behavior, recommendation))
        return tuple(item[2] for item in sorted(ranked)[:limit])


class SmoothedOperatorPolicy:
    """Softmax posterior means with a mandatory uniformly mixed exploration floor."""

    def __init__(
        self,
        posterior: BetaBinomialTable,
        *,
        quality_posterior: BetaBinomialTable | None = None,
        quality_weight: float = 0.35,
        exploration_floor: float = 0.20,
        temperature: float = 0.35,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= exploration_floor < 1.0 or temperature <= 0 or quality_weight < 0:
            raise ValueError("invalid exploration floor or temperature")
        self.posterior = posterior
        self.quality_posterior = quality_posterior or BetaBinomialTable()
        self.quality_weight = quality_weight
        self.exploration_floor = exploration_floor
        self.temperature = temperature
        self.seed = seed
        self._rng = random.Random(seed)
        self._tried: set[str] = set()

    def decide(self, available: Iterable[str], *, context_key: str) -> PolicyDecision:
        operators = tuple(sorted(set(available)))
        if not operators:
            raise ValueError("at least one operator must be available")
        unseen = tuple(operator for operator in operators if operator not in self._tried)
        if unseen:
            probabilities = {operator: (1.0 / len(unseen) if operator in unseen else 0.0) for operator in operators}
        else:
            utilities = []
            for operator in operators:
                mean, uncertainty, _ = self.posterior.posterior((f"{context_key}|{operator}", f"*|{operator}"))
                quality_mean, quality_uncertainty, quality_trials = self.quality_posterior.posterior(
                    (f"{context_key}|{operator}", f"*|{operator}")
                )
                quality_utility = quality_mean - quality_uncertainty if quality_trials else 0.0
                utilities.append((
                    operator,
                    (mean - uncertainty + self.quality_weight * quality_utility) / self.temperature,
                ))
            maximum = max(value for _, value in utilities)
            weights = {operator: math.exp(value - maximum) for operator, value in utilities}
            total = sum(weights.values())
            exploitation = {operator: value / total for operator, value in weights.items()}
            probabilities = {operator: (1.0 - self.exploration_floor) * exploitation[operator] + self.exploration_floor / len(operators) for operator in operators}
        pick = self._rng.random()
        running = 0.0
        chosen = operators[-1]
        for operator in operators:
            running += probabilities[operator]
            if pick <= running:
                chosen = operator
                break
        self._tried.add(chosen)
        return PolicyDecision(chosen, probabilities, probabilities[chosen], self.seed, context_key)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "kind": "smoothed_contextual_softmax@2",
            "exploration_floor": self.exploration_floor,
            "temperature": self.temperature,
            "quality_weight": self.quality_weight,
            "seed": self.seed,
            "posterior": self.posterior.to_mapping(),
            "quality_posterior": self.quality_posterior.to_mapping(),
        }


class LearnedQualityDiversityPolicy:
    """Drop-in ``ResearchPolicy`` that changes only the operator distribution.

    QD still owns feasible support, target-cell/parent selection, archive state,
    and all evaluator boundaries.  This wrapper merely selects an operator from
    a frozen, logged advisory distribution and retains its exact propensity.
    """

    def __init__(
        self,
        baseline: QualityDiversityPolicy,
        advisor: SmoothedOperatorPolicy,
        *,
        representation: str = "formula",
        context_key: str | None = None,
        memory: CaseMemory | None = None,
        policy_id: str | None = None,
        policy_digest: str | None = None,
        policy_version: str = "formula-memory-bandit@2",
        store: LearningArtifactStore | None = None,
        program_advisor: ProgramValueAdvisor | None = None,
    ) -> None:
        self.baseline = baseline
        self.advisor = advisor
        self.representation = representation
        self.context_key = context_key or f"*|{representation}"
        self.memory = memory or CaseMemory()
        self.policy_id = policy_id
        self.policy_digest = policy_digest
        self._policy_version = policy_version
        self.store = store
        self.program_advisor = program_advisor
        self._last_decision: PolicyDecision | None = None

    @property
    def policy_version(self) -> str:
        return self._policy_version

    def select(self) -> SearchDirective:
        snapshot = self.baseline.search.refresh()
        available = self.baseline.search.feasible_operators(snapshot)
        decision = self.advisor.decide(available, context_key=self.context_key)
        self._last_decision = decision
        return self.baseline.select_with_operator(
            decision.operator, decision.probabilities, policy_version=self.policy_version,
        )

    def context(self, directive: SearchDirective, *, prior_validation_failure: dict[str, Any] | None = None) -> ResearchContext:
        context = self.baseline.context(directive, prior_validation_failure=prior_validation_failure)
        campaign_id = str(context.campaign.get("id", "")) or None
        profile_id = context.campaign.get("distribution_profile_id")
        profile = self._profile(str(profile_id)) if profile_id else None
        profile_parameters = profile.get("parameter_schema", {}) if isinstance(profile, Mapping) else {}
        profile_generators = profile_parameters.get("generators", {}) if isinstance(profile_parameters, Mapping) else {}
        profile_families = profile_generators.get("families", ()) if isinstance(profile_generators, Mapping) else ()
        query = " ".join(str(item) for item in (
            directive.operator,
            directive.focus,
            directive.rationale,
            context.campaign.get("goal"),
            *profile_families,
            profile_parameters.get("deployment_intent") if isinstance(profile_parameters, Mapping) else None,
            (prior_validation_failure or {}).get("errors"),
        ) if item)
        known_behaviors = {
            str(descriptors["formula_behavior_digest"])
            for candidate in context.candidates
            if isinstance(candidate, Mapping)
            for descriptors in (candidate.get("descriptors"),)
            if isinstance(descriptors, Mapping)
            and isinstance(descriptors.get("formula_behavior_digest"), str)
        }
        if self.representation == "formula":
            known_behaviors.add(normalize_formula("distance").probe_behavior_digest)
        retrieved = self.memory.retrieve_balanced(
            query,
            representation=self.representation,
            profile_id=str(profile_id) if profile_id else None,
            operator=directive.operator,
            exclude_campaign_id=campaign_id,
            positive_limit=8,
            negative_limit=8,
        )
        retrieved = tuple(
            item for item in retrieved
            if str(item.document.action.get("formula_behavior_digest")) not in known_behaviors
        )[:6]
        recommendations = (
            self.program_advisor.recommend(
                profile,
                exclude_behavior_digests=known_behaviors,
            )
            if self.program_advisor is not None else ()
        )
        recommendation_rows = tuple({
            "recommendation": item.to_mapping(),
            "retrieval_score": -item.conservative_gap_percent,
            "retrieval_reasons": ["program_value", "historical_generalization", "profile_family_match"],
        } for item in recommendations)
        return replace(context, retrieved_cases=(*recommendation_rows, *tuple({
            "case": item.document.to_mapping(),
            "retrieval_score": item.score,
            "retrieval_reasons": list(item.reasons),
        } for item in retrieved)))

    def _profile(self, profile_id: str) -> dict[str, Any] | None:
        if self.store is None:
            return None
        try:
            return self.store.read(f"distributions/{profile_id}.json")
        except (LearningStoreError, OSError):
            return None

    def record_validation_failure(self, directive: SearchDirective) -> None:
        self.baseline.record_validation_failure(directive)

    def decision_metadata(self, directive: SearchDirective) -> dict[str, Any]:
        metadata = self.baseline.decision_metadata(directive)
        metadata["policy_version"] = self.policy_version
        if self._last_decision is not None:
            metadata.update({
                "operator_probabilities": dict(self._last_decision.probabilities),
                "selection_probability": metadata.get("cell_selection_probability", 1.0) * metadata.get("parent_selection_probability", 1.0) * self._last_decision.selection_probability,
                "operator_probability": self._last_decision.selection_probability,
                "advisor_context_key": self._last_decision.context_key,
                "advisor_rng_seed": self._last_decision.rng_seed,
                "learned_policy_id": self.policy_id,
                "learned_policy_digest": self.policy_digest,
            })
        return metadata


def load_learned_policy(
    *,
    store: LearningArtifactStore,
    policy_id: str,
    baseline: QualityDiversityPolicy,
    model: str,
    representation: str,
    runtime_seed: int = 0,
) -> LearnedQualityDiversityPolicy:
    """Reconstruct a runnable advisor only from verified frozen artifacts."""
    policy = store.read(f"policies/{policy_id}/policy.json")
    if policy.get("kind") != "agent_policy" or policy.get("policy_kind") != "learned":
        raise LearningStoreError(f"{policy_id!r} is not a learned agent policy")
    if policy.get("representation") != representation:
        raise LearningStoreError(
            f"policy representation {policy.get('representation')!r} does not match {representation!r}"
        )
    if policy.get("feature_schema_digest") != FEATURE_SCHEMA_DIGEST:
        raise LearningStoreError("learned policy feature schema is incompatible with this runtime")
    parameter_digest = policy.get("parameter_blob_digest")
    memory_digest = policy.get("memory_snapshot_digest")
    if not isinstance(parameter_digest, str) or not isinstance(memory_digest, str):
        raise LearningStoreError("learned policy is missing parameter or memory blobs")
    training_digest = policy.get("training_dataset_digest")
    if not isinstance(training_digest, str):
        raise LearningStoreError("learned policy is missing its training-dataset reference")
    training_reference = store.read_json_blob(training_digest)
    if not isinstance(training_reference, Mapping) or not isinstance(training_reference.get("dataset_digest"), str):
        raise LearningStoreError("learned policy has an invalid training-dataset reference")
    parameters = store.read_json_blob(parameter_digest)
    memory_mapping = store.read_json_blob(memory_digest)
    if not isinstance(parameters, Mapping) or parameters.get("schema") not in {"agent-learning-policy-parameters@2", "agent-learning-policy-parameters@3", "agent-learning-policy-parameters@4", "agent-learning-policy-parameters@5"}:
        raise LearningStoreError("unsupported learned-policy parameter schema")
    operator_config = parameters.get("operator_policy", {})
    if not isinstance(operator_config, Mapping):
        raise LearningStoreError("learned policy has invalid operator configuration")
    posterior_mapping = parameters.get("posterior", {})
    quality_posterior_mapping = parameters.get("quality_posterior", {})
    predictor_mapping = parameters.get("validity_predictor", {})
    if not isinstance(posterior_mapping, Mapping) or not isinstance(quality_posterior_mapping, Mapping) or not isinstance(predictor_mapping, Mapping):
        raise LearningStoreError("learned policy has malformed fitted models")
    # Parse the validity model now even though v1 uses it only for proposal-pool
    # ranking. This prevents a corrupt unused component from hiding in a policy.
    LinearProbabilityModel.from_mapping(predictor_mapping)
    advisor = SmoothedOperatorPolicy(
        BetaBinomialTable.from_mapping(posterior_mapping),
        quality_posterior=BetaBinomialTable.from_mapping(quality_posterior_mapping),
        quality_weight=float(operator_config.get("quality_weight", 0.35)),
        exploration_floor=float(operator_config.get("exploration_floor", 0.20)),
        temperature=float(operator_config.get("temperature", 0.35)),
        seed=int(operator_config.get("seed", 0)) ^ int(runtime_seed),
    )
    program_mapping = parameters.get("program_value_model")
    program_advisor = ProgramValueAdvisor(program_mapping) if isinstance(program_mapping, Mapping) else None
    return LearnedQualityDiversityPolicy(
        baseline,
        advisor,
        representation=representation,
        context_key=f"{model}|{representation}",
        memory=CaseMemory.from_mapping(memory_mapping if isinstance(memory_mapping, Mapping) else {}),
        policy_id=policy_id,
        policy_digest=str(policy["content_digest"]),
        policy_version=str(policy["policy_version"]),
        store=store,
        program_advisor=program_advisor,
    )


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise LearningStoreError(f"program value model is missing {key!r}")
    return item


__all__ = ["LearnedQualityDiversityPolicy", "PolicyDecision", "ProgramRecommendation", "ProgramValueAdvisor", "SmoothedOperatorPolicy", "load_learned_policy"]
