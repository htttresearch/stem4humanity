"""Preregistered, campaign-level outer-evaluation planning and admission gates.

This module creates plans and evaluates already-recorded campaign summaries. It
does not start Ollama, submit a candidate, or access a holdout manifest itself.
That separation prevents an analytics command from becoming an evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from statistics import median
from typing import Any, Iterable, Literal, Mapping

from algofinder.agents.contracts import new_id
from algofinder.agents.learning.contracts import AgentEvaluation
from algofinder.agents.learning.store import LearningArtifactStore


Phase = Literal["outer-validation", "sealed-test"]
BASELINE_POLICY_ID = "policy-baseline-qd-map-elites-ucb3"


@dataclass(frozen=True)
class CampaignArm:
    campaign_key: str
    task_profile_id: str
    model: str
    policy_id: str
    seed: int
    outer_rounds: int
    max_transitions_per_round: int
    token_cap: int
    evaluated_candidate_cap: int
    wall_seconds: int

    def to_mapping(self) -> dict[str, Any]:
        return self.__dict__.copy()


def prepare_agent_evaluation(
    *,
    store: LearningArtifactStore,
    learned_policy_id: str,
    profile_ids: tuple[str, ...],
    models: tuple[str, ...],
    phase: Phase,
    evaluation_id: str | None = None,
) -> dict[str, Any]:
    """Freeze a balanced plan before any campaign begins."""
    if phase not in {"outer-validation", "sealed-test"}:
        raise ValueError("phase must be outer-validation or sealed-test")
    # The validation admission rule requires wins on at least three task
    # profiles, so fewer than four profiles would make the gate impossible or
    # leave no room for a single heterogeneous failure.
    expected_profiles = 4 if phase == "outer-validation" else 6
    if len(profile_ids) != expected_profiles:
        raise ValueError(f"{phase} requires exactly {expected_profiles} task profiles")
    if len(models) != 2 or len(set(models)) != 2:
        raise ValueError("v1 acceptance plan requires exactly two distinct model arms")
    # Reading verifies the frozen policy digest before it can be scheduled.
    policy = store.read(f"policies/{learned_policy_id}/policy.json")
    if policy.get("kind") != "agent_policy" or policy.get("policy_kind") != "learned":
        raise ValueError("outer evaluation requires a frozen learned policy")
    expected_visibility = "outer_validation" if phase == "outer-validation" else "sealed"
    for profile_id in profile_ids:
        profile = store.read(f"distributions/{profile_id}.json")
        parameters = profile.get("parameter_schema", {})
        if profile.get("kind") != "distribution_profile" or profile.get("id") != profile_id:
            raise ValueError(f"invalid frozen distribution profile {profile_id!r}")
        if not isinstance(parameters, Mapping) or parameters.get("visibility") != expected_visibility:
            raise ValueError(
                f"profile {profile_id!r} must have {expected_visibility!r} visibility for {phase}"
            )
    phase_seed = 9_100_000 if phase == "outer-validation" else 9_200_000
    rounds = 32 if phase == "outer-validation" else 48
    token_cap = 233_333 if phase == "outer-validation" else 350_000
    arm_order = _williams_order((BASELINE_POLICY_ID, learned_policy_id), models)
    arms: list[CampaignArm] = []
    for task_index, profile_id in enumerate(profile_ids):
        rotated_order = arm_order[task_index % len(arm_order):] + arm_order[:task_index % len(arm_order)]
        for position, (policy_id, model) in enumerate(rotated_order):
            key = f"{phase}-{task_index + 1:02d}-{position + 1:02d}"
            arms.append(CampaignArm(
                campaign_key=key,
                task_profile_id=profile_id,
                model=model,
                policy_id=policy_id,
                # Baseline and learned policies receive the same stochastic
                # campaign seed within each model/profile pair. Order is still
                # counterbalanced by the Williams rotation.
                seed=_arm_seed(f"{phase}:{profile_id}:{model}", phase_seed),
                outer_rounds=rounds,
                max_transitions_per_round=3,
                token_cap=token_cap,
                evaluated_candidate_cap=8,
                wall_seconds=2_700,
            ))
    evaluation = AgentEvaluation(
        evaluation_id=evaluation_id or new_id("agent-evaluation"),
        policy_id=learned_policy_id,
        distribution_profile_id=profile_ids[0],
        evidence_role="outer_validation" if phase == "outer-validation" else "sealed",
        metrics={"campaign_arms": [arm.to_mapping() for arm in arms]},
        paired_policy_id=BASELINE_POLICY_ID,
        phase=f"{phase}_planned",
        policy_arms=(BASELINE_POLICY_ID, learned_policy_id),
        model_arms=models,
        task_profile_ids=profile_ids,
        budgets={"outer_rounds": rounds, "max_transitions_per_round": 3, "token_cap": token_cap, "evaluated_candidate_cap": 8, "wall_seconds": 2_700},
        integrity={
            "policy_frozen": True,
            "profiles_frozen": True,
            "authority_only_scores": True,
            "holdouts_deferred": True,
        },
    )
    digest = store.write(evaluation)
    return {"evaluation_id": evaluation.evaluation_id, "content_digest": digest, "arms": [arm.to_mapping() for arm in arms]}


def finalize_agent_evaluation(
    *,
    store: LearningArtifactStore,
    planned_evaluation_id: str,
    campaign_summaries: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the preregistered campaign-level admission gate to paired results."""
    plan = store.read(f"evaluations/{planned_evaluation_id}.json")
    phase = str(plan.get("phase", ""))
    if not phase.endswith("_planned"):
        raise ValueError("evaluation record is not a planned preregistration")
    base_phase: Phase = "outer-validation" if phase.startswith("outer-validation") else "sealed-test"
    outcomes = [dict(item) for item in campaign_summaries]
    _validate_outcomes(plan, outcomes)
    decision = _admission_decision(base_phase, outcomes)
    result = AgentEvaluation(
        evaluation_id=new_id("agent-evaluation"),
        policy_id=str(plan["policy_id"]),
        distribution_profile_id=str(plan["distribution_profile_id"]),
        evidence_role=str(plan["evidence_role"]),
        metrics={"campaign_outcomes": outcomes, "scorecard": _scorecard(outcomes)},
        paired_policy_id=BASELINE_POLICY_ID,
        phase=f"{base_phase}_complete",
        policy_arms=tuple(str(item) for item in plan.get("policy_arms", ())),
        model_arms=tuple(str(item) for item in plan.get("model_arms", ())),
        task_profile_ids=tuple(str(item) for item in plan.get("task_profile_ids", ())),
        budgets=dict(plan.get("budgets", {})),
        integrity={"policy_frozen": True, "authority_only_scores": True, "all_campaigns_completed_before_holdout": True},
        decision=decision,
    )
    result = AgentEvaluation(
        **{
            **result.__dict__,
            "integrity": {
                **result.integrity,
                "planned_evaluation_id": planned_evaluation_id,
                "planned_evaluation_digest": plan["content_digest"],
            },
        }
    )
    digest = store.write(result)
    return {"evaluation_id": result.evaluation_id, "content_digest": digest, "decision": decision, "scorecard": result.metrics["scorecard"]}


def _williams_order(policies: tuple[str, str], models: tuple[str, str]) -> tuple[tuple[str, str], ...]:
    """Four-arm Williams order, rotated per task by the coordinator's caller."""
    return ((policies[0], models[0]), (policies[1], models[1]), (policies[1], models[0]), (policies[0], models[1]))


def _arm_seed(key: str, phase_seed: int) -> int:
    return phase_seed + int(sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 900_000


def _validate_outcomes(plan: Mapping[str, Any], outcomes: list[dict[str, Any]]) -> None:
    planned_arms = plan.get("metrics", {}).get("campaign_arms", ())
    expected_rows = {
        str(item["campaign_key"]): item
        for item in planned_arms if isinstance(item, Mapping) and item.get("campaign_key")
    }
    expected = set(expected_rows)
    supplied_keys = [str(item.get("campaign_key")) for item in outcomes]
    supplied = set(supplied_keys)
    if supplied != expected:
        raise ValueError(f"campaign outcome keys differ from preregistration; missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}")
    if len(supplied_keys) != len(supplied):
        raise ValueError("campaign outcomes contain duplicate campaign keys")
    for outcome in outcomes:
        key = str(outcome.get("campaign_key"))
        arm = expected_rows[key]
        for field in ("task_profile_id", "model", "policy_id"):
            if outcome.get(field) != arm.get(field):
                raise ValueError(
                    f"campaign {key} changed frozen {field}: expected={arm.get(field)!r}, got={outcome.get(field)!r}"
                )
        if outcome.get("integrity_passed") is not True:
            raise ValueError(f"campaign {key} did not pass integrity checks")
        if outcome.get("instance_rows_as_replicates"):
            raise ValueError("outer evaluation must use campaign-level, not instance-level, replicates")
        for field in (
            "accepted_normalized_novel", "duplicate_rate_percent",
            "champion_holdout_gap_percent", "tokens", "wall_seconds",
            "contract_failure_rate",
        ):
            value = outcome.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"campaign {key} has invalid required metric {field!r}")
            if field in {"accepted_normalized_novel", "duplicate_rate_percent", "tokens", "wall_seconds", "contract_failure_rate"} and value < 0:
                raise ValueError(f"campaign {key} has negative metric {field!r}")


def _admission_decision(phase: Phase, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in outcomes:
        by_pair.setdefault((str(row["model"]), str(row["task_profile_id"])), {})[str(row["policy_id"])] = row
    learned_ids = {str(row["policy_id"]) for row in outcomes} - {BASELINE_POLICY_ID}
    if len(learned_ids) != 1:
        raise ValueError("outer evaluation requires exactly one learned policy arm")
    learned_id = next(iter(learned_ids))
    decisions: dict[str, str] = {}
    for model in sorted({str(row["model"]) for row in outcomes}):
        pairs = [values for (candidate_model, _), values in by_pair.items() if candidate_model == model]
        if any(set(pair) != {BASELINE_POLICY_ID, learned_id} for pair in pairs):
            raise ValueError(f"model {model!r} has an incomplete policy pair")
        deltas = [_delta(pair[learned_id], pair[BASELINE_POLICY_ID]) for pair in pairs if learned_id in pair and BASELINE_POLICY_ID in pair]
        if phase == "outer-validation":
            novel_wins = sum(delta["accepted_novel"] >= 1.0 for delta in deltas)
            passes = novel_wins >= 3 and _median(d["holdout_gap"] for d in deltas) <= 0.10 and _median(d["token_ratio"] for d in deltas) <= 1.25 and _median(d["wall_ratio"] for d in deltas) <= 1.25
        else:
            duplicate_reduction = _median(d["duplicate_rate"] for d in deltas)
            novel_gain = _median(d["accepted_novel"] for d in deltas)
            novel_wins = sum(delta["accepted_novel"] > 0.0 for delta in deltas)
            passes = duplicate_reduction <= -10.0 and novel_gain >= 2.0 and novel_wins >= 4 and _median(d["holdout_gap"] for d in deltas) <= 0.10 and _median(d["token_ratio"] for d in deltas) <= 1.10 and _median(d["wall_ratio"] for d in deltas) <= 1.10 and all(d["failure_rate"] <= 0 for d in deltas)
        decisions[model] = "experimental" if passes else "rejected"
    return {"by_model": decisions, "default_policy_changed": False, "phase": phase}


def _delta(learned: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    def number(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
        value = row.get(key, default)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default
    return {
        "accepted_novel": number(learned, "accepted_normalized_novel") - number(baseline, "accepted_normalized_novel"),
        "duplicate_rate": number(learned, "duplicate_rate_percent") - number(baseline, "duplicate_rate_percent"),
        "holdout_gap": number(learned, "champion_holdout_gap_percent") - number(baseline, "champion_holdout_gap_percent"),
        "token_ratio": number(learned, "tokens", 1.0) / max(number(baseline, "tokens", 1.0), 1e-12),
        "wall_ratio": number(learned, "wall_seconds", 1.0) / max(number(baseline, "wall_seconds", 1.0), 1e-12),
        "failure_rate": number(learned, "contract_failure_rate") - number(baseline, "contract_failure_rate"),
    }


def _scorecard(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    return {"campaign_count": len(outcomes), "models": sorted({str(item["model"]) for item in outcomes}), "task_profiles": sorted({str(item["task_profile_id"]) for item in outcomes})}


def _median(values: Iterable[float]) -> float:
    data = list(values)
    return float(median(data)) if data else float("inf")


__all__ = ["BASELINE_POLICY_ID", "CampaignArm", "finalize_agent_evaluation", "prepare_agent_evaluation"]
