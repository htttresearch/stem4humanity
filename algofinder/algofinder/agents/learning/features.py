"""Approved state/action/outcome projections for learning datasets."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping

from algofinder.agents.learning.normalization import NormalizationError, normalize_formula
from algofinder.trace.serialize import canonical_dumps


FEATURE_SCHEMA_ID = "agent-learning-state-action-outcome@3"
FEATURE_SCHEMA_DIGEST = sha256(FEATURE_SCHEMA_ID.encode("utf-8")).hexdigest()


def project_attempt(record: Mapping[str, Any], *, episode: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Flatten only agent-visible provenance; no raw blobs or hidden metrics."""
    reward = record.get("reward", {}) if isinstance(record.get("reward"), Mapping) else {}
    action = record.get("policy_decision", {}) if isinstance(record.get("policy_decision"), Mapping) else {}
    model_identity = (episode or {}).get("model_identity", {})
    if not isinstance(model_identity, Mapping):
        model_identity = {}
    template_values = action.get("template_values", {}) if isinstance(action.get("template_values"), Mapping) else {}
    expression = template_values.get("priority_expression")
    public_metrics = reward.get("public_metric_vector", {}) if isinstance(reward.get("public_metric_vector"), Mapping) else {}
    runtime = reward.get("public_runtime_summary", {}) if isinstance(reward.get("public_runtime_summary"), Mapping) else {}
    objective_value = public_metrics.get("paired_gap_percent")
    terminal_status = str(record.get("terminal_status", ""))
    if "all_requested_gates_passed" in reward:
        passed = bool(reward["all_requested_gates_passed"])
    else:
        passed = terminal_status in {"archived", "evaluated_pass"}
    formula: dict[str, Any] = {}
    selected_expression = expression if isinstance(expression, str) else record.get("priority_expression")
    if isinstance(selected_expression, str):
        try:
            formula_identity = normalize_formula(selected_expression)
            selected_expression = formula_identity.exact_expression
            formula = {
                "formula_exact_digest": formula_identity.exact_digest,
                "formula_structural_digest": formula_identity.structural_digest,
                "formula_behavior_digest": formula_identity.probe_behavior_digest,
                "formula_features": formula_identity.features,
            }
        except NormalizationError:
            formula = {}
    return {
        "campaign_id": record.get("campaign_id"),
        "episode_id": record.get("episode_id"),
        "attempt_id": record.get("id") or record.get("attempt_id") or record.get("historical_attempt_id"),
        "attempt_number": record.get("attempt_number", record.get("outer_attempt_number")),
        "operator": record.get("operator", action.get("operator")),
        "parent_count": len(record.get("parent_ids", action.get("parent_ids", ()))),
        "terminal_status": record.get("terminal_status"),
        "candidate_id": record.get("candidate_id"),
        "evaluation_count": len(record.get("evaluation_ids", ())),
        "transition_count": len(record.get("transition_ids", ())),
        "backfill_status": record.get("backfill_status", "native_exact"),
        "backfill_confidence": record.get("backfill_confidence", 1.0),
        "model": model_identity.get("model", record.get("model")),
        "representation": model_identity.get("representation", record.get("representation")),
        "profile_id": (episode or {}).get("distribution_profile_id", record.get("profile_id")),
        "evidence_role": (episode or {}).get("evidence_role", record.get("evidence_role", "public_learning")),
        "valid": bool(record.get("candidate_id")) or str(record.get("terminal_status", "")).startswith("evaluated"),
        "passed": passed,
        "quality_improved": isinstance(objective_value, (int, float)) and not isinstance(objective_value, bool) and float(objective_value) < 0.0,
        "objective_value": objective_value,
        "objective_regret": objective_value,
        "ok_rate": public_metrics.get("ok_rate"),
        "public_strata": public_metrics.get("strata", ()),
        "runtime_p95_seconds": runtime.get("p95_wall_seconds"),
        "runtime_max_seconds": runtime.get("max_wall_seconds"),
        "runtime_timeout_count": runtime.get("timeout_runs", 0),
        "priority_expression": selected_expression,
        **formula,
        "selection_probability": record.get("selection_probability"),
        "split_group_id": record.get("split_group_id"),
        "source_digest": record.get("source_digest"),
        "prompt_digest": record.get("prompt_digest"),
        "search_policy_version": record.get("search_policy_version"),
        "tool_api_version": record.get("tool_api_version"),
    }


def feature_schema_digest(extra: Mapping[str, Any] | None = None) -> str:
    return sha256(canonical_dumps({"schema": FEATURE_SCHEMA_ID, "extra": dict(extra or {})}).encode("utf-8")).hexdigest()


__all__ = ["FEATURE_SCHEMA_DIGEST", "FEATURE_SCHEMA_ID", "feature_schema_digest", "project_attempt"]
