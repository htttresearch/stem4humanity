"""Append-only projections for pre-instrumentation campaign histories."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

from algofinder.agents.ledger import CampaignLedger
from algofinder.agents.learning.eligibility import ExperienceEligibilityReport
from algofinder.agents.learning.store import LearningArtifactStore
from algofinder.trace.serialize import canonical_dumps


class BackfillError(RuntimeError):
    """Raised when an historical campaign cannot safely be projected."""


def backfill_pilot_summary(
    *,
    campaigns_root: str | Path,
    campaign_id: str,
    eligibility: ExperienceEligibilityReport,
    learning_store: LearningArtifactStore,
) -> dict[str, Any]:
    """Project one historical pilot summary without modifying its campaign ledger.

    The historic loop records outer proposal rounds, but not exact retries,
    prompts, or propensities.  This function makes those missing dimensions
    explicit instead of inventing causal joins.
    """
    if eligibility.tier not in {"A", "B", "C"}:
        raise BackfillError("quarantined evidence cannot be backfilled")
    ledger = CampaignLedger(campaigns_root, campaign_id)
    summary_path = ledger.root / "pilot-summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackfillError(f"missing or invalid pilot summary for {campaign_id}") from exc
    if not isinstance(summary, dict) or summary.get("campaign_id") != campaign_id:
        raise BackfillError("pilot summary campaign id does not match ledger")
    candidates = {item["id"]: item for item in ledger.list("candidate")}
    evaluations = [
        item for item in ledger.list("evaluation")
        if item.get("zone") == "public" and item.get("stage") in {"gate_0", "gate_1"}
    ]
    evaluations_by_candidate: dict[str, list[str]] = {}
    final_public_by_candidate: dict[str, dict[str, Any]] = {}
    for item in evaluations:
        candidate_id = item.get("candidate_id")
        if isinstance(candidate_id, str):
            evaluations_by_candidate.setdefault(candidate_id, []).append(str(item["id"]))
            incumbent = final_public_by_candidate.get(candidate_id)
            if incumbent is None or str(item.get("stage")) > str(incumbent.get("stage")):
                final_public_by_candidate[candidate_id] = item

    agent_specs = ledger.list("agent_spec")
    agent_spec = agent_specs[0] if len(agent_specs) == 1 else {}
    sampling = agent_spec.get("sampling", {}) if isinstance(agent_spec.get("sampling"), dict) else {}
    model = agent_spec.get("model") if isinstance(agent_spec.get("model"), str) else None
    representation = sampling.get("representation") if isinstance(sampling.get("representation"), str) else None
    campaign_spec = ledger.campaign()
    profile_id = campaign_spec.get("distribution_profile_id")
    source_digest = eligibility.provenance.get("source_digest")
    model_seed = summary.get("model_seed")
    split_group_id = (
        f"{source_digest or 'unknown-source'}:{profile_id or 'unknown-profile'}:seed-{model_seed}"
        if isinstance(model_seed, int) else campaign_id
    )

    rows: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(summary.get("steps", ()), start=1):
        if not isinstance(raw, dict):
            continue
        candidate_id = raw.get("candidate_id") if isinstance(raw.get("candidate_id"), str) else None
        candidate = candidates.get(candidate_id) if candidate_id else None
        evaluated = candidate is not None
        descriptors = candidate.get("descriptors", {}) if candidate else {}
        template_state = descriptors.get("template_state", {}) if isinstance(descriptors, dict) else {}
        priority_expression = template_state.get("priority_expression") if isinstance(template_state, dict) else None
        final_public = final_public_by_candidate.get(candidate_id) if candidate_id else None
        archived = raw.get("status") == "archived"
        rows.append({
            "historical_attempt_id": f"backfill-{campaign_id}-{ordinal}",
            "campaign_id": campaign_id,
            "outer_attempt_number": ordinal,
            "operator": raw.get("operator"),
            "terminal_status": raw.get("status"),
            "outer_attempt_join": "exact",
            # The outer round is exact, but historical summaries do not expose
            # structured-generation retries or the exact model context.
            "backfill_status": "backfilled_partial",
            "backfill_confidence": 0.85 if evaluated else 0.65,
            "candidate_id": candidate_id,
            "hypothesis_id": candidate.get("hypothesis_id") if candidate else None,
            "evaluation_ids": evaluations_by_candidate.get(candidate_id, []) if candidate_id else [],
            "final_rejection_reasons": list(raw.get("validation_errors", ())),
            "transition_ids": [],
            "unresolved_hypothesis_ids": [] if evaluated else ["unknown-rejected-proposal"],
            "reconstructed_context": None,
            "per_attempt_token_cost": None,
            "policy_propensity": None,
            "selection_probability": None,
            "model": model,
            "representation": representation,
            "profile_id": profile_id,
            "split_group_id": split_group_id,
            "source_digest": source_digest,
            "prompt_digest": agent_spec.get("prompt_digest"),
            "search_policy_version": campaign_spec.get("search_policy_version"),
            "tool_api_version": agent_spec.get("tool_api_version"),
            "priority_expression": priority_expression,
            "policy_decision": {
                "operator": raw.get("operator"),
                "template_values": ({"priority_expression": priority_expression} if isinstance(priority_expression, str) else {}),
            },
            "reward": {
                "all_requested_gates_passed": archived,
                "public_metric_vector": dict(final_public.get("metric_vector", {})) if isinstance(final_public, dict) else {},
                "public_runtime_summary": _runtime_summary(final_public),
            },
        })
    model_usage = summary.get("model_usage", [])
    payload = {
        "campaign_id": campaign_id,
        "eligibility_digest": eligibility.to_mapping()["content_digest"],
        "source": str(summary_path),
        "model_usage_totals": _usage_totals(model_usage),
        "attempts": rows,
        "evidence_role": "historical_training" if eligibility.tier in {"A", "B"} else "diagnostic",
        "eligibility_report": eligibility.to_mapping(),
    }
    payload["projection_digest"] = sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()
    backfill_id = f"backfill-{campaign_id}"
    digest = learning_store.write_backfill(backfill_id, payload)
    return {"backfill_id": backfill_id, "content_digest": digest, **payload}


def _usage_totals(rows: Any) -> dict[str, int]:
    if not isinstance(rows, list):
        return {"prompt_tokens": 0, "completion_tokens": 0, "tokens": 0}
    prompt = sum(int(item.get("prompt_tokens", 0)) for item in rows if isinstance(item, dict))
    completion = sum(int(item.get("completion_tokens", 0)) for item in rows if isinstance(item, dict))
    return {"prompt_tokens": prompt, "completion_tokens": completion, "tokens": prompt + completion}


def _runtime_summary(evaluation: dict[str, Any] | None) -> dict[str, Any]:
    feedback = evaluation.get("feedback", {}) if isinstance(evaluation, dict) else {}
    raw = feedback.get("raw_runs", ()) if isinstance(feedback, dict) else ()
    runs = [item for item in raw if isinstance(item, dict)] if isinstance(raw, (list, tuple)) else []
    candidate = [item for item in runs if "control" not in item.get("tags", ())]
    walls = sorted(
        float(item["wall_seconds"])
        for item in candidate
        if item.get("status") == "ok"
        and isinstance(item.get("wall_seconds"), (int, float))
        and not isinstance(item.get("wall_seconds"), bool)
    )
    if not candidate:
        return {}
    percentile_index = max(0, math.ceil(0.95 * len(walls)) - 1) if walls else 0
    return {
        "candidate_runs": len(candidate),
        "ok_runs": sum(item.get("status") == "ok" for item in candidate),
        "timeout_runs": sum(item.get("status") == "timeout" for item in candidate),
        "p95_wall_seconds": walls[percentile_index] if walls else None,
        "max_wall_seconds": walls[-1] if walls else None,
    }


__all__ = ["BackfillError", "backfill_pilot_summary"]
