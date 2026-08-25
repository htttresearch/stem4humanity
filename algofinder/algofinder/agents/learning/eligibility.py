"""Conservative compatibility and leakage checks for learning evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Literal

from algofinder.agents.ledger import CampaignLedger, LedgerError
from algofinder.agents.learning.contracts import SCHEMA_ID, SCHEMA_VERSION
from algofinder.trace.serialize import canonical_dumps


EligibilityTier = Literal["A", "B", "C", "D"]


@dataclass(frozen=True)
class ExperienceEligibilityReport:
    """Immutable-style assessment made before a campaign enters a dataset."""

    campaign_id: str
    tier: EligibilityTier
    eligible_for: tuple[str, ...]
    reasons: tuple[str, ...]
    provenance: dict[str, Any]
    counts: dict[str, int]

    def to_mapping(self) -> dict[str, Any]:
        body = {
            "kind": "experience_eligibility_report",
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "id": f"eligibility-{self.campaign_id}",
            "campaign_id": self.campaign_id,
            "tier": self.tier,
            "eligible_for": list(self.eligible_for),
            "reasons": list(self.reasons),
            "provenance": self.provenance,
            "counts": self.counts,
        }
        body["content_digest"] = sha256(canonical_dumps(body).encode("utf-8")).hexdigest()
        return body


def registered_campaign_ids(summary_path: str | Path) -> frozenset[str]:
    """Extract campaign ids from a comparison summary without trusting filenames."""
    try:
        payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid comparison summary: {summary_path}") from exc
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        raise ValueError("comparison summary is not marked complete")
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "campaign_id" and isinstance(item, str):
                    found.add(item)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    return frozenset(found)


def assess_campaign(
    campaigns_root: str | Path,
    campaign_id: str,
    *,
    registered_final_campaign_ids: Iterable[str] = (),
    allow_diagnostic: bool = False,
) -> ExperienceEligibilityReport:
    """Verify immutable evidence and assign the strictest applicable tier.

    Tier A is a registered final-comparison campaign.  Tier B is a closed,
    compatible campaign with complete reconstructible provenance.  Tier C can
    only supply named diagnostic failure cases.  Tier D is quarantined.
    """
    root = Path(campaigns_root)
    ledger = CampaignLedger(root, campaign_id)
    reasons: list[str] = []
    counts: dict[str, int] = {}
    provenance: dict[str, Any] = {}
    try:
        spec = ledger.campaign()
        for kind in ("agent_run", "agent_episode", "research_attempt", "agent_transition", "hypothesis", "candidate", "experiment", "evaluation", "analysis", "decision"):
            counts[kind] = len(ledger.list(kind))
    except LedgerError as exc:
        return ExperienceEligibilityReport(
            campaign_id=campaign_id, tier="D", eligible_for=(),
            reasons=(f"immutable ledger verification failed: {exc}",), provenance={}, counts={},
        )

    source_snapshot = ledger.root / "provenance" / "experiment-sources.json"
    if not source_snapshot.is_file():
        reasons.append("missing experiment-source snapshot")
    else:
        try:
            source = json.loads(source_snapshot.read_text(encoding="utf-8"))
            if not isinstance(source, dict) or not source.get("source_digest"):
                reasons.append("invalid experiment-source snapshot")
            else:
                provenance["source_digest"] = source["source_digest"]
        except json.JSONDecodeError:
            reasons.append("invalid experiment-source snapshot JSON")
    provenance.update({
        "search_policy_version": spec.get("search_policy_version"),
        "tool_api_versions": sorted({item.get("tool_api_version") for item in ledger.list("agent_spec") if item.get("tool_api_version")}),
        "agent_models": sorted({str(item.get("model")) for item in ledger.list("agent_spec") if item.get("model")}),
        "campaign_state": _campaign_state(ledger.root),
    })
    if not provenance["tool_api_versions"]:
        reasons.append("missing agent tool API identity")
    if not provenance["agent_models"]:
        reasons.append("missing model identity")

    candidates = {item["id"]: item for item in ledger.list("candidate")}
    hypotheses = {item["id"] for item in ledger.list("hypothesis")}
    evaluations = ledger.list("evaluation")
    accepted = [item for item in candidates.values() if item.get("hypothesis_id") in hypotheses]
    dangling = [item["id"] for item in candidates.values() if item.get("hypothesis_id") not in hypotheses]
    counts["accepted_candidate_hypothesis_joins"] = len(accepted)
    counts["evaluations"] = len(evaluations)
    if dangling:
        reasons.append(f"{len(dangling)} candidates have no immutable hypothesis join")

    state = provenance["campaign_state"]
    registered = campaign_id in set(registered_final_campaign_ids)
    if state not in {"review", "closed"}:
        reasons.append(f"campaign lifecycle is {state!r}, not review or closed")
    if not reasons and registered:
        tier: EligibilityTier = "A"
        eligible = ("historical_training", "retrieval", "supervised_targets")
    elif not reasons and state == "closed":
        tier = "B"
        eligible = ("historical_training", "retrieval", "supervised_targets")
    elif allow_diagnostic and not reasons:
        tier = "C"
        eligible = ("diagnostic_failure_taxonomy",)
        reasons.append("not admitted for quality rewards or policy-value estimation")
    else:
        tier = "D"
        eligible = ()
        if state not in {"review", "closed"}:
            reasons.append(f"campaign lifecycle is {state!r}, not closed")
        if not reasons:
            reasons.append("not registered as final comparison and not explicitly admitted")
    return ExperienceEligibilityReport(
        campaign_id=campaign_id,
        tier=tier,
        eligible_for=eligible,
        reasons=tuple(reasons),
        provenance=provenance,
        counts=counts,
    )


def _campaign_state(root: Path) -> str:
    try:
        value = json.loads((root / "state.json").read_text(encoding="utf-8"))
        return str(value.get("lifecycle", "unknown")) if isinstance(value, dict) else "unknown"
    except (OSError, json.JSONDecodeError):
        return "unknown"


__all__ = ["ExperienceEligibilityReport", "EligibilityTier", "assess_campaign", "registered_campaign_ids"]
