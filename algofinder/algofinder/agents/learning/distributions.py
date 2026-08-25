"""Task-distribution profiles and transparent profile-distance calculations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Any, Mapping

from algofinder.agents.contracts import new_id
from algofinder.agents.learning.contracts import DistributionProfile
from algofinder.trace.serialize import canonical_dumps


@dataclass(frozen=True)
class ProfileSummary:
    profile: DistributionProfile
    numeric_features: dict[str, float]
    lineage_namespace: str
    visibility: str


def build_distribution_profile(
    *,
    family: str,
    version: str,
    manifest_digests: tuple[str, ...],
    lineage_namespace: str,
    generators: Mapping[str, Any],
    feature_summary: Mapping[str, float],
    seed_set: tuple[int, ...],
    split_rule: str,
    deployment_intent: str,
    allowed_specialization_axes: tuple[str, ...] = (),
    meaning_preserving_transformations: tuple[str, ...] = (),
    diagnostic_transformations: tuple[str, ...] = (),
    visibility: str = "public_learning",
    profile_id: str | None = None,
) -> ProfileSummary:
    """Create an immutable declaration that never incorporates evaluation scores."""
    if not lineage_namespace.strip() or not deployment_intent.strip() or not generators or not feature_summary:
        raise ValueError("distribution lineage, generators, feature summary, and deployment intent are required")
    if visibility not in {"public_learning", "outer_validation", "sealed"}:
        raise ValueError("invalid distribution-profile visibility")
    if not manifest_digests or len(set(manifest_digests)) != len(manifest_digests):
        raise ValueError("distribution profiles require unique manifest digests")
    if any(len(item) != 64 or any(char not in "0123456789abcdef" for char in item) for item in manifest_digests):
        raise ValueError("manifest digests must be sha256 hex values")
    if any(not _finite(value) for value in feature_summary.values()):
        raise ValueError("distribution feature summaries must be finite numeric values")
    parameters = {
        "manifest_digests": list(manifest_digests),
        "lineage_namespace": lineage_namespace,
        "generators": dict(generators),
        "feature_summary": dict(feature_summary),
        "deployment_intent": deployment_intent,
        "allowed_specialization_axes": list(allowed_specialization_axes),
        "meaning_preserving_transformations": list(meaning_preserving_transformations),
        "diagnostic_transformations": list(diagnostic_transformations),
        "visibility": visibility,
    }
    profile = DistributionProfile(
        profile_id=profile_id or new_id("profile"),
        family=family,
        version=version,
        parameter_schema=parameters,
        seed_set=seed_set,
        split_rule=split_rule,
    )
    numeric = {key: float(value) for key, value in feature_summary.items()}
    return ProfileSummary(profile, numeric, lineage_namespace, visibility)


def profile_distance(left: ProfileSummary, right: ProfileSummary) -> float:
    """Deterministic, scale-stabilized distance for retrieval/routing only."""
    if left.lineage_namespace != right.lineage_namespace:
        return float("inf")
    keys = sorted(set(left.numeric_features) | set(right.numeric_features))
    if not keys:
        return 0.0
    terms = []
    for key in keys:
        lhs = left.numeric_features.get(key, 0.0)
        rhs = right.numeric_features.get(key, 0.0)
        scale = max(1.0, abs(lhs), abs(rhs))
        terms.append(((lhs - rhs) / scale) ** 2)
    return math.sqrt(sum(terms) / len(terms))


def profile_feature_digest(summary: ProfileSummary) -> str:
    return sha256(canonical_dumps({"features": summary.numeric_features, "lineage": summary.lineage_namespace}).encode("utf-8")).hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


__all__ = ["ProfileSummary", "build_distribution_profile", "profile_distance", "profile_feature_digest"]
