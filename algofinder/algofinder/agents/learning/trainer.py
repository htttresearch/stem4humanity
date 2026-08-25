"""Reproducible dataset-to-policy build for the deliberately small v1 models."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable

from algofinder.agents.contracts import new_id
from algofinder.agents.learning.contracts import AgentPolicy, LearningRun
from algofinder.agents.learning.memory import CaseDocument, CaseMemory
from algofinder.agents.learning.models import BetaBinomialTable, ContinuousStatsTable, fit_logistic
from algofinder.agents.learning.normalization import NormalizationError, normalize_formula
from algofinder.agents.learning.store import LearningArtifactStore
from algofinder.trace.serialize import canonical_dumps


class TrainingError(RuntimeError):
    """Raised when data cannot support a reproducible v1 policy artifact."""


_TRAINING_EVIDENCE_ROLES = frozenset({"public_learning", "historical_training", "train"})


def train_policy_from_rows(
    *,
    store: LearningArtifactStore,
    rows: Iterable[dict[str, Any]],
    dataset_digest: str,
    representation: str,
    feature_schema_digest: str,
    seed: int = 75348,
    policy_id: str | None = None,
) -> dict[str, Any]:
    """Fit posteriors + a small validity predictor and freeze inspectable JSON.

    This is a ranking advisor only: emitted coefficients do not grant a policy
    permission to accept candidates or alter evaluator scores.
    """
    if len(dataset_digest) != 64 or any(char not in "0123456789abcdef" for char in dataset_digest):
        raise TrainingError("dataset_digest must be a sha256 hex digest")
    records = [
        dict(row) for row in rows
        if row.get("evidence_role") in _TRAINING_EVIDENCE_ROLES
        and row.get("representation") in {None, representation}
    ]
    train = [row for row in records if row.get("split", "train") == "train"]
    if not train:
        raise TrainingError("no non-sealed training rows available")
    posterior = BetaBinomialTable()
    quality_posterior = BetaBinomialTable()
    development_values = ContinuousStatsTable(prior_scale=0.20)
    generalization_values = ContinuousStatsTable(prior_scale=0.15)
    family_values = ContinuousStatsTable(prior_scale=0.25)
    runtime_values = ContinuousStatsTable(prior_mean=0.10, prior_scale=0.05)
    formula_catalog: dict[str, dict[str, Any]] = {}
    for row in train:
        operator = str(row.get("operator", "unknown"))
        model = str(row.get("model") or "*")
        rep = str(row.get("representation") or representation)
        valid = bool(row.get("valid", False))
        posterior.observe(f"{model}|{rep}|{operator}", valid)
        posterior.observe(f"*|{operator}", valid)
        if bool(row.get("valid", False)) and bool(row.get("passed", False)):
            quality = _quality_value(row)
            quality_success = quality is not None and quality < 0.0
            quality_posterior.observe(f"{model}|{rep}|{operator}", quality_success)
            quality_posterior.observe(f"*|{operator}", quality_success)
            identity = _formula_identity(row)
            if identity is not None:
                behavior, expression, exact, structural = identity
                catalog = formula_catalog.setdefault(behavior, {
                    "priority_expression": expression,
                    "formula_exact_digest": exact,
                    "formula_structural_digest": structural,
                    "formula_behavior_digest": behavior,
                    "development_observations": 0,
                    "generalization_observations": 0,
                })
                development = _number(row.get("objective_value"))
                if development is not None:
                    development_values.observe(f"formula|{behavior}", development)
                    catalog["development_observations"] += 1
                generalization = _number(row.get("generalization_objective_value"))
                if generalization is not None:
                    generalization_values.observe(f"formula|{behavior}", generalization)
                    catalog["generalization_observations"] += 1
                strata = _strata(row.get("generalization_strata")) or _strata(row.get("public_strata"))
                deployment_strata = [
                    stratum for stratum in strata
                    if stratum.get("split") in {None, "validation", "test_iid"}
                ]
                for stratum in deployment_strata or strata:
                    family = stratum.get("family")
                    gap = _number(stratum.get("paired_gap_percent"))
                    if isinstance(family, str) and gap is not None:
                        family_values.observe(f"formula|{behavior}|family|{family}", gap)
                runtime = _number(row.get("generalization_runtime_p95_seconds"))
                if runtime is None:
                    runtime = _number(row.get("runtime_p95_seconds"))
                if runtime is not None:
                    runtime_values.observe(f"formula|{behavior}", runtime)
    feature_names = ("parent_count", "transition_count", "attempt_number")
    predictor_rows = [{name: float(row.get(name) or 0.0) for name in feature_names} for row in train]
    predictor = fit_logistic(
        predictor_rows,
        [bool(row.get("valid", False)) for row in train],
        feature_names=feature_names,
        regularization=1.0,
    )
    # Retrieval is part of the policy, so dev/test cases must not enter its
    # frozen memory any more than they may enter fitted coefficients.
    # A large fraction of historical rejected rows have neither a formula nor
    # a recoverable failure cause.  Indexing those generic records made BM25
    # return pages of indistinguishable "validation_rejected" examples and
    # crowded out the actual program/outcome evidence.  Keep only cases whose
    # proposed program can be inspected and either imitated or avoided.
    documents = tuple(
        _case_document(row, representation)
        for row in train
        if isinstance(row.get("priority_expression"), str)
    )
    memory = CaseMemory(documents)
    memory_id = new_id("memory")
    memory_digest = store.write_memory(memory_id, memory.to_mapping())
    memory_blob = store.put_json_blob(memory.to_mapping())
    parameter_mapping = {
        "schema": "agent-learning-policy-parameters@5",
        "posterior": posterior.to_mapping(),
        "quality_posterior": quality_posterior.to_mapping(),
        "validity_predictor": predictor.to_mapping(),
        "program_value_model": {
            "development": development_values.to_mapping(),
            "generalization": generalization_values.to_mapping(),
            "family": family_values.to_mapping(),
            "runtime": runtime_values.to_mapping(),
            "formula_catalog": [formula_catalog[key] for key in sorted(formula_catalog)],
            "objective_direction": "minimize",
            "uncertainty_policy": "posterior mean plus one standard error",
        },
        "operator_policy": {
            "kind": "conservative_smoothed_contextual_softmax",
            "exploration_floor": 0.30,
            "temperature": 0.45,
            "minimum_forced_trials_per_operator": 1,
            "quality_weight": 0.50,
            "seed": seed,
        },
        "retrieval": {"positive_cases": 3, "negative_cases": 3, "recommendations": 4, "max_context_tokens": 1500},
    }
    parameter_blob = store.put_json_blob(parameter_mapping)
    dataset_blob = store.put_json_blob({"dataset_digest": dataset_digest})
    selected_policy_id = policy_id or new_id("policy")
    learning_run_id = new_id("learning-run")
    policy = AgentPolicy(
        policy_id=selected_policy_id,
        policy_kind="learned",
        policy_version="formula-profile-value@5",
        representation=representation,
        feature_schema_digest=feature_schema_digest,
        normalizer_digest=None,
        parameter_blob_digest=parameter_blob,
        memory_id=memory_id,
        memory_snapshot_digest=memory_blob,
        training_dataset_digest=dataset_blob,
        evidence_role="public_learning",
    )
    policy_digest = store.write(policy)
    code_digest = _training_code_digest()
    run = LearningRun(
        learning_run_id=learning_run_id,
        command="algofinder-agent-learning train",
        input_digests={"dataset_manifest": dataset_digest},
        output_digests={"policy": policy_digest, "memory": memory_digest},
        config={"seed": seed, "representation": representation, "methods": ["separate_validity_and_quality_posteriors", "deployment_conditioned_program_value", "continuous_program_value_with_uncertainty", "family_conditioned_value", "runtime_tail_model", "program_identified_case_memory"]},
        code_digest=code_digest,
        status="completed",
    )
    run_digest = store.write(run)
    return {
        "learning_run_id": learning_run_id,
        "learning_run_digest": run_digest,
        "policy_id": selected_policy_id,
        "policy_digest": policy_digest,
        "memory_id": memory_id,
        "memory_digest": memory_digest,
        "memory_blob_digest": memory_blob,
        "parameter_blob_digest": parameter_blob,
        "training_rows": len(train),
    }


def _case_document(row: dict[str, Any], representation: str) -> CaseDocument:
    outcome = {
        "valid": bool(row.get("valid", False)),
        "passed": bool(row.get("passed", False)),
        "terminal_status": row.get("terminal_status"),
        "quality_improved": bool(row.get("quality_improved", False)),
        "objective_value": row.get("objective_value"),
        "generalization_objective_value": row.get("generalization_objective_value"),
        "runtime_p95_seconds": row.get("runtime_p95_seconds"),
        "generalization_runtime_p95_seconds": row.get("generalization_runtime_p95_seconds"),
    }
    action = {
        "operator": row.get("operator"),
        "parent_count": row.get("parent_count"),
        "priority_expression": row.get("priority_expression"),
        "formula_exact_digest": row.get("formula_exact_digest"),
        "formula_structural_digest": row.get("formula_structural_digest"),
        "formula_behavior_digest": row.get("formula_behavior_digest"),
    }
    families = sorted({str(item.get("family")) for item in (_strata(row.get("generalization_strata")) or _strata(row.get("public_strata"))) if item.get("family")})
    text = " ".join(str(item) for item in (
        row.get("operator"), row.get("terminal_status"), row.get("representation"),
        row.get("model"), row.get("priority_expression"),
        "quality_improved" if (_quality_value(row) or 0.0) < 0.0 else None,
        "quality_degraded" if (_quality_value(row) or 0.0) > 0.0 else None,
        *families,
    ) if item)
    return CaseDocument(
        case_id=str(row.get("attempt_id")), campaign_id=str(row.get("campaign_id")),
        representation=str(row.get("representation") or representation), evidence_role=str(row.get("evidence_role") or "historical_training"),
        profile_id=row.get("profile_id") if isinstance(row.get("profile_id"), str) else None,
        operator=str(row.get("operator")) if row.get("operator") else None,
        text=text, action=action, outcome=outcome,
        confidence=float(row.get("backfill_confidence", 1.0)),
    )


def _training_code_digest() -> str:
    files = ("trainer.py", "models.py", "memory.py", "features.py", "policy.py")
    root = Path(__file__).parent
    payload = {
        name: sha256((root / name).read_bytes()).hexdigest()
        for name in files
    }
    return sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) else None


def _quality_value(row: dict[str, Any]) -> float | None:
    generalization = _number(row.get("generalization_objective_value"))
    return generalization if generalization is not None else _number(row.get("objective_value"))


def _strata(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, (list, tuple)) else []


def _formula_identity(row: dict[str, Any]) -> tuple[str, str, str, str] | None:
    expression = row.get("priority_expression")
    if not isinstance(expression, str):
        return None
    try:
        identity = normalize_formula(expression)
    except NormalizationError:
        return None
    return (
        identity.probe_behavior_digest,
        identity.exact_expression,
        identity.exact_digest,
        identity.structural_digest,
    )


__all__ = ["TrainingError", "train_policy_from_rows"]
