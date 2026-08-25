"""Prospective matched comparison of a learned policy and its QD baseline.

The assay deliberately keeps the proposal model fixed.  It first learns only
from the 9B model's registered campaigns in a completed historical comparison,
then runs baseline and learned search on four newly sampled development and
holdout task profiles.  All searches finish before any holdout is opened.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import comb
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable
import warnings

from algofinder.agents.experiment_manifests import (
    HOLDOUT_SPLITS,
    prepare_fresh_learning_manifests,
)
from algofinder.agents.learning.backfill import backfill_pilot_summary
from algofinder.agents.learning.dataset import (
    build_dataset,
    historical_backfill_rows,
)
from algofinder.agents.learning.distributions import build_distribution_profile
from algofinder.agents.learning.eligibility import assess_campaign
from algofinder.agents.learning.features import FEATURE_SCHEMA_DIGEST
from algofinder.agents.learning.store import LearningArtifactStore
from algofinder.agents.learning.trainer import train_policy_from_rows
from algofinder.agents.ollama_model import ollama_model_identity, unload_ollama_model
from algofinder.agents.pilot import (
    _experiment_source_provenance,
    finalize_deferred_holdout,
    run_pilot,
)


BASELINE_ARM = "baseline-qd"
LEARNED_ARM = "learned-policy"
DEFAULT_REPLICATE_SEEDS = (950_009, 950_071, 950_129, 950_191)
DEFAULT_FRESH_SEED_OFFSETS = (11_610_000, 11_710_000, 11_810_000, 11_910_000)


class LearnedPolicyComparisonError(RuntimeError):
    """Raised when a comparison checkpoint cannot be trusted."""


def run_learned_policy_comparison(
    *,
    repo_root: Path,
    experiment_root: Path,
    historical_summary_path: Path,
    additional_historical_summary_paths: tuple[Path, ...] = (),
    source_manifest: Path,
    endpoint: str,
    model: str,
    candidates_per_campaign: int,
    per_cell_budget_seconds: float,
    replicate_seeds: tuple[int, ...] = DEFAULT_REPLICATE_SEEDS,
    fresh_seed_offsets: tuple[int, ...] = DEFAULT_FRESH_SEED_OFFSETS,
    warehouse_mode: str = "off",
) -> dict[str, object]:
    """Run or resume the frozen one-model, two-policy comparison.

    ``warehouse_mode='dev'`` queues a best-effort incremental, idempotent
    warehouse ingest after each committed comparison checkpoint.  ``'prod'``
    performs no synchronous DuckDB work during search and does one final ingest
    after the comparison completes.  ``'off'`` preserves the lightweight API
    for callers that manage ingestion independently.
    """
    repo_root = repo_root.resolve()
    experiment_root = experiment_root.resolve()
    historical_summary_paths = tuple(
        dict.fromkeys(path.resolve() for path in (historical_summary_path, *additional_historical_summary_paths))
    )
    source_manifest = source_manifest.resolve()
    if candidates_per_campaign < 1 or per_cell_budget_seconds <= 0:
        raise ValueError("candidate count and evaluator budget must be positive")
    if len(replicate_seeds) != 4 or len(set(replicate_seeds)) != 4:
        raise ValueError("exactly four unique replicate seeds are required")
    if len(fresh_seed_offsets) != 4 or len(set(fresh_seed_offsets)) != 4:
        raise ValueError("exactly four unique fresh seed offsets are required")
    if warehouse_mode not in {"off", "dev", "prod"}:
        raise ValueError("warehouse_mode must be 'off', 'dev', or 'prod'")
    experiment_root.mkdir(parents=True, exist_ok=True)
    summary_path = experiment_root / "comparison-summary.json"

    if summary_path.exists():
        summary = _load_checkpoint(
            summary_path=summary_path,
            experiment_root=experiment_root,
            repo_root=repo_root,
            endpoint=endpoint,
            model=model,
        )
        if summary.get("complete") is True:
            _ingest_checkpoint(repo_root, experiment_root, warehouse_mode, final=True)
            return {**summary, "summary_path": str(summary_path)}
    else:
        summary = _initialize(
            repo_root=repo_root,
            experiment_root=experiment_root,
            historical_summary_paths=historical_summary_paths,
            source_manifest=source_manifest,
            endpoint=endpoint,
            model=model,
            candidates_per_campaign=candidates_per_campaign,
            per_cell_budget_seconds=per_cell_budget_seconds,
            replicate_seeds=replicate_seeds,
            fresh_seed_offsets=fresh_seed_offsets,
        )
        _checkpoint(summary_path, summary, repo_root, experiment_root, warehouse_mode)

    plan = _mapping(summary, "plan")
    runs = [dict(row) for row in _list_of_mappings(summary, "runs")]
    schedule = _list_of_mappings(plan, "schedule")
    completed_keys = {str(row.get("campaign_key")) for row in runs}
    campaign_store = experiment_root / "campaigns"
    campaign_store.mkdir(exist_ok=True)
    learning_root = Path(str(summary["learning_root"])).resolve()
    learned_policy_id = str(_mapping(summary, "learning")["policy_id"])

    if summary.get("phase") in {"preregistered", "development_search"}:
        summary["phase"] = "development_search"
        _checkpoint(summary_path, summary, repo_root, experiment_root, warehouse_mode)
        for scheduled in schedule:
            campaign_key = str(scheduled["campaign_key"])
            if campaign_key in completed_keys:
                continue
            profile = _profile_for(summary, int(scheduled["replicate_index"]))
            arm = str(scheduled["arm"])
            try:
                result = run_pilot(
                    repo_root=repo_root,
                    campaigns_root=campaign_store,
                    manifest_path=Path(str(_mapping(profile, "development")["path"])),
                    holdout_manifest_path=Path(str(_mapping(profile, "holdout")["path"])),
                    model_name=model,
                    endpoint=endpoint,
                    target_candidates=int(plan["candidates_per_campaign"]),
                    per_cell_budget_seconds=float(plan["per_cell_budget_seconds"]),
                    representation="formula",
                    backend="model",
                    model_seed=int(scheduled["model_seed"]),
                    holdout_splits=HOLDOUT_SPLITS,
                    defer_holdout=True,
                    learning_root=learning_root,
                    learned_policy_id=learned_policy_id if arm == LEARNED_ARM else None,
                    distribution_profile_id=str(profile["profile_id"]),
                    evidence_role="outer_validation",
                    evidence_partition_id=f"partition-outer-validation-{int(scheduled['replicate_index']) + 1:02d}",
                )
                runs.append(_run_record(scheduled=scheduled, profile=profile, result=result))
                completed_keys.add(campaign_key)
                summary["runs"] = runs
                summary["analysis"] = _analysis(runs)
                _checkpoint(summary_path, summary, repo_root, experiment_root, warehouse_mode)
            finally:
                unload_ollama_model(endpoint=endpoint, model=model)

        if len(runs) != len(schedule):
            raise LearnedPolicyComparisonError("development schedule did not complete")
        summary["all_searches_completed_at"] = _now()
        summary["phase"] = "deferred_holdout"
        summary["analysis"] = _analysis(runs)
        _checkpoint(summary_path, summary, repo_root, experiment_root, warehouse_mode)

    if summary.get("phase") != "deferred_holdout":
        raise LearnedPolicyComparisonError(f"unknown comparison phase: {summary.get('phase')!r}")
    for run in runs:
        if run.get("holdout_finalized") is True:
            continue
        profile = _profile_for(summary, int(run["replicate_index"]))
        result = finalize_deferred_holdout(
            repo_root=repo_root,
            campaigns_root=campaign_store,
            campaign_id=str(run["campaign_id"]),
            holdout_manifest_path=Path(str(_mapping(profile, "holdout")["path"])),
            per_cell_budget_seconds=float(plan["per_cell_budget_seconds"]),
        )
        run["holdout"] = result.get("holdout")
        run["campaign_state"] = result.get("campaign_state")
        run["budget"] = result.get("budget")
        run["holdout_finalized"] = True
        summary["runs"] = runs
        summary["analysis"] = _analysis(runs)
        _checkpoint(summary_path, summary, repo_root, experiment_root, warehouse_mode)

    summary["phase"] = "complete"
    summary["complete"] = True
    summary["completed_at"] = _now()
    summary["analysis"] = _analysis(runs)
    _checkpoint(summary_path, summary, repo_root, experiment_root, warehouse_mode)
    _ingest_checkpoint(repo_root, experiment_root, warehouse_mode, final=True)
    return {**summary, "summary_path": str(summary_path)}


def _initialize(
    *,
    repo_root: Path,
    experiment_root: Path,
    historical_summary_paths: tuple[Path, ...],
    source_manifest: Path,
    endpoint: str,
    model: str,
    candidates_per_campaign: int,
    per_cell_budget_seconds: float,
    replicate_seeds: tuple[int, ...],
    fresh_seed_offsets: tuple[int, ...],
) -> dict[str, object]:
    learning_root = experiment_root / "learning"
    store = LearningArtifactStore(learning_root).create()
    eligibility: list[dict[str, object]] = []
    registered: set[str] = set()
    for historical_summary_path in historical_summary_paths:
        historical_campaigns_root = historical_summary_path.parent / "campaigns"
        summary_campaigns = _registered_run_campaign_ids(historical_summary_path)
        for campaign_id in sorted(summary_campaigns):
            if campaign_id in registered:
                continue
            report = assess_campaign(
                historical_campaigns_root,
                campaign_id,
                registered_final_campaign_ids=summary_campaigns,
            )
            if report.tier != "A":
                raise LearnedPolicyComparisonError(
                    f"historical campaign {campaign_id} is not registered tier A: {report.reasons}"
                )
            projection = backfill_pilot_summary(
                campaigns_root=historical_campaigns_root,
                campaign_id=campaign_id,
                eligibility=report,
                learning_store=store,
            )
            registered.add(campaign_id)
            eligibility.append({
                "campaign_id": campaign_id,
                "source_summary": str(historical_summary_path),
                "tier": report.tier,
                "backfill_id": projection["backfill_id"],
                "projection_digest": projection["projection_digest"],
            })

    # This is a deliberately strict self-learning test: the frozen policy sees
    # only historical attempts made by the same 9B model under evaluation.
    model_rows = [row for row in historical_backfill_rows(store) if row.get("model") == model]
    _annotate_historical_holdouts(model_rows, historical_summary_paths)
    if not model_rows:
        raise LearnedPolicyComparisonError(f"no eligible historical rows exist for {model!r}")
    dataset_id = "qwen35-9b-self-history-v5"
    dataset = build_dataset(
        store=store,
        rows=model_rows,
        dataset_id=dataset_id,
        split_seed=75_348,
        source_description="registered prior qwen3.5:9b attempts with consumed historical champion test_iid labels; all prospective deployment and diagnostic profiles remain fresh",
    )
    dataset_manifest = store.read(f"datasets/{dataset_id}/manifest.json")
    training_rows = _read_parquet_rows(learning_root / "datasets" / dataset_id / "attempts.parquet")
    policy = train_policy_from_rows(
        store=store,
        rows=training_rows,
        dataset_digest=str(dataset_manifest["content_digest"]),
        representation="formula",
        feature_schema_digest=FEATURE_SCHEMA_DIGEST,
        seed=75_348,
        policy_id="policy-qwen35-9b-self-history-v5",
    )

    profiles: list[dict[str, object]] = []
    manifests_root = experiment_root / "manifests"
    manifests_root.mkdir(exist_ok=False)
    for index, offset in enumerate(fresh_seed_offsets):
        suite_name = f"ml_ood_qwen9b_learning_v5_outer_validation_r{index + 1}_20260824"
        manifests = prepare_fresh_learning_manifests(
            source_manifest=source_manifest,
            output_dir=manifests_root / f"replicate-{index + 1:02d}",
            seed_offset=offset,
            suite_name=suite_name,
        )
        development_path = Path(str(_mapping(manifests, "development")["path"]))
        holdout_path = Path(str(_mapping(manifests, "holdout")["path"]))
        development_rows = _manifest_rows(development_path)
        holdout_rows = _manifest_rows(holdout_path)
        rows = development_rows + holdout_rows
        # The target deployment profile is the same biased family support seen
        # during search. Family and parameter shifts remain frozen diagnostics.
        families = sorted({str(row.get("family")) for row in development_rows})
        diagnostic_families = sorted({
            str(row.get("family")) for row in holdout_rows
            if row.get("split") != "test_iid"
        })
        city_counts = [len(_mapping(row, "data").get("points", ())) for row in rows]
        profile_id = f"profile-qwen9b-learning-v5-r{index + 1}"
        profile = build_distribution_profile(
            family="tsp-ml-ood-biased",
            version=f"fresh-outer-validation-r{index + 1}-v1",
            manifest_digests=(
                str(_mapping(manifests, "development")["digest"]),
                str(_mapping(manifests, "holdout")["digest"]),
            ),
            lineage_namespace=suite_name,
            generators={
                "registry": "algofinder.benchmark.suites._ml_ood_cells",
                "families": families,
                "diagnostic_families": diagnostic_families,
                "cell_count": len(manifests.get("cells", ())),
            },
            feature_summary={
                "development_instances": 28.0,
                "holdout_instances": 114.0,
                "generator_family_count": float(len(families)),
                "mean_city_count": mean(city_counts),
            },
            seed_set=tuple(sorted({int(row["seed"]) for row in rows})),
            split_rule="fresh validation search; deferred test_iid/family_ood/parameter_ood champion evaluation",
            deployment_intent="learn city/distribution-specific TSP proposal biases without claiming general ETSP superiority",
            allowed_specialization_axes=("generator_family", "city_count", "geometry_distribution"),
            meaning_preserving_transformations=("translation", "rotation", "reflection", "uniform_scaling"),
            diagnostic_transformations=("family_shift", "parameter_shift"),
            visibility="outer_validation",
            profile_id=profile_id,
        )
        profile_digest = store.write(profile.profile)
        profiles.append({
            "replicate_index": index,
            "profile_id": profile_id,
            "profile_digest": profile_digest,
            "seed_offset": offset,
            "suite_name": suite_name,
            "development": manifests["development"],
            "holdout": manifests["holdout"],
            "manifest_provenance": manifests["metadata_path"],
        })

    schedule: list[dict[str, object]] = []
    for index, seed in enumerate(replicate_seeds):
        order = (BASELINE_ARM, LEARNED_ARM) if index % 2 == 0 else (LEARNED_ARM, BASELINE_ARM)
        for position, arm in enumerate(order):
            schedule.append({
                "campaign_key": f"r{index + 1:02d}-{arm}",
                "replicate_index": index,
                "model_seed": seed,
                "order_position": position,
                "arm": arm,
                "policy_id": "baseline-qd-map-elites-ucb-v3" if arm == BASELINE_ARM else policy["policy_id"],
                "profile_id": profiles[index]["profile_id"],
            })

    identity = ollama_model_identity(endpoint=endpoint, model=model)
    source_provenance = _experiment_source_provenance(repo_root)
    plan = {
        "model": model,
        "model_identity": identity,
        "policy_arms": [BASELINE_ARM, LEARNED_ARM],
        "replicate_seeds": list(replicate_seeds),
        "fresh_seed_offsets": list(fresh_seed_offsets),
        "candidates_per_campaign": candidates_per_campaign,
        "max_proposal_rounds_per_campaign": candidates_per_campaign * 8,
        "per_cell_budget_seconds": per_cell_budget_seconds,
        "representation": "formula",
        "primary_outcome": "paired fresh test_iid champion gap on the declared biased deployment distribution: learned minus baseline",
        "secondary_outcomes": [
            "development champion gap",
            "family-shift and parameter-shift diagnostic gaps",
            "all-splits aggregate gap",
            "accepted normalized novel candidates",
            "proposal attempts",
            "model tokens",
            "campaign wall seconds",
        ],
        "positive_result_rule": "descriptive preliminary evidence only: median held-out delta < 0 and learned wins at least 3 of 4 matched profiles",
        "order_rule": "AB/BA policy order alternation across four matched profiles",
        "holdout_rule": "finish all eight searches before opening any profile holdout",
        "gpu_rule": "one qwen3.5:9b campaign at a time; unload at every campaign boundary",
        "learning_rule": "train only on registered qwen3.5:9b history; consumed integrity-valid champion test_iid results may label their selected historical formulas; retain OOD results as diagnostics; exclude every fresh prospective profile",
        "inference_unit": "matched campaign/profile pair",
        "inference_scope": "exploratory; four pairs cannot establish conventional statistical significance",
        "schedule": schedule,
    }
    return {
        "schema_id": "algofinder.agents.learned-policy-comparison",
        "schema_version": "5",
        "experiment_root": str(experiment_root),
        "summary_path": str(experiment_root / "comparison-summary.json"),
        "started_at": _now(),
        "phase": "preregistered",
        "complete": False,
        "repo_root": str(repo_root),
        "source_manifest": str(source_manifest),
        "historical_summaries": [str(path) for path in historical_summary_paths],
        "learning_root": str(learning_root),
        "learning": {
            "registered_campaign_count": len(registered),
            "eligible_campaigns": eligibility,
            "model_specific_rows": len(model_rows),
            "dataset_id": dataset_id,
            "dataset_manifest_digest": dataset_manifest["content_digest"],
            "dataset_split_plan": dataset.get("split_plan"),
            "policy_id": policy["policy_id"],
            "policy_digest": policy["policy_digest"],
            "policy_version": "formula-profile-value@5",
            "train_rows": policy["training_rows"],
        },
        "profiles": profiles,
        "plan": plan,
        "experiment_source_provenance": source_provenance,
        "runs": [],
        "analysis": _analysis([]),
    }


def _load_checkpoint(
    *,
    summary_path: Path,
    experiment_root: Path,
    repo_root: Path,
    endpoint: str,
    model: str,
) -> dict[str, object]:
    summary = _read_object(summary_path, "comparison summary")
    _require(summary.get("schema_id") == "algofinder.agents.learned-policy-comparison", "unsupported comparison summary")
    _require(Path(str(summary.get("experiment_root"))).resolve() == experiment_root, "checkpoint belongs to another experiment root")
    plan = _mapping(summary, "plan")
    _require(plan.get("model") == model, "model differs from the frozen plan")
    recorded_identity = _mapping(plan, "model_identity")
    current_identity = ollama_model_identity(endpoint=endpoint, model=model)
    _require(current_identity.get("blob_digest") == recorded_identity.get("blob_digest"), "Ollama model blob changed")
    recorded_source = _mapping(summary, "experiment_source_provenance")
    current_source = _experiment_source_provenance(repo_root)
    _require(current_source.get("source_digest") == recorded_source.get("source_digest"), "experiment source code changed")
    _verify_profiles(summary)
    keys = [str(row.get("campaign_key")) for row in _list_of_mappings(summary, "runs")]
    _require(len(keys) == len(set(keys)), "checkpoint contains duplicate completed runs")
    summary.setdefault("resume_events", []).append({"resumed_at": _now(), "completed_runs": len(keys), "phase": summary.get("phase")})
    return summary


def _verify_profiles(summary: dict[str, object]) -> None:
    for profile in _list_of_mappings(summary, "profiles"):
        for name in ("development", "holdout"):
            item = _mapping(profile, name)
            path = Path(str(item.get("path"))).resolve()
            _require(path.is_file(), f"missing frozen {name} manifest: {path}")
            _require(_digest(path) == item.get("digest"), f"frozen {name} manifest changed")


def _run_record(
    *,
    scheduled: dict[str, object],
    profile: dict[str, object],
    result: dict[str, object],
) -> dict[str, object]:
    steps = result.get("steps", [])
    status_counts = Counter(
        str(row.get("status")) for row in steps if isinstance(row, dict)
    )
    operator_counts = Counter(
        str(row.get("operator")) for row in steps if isinstance(row, dict)
    )
    return {
        **scheduled,
        "model": result["model"],
        "profile_id": profile["profile_id"],
        "campaign_id": result["campaign_id"],
        "campaign_root": result["campaign_root"],
        "evaluated_candidates": result["evaluated_candidates"],
        "proposal_attempts": len(steps) if isinstance(steps, list) else None,
        "status_counts": dict(status_counts),
        "operator_counts": dict(operator_counts),
        "campaign_state": result["campaign_state"],
        "learned_policy_id": result.get("learned_policy_id"),
        "learned_policy_digest": result.get("learned_policy_digest"),
        "distribution_profile_digest": result.get("distribution_profile_digest"),
        "holdout": result.get("holdout"),
        "holdout_finalized": False,
        "model_usage_totals": _usage_totals(result.get("model_usage")),
        "budget": result.get("budget"),
    }


def _analysis(runs: list[dict[str, object]]) -> dict[str, object]:
    by_arm: dict[str, dict[str, object]] = {}
    for arm in (BASELINE_ARM, LEARNED_ARM):
        selected = [row for row in runs if row.get("arm") == arm]
        dev = [value for row in selected if (value := _development_gap(row)) is not None]
        holdout = [value for row in selected if (value := _holdout_gap(row)) is not None]
        aggregate_holdout = [
            value for row in selected
            if (value := _holdout_split_gap(row, None)) is not None
        ]
        family_ood = [
            value for row in selected
            if (value := _holdout_split_gap(row, "test_family_ood")) is not None
        ]
        parameter_ood = [
            value for row in selected
            if (value := _holdout_split_gap(row, "test_parameter_ood")) is not None
        ]
        tokens = [_number(_mapping_or_empty(row, "model_usage_totals").get("tokens")) for row in selected]
        walls = [_actual_budget(row, "wall_seconds") for row in selected]
        by_arm[arm] = {
            "search_campaigns_complete": len(selected),
            "holdout_campaigns_complete": sum(row.get("holdout_finalized") is True for row in selected),
            "mean_development_gap_percent": mean(dev) if dev else None,
            "median_development_gap_percent": median(dev) if dev else None,
            "mean_holdout_gap_percent": mean(holdout) if holdout else None,
            "median_holdout_gap_percent": median(holdout) if holdout else None,
            "holdout_primary_split": "test_iid",
            "mean_all_splits_holdout_gap_percent": mean(aggregate_holdout) if aggregate_holdout else None,
            "mean_family_ood_gap_percent": mean(family_ood) if family_ood else None,
            "mean_parameter_ood_gap_percent": mean(parameter_ood) if parameter_ood else None,
            "development_improvements_over_solver_baseline": sum(value < 0 for value in dev),
            "holdout_improvements_over_solver_baseline": sum(value < 0 for value in holdout),
            "accepted_normalized_novel": sum(int(row.get("evaluated_candidates", 0)) for row in selected),
            "proposal_attempts": sum(int(row.get("proposal_attempts", 0)) for row in selected),
            "model_tokens": sum(value for value in tokens if value is not None),
            "campaign_wall_seconds": sum(value for value in walls if value is not None),
        }

    pairs: list[dict[str, object]] = []
    for index in range(4):
        baseline = next((row for row in runs if row.get("replicate_index") == index and row.get("arm") == BASELINE_ARM), None)
        learned = next((row for row in runs if row.get("replicate_index") == index and row.get("arm") == LEARNED_ARM), None)
        if baseline is None or learned is None:
            continue
        row: dict[str, object] = {
            "replicate_index": index,
            "profile_id": baseline.get("profile_id"),
            "model_seed": baseline.get("model_seed"),
        }
        for label, extractor in (("development", _development_gap), ("holdout", _holdout_gap)):
            base_value, learned_value = extractor(baseline), extractor(learned)
            row[f"baseline_{label}_gap_percent"] = base_value
            row[f"learned_{label}_gap_percent"] = learned_value
            row[f"learned_minus_baseline_{label}_percent"] = (
                learned_value - base_value
                if learned_value is not None and base_value is not None else None
            )
        for split in ("test_family_ood", "test_parameter_ood", None):
            label = "all_splits" if split is None else split.removeprefix("test_")
            base_value = _holdout_split_gap(baseline, split)
            learned_value = _holdout_split_gap(learned, split)
            row[f"baseline_{label}_gap_percent"] = base_value
            row[f"learned_{label}_gap_percent"] = learned_value
            row[f"learned_minus_baseline_{label}_percent"] = (
                learned_value - base_value
                if learned_value is not None and base_value is not None else None
            )
        pairs.append(row)

    deltas = [
        float(row["learned_minus_baseline_holdout_percent"])
        for row in pairs
        if isinstance(row.get("learned_minus_baseline_holdout_percent"), (int, float))
    ]
    wins = sum(value < 0 for value in deltas)
    losses = sum(value > 0 for value in deltas)
    matched_baseline = [
        float(row["baseline_holdout_gap_percent"])
        for row in pairs
        if isinstance(row.get("learned_minus_baseline_holdout_percent"), (int, float))
        and isinstance(row.get("baseline_holdout_gap_percent"), (int, float))
    ]
    matched_learned = [
        float(row["learned_holdout_gap_percent"])
        for row in pairs
        if isinstance(row.get("learned_minus_baseline_holdout_percent"), (int, float))
        and isinstance(row.get("learned_holdout_gap_percent"), (int, float))
    ]
    by_arm[BASELINE_ARM].update({
        "matched_integrity_valid_holdout_campaigns": len(matched_baseline),
        "matched_integrity_valid_mean_holdout_gap_percent": mean(matched_baseline) if matched_baseline else None,
        "matched_integrity_valid_median_holdout_gap_percent": median(matched_baseline) if matched_baseline else None,
    })
    by_arm[LEARNED_ARM].update({
        "matched_integrity_valid_holdout_campaigns": len(matched_learned),
        "matched_integrity_valid_mean_holdout_gap_percent": mean(matched_learned) if matched_learned else None,
        "matched_integrity_valid_median_holdout_gap_percent": median(matched_learned) if matched_learned else None,
    })
    preliminary_positive = len(deltas) == 4 and median(deltas) < 0 and wins >= 3
    return {
        "primary_holdout_split": "test_iid",
        "by_arm": by_arm,
        "matched_pairs": pairs,
        "complete_holdout_pairs": len(deltas),
        "learned_holdout_wins": wins,
        "ties": sum(value == 0 for value in deltas),
        "baseline_holdout_wins": losses,
        "mean_learned_minus_baseline_holdout_percent": mean(deltas) if deltas else None,
        "median_learned_minus_baseline_holdout_percent": median(deltas) if deltas else None,
        "two_sided_exact_sign_p": _two_sided_sign_p(wins, losses),
        "preliminary_positive": preliminary_positive,
        "interpretation": "negative paired test_iid gaps favor the learned policy on the declared biased deployment distribution; OOD and all-splits gaps are diagnostics; four pairs are descriptive, not a confirmatory significance test",
        "pseudo_replication_avoided": True,
    }


def _development_gap(run: dict[str, object]) -> float | None:
    holdout = run.get("holdout")
    value = holdout.get("development_paired_gap_percent") if isinstance(holdout, dict) else None
    return _number(value)


def _holdout_gap(run: dict[str, object]) -> float | None:
    return _holdout_split_gap(run, "test_iid")


def _holdout_split_gap(run: dict[str, object], split: str | None) -> float | None:
    holdout = run.get("holdout")
    if not isinstance(holdout, dict) or holdout.get("gate") != "pass":
        return None
    metrics = holdout.get("metric_vector")
    if not isinstance(metrics, dict):
        return None
    if split is None:
        return _number(metrics.get("paired_gap_percent"))
    return _metric_split_gap(metrics, split)


def _metric_split_gap(metrics: dict[str, Any], split: str) -> float | None:
    strata = metrics.get("strata")
    if not isinstance(strata, list):
        return None
    weighted_sum = 0.0
    comparisons = 0
    for row in strata:
        if not isinstance(row, dict) or row.get("split") != split:
            continue
        gap = _number(row.get("paired_gap_percent"))
        count = row.get("comparisons")
        if gap is None or not isinstance(count, int) or isinstance(count, bool) or count < 1:
            continue
        weighted_sum += gap * count
        comparisons += count
    return weighted_sum / comparisons if comparisons else None


def _actual_budget(run: dict[str, object], key: str) -> float | None:
    budget = run.get("budget")
    actual = budget.get("actual") if isinstance(budget, dict) else None
    return _number(actual.get(key)) if isinstance(actual, dict) else None


def _usage_totals(raw: object) -> dict[str, int]:
    rows = raw if isinstance(raw, list) else []
    prompt = sum(int(row.get("prompt_tokens", 0)) for row in rows if isinstance(row, dict))
    completion = sum(int(row.get("completion_tokens", 0)) for row in rows if isinstance(row, dict))
    return {"prompt_tokens": prompt, "completion_tokens": completion, "tokens": prompt + completion}


def _registered_run_campaign_ids(summary_path: Path) -> frozenset[str]:
    summary = _read_object(summary_path, "historical comparison summary")
    _require(summary.get("complete") is True, f"historical comparison is incomplete: {summary_path}")
    runs = summary.get("runs")
    _require(isinstance(runs, list), "historical comparison has no run registry")
    return frozenset(
        str(item["campaign_id"])
        for item in runs
        if isinstance(item, dict) and isinstance(item.get("campaign_id"), str)
    )


def _annotate_historical_holdouts(
    rows: list[dict[str, Any]],
    summary_paths: tuple[Path, ...],
) -> None:
    """Attach consumed, integrity-valid champion holdouts as retrospective labels.

    These labels are safe only because every prospective manifest is regenerated
    from new seed offsets.  Failed holdouts and non-champion attempts remain
    development-only observations.
    """
    labels: dict[tuple[str, str], dict[str, Any]] = {}
    for summary_path in summary_paths:
        summary = _read_object(summary_path, "historical comparison summary")
        _require(summary.get("complete") is True, f"historical comparison is incomplete: {summary_path}")
        source_digest = _digest(summary_path)
        campaigns_root = summary_path.parent / "campaigns"
        for run in summary.get("runs", ()):
            if not isinstance(run, dict):
                continue
            campaign_id = run.get("campaign_id")
            holdout = run.get("holdout")
            if not isinstance(campaign_id, str) or not isinstance(holdout, dict) or holdout.get("gate") != "pass":
                continue
            candidate_id = holdout.get("selected_candidate_id")
            metrics = holdout.get("metric_vector")
            aggregate_gap = metrics.get("paired_gap_percent") if isinstance(metrics, dict) else None
            deployment_gap = _metric_split_gap(metrics, "test_iid") if isinstance(metrics, dict) else None
            if deployment_gap is None:
                deployment_gap = _number(aggregate_gap)
            if not isinstance(candidate_id, str) or deployment_gap is None:
                continue
            labels[(campaign_id, candidate_id)] = {
                "generalization_objective_value": deployment_gap,
                "aggregate_generalization_objective_value": _number(aggregate_gap),
                "generalization_strata": metrics.get("strata", ()) if isinstance(metrics, dict) else (),
                "generalization_ok_rate": metrics.get("ok_rate") if isinstance(metrics, dict) else None,
                "generalization_runtime_p95_seconds": _historical_holdout_p95(
                    campaigns_root, campaign_id, holdout.get("evaluation_id")
                ),
                "generalization_source_summary_digest": source_digest,
                "generalization_label_role": "consumed_historical_test_iid_holdout",
            }
    for row in rows:
        label = labels.get((str(row.get("campaign_id")), str(row.get("candidate_id"))))
        if label is not None:
            row.update(label)


def _historical_holdout_p95(
    campaigns_root: Path,
    campaign_id: str,
    evaluation_id: object,
) -> float | None:
    if not isinstance(evaluation_id, str):
        return None
    path = campaigns_root / campaign_id / "evaluations" / f"{evaluation_id}.json"
    try:
        evaluation = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    feedback = evaluation.get("feedback", {}) if isinstance(evaluation, dict) else {}
    raw = feedback.get("raw_runs", ()) if isinstance(feedback, dict) else ()
    walls = sorted(
        float(item["wall_seconds"])
        for item in raw
        if isinstance(item, dict)
        and "control" not in item.get("tags", ())
        and item.get("status") == "ok"
        and isinstance(item.get("wall_seconds"), (int, float))
        and not isinstance(item.get("wall_seconds"), bool)
    ) if isinstance(raw, list) else []
    return walls[max(0, int(0.95 * len(walls)) - 1)] if walls else None


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise LearnedPolicyComparisonError("DuckDB is required for policy training") from exc
    with duckdb.connect() as connection:
        cursor = connection.execute("SELECT * FROM read_parquet(?)", [str(path)])
        columns = [item[0] for item in cursor.description]
        return [dict(zip(columns, values)) for values in cursor.fetchall()]


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LearnedPolicyComparisonError(f"cannot read generated manifest {path}") from exc
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise LearnedPolicyComparisonError(f"generated manifest is not a JSON object list: {path}")
    return rows


def _profile_for(summary: dict[str, object], replicate_index: int) -> dict[str, object]:
    profiles = _list_of_mappings(summary, "profiles")
    profile = next((row for row in profiles if row.get("replicate_index") == replicate_index), None)
    if profile is None:
        raise LearnedPolicyComparisonError(f"missing profile for replicate {replicate_index}")
    return profile


def _two_sided_sign_p(wins: int, losses: int) -> float | None:
    count = wins + losses
    if count == 0:
        return None
    lower = min(wins, losses)
    return min(1.0, 2.0 * sum(comb(count, index) for index in range(lower + 1)) / (2**count))


def _mapping(value: dict[str, object], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise LearnedPolicyComparisonError(f"checkpoint field {key!r} is not an object")
    return item


def _mapping_or_empty(value: dict[str, object], key: str) -> dict[str, Any]:
    item = value.get(key)
    return item if isinstance(item, dict) else {}


def _list_of_mappings(value: dict[str, object], key: str) -> list[dict[str, Any]]:
    item = value.get(key)
    if not isinstance(item, list) or any(not isinstance(row, dict) for row in item):
        raise LearnedPolicyComparisonError(f"checkpoint field {key!r} is not an object list")
    return item


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LearnedPolicyComparisonError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise LearnedPolicyComparisonError(f"{label} is not a JSON object")
    return value


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    summary["updated_at"] = _now()
    encoded = json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _checkpoint(
    path: Path,
    summary: dict[str, object],
    repo_root: Path,
    experiment_root: Path,
    warehouse_mode: str,
) -> None:
    """Commit a source checkpoint before best-effort dev-mode ingestion."""
    _write_summary(path, summary)
    _ingest_checkpoint(repo_root, experiment_root, warehouse_mode, final=False)


def _ingest_checkpoint(
    repo_root: Path,
    experiment_root: Path,
    warehouse_mode: str,
    *,
    final: bool,
) -> None:
    if warehouse_mode == "off" or (warehouse_mode == "prod" and not final):
        return
    try:
        # Deliberately lazy: an experiment remains valid if a local analytics
        # dependency is unavailable or an ingest is interrupted.  The next
        # idempotent pass resumes from the immutable source evidence.
        from algofinder.store.warehouse import incremental_ingest

        incremental_ingest(
            repo_root=repo_root,
            experiment_root=experiment_root,
            mode="prod" if warehouse_mode == "prod" else "dev",
        )
    except Exception as exc:  # pragma: no cover - protects expensive searches
        warnings.warn(f"warehouse ingestion deferred: {exc}", RuntimeWarning, stacklevel=2)


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LearnedPolicyComparisonError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--historical-summary", type=Path, action="append", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--candidates-per-campaign", type=int, default=12)
    parser.add_argument("--budget-seconds", type=float, default=0.20)
    parser.add_argument(
        "--warehouse-mode", choices=("off", "dev", "prod"), default="dev",
        help="dev ingests each committed checkpoint; prod ingests once after completion",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_learned_policy_comparison(
        repo_root=args.repo_root,
        experiment_root=args.experiment_root,
        historical_summary_path=args.historical_summary[0],
        additional_historical_summary_paths=tuple(args.historical_summary[1:]),
        source_manifest=args.source_manifest,
        endpoint=args.endpoint,
        model=args.model,
        candidates_per_campaign=args.candidates_per_campaign,
        per_cell_budget_seconds=args.budget_seconds,
        warehouse_mode=args.warehouse_mode,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "BASELINE_ARM",
    "DEFAULT_FRESH_SEED_OFFSETS",
    "DEFAULT_REPLICATE_SEEDS",
    "LEARNED_ARM",
    "LearnedPolicyComparisonError",
    "run_learned_policy_comparison",
]
