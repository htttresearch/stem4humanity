"""Counterbalanced, deferred-holdout comparison of three local models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations, permutations
import json
from math import comb
from pathlib import Path
from statistics import mean, median
from typing import Any

from algofinder.agents.experiment_manifests import (
    HOLDOUT_SPLITS,
    prepare_three_model_manifests,
)
from algofinder.agents.ollama_model import ollama_model_identity, unload_ollama_model
from algofinder.agents.pilot import finalize_deferred_holdout, run_pilot


DEFAULT_REPLICATE_SEEDS = (1181, 2713, 4219, 5779, 6971, 8089)


@dataclass(frozen=True)
class ModelArm:
    label: str
    model: str

    def __post_init__(self) -> None:
        if not self.label.strip() or not self.model.strip():
            raise ValueError("model-arm label and model name are required")


@dataclass(frozen=True)
class ThreeModelComparisonPlan:
    arms: tuple[ModelArm, ModelArm, ModelArm]
    replicate_seeds: tuple[int, ...] = DEFAULT_REPLICATE_SEEDS
    candidates_per_campaign: int = 12
    per_cell_budget_seconds: float = 0.20
    holdout_splits: tuple[str, ...] = HOLDOUT_SPLITS
    fresh_seed_offset: int = 8_220_000

    def __post_init__(self) -> None:
        if len({arm.label for arm in self.arms}) != 3:
            raise ValueError("three unique arm labels are required")
        if len({arm.model for arm in self.arms}) != 3:
            raise ValueError("three unique model names are required")
        if len(self.replicate_seeds) != 6 or len(set(self.replicate_seeds)) != 6:
            raise ValueError("exactly six unique seeds are required for complete order counterbalancing")
        if any(seed < 0 for seed in self.replicate_seeds):
            raise ValueError("replicate seeds must be non-negative")
        if self.candidates_per_campaign < 1 or self.per_cell_budget_seconds <= 0:
            raise ValueError("candidate count and evaluator budget must be positive")
        if tuple(self.holdout_splits) != HOLDOUT_SPLITS:
            raise ValueError("the preregistered comparison uses all three ML/OOD test splits")

    def ordered_arms(self, replicate_index: int) -> tuple[ModelArm, ModelArm, ModelArm]:
        orders = tuple(permutations(self.arms))
        return orders[replicate_index]

    def to_mapping(self) -> dict[str, object]:
        return {
            "arms": [{"label": arm.label, "model": arm.model} for arm in self.arms],
            "replicate_seeds": list(self.replicate_seeds),
            "candidates_per_campaign": self.candidates_per_campaign,
            "max_proposal_rounds_per_campaign": self.candidates_per_campaign * 8,
            "per_cell_budget_seconds": self.per_cell_budget_seconds,
            "representation": "formula",
            "development_split": "validation",
            "holdout_splits": list(self.holdout_splits),
            "fresh_seed_offset": self.fresh_seed_offset,
            "primary_outcome": "fresh-held-out paired_gap_percent of the development-selected champion",
            "selection_rule": "lowest development paired_gap_percent; proposal ordinal then patch digest tie break",
            "order_rule": "all six model-order permutations, one per matched replicate seed",
            "holdout_rule": "finish all 18 searches before opening any held-out outcome",
            "gpu_rule": "run serially and unload every model after its campaign",
            "inference_unit": "complete campaign",
            "novelty_rule": "reject any normalized template state already evaluated in that campaign",
            "inference_scope": "exploratory because each model has six campaign-level replicates",
        }


def run_three_model_comparison(
    *,
    repo_root: Path,
    campaigns_root: Path,
    source_manifest: Path,
    endpoint: str,
    plan: ThreeModelComparisonPlan,
) -> dict[str, object]:
    """Run all searches first, then evaluate one frozen champion per campaign."""
    repo_root = repo_root.resolve()
    campaigns_root = campaigns_root.resolve()
    campaigns_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    experiment_root = campaigns_root / f"tsp-three-model-comparison-{stamp}"
    experiment_root.mkdir(parents=False, exist_ok=False)
    campaign_store = experiment_root / "campaigns"
    campaign_store.mkdir()

    manifests = prepare_three_model_manifests(
        source_manifest=source_manifest,
        output_dir=experiment_root / "manifests",
        seed_offset=plan.fresh_seed_offset,
    )
    development_path = Path(str(_nested(manifests, "development", "path")))
    holdout_path = Path(str(_nested(manifests, "holdout", "path")))
    identities = {
        arm.label: ollama_model_identity(endpoint=endpoint, model=arm.model)
        for arm in plan.arms
    }
    blob_digests = {str(item.get("blob_digest")) for item in identities.values()}
    if len(blob_digests) != 3:
        raise RuntimeError("comparison arms did not resolve to three distinct Ollama blobs")

    summary_path = experiment_root / "comparison-summary.json"
    started_at = datetime.now(timezone.utc).isoformat()
    runs: list[dict[str, object]] = []
    base: dict[str, object] = {
        "schema_id": "algofinder.agents.three-model-comparison",
        "schema_version": "1",
        "experiment_root": str(experiment_root),
        "started_at": started_at,
        "phase": "preregistered",
        "complete": False,
        "plan": plan.to_mapping(),
        "model_identities": identities,
        "manifests": manifests,
        "runs": runs,
        "analysis": _analysis(plan, runs),
    }
    _write_summary(summary_path, base)

    for arm in plan.arms:
        unload_ollama_model(endpoint=endpoint, model=arm.model)

    base["phase"] = "development_search"
    _write_summary(summary_path, base)
    for replicate_index, seed in enumerate(plan.replicate_seeds):
        for order_position, arm in enumerate(plan.ordered_arms(replicate_index)):
            try:
                result = run_pilot(
                    repo_root=repo_root,
                    campaigns_root=campaign_store,
                    manifest_path=development_path,
                    holdout_manifest_path=holdout_path,
                    model_name=arm.model,
                    endpoint=endpoint,
                    target_candidates=plan.candidates_per_campaign,
                    per_cell_budget_seconds=plan.per_cell_budget_seconds,
                    representation="formula",
                    backend="model",
                    model_seed=seed,
                    holdout_splits=plan.holdout_splits,
                    defer_holdout=True,
                )
                runs.append(_run_record(
                    arm=arm,
                    seed=seed,
                    replicate_index=replicate_index,
                    order_position=order_position,
                    result=result,
                ))
                base["runs"] = runs
                base["analysis"] = _analysis(plan, runs)
                _write_summary(summary_path, base)
            finally:
                unload_ollama_model(endpoint=endpoint, model=arm.model)

    base["all_searches_completed_at"] = datetime.now(timezone.utc).isoformat()
    base["phase"] = "deferred_holdout"
    _write_summary(summary_path, base)
    for run in runs:
        result = finalize_deferred_holdout(
            repo_root=repo_root,
            campaigns_root=campaign_store,
            campaign_id=str(run["campaign_id"]),
            holdout_manifest_path=holdout_path,
            per_cell_budget_seconds=plan.per_cell_budget_seconds,
        )
        run["holdout"] = result.get("holdout")
        run["campaign_state"] = result.get("campaign_state")
        run["budget"] = result.get("budget")
        base["analysis"] = _analysis(plan, runs)
        _write_summary(summary_path, base)

    base["phase"] = "complete"
    base["complete"] = True
    base["completed_at"] = datetime.now(timezone.utc).isoformat()
    base["analysis"] = _analysis(plan, runs)
    _write_summary(summary_path, base)
    return {**base, "summary_path": str(summary_path)}


def _run_record(
    *,
    arm: ModelArm,
    seed: int,
    replicate_index: int,
    order_position: int,
    result: dict[str, object],
) -> dict[str, object]:
    return {
        "arm": arm.label,
        "model": arm.model,
        "model_seed": seed,
        "replicate_index": replicate_index,
        "order_position": order_position,
        "campaign_id": result["campaign_id"],
        "campaign_root": result["campaign_root"],
        "evaluated_candidates": result["evaluated_candidates"],
        "campaign_state": result["campaign_state"],
        "holdout": result.get("holdout"),
        "model_usage": result.get("model_usage", []),
        "budget": result.get("budget"),
    }


def _heldout_gap(run: dict[str, object]) -> float | None:
    holdout = run.get("holdout")
    if not isinstance(holdout, dict) or holdout.get("gate") != "pass":
        return None
    metrics = holdout.get("metric_vector")
    value = metrics.get("paired_gap_percent") if isinstance(metrics, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def _analysis(plan: ThreeModelComparisonPlan, runs: list[dict[str, object]]) -> dict[str, object]:
    by_model: dict[str, dict[str, object]] = {}
    for arm in plan.arms:
        model_runs = [row for row in runs if row.get("arm") == arm.label]
        gaps = [gap for row in model_runs if (gap := _heldout_gap(row)) is not None]
        by_model[arm.label] = {
            "model": arm.model,
            "campaigns_started": len(model_runs),
            "heldout_campaigns_complete": len(gaps),
            "mean_heldout_gap_percent": mean(gaps) if gaps else None,
            "median_heldout_gap_percent": median(gaps) if gaps else None,
            "heldout_gaps_percent": gaps,
        }

    paired: dict[str, object] = {}
    for first, second in combinations(plan.arms, 2):
        rows: list[dict[str, object]] = []
        for seed in plan.replicate_seeds:
            left = next(
                (row for row in runs if row.get("arm") == first.label and row.get("model_seed") == seed),
                None,
            )
            right = next(
                (row for row in runs if row.get("arm") == second.label and row.get("model_seed") == seed),
                None,
            )
            if left is None or right is None:
                continue
            left_gap = _heldout_gap(left)
            right_gap = _heldout_gap(right)
            if left_gap is None or right_gap is None:
                rows.append({"seed": seed, "status": "incomplete"})
                continue
            rows.append({
                "seed": seed,
                f"{first.label}_gap_percent": left_gap,
                f"{second.label}_gap_percent": right_gap,
                "second_minus_first_percent": right_gap - left_gap,
            })
        deltas = [
            float(row["second_minus_first_percent"])
            for row in rows if isinstance(row.get("second_minus_first_percent"), (int, float))
        ]
        wins = sum(delta < 0 for delta in deltas)
        losses = sum(delta > 0 for delta in deltas)
        paired[f"{second.label}_minus_{first.label}"] = {
            "replicates": rows,
            "complete_pairs": len(deltas),
            f"{second.label}_wins": wins,
            "ties": sum(delta == 0 for delta in deltas),
            f"{first.label}_wins": losses,
            "mean_difference_percent": mean(deltas) if deltas else None,
            "median_difference_percent": median(deltas) if deltas else None,
            "two_sided_exact_sign_p": _two_sided_sign_p(wins, losses),
            "interpretation": f"negative differences favor {second.label}",
        }
    return {
        "by_model": by_model,
        "pairwise_matched": paired,
        "inference_scope": "exploratory: six complete campaign-level replicates per model",
        "pseudo_replication_avoided": True,
    }


def _two_sided_sign_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    lower = min(wins, losses)
    return min(1.0, 2.0 * sum(comb(n, index) for index in range(lower + 1)) / (2**n))


def _nested(mapping: dict[str, object], first: str, second: str) -> object:
    value = mapping.get(first)
    if not isinstance(value, dict) or second not in value:
        raise RuntimeError(f"manifest metadata lacks {first}.{second}")
    return value[second]


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    summary["updated_at"] = datetime.now(timezone.utc).isoformat()
    encoded = json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--campaigns-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model-1p5b", default="qwen2.5-coder:1.5b")
    parser.add_argument("--model-7b", default="qwen2.5-coder:7b")
    parser.add_argument("--model-9b", default="qwen3.5:9b")
    parser.add_argument("--candidates-per-campaign", type=int, default=12)
    parser.add_argument("--budget-seconds", type=float, default=0.20)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    plan = ThreeModelComparisonPlan(
        arms=(
            ModelArm("coder-1p5b", args.model_1p5b),
            ModelArm("coder-7b", args.model_7b),
            ModelArm("general-9b", args.model_9b),
        ),
        candidates_per_campaign=args.candidates_per_campaign,
        per_cell_budget_seconds=args.budget_seconds,
    )
    result = run_three_model_comparison(
        repo_root=args.repo_root,
        campaigns_root=args.campaigns_root,
        source_manifest=args.source_manifest,
        endpoint=args.endpoint,
        plan=plan,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["ModelArm", "ThreeModelComparisonPlan", "run_three_model_comparison"]
