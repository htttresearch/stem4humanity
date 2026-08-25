"""Preregistered, paired comparison of two local proposal models.

The unit of inference is a complete campaign, not an individual candidate.
Each model receives the same formula-only search language and development
feedback.  One development-selected candidate per campaign is then evaluated
once on the test splits, after the search loop has ended, so held-out evidence
cannot influence either model's proposals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from algofinder.agents.ollama_model import unload_ollama_model
from algofinder.agents.pilot import run_pilot


DEFAULT_REPLICATE_SEEDS = (1181, 2713, 4219, 5779, 6971)
DEFAULT_HOLDOUT_SPLITS = ("test_iid", "test_family_ood", "test_parameter_ood")


@dataclass(frozen=True)
class ModelComparisonPlan:
    """Fixed design for a two-model formula-search ablation."""

    control_model: str
    treatment_model: str
    replicate_seeds: tuple[int, ...] = DEFAULT_REPLICATE_SEEDS
    candidates_per_campaign: int = 12
    per_cell_budget_seconds: float = 0.20
    holdout_splits: tuple[str, ...] = DEFAULT_HOLDOUT_SPLITS

    def __post_init__(self) -> None:
        if not self.control_model.strip() or not self.treatment_model.strip():
            raise ValueError("both comparison model names are required")
        if self.control_model == self.treatment_model:
            raise ValueError("comparison models must differ")
        if len(self.replicate_seeds) < 2 or len(set(self.replicate_seeds)) != len(self.replicate_seeds):
            raise ValueError("use at least two unique replicate seeds")
        if any(seed < 0 for seed in self.replicate_seeds):
            raise ValueError("replicate seeds must be non-negative")
        if self.candidates_per_campaign < 1 or self.per_cell_budget_seconds <= 0:
            raise ValueError("candidate count and evaluator budget must be positive")
        if not self.holdout_splits:
            raise ValueError("a real model comparison requires a held-out suite")

    def to_mapping(self) -> dict[str, object]:
        return {
            "control_model": self.control_model,
            "treatment_model": self.treatment_model,
            "replicate_seeds": list(self.replicate_seeds),
            "candidates_per_campaign": self.candidates_per_campaign,
            "per_cell_budget_seconds": self.per_cell_budget_seconds,
            "representation": "formula",
            "development_split": "validation",
            "holdout_splits": list(self.holdout_splits),
            "primary_outcome": "held-out paired_gap_percent of the development-selected campaign champion",
            "selection_rule": "lowest development paired_gap_percent; proposal ordinal then patch digest tie break",
            "order_rule": (
                "counterbalance arm order across replicate seeds; explicitly unload the resident "
                "model after every campaign before loading the other arm"
            ),
            "gpu_lifecycle": (
                "exactly one comparison model is resident at a time and is offloaded with "
                "Ollama keep_alive=0 after its campaign"
            ),
        }


def run_model_comparison(
    *,
    repo_root: Path,
    campaigns_root: Path,
    manifest_path: Path,
    endpoint: str,
    plan: ModelComparisonPlan,
) -> dict[str, object]:
    """Run all campaigns serially and retain campaign-level paired outcomes."""
    campaigns_root = campaigns_root.resolve()
    campaigns_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_path = campaigns_root / f"tsp-formula-model-comparison-{stamp}.json"
    runs: list[dict[str, object]] = []

    for replicate_index, seed in enumerate(plan.replicate_seeds):
        arms = (("control", plan.control_model), ("treatment", plan.treatment_model))
        if replicate_index % 2:
            arms = tuple(reversed(arms))
        for arm, model in arms:
            try:
                result = run_pilot(
                    repo_root=repo_root,
                    campaigns_root=campaigns_root,
                    manifest_path=manifest_path,
                    model_name=model,
                    endpoint=endpoint,
                    target_candidates=plan.candidates_per_campaign,
                    per_cell_budget_seconds=plan.per_cell_budget_seconds,
                    representation="formula",
                    backend="model",
                    model_seed=seed,
                    holdout_splits=plan.holdout_splits,
                )
                runs.append(_run_record(arm=arm, model=model, seed=seed, result=result))
                _write_summary(output_path, plan=plan, runs=runs, complete=False)
            finally:
                # The next arm must never begin with prior model weights
                # retained on a small GPU.
                unload_ollama_model(endpoint=endpoint, model=model)

    summary = _write_summary(output_path, plan=plan, runs=runs, complete=True)
    return {**summary, "summary_path": str(output_path)}


def _run_record(*, arm: str, model: str, seed: int, result: dict[str, object]) -> dict[str, object]:
    holdout = result.get("holdout")
    return {
        "arm": arm,
        "model": model,
        "model_seed": seed,
        "campaign_id": result["campaign_id"],
        "campaign_root": result["campaign_root"],
        "evaluated_candidates": result["evaluated_candidates"],
        "holdout": holdout,
        "model_usage": result.get("model_usage", []),
    }


def _heldout_gap(run: dict[str, object]) -> float | None:
    holdout = run.get("holdout")
    if not isinstance(holdout, dict) or holdout.get("gate") != "pass":
        return None
    metrics = holdout.get("metric_vector")
    value = metrics.get("paired_gap_percent") if isinstance(metrics, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def _analysis(plan: ModelComparisonPlan, runs: list[dict[str, object]]) -> dict[str, object]:
    paired: list[dict[str, object]] = []
    for seed in plan.replicate_seeds:
        control = next(
            (row for row in runs if row["model_seed"] == seed and row["arm"] == "control"), None
        )
        treatment = next(
            (row for row in runs if row["model_seed"] == seed and row["arm"] == "treatment"), None
        )
        if control is None or treatment is None:
            continue
        control_gap = _heldout_gap(control)
        treatment_gap = _heldout_gap(treatment)
        if control_gap is None or treatment_gap is None:
            paired.append({"seed": seed, "status": "incomplete"})
            continue
        paired.append({
            "seed": seed,
            "control_gap_percent": control_gap,
            "treatment_gap_percent": treatment_gap,
            "treatment_minus_control_percent": treatment_gap - control_gap,
        })

    deltas = [
        float(row["treatment_minus_control_percent"])
        for row in paired if isinstance(row.get("treatment_minus_control_percent"), (int, float))
    ]
    return {
        "paired_campaigns": paired,
        "paired_campaigns_complete": len(deltas),
        "treatment_wins": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "control_wins": sum(delta > 0 for delta in deltas),
        "mean_treatment_minus_control_percent": mean(deltas) if deltas else None,
        "median_treatment_minus_control_percent": median(deltas) if deltas else None,
        "inference_scope": (
            "confirmatory" if len(deltas) >= 20 else
            "exploratory: fewer than 20 complete paired campaigns"
        ),
        "interpretation": (
            "negative paired differences favor the treatment model; these are campaign-level "
            "replicates, not pseudo-replicated candidate-level observations"
        ),
    }


def _write_summary(
    output_path: Path,
    *,
    plan: ModelComparisonPlan,
    runs: list[dict[str, object]],
    complete: bool,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "schema_id": "algofinder.agents.model-comparison",
        "schema_version": "1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "plan": plan.to_mapping(),
        "runs": runs,
        "analysis": _analysis(plan, runs),
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--campaigns-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--control-model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--treatment-model", default="qwen3.5:9b")
    parser.add_argument("--replicate-seed", action="append", type=int, default=[])
    parser.add_argument("--candidates-per-campaign", type=int, default=12)
    parser.add_argument("--budget-seconds", type=float, default=0.20)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    plan = ModelComparisonPlan(
        control_model=args.control_model,
        treatment_model=args.treatment_model,
        replicate_seeds=tuple(args.replicate_seed) or DEFAULT_REPLICATE_SEEDS,
        candidates_per_campaign=args.candidates_per_campaign,
        per_cell_budget_seconds=args.budget_seconds,
    )
    result = run_model_comparison(
        repo_root=args.repo_root,
        campaigns_root=args.campaigns_root,
        manifest_path=args.manifest,
        endpoint=args.endpoint,
        plan=plan,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["ModelComparisonPlan", "run_model_comparison"]
