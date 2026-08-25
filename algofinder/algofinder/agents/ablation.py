"""Matched-arm orchestration for formula, HIR, and full-code comparisons.

This module keeps the experimental comparison honest: every enabled arm uses
the same manifest, seed, candidate count, and per-cell evaluator budget.  A
strong-model full-code lane is represented explicitly and is skipped until a
model name is supplied; it is never silently substituted with the 1.5B model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Literal

from algofinder.agents.pilot import run_pilot


ArmName = Literal["formula-1p5b", "hir-evolution", "hir-1p5b", "full-code-strong"]


@dataclass(frozen=True)
class AblationArm:
    name: ArmName
    representation: Literal["formula", "hir", "full-code"]
    backend: Literal["model", "evolution"]
    model: str | None
    enabled: bool

    def __post_init__(self) -> None:
        if self.backend == "evolution" and self.representation != "hir":
            raise ValueError("only the HIR arm may use the evolution backend")
        if self.backend == "model" and self.enabled and not self.model:
            raise ValueError(f"enabled model arm {self.name} requires a model name")
        if self.name == "full-code-strong" and self.representation != "full-code":
            raise ValueError("full-code-strong must use the full-code representation")


@dataclass(frozen=True)
class MatchedAblation:
    candidates_per_arm: int
    per_cell_budget_seconds: float
    seed: int
    arms: tuple[AblationArm, ...]

    def __post_init__(self) -> None:
        if self.candidates_per_arm < 1:
            raise ValueError("candidates_per_arm must be positive")
        if self.per_cell_budget_seconds <= 0:
            raise ValueError("per_cell_budget_seconds must be positive")
        enabled = [arm for arm in self.arms if arm.enabled]
        if len(enabled) < 3:
            raise ValueError("a matched ablation needs formula, HIR evolution, and HIR model arms")
        names = [arm.name for arm in self.arms]
        if len(set(names)) != len(names):
            raise ValueError("ablation arm names must be unique")


def default_ablation(*, local_model: str, strong_model: str | None = None, candidates_per_arm: int = 4, per_cell_budget_seconds: float = 0.20) -> MatchedAblation:
    """Create the four requested arms, leaving unavailable strong control disabled."""
    return MatchedAblation(
        candidates_per_arm=candidates_per_arm,
        per_cell_budget_seconds=per_cell_budget_seconds,
        seed=0,
        arms=(
            AblationArm("formula-1p5b", "formula", "model", local_model, True),
            AblationArm("hir-evolution", "hir", "evolution", None, True),
            AblationArm("hir-1p5b", "hir", "model", local_model, True),
            AblationArm("full-code-strong", "full-code", "model", strong_model, strong_model is not None),
        ),
    )


def run_matched_ablation(
    *,
    repo_root: Path,
    campaigns_root: Path,
    manifest_path: Path,
    local_endpoint: str,
    spec: MatchedAblation,
) -> dict[str, object]:
    """Run enabled arms serially and persist a compact, linked summary."""
    runs: list[dict[str, object]] = []
    for arm in spec.arms:
        if not arm.enabled:
            runs.append({"arm": arm.name, "status": "skipped", "reason": "no strong model configured"})
            continue
        result = run_pilot(
            repo_root=repo_root,
            campaigns_root=campaigns_root,
            manifest_path=manifest_path,
            model_name=arm.model or "",
            endpoint=local_endpoint,
            target_candidates=spec.candidates_per_arm,
            per_cell_budget_seconds=spec.per_cell_budget_seconds,
            representation=arm.representation,
            backend=arm.backend,
            model_seed=spec.seed,
        )
        runs.append({
            "arm": arm.name,
            "status": "complete",
            "campaign_id": result["campaign_id"],
            "campaign_root": result["campaign_root"],
            "evaluated_candidates": result["evaluated_candidates"],
        })
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    summary = {
        "schema_id": "algofinder.agents.matched-ablation",
        "schema_version": "1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spec": {**asdict(spec), "arms": [asdict(arm) for arm in spec.arms]},
        "runs": runs,
    }
    output = campaigns_root / f"tsp-matched-ablation-{stamp}.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    summary["summary_path"] = str(output)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a matched AlgoFinder representation ablation")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--campaigns-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--local-model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--strong-model")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--candidates-per-arm", type=int, default=4)
    parser.add_argument("--budget-seconds", type=float, default=0.20)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    spec = default_ablation(
        local_model=args.local_model,
        strong_model=args.strong_model,
        candidates_per_arm=args.candidates_per_arm,
        per_cell_budget_seconds=args.budget_seconds,
    )
    result = run_matched_ablation(
        repo_root=Path(args.repo_root),
        campaigns_root=Path(args.campaigns_root),
        manifest_path=Path(args.manifest),
        local_endpoint=args.endpoint,
        spec=spec,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["AblationArm", "MatchedAblation", "default_ablation", "run_matched_ablation"]
