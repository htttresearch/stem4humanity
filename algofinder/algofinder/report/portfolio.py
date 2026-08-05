"""Portfolio report: virtual-best opportunity, regret, feature cost.

Given benchmark runs and feature records, quantify: (1) per-instance
virtual-best performance and per-solver share of wins, (2) mean regret
(gap of each solver against the virtual best on the instances it solved),
(3) the cost of the feature layer (timing and peak RSS per feature set),
and (4) per-split performance so out-of-distribution degradation is
visible when both splits are benchmarked.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from algofinder.features import FEATURE_SETS, instance_id
from algofinder.features.storage import read_feature_set
from algofinder.harness.benchmark import BenchmarkRun
from algofinder.harness.leaderboard import summarize
from algofinder.problems.base import Instance


def _flatten(values: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten a nested feature dict into dot-prefixed scalars."""
    flat: dict[str, float] = {}
    for key, value in sorted(values.items()):
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[name] = float(value)
    return flat


def portfolio(
    runs: Iterable[BenchmarkRun],
    instances: Iterable[Instance],
    instances_dir: str | Path,
) -> dict[str, Any]:
    runs = list(runs)
    digest_by_name = {instance.name: instance_id(instance) for instance in instances}

    ok = [run for run in runs if run.status == "ok" and run.cost is not None]
    per_instance: dict[str, list[BenchmarkRun]] = defaultdict(list)
    for run in ok:
        per_instance[run.instance].append(run)

    virtual_best: dict[str, float] = {}
    for instance, cells in per_instance.items():
        virtual_best[instance] = min(float(cell.cost) for cell in cells)

    wins: dict[str, set[str]] = defaultdict(set)
    regret_by_solver: dict[str, list[float]] = defaultdict(list)
    gap_by_split: dict[str, list[float]] = defaultdict(list)
    for instance, cells in sorted(per_instance.items()):
        best = virtual_best[instance]
        for cell in cells:
            cost = float(cell.cost)
            gap = (cost - best) / best * 100.0 if best != 0.0 else (0.0 if cost == 0.0 else float("inf"))
            regret_by_solver[cell.solver].append(gap)
            gap_by_split[cell.split or "test"].append(gap)
            if gap <= 0.0:
                wins[cell.solver].add(instance)

    solver_rows: list[dict[str, Any]] = []
    for solver in sorted(regret_by_solver):
        gaps = regret_by_solver[solver]
        solver_rows.append(
            {
                "solver": solver,
                "instances": len(gaps),
                "wins": len(wins[solver]),
                "mean_regret_percent": round(mean(gaps), 4),
                "median_regret_percent": round(_median(gaps), 4),
                "max_regret_percent": round(max(gaps), 4),
            }
        )

    feature_cost: dict[str, dict[str, Any]] = {}
    for feature_set in FEATURE_SETS:
        records = read_feature_set(instances_dir, feature_set.id)
        timings = [record.get("timing_seconds") for record in records if record.get("timing_seconds") is not None]
        rss = [record.get("peak_rss_bytes") for record in records if record.get("peak_rss_bytes") is not None]
        feature_cost[feature_set.id] = {
            "records": len(records),
            "mean_timing_seconds": round(mean(timings), 6) if timings else None,
            "median_timing_seconds": round(_median(timings), 6) if timings else None,
            "mean_peak_rss_bytes": int(mean(rss)) if rss else None,
        }

    feature_values: dict[str, dict[str, float]] = {}
    for feature_set in FEATURE_SETS:
        flat_accum: dict[str, list[float]] = defaultdict(list)
        for record in read_feature_set(instances_dir, feature_set.id):
            if "error" in record.get("status_by_feature", {}):
                continue
            for name, value in _flatten(record.get("values", {})).items():
                flat_accum[name].append(value)
        feature_values[feature_set.id] = {
            name: {
                "mean": round(mean(values), 6),
                "min": round(min(values), 6),
                "max": round(max(values), 6),
            }
            for name, values in sorted(flat_accum.items())
        }

    leaderboard = summarize(runs)
    split_gaps: dict[str, dict[str, Any]] = {}
    for split, gaps in sorted(gap_by_split.items()):
        split_gaps[split] = {
            "ok_runs": len(gaps),
            "mean_gap_percent_vs_vbs": round(mean(gaps), 4),
        }

    all_regrets = [
        gap for solver_gaps in regret_by_solver.values() for gap in solver_gaps
    ]

    return {
        "instances": len(per_instance),
        "ok_runs": len(ok),
        "splits_present": sorted(gap_by_split),
        "virtual_best_instances": len(virtual_best),
        "solvers": solver_rows,
        "mean_regret_percent": round(mean(all_regrets), 4) if all_regrets else None,
        "feature_cost": feature_cost,
        "feature_values": feature_values,
        "split_gaps": split_gaps,
        "leaderboard_rows": leaderboard,
    }


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def render_portfolio(summary: dict[str, Any]) -> str:
    lines = ["# Portfolio report", ""]
    lines.append(
        f"- instances with ok runs: {summary['instances']} "
        f"(ok runs: {summary['ok_runs']})"
    )
    lines.append(
        f"- splits present in runs: {', '.join(summary['splits_present']) or 'none'}"
    )
    lines.append(f"- virtual-best instances: {summary['virtual_best_instances']}")
    lines.append("")
    lines.append("## Virtual best vs solvers")
    lines.append("")
    lines.append("| Solver | Instances | Wins | Mean regret % | Median regret % | Max regret % |")
    lines.append("|---|---|---|---|---|---|")
    for row in sorted(summary["solvers"], key=lambda item: (item["mean_regret_percent"] is None, item["mean_regret_percent"])):
        lines.append(
            f"| {row['solver']} | {row['instances']} | {row['wins']} "
            f"| {row['mean_regret_percent']} | {row['median_regret_percent']} "
            f"| {row['max_regret_percent']} |"
        )
    lines.append("")
    lines.append("## Feature cost")
    lines.append("")
    lines.append("| Feature set | Records | Mean timing s | Median timing s | Mean peak RSS B |")
    lines.append("|---|---|---|---|---|")
    for set_id, cost in sorted(summary["feature_cost"].items()):
        lines.append(
            f"| {set_id} | {cost['records']} | {cost['mean_timing_seconds']} "
            f"| {cost['median_timing_seconds']} | {cost['mean_peak_rss_bytes']} |"
        )
    lines.append("")
    lines.append("## Per-split gap vs virtual best")
    lines.append("")
    for split, gaps in sorted(summary["split_gaps"].items()):
        lines.append(f"- {split}: {gaps['ok_runs']} ok runs, mean gap {gaps['mean_gap_percent_vs_vbs']}%")
    return "\n".join(lines)


__all__ = ["_flatten", "portfolio", "render_portfolio"]
