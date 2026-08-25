"""Evaluator-owned public-suite worker.

This module intentionally imports only the candidate worktree's AlgoFinder
package after it has been placed first on ``sys.path``. It is launched by the
authority, not exposed as an agent tool, and writes its result to stdout for
the authority to persist and redact.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any


def _load_candidate_solvers(entrypoints: list[str]) -> None:
    import algofinder.solvers.registry  # noqa: F401
    from algofinder.solvers.base import Solver, all_solvers, register_solver

    for entrypoint in entrypoints:
        module_name, separator, class_name = entrypoint.partition(":")
        if not separator or not module_name or not class_name:
            raise ValueError(f"invalid candidate entrypoint {entrypoint!r}")
        cls = getattr(importlib.import_module(module_name), class_name)
        if not isinstance(cls, type) or not issubclass(cls, Solver):
            raise TypeError(f"candidate entrypoint is not a Solver subclass: {entrypoint}")
        if cls.__module__ != module_name:
            raise ValueError(
                f"candidate entrypoint must resolve to its declared candidate module: {entrypoint}"
            )
        if not isinstance(cls.id, str) or not cls.id.strip():
            raise ValueError(f"candidate solver has an empty or invalid id: {entrypoint}")
        registered = all_solvers().get(cls.id)
        if registered is not None:
            raise ValueError(f"candidate solver id conflicts with existing solver: {cls.id}")
        register_solver(cls)


def run_public_suite(
    *, package_root: Path, manifest: Path, entrypoints: list[str],
    budget_seconds: float | None, timeout_seconds: float, seed: int,
    splits: tuple[str, ...], baseline_solver_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    sys.path.insert(0, str(package_root))
    from algofinder.harness.benchmark import run_benchmark
    from algofinder.instances.base import load_manifest

    _load_candidate_solvers(entrypoints)
    instances = load_manifest(manifest)
    solver_ids = []
    for entrypoint in entrypoints:
        cls = getattr(importlib.import_module(entrypoint.partition(":")[0]), entrypoint.partition(":")[2])
        solver_ids.append(cls.id)
    requested_solver_ids = tuple(dict.fromkeys([*solver_ids, *baseline_solver_ids]))
    runs = run_benchmark(
        instances,
        budget_seconds=budget_seconds,
        timeout_seconds=timeout_seconds,
        solver_include=requested_solver_ids,
        splits=splits,
        seed=seed,
    )
    mappings = [run.to_mapping() for run in runs]
    candidate_runs = [run for run in mappings if run["solver"] in solver_ids]
    baseline_runs = [run for run in mappings if run["solver"] in baseline_solver_ids]
    statuses = [run["status"] for run in candidate_runs]
    gaps = [run["gap_percent"] for run in candidate_runs if run.get("gap_percent") is not None]
    wall = [run["wall_seconds"] for run in candidate_runs if run.get("wall_seconds") is not None]
    hard_failure = any(status in {"invalid", "error", "timeout", "memory_limit"} for status in statuses)
    baseline_failure = any(
        run["status"] in {"invalid", "error", "timeout", "memory_limit"}
        for run in baseline_runs
    )
    paired: dict[str, dict[str, Any]] = {}
    paired_gaps: list[float] = []
    for baseline_id in baseline_solver_ids:
        baseline_by_instance = {
            (run["instance"], run["split"]): run
            for run in baseline_runs
            if run["solver"] == baseline_id and run["status"] == "ok" and run.get("cost") is not None
        }
        comparisons: list[float] = []
        strata: dict[tuple[str, str], list[float]] = {}
        wins = ties = losses = 0
        for run in candidate_runs:
            if run["status"] != "ok" or run.get("cost") is None:
                continue
            baseline = baseline_by_instance.get((run["instance"], run["split"]))
            if baseline is None or not baseline.get("cost"):
                continue
            gap = (float(run["cost"]) - float(baseline["cost"])) / float(baseline["cost"]) * 100.0
            comparisons.append(gap)
            strata.setdefault((str(run.get("split", "")), str(run.get("family", ""))), []).append(gap)
            if gap < -1e-9:
                wins += 1
            elif gap > 1e-9:
                losses += 1
            else:
                ties += 1
        mean_gap = (sum(comparisons) / len(comparisons)) if comparisons else None
        if mean_gap is not None:
            paired_gaps.append(mean_gap)
        paired[baseline_id] = {
            "paired_gap_percent": mean_gap,
            "mean_improvement_percent": -mean_gap if mean_gap is not None else None,
            "comparisons": len(comparisons),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "strata": [
                {
                    "split": split,
                    "family": family,
                    "comparisons": len(values),
                    "paired_gap_percent": sum(values) / len(values),
                    "wins": sum(value < -1e-9 for value in values),
                    "ties": sum(abs(value) <= 1e-9 for value in values),
                    "losses": sum(value > 1e-9 for value in values),
                }
                for (split, family), values in sorted(strata.items())
            ],
        }
    gate = "blocked" if baseline_failure else ("fail" if hard_failure else "pass")
    measured_runs = [
        run for run in mappings
        if isinstance(run.get("wall_seconds"), (int, float))
    ]
    total_wall = sum(float(run["wall_seconds"]) for run in measured_runs)
    total_cpu = sum(
        float(run["cpu_seconds"])
        for run in mappings
        if isinstance(run.get("cpu_seconds"), (int, float))
    )
    stratum_summaries = next(iter(paired.values())).get("strata", []) if paired else []
    return {
        "gate": gate,
        "reason": "baseline" if baseline_failure else ("correctness" if hard_failure else "ok"),
        "metric_vector": {
            "runs": len(candidate_runs), "ok_rate": (statuses.count("ok") / len(statuses)) if statuses else 0.0,
            "mean_gap_percent": (sum(gaps) / len(gaps)) if gaps else None,
            "paired_gap_percent": (sum(paired_gaps) / len(paired_gaps)) if paired_gaps else None,
            "mean_wall_seconds": (sum(wall) / len(wall)) if wall else None,
            "total_solver_wall_seconds": total_wall,
            "total_solver_cpu_seconds": total_cpu,
            "solver_cells": len(measured_runs),
            "strata": stratum_summaries,
        },
        "paired_baselines": paired,
        "raw_runs": mappings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--entrypoint", action="append", default=[])
    parser.add_argument("--baseline-solver", action="append", default=[])
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--budget-seconds", type=float)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        result = run_public_suite(
            package_root=Path(args.package_root), manifest=Path(args.manifest),
            entrypoints=list(args.entrypoint), budget_seconds=args.budget_seconds,
            timeout_seconds=args.timeout_seconds, seed=args.seed,
            splits=tuple(args.split) or ("test",),
            baseline_solver_ids=tuple(args.baseline_solver),
        )
    except Exception as exc:
        result = {"gate": "blocked", "reason": "build", "metric_vector": {}, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":  # pragma: no cover
    main()
