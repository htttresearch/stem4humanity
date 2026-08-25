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
    from algofinder.solvers.base import Solver, all_solvers, register_solver

    for entrypoint in entrypoints:
        module_name, separator, class_name = entrypoint.partition(":")
        if not separator or not module_name or not class_name:
            raise ValueError(f"invalid candidate entrypoint {entrypoint!r}")
        cls = getattr(importlib.import_module(module_name), class_name)
        if not isinstance(cls, type) or not issubclass(cls, Solver):
            raise TypeError(f"candidate entrypoint is not a Solver subclass: {entrypoint}")
        registered = all_solvers().get(cls.id)
        if registered is None:
            register_solver(cls)
        elif registered is not cls:
            raise ValueError(f"candidate solver id conflicts with existing solver: {cls.id}")


def run_public_suite(
    *, package_root: Path, manifest: Path, entrypoints: list[str],
    budget_seconds: float | None, timeout_seconds: float, seed: int,
    splits: tuple[str, ...],
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
    runs = run_benchmark(
        instances,
        budget_seconds=budget_seconds,
        timeout_seconds=timeout_seconds,
        solver_include=tuple(solver_ids),
        splits=splits,
        seed=seed,
    )
    mappings = [run.to_mapping() for run in runs]
    statuses = [run["status"] for run in mappings]
    gaps = [run["gap_percent"] for run in mappings if run.get("gap_percent") is not None]
    wall = [run["wall_seconds"] for run in mappings if run.get("wall_seconds") is not None]
    hard_failure = any(status in {"invalid", "error", "timeout", "memory_limit"} for status in statuses)
    return {
        "gate": "fail" if hard_failure else "pass",
        "reason": "correctness" if hard_failure else "ok",
        "metric_vector": {
            "runs": len(mappings), "ok_rate": (statuses.count("ok") / len(statuses)) if statuses else 0.0,
            "mean_gap_percent": (sum(gaps) / len(gaps)) if gaps else None,
            "mean_wall_seconds": (sum(wall) / len(wall)) if wall else None,
        },
        "raw_runs": mappings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--entrypoint", action="append", default=[])
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
        )
    except Exception as exc:
        result = {"gate": "blocked", "reason": "build", "metric_vector": {}, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":  # pragma: no cover
    main()
