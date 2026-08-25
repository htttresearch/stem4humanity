"""Adapter from frozen candidate worktrees to the existing benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Literal

from algofinder.agents.contracts import Candidate, ExperimentSpec
from algofinder.agents.sandbox import SandboxError, SandboxPolicy, SandboxRunner
from algofinder.agents.workspace import CandidateWorkspace


class BenchmarkAdapterError(RuntimeError):
    """Raised when evaluator-owned benchmark execution cannot be trusted."""


class PublicBenchmarkAdapter:
    """Run authority-gated local suites through AlgoFinder's existing harness.

    This adapter is deliberately public-only. Validation/challenge execution is
    refused until the deployment supplies a container profile that mounts the
    package, worker, and hidden input paths with no host escape route.
    """

    def __init__(
        self,
        *,
        authority_package_root: str | Path,
        runner: SandboxRunner | None = None,
        allow_trusted_local: bool = False,
        baseline_solver_ids: tuple[str, ...] = (),
        cell_timeout_seconds: float | None = None,
    ) -> None:
        self.authority_package_root = Path(authority_package_root).resolve()
        self.runner = runner or SandboxRunner()
        self.allow_trusted_local = allow_trusted_local
        self.baseline_solver_ids = baseline_solver_ids
        self.cell_timeout_seconds = cell_timeout_seconds

    def evaluate(
        self,
        *,
        workspace: CandidateWorkspace,
        candidate: Candidate,
        experiment: ExperimentSpec,
        manifest_path: str | Path,
        splits: tuple[str, ...],
        timeout_seconds: float,
        zone: Literal["public", "validation", "challenge"] = "public",
    ) -> dict[str, Any]:
        if experiment.stage not in {"gate_1", "gate_2"}:
            raise BenchmarkAdapterError("local benchmark adapter only runs gate_1 or gate_2")
        trusted_templates = {"tsp-edge-formula-v1", "tsp-hir-v1"}
        if zone != "public" and candidate.descriptors.get("candidate_template") not in trusted_templates:
            raise BenchmarkAdapterError(
                "arbitrary candidate code may not share a process or filesystem with a sealed manifest"
            )
        if not candidate.solver_entrypoints:
            raise BenchmarkAdapterError("candidate has no declared solver entrypoint")
        manifest = Path(manifest_path).resolve()
        if not manifest.is_file():
            raise BenchmarkAdapterError(f"public suite manifest not found: {manifest}")
        if candidate.descriptors.get("candidate_template") not in trusted_templates:
            try:
                manifest_rows = json.loads(manifest.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise BenchmarkAdapterError(f"suite manifest is not valid JSON: {exc}") from exc
            if not isinstance(manifest_rows, list) or any(
                not isinstance(row, dict) or row.get("split", "test") not in splits
                for row in manifest_rows
            ):
                raise BenchmarkAdapterError(
                    "arbitrary candidate code requires a split-specific public manifest; "
                    "the supplied file also contains non-public instances"
                )
        worker = self.authority_package_root / "algofinder" / "agents" / "evaluator_worker.py"
        if not worker.is_file():
            raise BenchmarkAdapterError(f"evaluator worker is missing: {worker}")
        package_root = workspace.worktree / "algofinder"
        runtime_paths: tuple[Path, ...] = ()
        cell_timeout = self.cell_timeout_seconds or timeout_seconds
        if self.runner.isolated_available:
            # The worker is self-contained standard-library code.  Supplying
            # it through ``-c`` lets Bubblewrap expose only the candidate
            # worktree, the explicit public manifest, and this virtualenv.
            runtime = Path(sys.executable).absolute().parent.parent
            base_runtime = Path(sys.base_prefix).resolve()
            site_packages = f"/runtime/0/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
            command = [
                "/usr/bin/env", f"PYTHONPATH={site_packages}",
                "/runtime/1/bin/python3", "-c", worker.read_text(encoding="utf-8"),
                "--package-root", "/workspace/algofinder", "--manifest", "/inputs/0",
                "--timeout-seconds", str(cell_timeout), "--seed", str(experiment.seeds[0]),
            ]
            runtime_paths = (runtime, base_runtime)
        else:
            command = [
                sys.executable, str(worker), "--package-root", str(package_root),
                "--manifest", str(manifest),
                "--timeout-seconds", str(cell_timeout), "--seed", str(experiment.seeds[0]),
            ]
        if experiment.budget_seconds is not None:
            command.extend(("--budget-seconds", str(experiment.budget_seconds)))
        for split in splits:
            command.extend(("--split", split))
        for entrypoint in candidate.solver_entrypoints:
            command.extend(("--entrypoint", entrypoint))
        for solver_id in self.baseline_solver_ids:
            command.extend(("--baseline-solver", solver_id))
        try:
            result = self.runner.run(
                command, worktree=workspace.worktree,
                policy=SandboxPolicy(
                    zone=zone, timeout_seconds=timeout_seconds,
                    trusted_local=self.allow_trusted_local,
                ),
                readable_paths=(manifest,),
                runtime_paths=runtime_paths,
            )
        except SandboxError as exc:
            raise BenchmarkAdapterError(str(exc)) from exc
        if result.timed_out:
            return {"gate": "fail", "reason": "resource", "metric_vector": {"timed_out": True}}
        if result.returncode != 0:
            return {"gate": "blocked", "reason": "build", "metric_vector": {}, "error": result.stderr}
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            detail = result.stdout[:500].replace("\n", " ")
            raise BenchmarkAdapterError(
                f"evaluator worker did not return JSON: {detail!r}"
            ) from exc

    def reserved_evaluation_seconds(
        self,
        *,
        manifest_path: str | Path,
        splits: tuple[str, ...],
        candidate: Candidate,
        per_cell_budget_seconds: float | None,
    ) -> float:
        """Return the hard-cap reservation for every candidate/baseline cell."""
        if per_cell_budget_seconds is None:
            return 0.0
        manifest = Path(manifest_path).resolve()
        try:
            rows = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkAdapterError(f"cannot inspect suite manifest for budgeting: {exc}") from exc
        if not isinstance(rows, list):
            raise BenchmarkAdapterError("suite manifest must be a JSON list")
        instances = sum(
            isinstance(row, dict) and row.get("split", "test") in splits
            for row in rows
        )
        solvers = len(tuple(dict.fromkeys((*candidate.solver_entrypoints, *self.baseline_solver_ids))))
        return float(per_cell_budget_seconds) * instances * solvers


__all__ = ["BenchmarkAdapterError", "PublicBenchmarkAdapter"]
