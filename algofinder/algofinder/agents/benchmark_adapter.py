"""Adapter from frozen candidate worktrees to the existing benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from algofinder.agents.contracts import Candidate, ExperimentSpec
from algofinder.agents.sandbox import SandboxError, SandboxPolicy, SandboxRunner
from algofinder.agents.workspace import CandidateWorkspace


class BenchmarkAdapterError(RuntimeError):
    """Raised when evaluator-owned benchmark execution cannot be trusted."""


class PublicBenchmarkAdapter:
    """Run gate-1 smoke suites through AlgoFinder's existing harness.

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
    ) -> None:
        self.authority_package_root = Path(authority_package_root).resolve()
        self.runner = runner or SandboxRunner()
        self.allow_trusted_local = allow_trusted_local

    def evaluate(
        self,
        *,
        workspace: CandidateWorkspace,
        candidate: Candidate,
        experiment: ExperimentSpec,
        manifest_path: str | Path,
        splits: tuple[str, ...],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        if experiment.stage != "gate_1":
            raise BenchmarkAdapterError("PublicBenchmarkAdapter only runs gate_1")
        if not candidate.solver_entrypoints:
            raise BenchmarkAdapterError("candidate has no declared solver entrypoint")
        manifest = Path(manifest_path).resolve()
        if not manifest.is_file():
            raise BenchmarkAdapterError(f"public suite manifest not found: {manifest}")
        if self.runner.isolated_available:
            raise BenchmarkAdapterError(
                "Bubblewrap adapter wiring is deployment-specific; use the sealed evaluator service instead"
            )
        worker = self.authority_package_root / "algofinder" / "agents" / "evaluator_worker.py"
        if not worker.is_file():
            raise BenchmarkAdapterError(f"evaluator worker is missing: {worker}")
        package_root = workspace.worktree / "algofinder"
        command = [
            sys.executable, str(worker), "--package-root", str(package_root),
            "--manifest", str(manifest),
            "--timeout-seconds", str(timeout_seconds), "--seed", str(experiment.seeds[0]),
        ]
        if experiment.budget_seconds is not None:
            command.extend(("--budget-seconds", str(experiment.budget_seconds)))
        for split in splits:
            command.extend(("--split", split))
        for entrypoint in candidate.solver_entrypoints:
            command.extend(("--entrypoint", entrypoint))
        try:
            result = self.runner.run(
                command, worktree=workspace.worktree,
                policy=SandboxPolicy(
                    zone="public", timeout_seconds=timeout_seconds,
                    trusted_local=self.allow_trusted_local,
                ),
                readable_paths=(manifest,),
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
            raise BenchmarkAdapterError("evaluator worker did not return JSON") from exc


__all__ = ["BenchmarkAdapterError", "PublicBenchmarkAdapter"]
