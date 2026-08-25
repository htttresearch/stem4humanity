"""Disposable candidate worktrees and a strict source-change allowlist.

Candidate code never edits the campaign base.  A worktree starts at the fixed
campaign commit, optionally receives immutable parent patches, and is frozen
into a normalized patch before it can enter the ledger.  This is isolation for
source state; execution isolation belongs to :mod:`algofinder.agents.sandbox`.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import fnmatch
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Iterable

from algofinder.agents.contracts import Candidate, ContractError, new_id, utc_now
from algofinder.agents.ledger import CampaignLedger, LedgerError
from algofinder.trace.serialize import canonical_dumps, strict_dumps


class WorkspaceError(RuntimeError):
    """Raised when a candidate workspace or patch violates its contract."""


@dataclass(frozen=True)
class PatchPolicy:
    """Git-relative paths a candidate is permitted to change."""

    allowed_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.allowed_paths:
            raise WorkspaceError("candidate patch allowlist cannot be empty")
        for pattern in self.allowed_paths:
            if not pattern or pattern.startswith("/") or ".." in PurePosixPath(pattern).parts:
                raise WorkspaceError(f"unsafe patch allowlist pattern {pattern!r}")

    def allows(self, path: str) -> bool:
        normalized = _relative_path(path)
        return any(_matches(normalized, pattern) for pattern in self.allowed_paths)

    def validate_paths(self, paths: Iterable[str]) -> list[str]:
        disallowed = sorted({_relative_path(path) for path in paths if not self.allows(path)})
        if disallowed:
            raise WorkspaceError(
                "candidate patch changes protected benchmark/control-plane files: "
                + ", ".join(disallowed)
            )
        return sorted({_relative_path(path) for path in paths})


@dataclass(frozen=True)
class CandidateWorkspace:
    candidate_id: str
    worktree: Path
    base_commit: str
    parent_ids: tuple[str, ...]
    hypothesis_id: str
    generation_operator: str

    @property
    def metadata_path(self) -> Path:
        # Keep service metadata outside the Git worktree so it can never be
        # mistaken for an agent-authored candidate file.
        return self.worktree.parent / f"{self.candidate_id}.workspace.json"


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    changed_paths: tuple[str, ...]
    patch_digest: str | None
    build_digest: str | None
    errors: tuple[str, ...]
    commands: tuple[dict[str, Any], ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "changed_paths": list(self.changed_paths),
            "patch_digest": self.patch_digest,
            "build_digest": self.build_digest,
            "errors": list(self.errors),
            "commands": list(self.commands),
        }


class WorkspaceService:
    """Creates, validates, and freezes candidate source worktrees."""

    def __init__(
        self,
        *,
        source_root: str | Path,
        package_root: str | Path,
        ledger: CampaignLedger,
        policy: PatchPolicy,
    ) -> None:
        self.source_root = Path(source_root).resolve()
        self.package_root = Path(package_root).resolve()
        self.ledger = ledger
        self.policy = policy
        self._git_root = self._git("rev-parse", "--show-toplevel", cwd=self.source_root).strip()
        if Path(self._git_root).resolve() != self.source_root:
            raise WorkspaceError("source_root must be the Git worktree root")
        try:
            self.package_rel = self.package_root.relative_to(self.source_root)
        except ValueError as exc:
            raise WorkspaceError("package_root must be below source_root") from exc

    def create(
        self,
        *,
        hypothesis_id: str,
        parent_ids: Iterable[str] = (),
        generation_operator: str = "invent",
        candidate_id: str | None = None,
    ) -> CandidateWorkspace:
        """Create a worktree at the campaign base and replay parent patches."""
        if generation_operator not in {"invent", "mutate", "recombine", "repair", "tune", "distill"}:
            raise WorkspaceError(f"unsupported generation operator {generation_operator!r}")
        candidate_id = candidate_id or new_id("candidate")
        parents = tuple(parent_ids)
        target = self.ledger.root / "workspaces" / candidate_id
        if target.exists():
            return self._load_workspace(target)
        base_commit = str(self.ledger.campaign()["base_commit"])
        self._git("worktree", "add", "--detach", str(target), base_commit, cwd=self.source_root)
        workspace = CandidateWorkspace(
            candidate_id=candidate_id,
            worktree=target,
            base_commit=base_commit,
            parent_ids=parents,
            hypothesis_id=hypothesis_id,
            generation_operator=generation_operator,
        )
        try:
            for parent_id in parents:
                patch = self.ledger.root / "candidates" / parent_id / "change.patch"
                if not patch.is_file():
                    raise WorkspaceError(f"parent candidate has no frozen patch: {parent_id}")
                self._apply_patch_file(workspace, patch)
            workspace.metadata_path.write_text(
                strict_dumps({
                    "schema_id": "algofinder.agents.workspace",
                    "schema_version": "1",
                    "candidate_id": candidate_id,
                    "base_commit": base_commit,
                    "parent_ids": list(parents),
                    "hypothesis_id": hypothesis_id,
                    "generation_operator": generation_operator,
                    "created_at": utc_now(),
                }) + "\n",
                encoding="utf-8",
            )
        except Exception:
            self._git("worktree", "remove", "--force", str(target), cwd=self.source_root, check=False)
            raise
        return workspace

    def apply_patch(self, workspace: CandidateWorkspace, patch: str) -> None:
        """Apply a proposed unified diff after checking its touched paths."""
        self._ensure_mutable(workspace)
        incoming = workspace.worktree.parent / f".{workspace.candidate_id}.incoming.patch"
        incoming.write_text(patch, encoding="utf-8")
        try:
            self._validate_patch_headers(patch)
            self._apply_patch_file(workspace, incoming)
        finally:
            incoming.unlink(missing_ok=True)

    def validate(self, workspace: CandidateWorkspace) -> ValidationReport:
        """Run gate-0 checks without evaluating solver performance."""
        errors: list[str] = []
        commands: list[dict[str, Any]] = []
        try:
            paths = tuple(self.changed_paths(workspace))
        except WorkspaceError as exc:
            return ValidationReport(False, (), None, None, (str(exc),), ())
        if not paths:
            errors.append("candidate patch is empty")
        try:
            self.policy.validate_paths(paths)
        except WorkspaceError as exc:
            errors.append(str(exc))
        diff_check = self._run(("git", "diff", "--check", workspace.base_commit, "--"), cwd=workspace.worktree)
        commands.append(diff_check)
        if diff_check["returncode"] != 0:
            errors.append("git diff --check failed: " + diff_check["stderr"].strip())
        changed_python = [str(workspace.worktree / path) for path in paths if path.endswith(".py")]
        if changed_python:
            syntax_check = (
                "from pathlib import Path; import sys; "
                "[compile(Path(p).read_text(encoding='utf-8'), p, 'exec') for p in sys.argv[1:]]"
            )
            compile_result = self._run((sys.executable, "-B", "-c", syntax_check, *changed_python), cwd=workspace.worktree)
            commands.append(compile_result)
            if compile_result["returncode"] != 0:
                errors.append("Python compile check failed")
        patch = self.normalized_patch(workspace)
        patch_digest = sha256(patch.encode("utf-8")).hexdigest() if patch else None
        build_digest = self.build_digest(workspace, patch_digest) if patch_digest else None
        return ValidationReport(
            passed=not errors,
            changed_paths=paths,
            patch_digest=patch_digest,
            build_digest=build_digest,
            errors=tuple(errors),
            commands=tuple(commands),
        )

    def freeze(
        self,
        workspace: CandidateWorkspace,
        *,
        solver_entrypoints: Iterable[str] = (),
        descriptors: dict[str, Any] | None = None,
        producing_agent_run_id: str | None = None,
    ) -> Candidate:
        """Validate and write a final, immutable candidate record plus patch."""
        self._ensure_mutable(workspace)
        report = self.validate(workspace)
        if not report.passed or report.patch_digest is None or report.build_digest is None:
            raise WorkspaceError("candidate cannot be frozen: " + "; ".join(report.errors))
        candidate = Candidate(
            candidate_id=workspace.candidate_id,
            campaign_id=self.ledger.campaign_id,
            hypothesis_id=workspace.hypothesis_id,
            parent_ids=workspace.parent_ids,
            generation_operator=workspace.generation_operator,  # type: ignore[arg-type]
            solver_entrypoints=tuple(solver_entrypoints),
            producing_agent_run_id=producing_agent_run_id,
            patch_digest=report.patch_digest,
            build_digest=report.build_digest,
            descriptors=descriptors or {},
        )
        candidate_dir = self.ledger.root / "candidates" / candidate.candidate_id
        candidate_dir.mkdir(exist_ok=True)
        patch_path = candidate_dir / "change.patch"
        _write_once(patch_path, self.normalized_patch(workspace))
        _write_once(candidate_dir / "build.json", strict_dumps(report.to_mapping()) + "\n")
        self.ledger.write(candidate)
        return candidate

    def changed_paths(self, workspace: CandidateWorkspace) -> list[str]:
        self._ensure_workspace(workspace)
        untracked = [
            path for path in self._git("ls-files", "--others", "--exclude-standard", "-z", cwd=workspace.worktree).split("\0")
            if path and not _generated_path(path)
        ]
        if untracked:
            self.policy.validate_paths(untracked)
            # Intent-to-add makes a new source file visible to git diff without
            # creating a committed/staged candidate change in the base worktree.
            self._git("add", "-N", "--", *untracked, cwd=workspace.worktree)
        output = self._git("diff", "--name-only", "--diff-filter=ACDMRTUXB", workspace.base_commit, "--", cwd=workspace.worktree)
        paths = [line for line in output.splitlines() if line]
        return self.policy.validate_paths(paths)

    def normalized_patch(self, workspace: CandidateWorkspace) -> str:
        self._ensure_workspace(workspace)
        patch = self._git("diff", "--binary", "--no-ext-diff", workspace.base_commit, "--", cwd=workspace.worktree)
        self._validate_patch_headers(patch)
        return patch

    def build_digest(self, workspace: CandidateWorkspace, patch_digest: str | None = None) -> str:
        """Digest build-relevant source identity without recording environment secrets."""
        package_pyproject = workspace.worktree / self.package_rel / "pyproject.toml"
        pyproject_digest = sha256(package_pyproject.read_bytes()).hexdigest() if package_pyproject.exists() else None
        payload = {
            "base_commit": workspace.base_commit,
            "patch_digest": patch_digest or sha256(self.normalized_patch(workspace).encode("utf-8")).hexdigest(),
            "pyproject_digest": pyproject_digest,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        }
        return sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()

    def _load_workspace(self, target: Path) -> CandidateWorkspace:
        metadata_path = target / ".algofinder-candidate.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise WorkspaceError(f"workspace exists but metadata is invalid: {target}") from exc
        return CandidateWorkspace(
            candidate_id=metadata["candidate_id"],
            worktree=target,
            base_commit=metadata["base_commit"],
            parent_ids=tuple(metadata["parent_ids"]),
            hypothesis_id=metadata["hypothesis_id"],
            generation_operator=metadata["generation_operator"],
        )

    def _ensure_workspace(self, workspace: CandidateWorkspace) -> None:
        if not workspace.worktree.is_dir() or not workspace.metadata_path.is_file():
            raise WorkspaceError("candidate workspace does not exist")
        try:
            workspace.worktree.resolve().relative_to((self.ledger.root / "workspaces").resolve())
        except ValueError as exc:
            raise WorkspaceError("candidate workspace escapes campaign workspaces") from exc

    def _ensure_mutable(self, workspace: CandidateWorkspace) -> None:
        self._ensure_workspace(workspace)
        if (self.ledger.root / "candidates" / f"{workspace.candidate_id}.json").exists():
            raise WorkspaceError("frozen candidates are immutable; create a child candidate")
        if self.ledger.query("evaluation", candidate_id=workspace.candidate_id):
            raise WorkspaceError("evaluated candidates are immutable; create a child candidate")

    def _apply_patch_file(self, workspace: CandidateWorkspace, path: Path) -> None:
        check = self._run(("git", "apply", "--check", "--index", "--whitespace=nowarn", str(path)), cwd=workspace.worktree)
        if check["returncode"] != 0:
            raise WorkspaceError("patch does not apply cleanly: " + check["stderr"].strip())
        apply = self._run(("git", "apply", "--index", "--whitespace=nowarn", str(path)), cwd=workspace.worktree)
        if apply["returncode"] != 0:
            raise WorkspaceError("patch application failed: " + apply["stderr"].strip())
        self.changed_paths(workspace)

    def _validate_patch_headers(self, patch: str) -> None:
        paths: list[str] = []
        for line in patch.splitlines():
            if line.startswith("+++ b/") or line.startswith("--- a/"):
                paths.append(line[6:])
            elif line.startswith("+++ /dev/null") or line.startswith("--- /dev/null"):
                continue
        self.policy.validate_paths(paths)

    def _git(self, *args: str, cwd: Path, check: bool = True) -> str:
        result = subprocess.run(
            ("git", *args), cwd=cwd, text=True, capture_output=True, check=False
        )
        if check and result.returncode != 0:
            raise WorkspaceError(result.stderr.strip() or f"git {' '.join(args)} failed")
        return result.stdout

    @staticmethod
    def _run(args: tuple[str, ...], *, cwd: Path) -> dict[str, Any]:
        result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
        return {
            "argv": list(args),
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        }


def _relative_path(path: str) -> str:
    value = PurePosixPath(path)
    if value.is_absolute() or ".." in value.parts or str(value) in ("", "."):
        raise WorkspaceError(f"unsafe candidate path {path!r}")
    return str(value)


def _matches(path: str, pattern: str) -> bool:
    pattern = pattern.replace("\\", "/")
    if pattern.endswith("/**"):
        root = pattern[:-3].rstrip("/")
        return path == root or path.startswith(root + "/")
    return fnmatch.fnmatchcase(path, pattern)


def _generated_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return "__pycache__" in parts or path.endswith((".pyc", ".pyo"))


def _write_once(path: Path, content: str) -> None:
    """Atomically create a frozen auxiliary artifact, accepting exact replay."""
    encoded = content.encode("utf-8")
    try:
        with open(path, "x", encoding="utf-8") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise WorkspaceError(f"immutable artifact collision: {path}")


__all__ = [
    "CandidateWorkspace", "PatchPolicy", "ValidationReport", "WorkspaceError", "WorkspaceService",
]
