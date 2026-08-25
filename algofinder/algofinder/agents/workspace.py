"""Disposable candidate worktrees and a strict source-change allowlist.

Candidate code never edits the campaign base.  A worktree starts at the fixed
campaign commit, optionally receives immutable parent patches, and is frozen
into a normalized patch before it can enter the ledger.  This is isolation for
source state; execution isolation belongs to :mod:`algofinder.agents.sandbox`.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
import fnmatch
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Iterable

from algofinder.agents.contracts import Candidate, ContractError, new_id, utc_now
from algofinder.agents.candidate_templates import (
    TemplateError,
    apply_values_to_template,
    describe_rendered_template,
    get_template,
    render_template,
    template_relative_path,
    validate_rendered_template,
    validate_values,
)
from algofinder.agents.ledger import CampaignLedger, LedgerError
from algofinder.trace.serialize import canonical_dumps, strict_dumps


class WorkspaceError(RuntimeError):
    """Raised when a candidate workspace or patch violates its contract."""


@dataclass(frozen=True)
class PatchPolicy:
    """Git-relative paths a candidate is permitted to change."""

    allowed_paths: tuple[str, ...]
    forbidden_imports: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.allowed_paths:
            raise WorkspaceError("candidate patch allowlist cannot be empty")
        for pattern in self.allowed_paths:
            if not pattern or pattern.startswith("/") or ".." in PurePosixPath(pattern).parts:
                raise WorkspaceError(f"unsafe patch allowlist pattern {pattern!r}")
        if any(not item or item.startswith(".") for item in self.forbidden_imports):
            raise WorkspaceError("forbidden imports must be absolute module names")

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

    def validate_imports(self, path: str, source: str) -> None:
        """Reject explicit delegation to campaign-excluded solver modules."""
        if not self.forbidden_imports:
            return
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            return  # The ordinary compile gate reports syntax errors.
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)
        violations = sorted(
            name for name in imported
            if any(name == blocked or name.startswith(blocked + ".") for blocked in self.forbidden_imports)
        )
        if violations:
            raise WorkspaceError(
                "candidate imports campaign-excluded solver code: " + ", ".join(violations)
            )


@dataclass(frozen=True)
class CandidateWorkspace:
    candidate_id: str
    worktree: Path
    base_commit: str
    parent_ids: tuple[str, ...]
    hypothesis_id: str
    generation_operator: str
    template_id: str | None = None

    @property
    def metadata_path(self) -> Path:
        # Keep service metadata outside the Git worktree so it can never be
        # mistaken for an agent-authored candidate file.
        return self.worktree.parent / f"{self.candidate_id}.workspace.json"

    @property
    def template_entrypoints(self) -> tuple[str, ...]:
        return () if self.template_id is None else get_template(self.template_id).entrypoints


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
        template_id: str | None = None,
    ) -> CandidateWorkspace:
        """Create a worktree at the campaign base and replay parent patches."""
        if generation_operator not in {"invent", "mutate", "recombine", "repair", "tune", "distill"}:
            raise WorkspaceError(f"unsupported generation operator {generation_operator!r}")
        candidate_id = candidate_id or new_id("candidate")
        parents = tuple(parent_ids)
        if template_id is not None:
            try:
                get_template(template_id)
            except TemplateError as exc:
                raise WorkspaceError(str(exc)) from exc
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
            template_id=template_id,
        )
        workspace.metadata_path.write_text(
            strict_dumps({
                "schema_id": "algofinder.agents.workspace",
                "schema_version": "1",
                "candidate_id": candidate_id,
                "base_commit": base_commit,
                "parent_ids": list(parents),
                "hypothesis_id": hypothesis_id,
                "generation_operator": generation_operator,
                "template_id": template_id,
                "created_at": utc_now(),
            }) + "\n",
            encoding="utf-8",
        )
        try:
            # A typed-template recombination applies one edit to the first
            # parent's rendered state. Additional parents are evidence donors;
            # replaying their complete patches would overwrite/conflict on the
            # same authority-owned template file.
            replay_parents = parents[:1] if template_id is not None else parents
            for parent_id in replay_parents:
                patch = self.ledger.root / "candidates" / parent_id / "change.patch"
                if not patch.is_file():
                    raise WorkspaceError(f"parent candidate has no frozen patch: {parent_id}")
                self._apply_patch_file(workspace, patch)
            if template_id is not None:
                self._seed_template(workspace)
        except Exception:
            workspace.metadata_path.unlink(missing_ok=True)
            self._git("worktree", "remove", "--force", str(target), cwd=self.source_root, check=False)
            raise
        return workspace

    def load(self, candidate_id: str) -> CandidateWorkspace:
        """Open an existing immutable candidate workspace for authority-only replay."""
        target = self.ledger.root / "workspaces" / candidate_id
        if not target.is_dir():
            raise WorkspaceError(f"candidate workspace does not exist: {candidate_id}")
        return self._load_workspace(target)

    def discard(self, workspace: CandidateWorkspace) -> None:
        """Remove a mutable workspace rejected before candidate freezing."""
        self._ensure_workspace(workspace)
        try:
            self.ledger.read("candidate", workspace.candidate_id)
        except LedgerError:
            pass
        else:
            raise WorkspaceError("cannot discard a frozen candidate workspace")
        self._git(
            "worktree", "remove", "--force", str(workspace.worktree),
            cwd=self.source_root,
        )
        workspace.metadata_path.unlink(missing_ok=True)

    def read_base_sources(
        self,
        paths: Iterable[str],
        *,
        max_total_bytes: int = 80_000,
    ) -> list[dict[str, str]]:
        """Read a bounded solver-source snapshot from the immutable base commit.

        This is a read-only research aid, not a general filesystem tool. Paths
        must be Git-relative Python files below this project's solver package;
        their bytes come from the campaign's pinned base commit, never from a
        mutable candidate worktree.
        """
        if max_total_bytes < 1:
            raise WorkspaceError("max_total_bytes must be positive")
        base_commit = str(self.ledger.campaign()["base_commit"])
        solver_root = str(self.package_rel / "algofinder" / "solvers").replace("\\", "/")
        result: list[dict[str, str]] = []
        used = 0
        for raw_path in paths:
            path = _relative_path(raw_path)
            if not path.startswith(solver_root + "/") or not path.endswith(".py"):
                raise WorkspaceError(f"source read is restricted to Python solver files: {path!r}")
            content = self._git("show", f"{base_commit}:{path}", cwd=self.source_root)
            size = len(content.encode("utf-8"))
            if used + size > max_total_bytes:
                raise WorkspaceError(
                    f"source snapshot exceeds {max_total_bytes} bytes at {path!r}"
                )
            result.append({"path": path, "content": content})
            used += size
        return result

    def apply_patch(self, workspace: CandidateWorkspace, patch: str) -> None:
        """Apply a proposed unified diff after checking its touched paths."""
        self._ensure_mutable(workspace)
        if workspace.template_id is not None:
            raise WorkspaceError("templated candidates accept constrained template values, not patches")
        incoming = workspace.worktree.parent / f".{workspace.candidate_id}.incoming.patch"
        incoming.write_text(patch, encoding="utf-8")
        try:
            self._validate_patch_headers(patch)
            self._apply_patch_file(workspace, incoming)
        finally:
            incoming.unlink(missing_ok=True)

    def apply_template_values(
        self,
        workspace: CandidateWorkspace,
        *,
        values: dict[str, str],
    ) -> None:
        """Fill a trusted scaffold's narrow model-owned gaps."""
        self._ensure_mutable(workspace)
        if workspace.template_id is None:
            raise WorkspaceError("candidate workspace was not created from a template")
        try:
            path = workspace.worktree / template_relative_path(workspace.template_id)
            if not path.is_file():
                raise WorkspaceError("candidate template source is missing")
            path.write_text(
                apply_values_to_template(
                    workspace.template_id,
                    path.read_text(encoding="utf-8"),
                    values,
                ),
                encoding="utf-8",
            )
        except TemplateError as exc:
            raise WorkspaceError(str(exc)) from exc

    def describe_template(self, workspace: CandidateWorkspace) -> dict[str, Any]:
        """Describe the exact rendered template currently in a workspace."""
        self._ensure_workspace(workspace)
        if workspace.template_id is None:
            return {}
        try:
            path = workspace.worktree / template_relative_path(workspace.template_id)
            return describe_rendered_template(
                workspace.template_id,
                path.read_text(encoding="utf-8"),
            )
        except (OSError, TemplateError) as exc:
            raise WorkspaceError(f"candidate template cannot be described: {exc}") from exc

    def validate(
        self,
        workspace: CandidateWorkspace,
        *,
        solver_entrypoints: Iterable[str] = (),
    ) -> ValidationReport:
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
        for path in paths:
            if not path.endswith(".py"):
                continue
            try:
                self.policy.validate_imports(
                    path,
                    (workspace.worktree / path).read_text(encoding="utf-8"),
                )
            except WorkspaceError as exc:
                errors.append(str(exc))
        try:
            self._validate_candidate_entrypoints(
                workspace, paths, tuple(solver_entrypoints)
            )
        except WorkspaceError as exc:
            errors.append(str(exc))
        if workspace.template_id is not None:
            try:
                template_path = workspace.worktree / template_relative_path(workspace.template_id)
                validate_rendered_template(
                    workspace.template_id, template_path.read_text(encoding="utf-8")
                )
            except (OSError, TemplateError) as exc:
                errors.append(f"candidate template validation failed: {exc}")
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
        producing_episode_id: str | None = None,
    ) -> Candidate:
        """Validate and write a final, immutable candidate record plus patch."""
        self._ensure_mutable(workspace)
        entrypoints = tuple(solver_entrypoints)
        report = self.validate(workspace, solver_entrypoints=entrypoints)
        if not report.passed or report.patch_digest is None or report.build_digest is None:
            raise WorkspaceError("candidate cannot be frozen: " + "; ".join(report.errors))
        candidate = Candidate(
            candidate_id=workspace.candidate_id,
            campaign_id=self.ledger.campaign_id,
            hypothesis_id=workspace.hypothesis_id,
            parent_ids=workspace.parent_ids,
            generation_operator=workspace.generation_operator,  # type: ignore[arg-type]
            solver_entrypoints=entrypoints,
            producing_agent_run_id=producing_agent_run_id,
            producing_episode_id=producing_episode_id,
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
        metadata_path = target.parent / f"{target.name}.workspace.json"
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
            template_id=metadata.get("template_id"),
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

    def _seed_template(self, workspace: CandidateWorkspace) -> None:
        """Add a framework-owned scaffold after parent replay, if necessary."""
        assert workspace.template_id is not None
        source_path = workspace.worktree / template_relative_path(workspace.template_id)
        if source_path.exists():
            try:
                validate_rendered_template(
                    workspace.template_id, source_path.read_text(encoding="utf-8")
                )
            except (OSError, TemplateError) as exc:
                raise WorkspaceError(f"parent candidate is not a valid template instance: {exc}") from exc
            return
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(render_template(workspace.template_id), encoding="utf-8")

    def _validate_patch_headers(self, patch: str) -> None:
        paths: list[str] = []
        for line in patch.splitlines():
            if line.startswith("+++ b/") or line.startswith("--- a/"):
                paths.append(line[6:])
            elif line.startswith("+++ /dev/null") or line.startswith("--- /dev/null"):
                continue
        self.policy.validate_paths(paths)

    def _validate_candidate_entrypoints(
        self,
        workspace: CandidateWorkspace,
        changed_paths: tuple[str, ...],
        entrypoints: tuple[str, ...],
    ) -> None:
        """Prove entrypoints are new, changed candidate solvers before evaluation.

        This is deliberately a Gate-0 static check.  The declared class must
        live in a Python file changed by this candidate, directly subclass the
        public ``Solver`` contract, carry a literal non-colliding id, and avoid
        importing any already-registered solver implementation.  The evaluator
        repeats the identity/collision check after import in its own sandbox.
        """
        if not entrypoints:
            return
        changed = set(changed_paths)
        registered_modules, registered_ids = _registered_solver_identity()
        seen_ids: set[str] = set()
        seen_entrypoints: set[str] = set()
        for entrypoint in entrypoints:
            if entrypoint in seen_entrypoints:
                raise WorkspaceError(f"duplicate candidate entrypoint: {entrypoint}")
            seen_entrypoints.add(entrypoint)
            module_name, separator, class_name = entrypoint.partition(":")
            if (
                not separator
                or not module_name
                or not class_name
                or ":" in class_name
                or any(char in module_name + class_name for char in "*?[]")
            ):
                raise WorkspaceError(
                    f"candidate entrypoint must use a concrete module:Class name: {entrypoint!r}"
                )
            module_path = self._entrypoint_module_path(module_name)
            if module_path not in changed:
                raise WorkspaceError(
                    f"candidate entrypoint must belong to a changed candidate module: {entrypoint}"
                )
            source_path = workspace.worktree / module_path
            try:
                source = source_path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=module_path)
            except (OSError, SyntaxError) as exc:
                raise WorkspaceError(
                    f"candidate entrypoint module cannot be inspected: {module_path}: {exc}"
                ) from exc
            self._reject_solver_delegation(module_path, tree, registered_modules)
            candidate_class = next(
                (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name),
                None,
            )
            if candidate_class is None:
                raise WorkspaceError(
                    f"candidate entrypoint class is not defined in its changed module: {entrypoint}"
                )
            if not _directly_subclasses_solver(tree, candidate_class):
                raise WorkspaceError(
                    f"candidate entrypoint must be a new direct Solver subclass: {entrypoint}"
                )
            solver_id = _literal_solver_id(candidate_class)
            if not solver_id:
                raise WorkspaceError(
                    f"candidate Solver class must declare a non-empty literal id: {entrypoint}"
                )
            if solver_id in registered_ids or solver_id in seen_ids:
                raise WorkspaceError(
                    f"candidate Solver id collides with an existing or proposed solver: {solver_id!r}"
                )
            seen_ids.add(solver_id)

    def _entrypoint_module_path(self, module_name: str) -> str:
        module_parts = tuple(module_name.split("."))
        if (
            len(module_parts) < 2
            or any(not part.isidentifier() for part in module_parts)
            or module_parts[0] != "algofinder"
        ):
            raise WorkspaceError(f"candidate entrypoint module is outside the project package: {module_name!r}")
        return str(PurePosixPath(*self.package_rel.parts, *module_parts).with_suffix(".py"))

    @staticmethod
    def _reject_solver_delegation(
        path: str,
        tree: ast.AST,
        registered_modules: set[str],
    ) -> None:
        violations: set[str] = set()
        allowed_base_names = {
            "Solver", "SolverResult", "SolverCapabilities", "InapplicableError", "UnsupportedError",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in registered_modules or any(
                        alias.name.startswith(module + ".") for module in registered_modules
                    ):
                        violations.add(alias.name)
                    if alias.name == "importlib":
                        violations.add("importlib")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "algofinder.solvers.base":
                    forbidden = [alias.name for alias in node.names if alias.name not in allowed_base_names]
                    violations.update(f"{node.module}.{name}" for name in forbidden)
                elif node.module in registered_modules or any(
                    node.module.startswith(module + ".") for module in registered_modules
                ):
                    violations.add(node.module)
                elif node.module == "importlib":
                    violations.add(node.module)
            elif isinstance(node, ast.Call):
                function = node.func
                if isinstance(function, ast.Name) and function.id == "__import__":
                    violations.add("__import__")
                elif isinstance(function, ast.Attribute) and function.attr == "import_module":
                    violations.add("import_module")
        if violations:
            raise WorkspaceError(
                "candidate may not invoke, wrap, or delegate to registered solver code: "
                + ", ".join(sorted(violations))
            )

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
    if (
        value.is_absolute()
        or ".." in value.parts
        or str(value) in ("", ".")
        or any(character in str(value) for character in "*?[]")
    ):
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


def _registered_solver_identity() -> tuple[set[str], set[str]]:
    """Return authority-known solver modules and ids without candidate imports."""
    import algofinder.solvers.registry  # noqa: F401
    from algofinder.solvers.base import all_solvers

    solvers = all_solvers()
    return ({cls.__module__ for cls in solvers.values()}, set(solvers))


def _directly_subclasses_solver(tree: ast.AST, candidate: ast.ClassDef) -> bool:
    """Recognize direct ``Solver`` bases, including an explicit alias import."""
    solver_names = {"Solver"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module != "algofinder.solvers.base":
            continue
        for alias in node.names:
            if alias.name == "Solver":
                solver_names.add(alias.asname or alias.name)
    for base in candidate.bases:
        if isinstance(base, ast.Name) and base.id in solver_names:
            return True
        if (
            isinstance(base, ast.Attribute)
            and base.attr == "Solver"
            and isinstance(base.value, ast.Name)
            and base.value.id in {"base", "solvers_base"}
        ):
            return True
    return False


def _literal_solver_id(candidate: ast.ClassDef) -> str | None:
    for statement in candidate.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "id" for target in statement.targets):
            continue
        if isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            return statement.value.value.strip() or None
        return None
    return None


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
