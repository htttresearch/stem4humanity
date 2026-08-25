"""Execution boundary for untrusted candidate code.

This module intentionally fails closed.  A plain subprocess is useful for
maintainer-owned local smoke checks, but it is not a security boundary; it is
never permitted for validation or challenge evaluation.  A deployment must
provide an isolated backend (currently Bubblewrap) before it can run untrusted
candidate code against non-public data.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
from typing import Literal, Sequence


class SandboxError(RuntimeError):
    """Raised when a requested candidate execution lacks an enforceable boundary."""


FeedbackZone = Literal["public", "validation", "challenge"]


@dataclass(frozen=True)
class SandboxPolicy:
    zone: FeedbackZone
    network_policy: Literal["deny", "allowlist"] = "deny"
    timeout_seconds: float = 60.0
    memory_bytes: int | None = None
    trusted_local: bool = False

    def __post_init__(self) -> None:
        if self.zone not in ("public", "validation", "challenge"):
            raise SandboxError(f"unknown zone {self.zone!r}")
        if self.timeout_seconds <= 0:
            raise SandboxError("sandbox timeout must be positive")
        if self.memory_bytes is not None and self.memory_bytes <= 0:
            raise SandboxError("sandbox memory limit must be positive")
        if self.zone != "public" and self.trusted_local:
            raise SandboxError("trusted-local execution is only permitted for public suites")


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class SandboxRunner:
    """Run candidate code only when its requested boundary is enforceable."""

    def __init__(self, *, bwrap_path: str | None = None) -> None:
        self.bwrap_path = bwrap_path or shutil.which("bwrap")

    @property
    def isolated_available(self) -> bool:
        return self.bwrap_path is not None

    def validate_workspace(self, worktree: str | Path) -> None:
        """Reject symlinks that could smuggle protected paths into a worktree."""
        root = Path(worktree).resolve()
        for path in root.rglob("*"):
            if not path.is_symlink():
                continue
            target = path.resolve(strict=False)
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise SandboxError(f"candidate workspace contains escaping symlink: {path}") from exc

    def run(
        self,
        command: Sequence[str],
        *,
        worktree: str | Path,
        policy: SandboxPolicy,
        readable_paths: Sequence[str | Path] = (),
        runtime_paths: Sequence[str | Path] = (),
        writable_dir: str | Path | None = None,
    ) -> SandboxResult:
        """Run a command with no ambient secrets; sealed zones require isolation.

        ``readable_paths`` is intentionally explicit.  The authority supplies
        only the suite inputs needed by the evaluator; agents never receive the
        resulting host paths through their tool response.
        """
        if not command:
            raise SandboxError("sandbox command cannot be empty")
        root = Path(worktree).resolve()
        self.validate_workspace(root)
        readable = [Path(path).resolve() for path in readable_paths]
        runtimes = [Path(path).resolve() for path in runtime_paths]
        if policy.zone != "public" and not self.isolated_available:
            raise SandboxError(
                "sealed evaluation requires Bubblewrap (bwrap); refusing unsafe host execution"
            )
        if self.isolated_available:
            return self._run_bwrap(command, root, policy, readable, runtimes, writable_dir)
        if not policy.trusted_local:
            raise SandboxError("no isolated backend available; enable trusted_local only for public maintainer smoke checks")
        return self._run_local(command, root, policy)

    def _run_local(self, command: Sequence[str], root: Path, policy: SandboxPolicy) -> SandboxResult:
        environment = {"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1", "HOME": str(root / ".sandbox-home")}
        try:
            result = subprocess.run(
                list(command), cwd=root, env=environment, text=True,
                capture_output=True, timeout=policy.timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(124, exc.stdout or "", exc.stderr or "", timed_out=True)
        return SandboxResult(result.returncode, result.stdout[-8000:], result.stderr[-8000:])

    def _run_bwrap(
        self,
        command: Sequence[str],
        root: Path,
        policy: SandboxPolicy,
        readable: Sequence[Path],
        runtimes: Sequence[Path],
        writable_dir: str | Path | None,
    ) -> SandboxResult:
        """Minimal Bubblewrap profile: candidate source is read-only and networkless."""
        assert self.bwrap_path is not None
        args = [self.bwrap_path, "--die-with-parent", "--new-session", "--unshare-net", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
        for directory in ("/usr", "/usr/local", "/bin", "/lib", "/lib64", "/etc"):
            path = Path(directory)
            if path.exists():
                args.extend(["--ro-bind", directory, directory])
        args.extend(["--ro-bind", str(root), "/workspace", "--chdir", "/workspace"])
        if readable:
            args.extend(["--dir", "/inputs"])
        for index, path in enumerate(readable):
            args.extend(["--ro-bind", str(path), f"/inputs/{index}"])
        if runtimes:
            args.extend(["--dir", "/runtime"])
        for index, path in enumerate(runtimes):
            args.extend(["--ro-bind", str(path), f"/runtime/{index}"])
        if writable_dir is not None:
            output = Path(writable_dir).resolve()
            output.mkdir(parents=True, exist_ok=True)
            args.extend(["--bind", str(output), "/output"])
        args.extend(["--setenv", "HOME", "/tmp", "--setenv", "PYTHONNOUSERSITE", "1", "--"])
        args.extend(command)
        try:
            result = subprocess.run(args, text=True, capture_output=True, timeout=policy.timeout_seconds, check=False)
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(124, exc.stdout or "", exc.stderr or "", timed_out=True)
        return SandboxResult(result.returncode, result.stdout[-8000:], result.stderr[-8000:])


__all__ = ["SandboxError", "SandboxPolicy", "SandboxResult", "SandboxRunner"]
