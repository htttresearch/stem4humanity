"""Content-addressed artifact store.

Large immutable values (instance snapshots, problem states, solver state
checkpoints, dense arrays) are stored once by content hash and referenced
from events. Publishing is idempotent and race-safe: write to a
run-unique temporary name, then atomically claim the final hash name;
if another worker already published it, reuse it.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from algofinder.trace.serialize import sha256_hex, strict_dumps


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to a stored artifact, with self-verifying metadata."""

    media_type: str
    bytes: int
    sha256: str
    dtype: str | None = None
    shape: list[int] | None = None
    path_rel: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {
            "media_type": self.media_type,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "dtype": self.dtype,
            "shape": self.shape,
            "path_rel": self.path_rel,
        }


class ArtifactStore:
    """Content-addressed JSON/NumPy artifacts under a session directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.instances_dir = root / "instances"
        self.states_dir = root / "problem-states"
        self.artifacts_dir = root / "artifacts"
        for directory in (self.instances_dir, self.states_dir, self.artifacts_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def publish_json(
        self, section: str, value: Any, *, media_type: str = "application/json"
    ) -> ArtifactRef:
        """Store one JSON value under its content hash; return its reference."""
        if section not in ("instances", "problem-states", "artifacts"):
            raise ValueError(f"unknown artifact section: {section}")
        directory = {
            "instances": self.instances_dir,
            "problem-states": self.states_dir,
            "artifacts": self.artifacts_dir,
        }[section]
        payload = strict_dumps(value).encode("utf-8")
        digest = sha256_hex(value)
        return self._claim(directory, digest, payload, media_type, "json")

    def publish_npy(self, array: np.ndarray) -> ArtifactRef:
        """Store a dense numeric array as ``.npy`` (no pickling)."""
        import io

        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        payload = buffer.getvalue()
        digest = sha256_hex(payload)
        ref = self._claim(
            self.artifacts_dir, digest, payload, "application/octet-stream", "npy"
        )
        return ArtifactRef(
            media_type="application/x-npy",
            bytes=ref.bytes,
            sha256=ref.sha256,
            dtype=str(array.dtype),
            shape=list(array.shape),
            path_rel=ref.path_rel,
        )

    def load_json(self, ref: ArtifactRef) -> Any:
        path = self._resolve(ref)
        return json.loads(path.read_text(encoding="utf-8"))

    def _claim(
        self, directory: Path, digest: str, payload: bytes, media_type: str, suffix: str
    ) -> ArtifactRef:
        target = directory / f"{digest}.{suffix}"
        if not target.exists():
            tmp = directory / f".{digest}.{suffix}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
            tmp.write_bytes(payload)
            os.replace(tmp, target)
        return ArtifactRef(
            media_type=media_type,
            bytes=len(payload),
            sha256=digest,
            path_rel=str(target.relative_to(self.root)),
        )

    def _resolve(self, ref: ArtifactRef) -> Path:
        if ref.path_rel:
            return self.root / ref.path_rel
        raise ValueError("artifact reference has no path")


__all__ = ["ArtifactStore", "ArtifactRef"]
