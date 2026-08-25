"""Immutable filesystem store for learning artifacts outside campaign ledgers."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Protocol

from algofinder.agents.learning.contracts import SCHEMA_ID, SCHEMA_VERSION, learning_digest
from algofinder.trace.serialize import canonical_dumps, strict_dumps, to_json_value


class LearningStoreError(RuntimeError):
    """Raised for a tampered or conflicting learning artifact."""


class HasMapping(Protocol):
    def to_mapping(self) -> dict[str, Any]: ...


_ROUTES = {
    "distribution_profile": "distributions/{id}.json",
    "learning_run": "runs/{id}.json",
    "agent_evaluation": "evaluations/{id}.json",
    "agent_policy": "policies/{id}/policy.json",
}


class LearningArtifactStore:
    """Content-addressed blobs plus immutable global learning records.

    This store is intentionally separate from campaign evidence: backfills and
    derived datasets can be added without modifying a completed campaign.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def create(self) -> "LearningArtifactStore":
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in ("distributions", "datasets", "runs", "memories", "policies", "evaluations", "backfills", "blobs/sha256"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        return self

    def write_backfill(self, backfill_id: str, mapping: dict[str, Any]) -> str:
        if not backfill_id or "/" in backfill_id or ".." in backfill_id:
            raise LearningStoreError("unsafe backfill id")
        payload = {
            "kind": "experience_backfill",
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "id": backfill_id,
            **to_json_value(mapping),
        }
        return self._write_immutable(self.root / "backfills" / f"{backfill_id}.json", payload)

    def write(self, record: HasMapping) -> str:
        mapping = record.to_mapping()
        kind = mapping.get("kind")
        record_id = mapping.get("id")
        if kind not in _ROUTES or not isinstance(record_id, str):
            raise LearningStoreError(f"unsupported learning record kind {kind!r}")
        return self._write_immutable(self.root / _ROUTES[kind].format(id=record_id), mapping)

    def write_dataset_manifest(self, dataset_id: str, manifest: dict[str, Any]) -> str:
        if not dataset_id or "/" in dataset_id or ".." in dataset_id:
            raise LearningStoreError("unsafe dataset id")
        payload = {
            "kind": "learning_dataset",
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "id": dataset_id,
            **to_json_value(manifest),
        }
        return self._write_immutable(self.root / "datasets" / dataset_id / "manifest.json", payload)

    def write_memory(self, memory_id: str, mapping: dict[str, Any]) -> str:
        if not memory_id or "/" in memory_id or ".." in memory_id:
            raise LearningStoreError("unsafe memory id")
        payload = {
            "kind": "agent_memory",
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "id": memory_id,
            **to_json_value(mapping),
        }
        return self._write_immutable(self.root / "memories" / f"{memory_id}.json", payload)

    def put_blob(self, value: str | bytes) -> str:
        self.create()
        data = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        digest = sha256(data).hexdigest()
        path = self.root / "blobs" / "sha256" / digest
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if path.read_bytes() != data:
                raise LearningStoreError("blob digest collision")
            return f"sha256:{digest}"
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        return f"sha256:{digest}"

    def put_json_blob(self, value: Any) -> str:
        return self.put_blob(canonical_dumps(to_json_value(value)))

    def read_blob(self, digest: str) -> bytes:
        """Read and verify a content-addressed blob."""
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise LearningStoreError("blob digest must use sha256:<hex>")
        value = digest.removeprefix("sha256:")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise LearningStoreError("invalid blob digest")
        try:
            data = (self.root / "blobs" / "sha256" / value).read_bytes()
        except OSError as exc:
            raise LearningStoreError(f"missing learning blob: {digest}") from exc
        if sha256(data).hexdigest() != value:
            raise LearningStoreError(f"learning blob digest mismatch: {digest}")
        return data

    def read_json_blob(self, digest: str) -> Any:
        try:
            return json.loads(self.read_blob(digest).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LearningStoreError(f"learning blob is not valid JSON: {digest}") from exc

    def read(self, relative_path: str | Path) -> dict[str, Any]:
        path = (self.root / relative_path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise LearningStoreError("artifact path escapes store root") from exc
        try:
            mapping = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LearningStoreError(f"cannot read learning artifact {path}") from exc
        if not isinstance(mapping, dict):
            raise LearningStoreError("learning artifact is not an object")
        recorded = mapping.pop("content_digest", None)
        actual = learning_digest(mapping)
        if recorded != actual:
            raise LearningStoreError(f"learning artifact digest mismatch: {path}")
        mapping["content_digest"] = actual
        return mapping

    def _write_immutable(self, path: Path, mapping: dict[str, Any]) -> str:
        self.create()
        digest = learning_digest(mapping)
        payload = {**mapping, "content_digest": digest}
        encoded = (strict_dumps(payload) + "\n").encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            current = self.read(path.relative_to(self.root))
            if current["content_digest"] != digest:
                raise LearningStoreError(f"immutable artifact id collision: {path.name}")
            return digest
        try:
            os.write(fd, encoded)
            os.fsync(fd)
        finally:
            os.close(fd)
        return digest


__all__ = ["LearningArtifactStore", "LearningStoreError"]
