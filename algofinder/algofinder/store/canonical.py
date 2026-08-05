"""Canonical store: where the immutable system of record lives.

Tier 1 of the storage design. These functions locate and digest the
canonical artifacts — instance manifests, benchmark run records, reference
registries — that are the single source of truth. Nothing in this module
writes to them; it only points at them and hashes them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from algofinder.trace.serialize import sha256_hex

SCHEMA_VERSION = "algofinder.store.v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: str | Path) -> str:
    """Digest of the raw bytes of a canonical file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_instances_manifests(instances_dir: str | Path) -> list[Path]:
    """Canonical instance manifests under a directory (``*.json``)."""
    return sorted(Path(instances_dir).glob("*.json"))


def find_results_files(results_dir: str | Path, glob_pattern: str = "*.json") -> list[Path]:
    """Canonical benchmark run-record files under a directory."""
    return sorted(Path(results_dir).glob(glob_pattern))


def references_paths() -> tuple[Path, Path]:
    """Canonical reference registry (append-only log) and snapshot."""
    root = PROJECT_ROOT / "data" / "references"
    return root / "registry.jsonl", root / "references_snapshot.json"


def run_digest(mapping: dict[str, Any]) -> str:
    """Content-addressed id of one run record (its canonical JSON)."""
    return sha256_hex(mapping)


def sessions_dir() -> Path:
    """Canonical dev-mode trace sessions."""
    return PROJECT_ROOT.parent / "private" / "sessions"


__all__ = [
    "PROJECT_ROOT",
    "SCHEMA_VERSION",
    "find_instances_manifests",
    "find_results_files",
    "references_paths",
    "run_digest",
    "sessions_dir",
    "sha256_file",
]
