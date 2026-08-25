"""Append-only registry: an inventory of what exists (tier 2).

One JSONL file per kind (``instances``, ``solvers``, ``runs``,
``environments``, ``features``). The indexer regenerates each file deterministically —
entries are written sorted by id, so a rerun over unchanged canonical
files produces byte-identical logs. The registry is *derived*: it indexes
the canonical store and is always rebuildable from it, but its digests
let a reader verify that a registry entry and the file it points at have
not drifted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from algofinder.trace.serialize import strict_dumps

SCHEMA_VERSION = "algofinder.registry.v1"

CORE_KINDS = ("instances", "solvers", "runs", "environments", "features")
CAMPAIGN_KINDS = (
    "campaigns", "agents", "hypotheses", "candidates", "experiments",
    "evaluations", "analyses", "decisions", "episodes",
)
KINDS = CORE_KINDS + CAMPAIGN_KINDS


class RegistryError(ValueError):
    """Raised on malformed or tampered registry content."""


def _path(root: Path, kind: str) -> Path:
    if kind not in KINDS:
        raise RegistryError(f"unknown registry kind {kind!r}")
    return Path(root) / f"{kind}.jsonl"


def write_kind(root: str | Path, kind: str, entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Regenerate one registry file, sorted by entry id (idempotent).

    Returns the sorted entry list. The output is deterministic: equal
    inputs produce byte-identical files.
    """
    path = _path(root, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        (dict(entry) for entry in entries),
        key=lambda entry: entry["id"],
    )
    lines = [strict_dumps(entry) + "\n" for entry in ordered]
    path.write_text("".join(lines), encoding="utf-8")
    return ordered


def read_kind(root: str | Path, kind: str) -> list[dict[str, Any]]:
    """Read all entries of one registry kind, in file order."""
    path = _path(root, kind)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RegistryError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(entry, dict) or "id" not in entry:
                raise RegistryError(f"{path}:{line_no}: entry missing 'id'")
            entries.append(entry)
    return entries


class Registry:
    """Read/write access to the registry directory (tier 2)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(self, kind: str, entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return write_kind(self.root, kind, entries)

    def read(self, kind: str) -> list[dict[str, Any]]:
        return read_kind(self.root, kind)

    def iter_entries(self, kind: str) -> Iterator[dict[str, Any]]:
        yield from self.read(kind)

    def verify(self, kind: str, check: Any) -> list[str]:
        """Cross-check every entry of a kind against a checker callback.

        ``check(entry)`` returns None when the entry is consistent and a
        human-readable problem string otherwise.
        """
        problems: list[str] = []
        for entry in self.read(kind):
            problem = check(entry)
            if problem is not None:
                problems.append(f"{kind}/{entry['id']}: {problem}")
        return problems


__all__ = [
    "CAMPAIGN_KINDS",
    "CORE_KINDS",
    "KINDS",
    "Registry",
    "RegistryError",
    "SCHEMA_VERSION",
    "read_kind",
    "write_kind",
]
