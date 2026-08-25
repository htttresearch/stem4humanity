"""Versioned solver manifests consumed by campaign infrastructure.

The manifest complements the existing :class:`Solver` interface; it does not
replace it.  Candidate workspaces can declare their entrypoint and searchable
constructor parameters without granting agents a second evaluation interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from algofinder.trace.serialize import to_json_value

SCHEMA_ID = "algofinder.solver-manifest"
SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class SolverManifest:
    solver_id: str
    entrypoint: str
    applies_to: tuple[str, ...]
    tags: tuple[str, ...]
    constructor_config: dict[str, Any] = field(default_factory=dict)
    parameter_space: dict[str, Any] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    manifest_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.solver_id or not self.entrypoint or ":" not in self.entrypoint:
            raise ValueError("solver manifests require a solver id and module:Class entrypoint")
        if len(set(self.applies_to)) != len(self.applies_to):
            raise ValueError("solver manifest applies_to contains duplicates")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("solver manifest dependencies contains duplicates")
        to_json_value(self.constructor_config)
        to_json_value(self.parameter_space)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "manifest_version": self.manifest_version,
            "solver_id": self.solver_id,
            "entrypoint": self.entrypoint,
            "applies_to": list(self.applies_to),
            "tags": list(self.tags),
            "constructor_config": to_json_value(self.constructor_config),
            "parameter_space": to_json_value(self.parameter_space),
            "dependencies": list(self.dependencies),
        }


__all__ = ["SCHEMA_ID", "SCHEMA_VERSION", "SolverManifest"]
