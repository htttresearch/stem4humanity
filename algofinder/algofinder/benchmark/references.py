"""Append-only reference registry and snapshots (spec sections 6.4 and 13 phase 2).

A reference record is immutable once appended. The registry log keeps every
record; ``snapshot`` resolves, per (instance, kind), the currently accepted
record (verified wins over imported_unverified; superseded/disputed are
excluded). Corrections are new records, never edits.

Verification rule (declared in code, spec section 7.1 step 8): a
``certified_optimum`` record is ``verified`` only when produced by the
exact-reference pipeline (held-karp on a small instance) or when an
accompanying tour artifact rescored through the shared objective oracle
equals the recorded value. Published optima without such an artifact are
``imported_unverified`` and must never be reported as proven.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from algofinder.benchmark.digests import reference_id, sha256_hex

SCHEMA_VERSION = "etsp.reference.v1"
REGISTRY_PATH = Path(__file__).resolve().parents[1] / "data" / "references" / "registry.jsonl"
SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "data" / "references" / "references_snapshot.json"

KINDS = frozenset({"certified_optimum", "upper_bound", "lower_bound", "asymptotic_diagnostic"})
STATUSES = frozenset({"verified", "imported_unverified", "superseded", "disputed"})


@dataclass(frozen=True)
class Reference:
    """One immutable reference record (etsp.reference.v1)."""

    instance_id: str
    objective_spec_id: str
    kind: str
    value: float
    provenance: dict[str, object]
    status: str = "imported_unverified"
    tour: tuple[int, ...] | None = None
    proof: dict[str, object] = field(default_factory=dict)
    created_at: str = "2026-08-02"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown reference kind: {self.kind}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown reference status: {self.status}")
        if self.kind == "certified_optimum" and self.tour is None:
            if self.status == "verified":
                raise ValueError(
                    "verified certified_optimum requires a tour artifact"
                )

    @property
    def tour_digest(self) -> str | None:
        if self.tour is None:
            return None
        payload = ",".join(str(city) for city in self.tour)
        return sha256_hex(payload.encode("utf-8"))

    def compute_reference_id(self) -> str:
        return reference_id(
            self.instance_id,
            self.kind,
            self.value,
            self.tour_digest,
            self.provenance,
        )

    def to_mapping(self) -> dict[str, object]:
        mapping = asdict(self)
        mapping["reference_id"] = self.compute_reference_id()
        if self.tour is not None:
            mapping["tour"] = list(self.tour)
        return mapping

    @classmethod
    def from_mapping(cls, mapping: dict[str, object]) -> "Reference":
        tour = mapping.get("tour")
        return cls(
            instance_id=mapping["instance_id"],
            objective_spec_id=mapping["objective_spec_id"],
            kind=mapping["kind"],
            value=float(mapping["value"]),
            provenance=dict(mapping["provenance"]),
            status=mapping["status"],
            tour=tuple(int(city) for city in tour) if tour else None,
            proof=dict(mapping.get("proof") or {}),
            created_at=mapping.get("created_at", "2026-08-02"),
            schema_version=mapping.get("schema_version", SCHEMA_VERSION),
        )


def rescore_reference(
    reference: Reference, points: NDArray[np.float64]
) -> tuple[bool, float]:
    """Rescore the reference's tour through the objective oracle.

    Returns ``(matches, rescored)``. The oracle comes from the objective
    registry, never from solver-reported costs.
    """
    from algofinder.benchmark.scoring import score_tour

    if reference.tour is None:
        return False, float("nan")
    rescored = score_tour(points, list(reference.tour), reference.objective_spec_id)
    matches = abs(rescored - reference.value) <= 1e-9 * max(
        1.0, abs(reference.value)
    )
    return matches, rescored


class ReferenceRegistry:
    """Append-only JSONL log plus resolved snapshot view."""

    def __init__(self, path: str | Path = REGISTRY_PATH) -> None:
        self.path = Path(path)

    def append(self, reference: Reference) -> bool:
        """Append the record; returns False if it already exists (idempotent)."""
        record = reference.to_mapping()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.load_all()
        if any(record["reference_id"] == item["reference_id"] for item in existing):
            return False
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return True

    def append_many(self, references: Sequence[Reference]) -> int:
        added = 0
        for reference in references:
            if self.append(reference):
                added += 1
        return added

    def load_all(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        records: list[dict[str, object]] = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def load_matching(
        self, instance_id: str, kind: str
    ) -> dict[str, object] | None:
        """Highest-status accepted record for (instance_id, kind), or None."""
        match = None
        for record in self.load_all():
            if record.get("instance_id") != instance_id or record.get("kind") != kind:
                continue
            if record["status"] in ("superseded", "disputed"):
                continue
            rank = {"verified": 2, "imported_unverified": 1}.get
            if match is None or rank(record["status"], 0) > rank(match["status"], 0):
                match = record
        return match

    def snapshot(self) -> dict[tuple[str, str], dict[str, object]]:
        """Resolve the currently accepted record per (instance_id, kind)."""
        accepted: dict[tuple[str, str], dict[str, object]] = {}
        for record in self.load_all():
            if record["status"] in ("superseded", "disputed"):
                continue
            key = (record["instance_id"], record["kind"])
            current = accepted.get(key)
            if current is None:
                accepted[key] = record
                continue
            rank = {"verified": 2, "imported_unverified": 1}.get
            if rank(record["status"], 0) > rank(current["status"], 0):
                accepted[key] = record
            elif rank(record["status"], 0) == rank(current["status"], 0):
                if record.get("created_at", "") >= current.get("created_at", ""):
                    accepted[key] = record
        return accepted

    def write_snapshot(self, path: str | Path = SNAPSHOT_PATH) -> dict[tuple[str, str], dict[str, object]]:
        snapshot = self.snapshot()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        records = sorted(
            (dict(record) for record in snapshot.values()),
            key=lambda item: (item["instance_id"], item["kind"]),
        )
        Path(path).write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        return snapshot


__all__ = [
    "SCHEMA_VERSION",
    "KINDS",
    "STATUSES",
    "REGISTRY_PATH",
    "SNAPSHOT_PATH",
    "Reference",
    "ReferenceRegistry",
    "rescore_reference",
]
