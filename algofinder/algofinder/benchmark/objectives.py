"""Objective specification registry (benchmark spec section 5).

An ``objective_spec`` names one precise scoring semantics for a tour:
the edge function, its accumulation, and the result type. Solvers produce
tours; the objective registry scores them, so solver-reported costs are never
trusted as the leaderboard oracle.

Every spec has a semantic version and conformance examples. Reference values
in manifests name the spec they were computed under; the same coordinates
under a different spec are a different objective sibling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


def _rounded_edge(delta: NDArray[np.float64]) -> NDArray[np.float64]:
    """``nint`` per TSPLIB: round to the nearest integer, halves up."""
    return np.floor(np.sqrt(np.einsum("ij,ij->i", delta, delta)) + 0.5)


@dataclass(frozen=True)
class ObjectiveSpec:
    """One versioned objective semantics."""

    spec_id: str
    family: str
    dimension: int
    result_type: str
    version: int
    edge_rounding: str
    conformance: tuple[tuple[tuple[float, ...], tuple[float, ...], float], ...]
    points_ndim: int = 2

    def full_id(self) -> str:
        return f"{self.spec_id}@{self.version}"

    def edge_length(self, delta: NDArray[np.float64]) -> float:
        """Length of one edge (a displacement vector) under this spec."""
        length = float(np.sqrt(np.einsum("i,i->", delta, delta)))
        if self.edge_rounding == "nint":
            return float(np.floor(length + 0.5))
        if self.edge_rounding == "ceil":
            return float(np.ceil(length))
        return length

    def edge_lengths(
        self, points: NDArray[np.float64], tour: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """Lengths of the closed-tour edges (n edges incl. the closure edge)."""
        delta = points[tour] - points[np.roll(tour, -1)]
        return np.asarray([self.edge_length(row) for row in delta], dtype=float)

    def tour_objective(
        self, points: NDArray[np.float64], tour: NDArray[np.int64]
    ) -> float:
        """Objective of a closed tour under this spec."""
        return float(self.edge_lengths(points, tour).sum())


_SPECS: dict[str, ObjectiveSpec] = {}


def register_spec(spec: ObjectiveSpec) -> ObjectiveSpec:
    full_id = spec.full_id()
    if full_id in _SPECS:
        raise ValueError(f"duplicate objective spec: {full_id}")
    _SPECS[full_id] = spec
    return spec


def get_objective(spec_id: str) -> ObjectiveSpec:
    """Look up an objective spec by ``name@version`` (default latest version)."""
    if "@" not in spec_id:
        candidates = [
            spec for full, spec in _SPECS.items() if full.startswith(spec_id + "@")
        ]
        if not candidates:
            raise KeyError(f"unknown objective spec: {spec_id}")
        return max(candidates, key=lambda spec: spec.version)
    try:
        return _SPECS[spec_id]
    except KeyError:
        raise KeyError(f"unknown objective spec: {spec_id}") from None


def all_objectives() -> dict[str, ObjectiveSpec]:
    return dict(_SPECS)


# --- first-release specs (benchmark spec section 5, v0.1 scope) ---

register_spec(
    ObjectiveSpec(
        spec_id="raw_l2_f64",
        family="euclidean",
        dimension=0,
        result_type="float64",
        version=1,
        edge_rounding="none",
        conformance=(
            ((0.0, 0.0), (3.0, 4.0), 5.0),
            ((0.0, 0.0), (1.0, 1.0), 2 ** 0.5),
        ),
    )
)

register_spec(
    ObjectiveSpec(
        spec_id="tsplib_euc_2d",
        family="rounded_euclidean",
        dimension=2,
        result_type="integer",
        version=1,
        edge_rounding="nint",
        conformance=(
            ((0, 0), (3, 4), 5.0),
            ((0, 0), (1, 1), 1.0),
            ((0, 0), (1, 2), 2.0),
            ((0, 0), (3, 3), 4.0),
        ),
    )
)

register_spec(
    ObjectiveSpec(
        spec_id="tsplib_euc_3d",
        family="rounded_euclidean",
        dimension=3,
        result_type="integer",
        version=1,
        edge_rounding="nint",
        points_ndim=3,
        conformance=(
            ((0, 0, 0), (3, 4, 0), 5.0),
            ((0, 0, 0), (1, 1, 1), 2.0),
        ),
    )
)

register_spec(
    ObjectiveSpec(
        spec_id="tsplib_ceil_2d",
        family="ceiling_euclidean",
        dimension=2,
        result_type="integer",
        version=1,
        edge_rounding="ceil",
        conformance=(
            ((0, 0), (3, 4), 5.0),
            ((0, 0), (1, 1), 2.0),
        ),
    )
)


def verify_conformance() -> list[str]:
    """Score the conformance examples of every registered spec."""
    failures: list[str] = []
    for spec in _SPECS.values():
        for left, right, expected in spec.conformance:
            delta = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
            actual = spec.edge_length(delta)
            if abs(actual - expected) > 1e-12:
                failures.append(
                    f"{spec.full_id()}: edge {left}-{right}: expected {expected}, "
                    f"got {actual}"
                )
    return failures


__all__ = [
    "ObjectiveSpec",
    "all_objectives",
    "get_objective",
    "register_spec",
    "verify_conformance",
]
