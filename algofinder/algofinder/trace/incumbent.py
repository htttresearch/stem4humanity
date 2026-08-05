"""Validated incumbent-trace ingestion (benchmark spec 6.6, 13 item 4).

The harness owns final feasibility and scoring: a solver's incumbent
stream is *suspected* output. Only candidates that pass
``ProblemState.verify`` and score cleanly through
``ProblemState.objective_value`` are accepted; invalid candidates are
dropped and counted, never recorded. Accepted incumbents are emitted as
strictly-improving events with the harness-scored cost and a stable
tour digest, so traces can never contain invalid tours.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

IncumbentObserver = Callable[[float, Any, float, str], None]


def tour_digest(tour: Any) -> str:
    """Stable digest of an ordered tour (node indices, as given)."""
    payload = json.dumps([int(node) for node in tour], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class Incumbent:
    """One accepted, harness-scored incumbent event."""

    t: float
    tour: list[int]
    cost: float
    tour_id: str

    def to_mapping(self) -> dict[str, Any]:
        return {"t": round(float(self.t), 6), "cost": float(self.cost), "tour_id": self.tour_id}


class IncumbentSink:
    """Receives solver incumbent candidates during one solve."""

    def emit(self, t: float, solution: Any) -> None:
        raise NotImplementedError

    def summary(self) -> dict[str, Any]:
        return {}


class NullIncumbentSink(IncumbentSink):
    """Prod-mode no-op: incumbents are neither validated nor recorded."""


class ValidatedIncumbentSink(IncumbentSink):
    """Validates every candidate; keeps only feasible, improving tours.

    ``state`` provides the authoritative ``verify`` and
    ``objective_value``. ``on_incumbent`` (optional) observes every
    accepted event, e.g. to write a trace span.
    """

    def __init__(
        self,
        state: Any,
        *,
        on_incumbent: IncumbentObserver | None = None,
        keep_tours: bool = False,
    ) -> None:
        self._state = state
        self._observer = on_incumbent
        self._keep_tours = keep_tours
        self._accepted: list[Incumbent] = []
        self._rejected_invalid = 0
        self._rejected_non_improving = 0
        self._accepted_candidates = 0
        self._last_error: str | None = None

    def emit(self, t: float, solution: Any) -> None:
        self._accepted_candidates += 1
        try:
            if not self._state.verify(solution):
                self._rejected_invalid += 1
                return
            cost = float(self._state.objective_value(solution))
        except Exception as exc:  # objective itself failed
            self._rejected_invalid += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            return
        if self._accepted and cost >= self._accepted[-1].cost:
            self._rejected_non_improving += 1
            return
        incumbent = Incumbent(
            t=float(t),
            tour=[int(node) for node in solution],
            cost=cost,
            tour_id=tour_digest(solution),
        )
        self._accepted.append(incumbent)
        if self._observer is not None:
            self._observer(incumbent.t, incumbent.tour, incumbent.cost, incumbent.tour_id)

    def summary(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "accepted_candidates": self._accepted_candidates,
            "accepted": len(self._accepted),
            "rejected_invalid": self._rejected_invalid,
            "rejected_non_improving": self._rejected_non_improving,
        }
        if self._accepted:
            record["first"] = self._accepted[0].to_mapping()
            record["best"] = self._accepted[-1].to_mapping()
            record["tour_ids"] = [item.tour_id for item in self._accepted]
            if self._keep_tours:
                record["tours"] = {
                    item.tour_id: item.tour for item in self._accepted
                }
        if self._last_error is not None:
            record["last_error"] = self._last_error
        return record


__all__ = ["Incumbent", "IncumbentSink", "NullIncumbentSink", "ValidatedIncumbentSink", "tour_digest"]
