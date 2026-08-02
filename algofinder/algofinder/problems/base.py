"""Shared contracts for optimization problem instances and states."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Instance:
    """A fully serializable problem instance descriptor."""

    name: str
    problem: str
    subproblem: str
    family: str
    seed: int
    data: dict[str, Any]
    best_known: float | None = None
    split: str = "test"

    def to_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "name": self.name,
            "problem": self.problem,
            "subproblem": self.subproblem,
            "family": self.family,
            "seed": self.seed,
            "data": self.data,
            "split": self.split,
        }
        if self.best_known is not None:
            mapping["best_known"] = self.best_known
        return mapping

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "Instance":
        return cls(
            name=str(mapping["name"]),
            problem=str(mapping["problem"]),
            subproblem=str(mapping["subproblem"]),
            family=str(mapping["family"]),
            seed=int(mapping["seed"]),
            data=dict(mapping["data"]),
            best_known=(
                float(mapping["best_known"]) if "best_known" in mapping else None
            ),
            split=str(mapping.get("split", "test")),
        )


class ProblemState(Protocol):
    """A concrete instance state that can verify and score solutions."""

    name: str

    def verify(self, solution: Any) -> bool: ...

    def objective_value(self, solution: Any) -> float: ...


class Problem(ABC):
    """A problem family with its subproblems and state construction rules."""

    id: str = ""
    display: str = ""
    subproblems: tuple[str, ...] = ()
    minimize: bool = True

    @abstractmethod
    def build_state(self, instance: Instance) -> ProblemState:
        """Construct a problem state from an instance descriptor."""

    def is_better(self, first: float, second: float) -> bool:
        """Return whether objective ``first`` dominates objective ``second``."""
        if self.minimize:
            return first < second
        return first > second

    def gap_percent(self, cost: float, reference: float) -> float:
        """Percentage gap of ``cost`` relative to ``reference`` (0 is best)."""
        if reference == 0.0:
            return 0.0 if cost == 0.0 else float("inf")
        if self.minimize:
            return 100.0 * (cost - reference) / reference
        return 100.0 * (reference - cost) / reference
