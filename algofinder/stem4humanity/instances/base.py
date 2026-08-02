"""Instance manifests: JSON serialization/deserialization and splits."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from stem4humanity.problems.base import Instance


def save_manifest(instances: Iterable[Instance], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [instance.to_mapping() for instance in instances]
    path.write_text(json.dumps(records, indent=2) + "\n")


def load_manifest(path: str | Path) -> list[Instance]:
    records = json.loads(Path(path).read_text())
    return [Instance.from_mapping(record) for record in records]


def split_manifests(instances: list[Instance]) -> dict[str, list[Instance]]:
    """Partition instances by their ``split`` field (default ``test``)."""
    splits: dict[str, list[Instance]] = {}
    for instance in instances:
        splits.setdefault(instance.split or "test", []).append(instance)
    return splits


def annotate_best_known(
    instances: list[Instance],
    solver: Any,
    problem: Any,
    budget_seconds: float | None = None,
) -> list[Instance]:
    """Fill in ``best_known`` using an exact solver (small instances only).

    The solver's claim is only accepted after independent verification:
    the solution must pass ``state.verify`` and its recomputed objective
    must match the reported cost within float tolerance.
    """
    updated = []
    for instance in instances:
        if instance.best_known is not None:
            updated.append(instance)
            continue
        try:
            state = problem.build_state(instance)
            result = solver().solve(state, budget_seconds=budget_seconds)
        except Exception:
            updated.append(instance)
            continue
        if not result.exact:
            updated.append(instance)
            continue
        try:
            verified = state.verify(result.solution)
            recomputed = state.objective_value(result.solution)
            verified = verified and abs(
                float(recomputed) - float(result.cost)
            ) <= 1e-9 * max(1.0, abs(float(result.cost)))
        except Exception:
            verified = False
        if verified:
            instance = replace(instance, best_known=float(result.cost))
        updated.append(instance)
    return updated
