"""Deterministic feature-fill generation (spec section 13.6).

Evolved instances are the most expensive way to fill coverage gaps; the
deterministic parameter scan is the deterministic middle ground: declared
bins (city-count ranges per Euclidean-TSP family) are scanned with
dedicated seeds that are never used by the final-evaluated corpus, and the
resulting instances are written to their own manifest with ``split =
"train"`` so they are never final-evaluated and never enter the test
folds.  Bins only produce instances whose names are not already present in
the canonical manifests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from algofinder.instances.base import load_manifest, save_manifest
from algofinder.instances.euclidean_tsp import generate_euclidean_instances
from algofinder.problems.base import Instance
from algofinder.store.canonical import find_instances_manifests

# (city_count range, family, count, seed) — seeds reserved for fill scans.
FILL_BINS: tuple[tuple[tuple[int, int], str, int, int], ...] = (
    ((200, 300), "uniform", 4, 6101),
    ((200, 300), "clustered", 4, 6201),
    ((200, 300), "corridor", 4, 6301),
    ((400, 500), "mixed", 4, 6401),
)

FILL_MANIFEST_NAME = "tsp_fill_manifests.json"


def _existing_names(instances_dir: str | Path) -> set[str]:
    names: set[str] = set()
    for path in find_instances_manifests(instances_dir):
        if Path(path).name == FILL_MANIFEST_NAME:
            continue
        names.update(instance.name for instance in load_manifest(path))
    return names


def generate_fill_instances(
    instances_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Deterministic scan over the declared fill bins (no evaluated configs)."""
    instances_dir = Path(instances_dir)
    existing = _existing_names(instances_dir)
    fill: list[Instance] = []
    skipped = 0
    for (lo, hi), family, count, seed in FILL_BINS:
        candidates = generate_euclidean_instances(
            (lo, hi), family, count=count, seed=seed, split="train"
        )
        for instance in candidates:
            if instance.name in existing:
                skipped += 1
                continue
            fill.append(instance)
    fill.sort(key=lambda instance: instance.name)
    target = (
        Path(output_path) if output_path is not None
        else instances_dir / FILL_MANIFEST_NAME
    )
    save_manifest(fill, target)
    return {
        "manifest": str(target),
        "bins": len(FILL_BINS),
        "instances": len(fill),
        "skipped_duplicates": skipped,
    }


__all__ = ["FILL_BINS", "FILL_MANIFEST_NAME", "generate_fill_instances"]
