"""Canonical storage of feature records (tier 1, spec section 6.7).

One append-style JSONL file per feature set under
``public/data/features/<feature_set_id>.jsonl``.  Files are written
deterministically (records sorted by instance digest), never rewritten in
place, and the registry indexes them like every other canonical artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from algofinder.features.base import FeatureSet, compute_feature_record
from algofinder.problems.base import Instance
from algofinder.trace.serialize import strict_dumps

FEATURES_SUBDIR = "features"


def features_dir(instances_dir: str | Path) -> Path:
    """Canonical feature directory, sibling of the instance manifests."""
    return Path(instances_dir) / FEATURES_SUBDIR


def feature_set_path(instances_dir: str | Path, feature_set_id: str) -> Path:
    return features_dir(instances_dir) / f"{feature_set_id}.jsonl"


def find_feature_files(instances_dir: str | Path) -> list[Path]:
    return sorted(features_dir(instances_dir).glob("*.jsonl"))


def write_feature_set(
    instances_dir: str | Path,
    feature_set: FeatureSet,
    instances: Iterable[Instance],
) -> Path:
    """Compute records for ``instances`` and write the canonical JSONL.

    Records are sorted by instance digest so the file is byte-identical
    across reruns over the same input.  Returns the written path.
    """
    records = [
        compute_feature_record(feature_set, instance)
        for instance in instances
        if feature_set.applies(instance)
    ]
    records.sort(key=lambda record: record["instance_id"])
    path = feature_set_path(instances_dir, feature_set.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [strict_dumps(record) + "\n" for record in records]
    path.write_text("".join(lines), encoding="utf-8")
    return path


def read_feature_set(instances_dir: str | Path, feature_set_id: str) -> list[dict[str, Any]]:
    """Read all records of one feature set, in file order."""
    path = feature_set_path(instances_dir, feature_set_id)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json_loads(line))
    return records


def json_loads(raw: str) -> dict[str, Any]:
    import json

    return json.loads(raw)


__all__ = [
    "FEATURES_SUBDIR",
    "feature_set_path",
    "features_dir",
    "find_feature_files",
    "read_feature_set",
    "write_feature_set",
]
