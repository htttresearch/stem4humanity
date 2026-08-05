"""Coverage audit for the phase-6 corpus (spec section 13.6).

Reports, per problem/subproblem/split, the instance counts, which feature
sets cover which instances, extraction failures, redundancy (same
instance digest from multiple manifests), and leakage (train/test digest
overlap).  Emits both a JSON summary and a markdown report.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from algofinder.features import FEATURE_SETS, instance_id
from algofinder.features.storage import find_feature_files, read_feature_set
from algofinder.instances.base import load_manifest
from algofinder.store.canonical import find_instances_manifests


def _load_corpus(instances_dir: str | Path) -> dict[str, Any]:
    manifests = find_instances_manifests(instances_dir)
    by_digest: dict[str, dict[str, Any]] = {}
    for path in manifests:
        for instance in load_manifest(path):
            digest = instance_id(instance)
            by_digest.setdefault(digest, {"name": instance.name, "sources": set()})
            by_digest[digest]["sources"].add(str(path))
    return {
        "manifests": [str(path) for path in manifests],
        "instances": by_digest,
    }


def _load_features(instances_dir: str | Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Feature records grouped by feature set, keyed by instance digest."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for path in find_feature_files(instances_dir):
        feature_set_id = path.stem
        by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in read_feature_set(instances_dir, feature_set_id):
            by_instance[record["instance_id"]].append(record)
        grouped[feature_set_id] = dict(by_instance)
    return grouped


def audit_corpus(instances_dir: str | Path) -> dict[str, Any]:
    corpus = _load_corpus(instances_dir)
    features = _load_features(instances_dir)

    cells: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    redundancy: list[dict[str, Any]] = []
    for digest, info in sorted(corpus["instances"].items()):
        if len(info["sources"]) > 1:
            redundancy.append(
                {
                    "instance_id": digest,
                    "name": info["name"],
                    "sources": sorted(info["sources"]),
                }
            )
    for path in corpus["manifests"]:
        for instance in load_manifest(path):
            cells[(instance.problem, instance.subproblem, instance.split or "test")].append(
                instance_id(instance)
            )

    coverage: list[dict[str, Any]] = []
    total_instances = 0
    holes: list[dict[str, Any]] = []
    for key in sorted(cells):
        problem, subproblem, split = key
        digests = sorted(set(cells[key]))
        total_instances += len(digests)
        set_coverage: dict[str, dict[str, Any]] = {}
        for feature_set in FEATURE_SETS:
            feature_set_id = feature_set.id
            applies = not feature_set.applies_to or (
                f"{problem}:{subproblem}" in feature_set.applies_to
            )
            covered = sum(
                1 for digest in digests if digest in features.get(feature_set_id, {})
            )
            failed = sum(
                1
                for digest in digests
                for record in features.get(feature_set_id, {}).get(digest, [])
                if "error" in record.get("status_by_feature", {})
            )
            set_coverage[feature_set_id] = {
                "applies": applies,
                "covered": covered,
                "failed": failed,
            }
            if applies and covered < len(digests):
                for digest in digests:
                    if digest not in features.get(feature_set_id, {}):
                        holes.append(
                            {
                                "instance_id": digest,
                                "problem": problem,
                                "subproblem": subproblem,
                                "split": split,
                                "feature_set_id": feature_set_id,
                            }
                        )
        coverage.append(
            {
                "problem": problem,
                "subproblem": subproblem,
                "split": split,
                "instances": len(digests),
                "feature_sets": set_coverage,
            }
        )

    leak_overlap: list[str] = []
    by_split: dict[str, set[str]] = defaultdict(set)
    for key, digests in cells.items():
        by_split[key[2]].update(digests)
    if "train" in by_split and "test" in by_split:
        leak_overlap = sorted(by_split["train"] & by_split["test"])

    return {
        "instances": total_instances,
        "manifests": corpus["manifests"],
        "coverage": coverage,
        "redundancy": redundancy,
        "holes": holes,
        "leak_overlap": leak_overlap,
        "feature_files": [str(path) for path in find_feature_files(instances_dir)],
    }


def render_audit(summary: dict[str, Any]) -> str:
    lines = ["# Corpus coverage audit", ""]
    lines.append(f"- instances: {summary['instances']}")
    lines.append(f"- manifests: {len(summary['manifests'])}")
    lines.append("")
    lines.append("## Coverage by problem:subproblem:split")
    lines.append("")
    set_ids = list(summary["coverage"][0]["feature_sets"]) if summary["coverage"] else []
    lines.append(
        "| Problem | Subproblem | Split | Instances | "
        + " | ".join(f"{set_id} covered" for set_id in set_ids)
        + " |"
    )
    lines.append("|" + "---|" * (4 + len(set_ids)))
    for row in summary["coverage"]:
        covered = " | ".join(
            str(row["feature_sets"][set_id]["covered"]) for set_id in set_ids
        )
        lines.append(
            f"| {row['problem']} | {row['subproblem']} | {row['split']} "
            f"| {row['instances']} | {covered} |"
        )
    lines.append("")
    lines.append(f"- redundancy (same digest in >1 manifest): {len(summary['redundancy'])}")
    for item in summary["redundancy"][:10]:
        lines.append(f"  - {item['name']}: {', '.join(item['sources'])}")
    lines.append("")
    lines.append(f"- coverage holes: {len(summary['holes'])}")
    for item in summary["holes"][:10]:
        lines.append(
            f"  - {item['instance_id'][:12]} ({item['problem']}:{item['subproblem']}, "
            f"{item['split']}): missing {item['feature_set_id']}"
        )
    lines.append("")
    lines.append(f"- train/test leak overlap: {len(summary['leak_overlap'])}")
    return "\n".join(lines)


__all__ = ["audit_corpus", "render_audit"]
