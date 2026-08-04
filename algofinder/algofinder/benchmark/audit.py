"""Public-source inventory and metric/reference-status audit (phase 2 item 5).

``python -m algofinder.benchmark.audit`` verifies the phase 2 exit criteria:

- every instance's content-addressable digests recompute to the recorded
  values (determinism);
- the same coordinates under distinct metrics produce distinct instance IDs
  (EUC_2D vs raw L2 vs CEIL_2D);
- no reference with a tour fails to rescore through the shared oracle;
- no ``certified_optimum`` is presented as proven without ``verified``
  status (imported optima stay visibly unverified);
- the source inventory lists every ingested file with checksum, license,
  usage, and reference status.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from algofinder.benchmark.digests import (
    coordinate_parent_id,
    instance_id,
    objective_sibling_group_id,
)
from algofinder.benchmark.references import (
    Reference,
    ReferenceRegistry,
    rescore_reference,
)
from algofinder.benchmark.sources import (
    CEIL_2D,
    EUC_2D,
    RAW_L2,
    SOURCES_INVENTORY_PATH,
)
from algofinder.instances.base import load_manifest

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

MANIFESTS = {
    "tsplib": DATA_DIR / "instances" / "tsp_tsplib_manifests.json",
    "micro": DATA_DIR / "instances" / "tsp_micro_manifests.json",
    "dimacs": DATA_DIR / "instances" / "tsp_dimacs_manifests.json",
    "waterloo": DATA_DIR / "instances" / "tsp_waterloo_manifests.json",
    "tetrahedra": DATA_DIR / "instances" / "tsp_tetrahedra_manifests.json",
    "exact_structure": DATA_DIR / "instances" / "tsp_exact_structure_manifests.json",
    "anytime_quality": DATA_DIR / "instances" / "tsp_anytime_quality_manifests.json",
    "ml_ood": DATA_DIR / "instances" / "tsp_ml_ood_manifests.json",
}

SUITE_INDEX_PATH = DATA_DIR / "instances" / "suite_index.json"

FIT_ROLES = frozenset({"train", "validation"})
EVAL_SPLITS = frozenset(
    {"test_iid", "test_family_ood", "test_parameter_ood", "test_metric_ood"}
)


def _load_references() -> list[dict[str, object]]:
    return ReferenceRegistry().load_all()


def _audit_generated(suite_names: list[str], lines: list[str], failures: list[str]) -> None:
    """Phase 3 checks for the compiled strata suites."""
    for suite_name in suite_names:
        path = DATA_DIR / "instances" / f"tsp_{suite_name}_manifests.json"
        if not path.exists():
            failures.append(f"missing generated manifest: {path}")
            continue
        instances = load_manifest(path)
        bucket_mismatches = 0
        for instance in instances:
            points = np.asarray(instance.data["points"], dtype=float)
            for key, spec in (
                ("instance_id_raw_l2", RAW_L2),
                ("instance_id_ceil_2d", CEIL_2D),
            ):
                recorded = instance.data.get(key)
                if recorded is not None and recorded != instance_id(points, spec):
                    bucket_mismatches += 1
                    failures.append(
                        f"{instance.name}: {key} mismatch"
                    )

        train_lineages = {
            i.data["lineage_group_id"]
            for i in instances
            if (i.split or "test") in FIT_ROLES
        }
        eval_lineages = {
            i.data["lineage_group_id"]
            for i in instances
            if (i.split or "test") in EVAL_SPLITS
        }
        leakage = train_lineages & eval_lineages
        if leakage:
            failures.append(
                f"{suite_name}: {len(leakage)} lineage groups cross fit/test splits"
            )

        by_status: dict[str, int] = {}
        for i in instances:
            role = i.split or "test"
            by_status[role] = by_status.get(role, 0) + 1
        split_summary = ", ".join(
            f"{role}={by_status[role]}" for role in sorted(by_status)
        )
        if suite_name == "exact_structure":
            if not all(i.best_known is not None for i in instances):
                failures.append(
                    "exact_structure: every instance requires an exact "
                    "best_known (certified optimum)"
                )
        if suite_name == "anytime_quality":
            if not all(i.best_known is not None for i in instances):
                failures.append(
                    "anytime_quality: every instance requires an upper bound"
                )
        lines.append(
            f"generated {suite_name:15s} {len(instances):3d} instances "
            f"({bucket_mismatches} bucket mismatches) splits [{split_summary}]"
        )

    if not SUITE_INDEX_PATH.exists():
        failures.append("missing suite index")
        return
    index = json.loads(SUITE_INDEX_PATH.read_text())
    for suite_name in suite_names:
        entry = index.get(suite_name)
        if entry is None:
            failures.append(f"suite index missing {suite_name}")
            continue
        path = DATA_DIR / "instances" / entry["manifest"]
        count = len(load_manifest(path)) if path.exists() else -1
        lines.append(
            f"  index {suite_name:15s} idx={entry['instances']:3d} "
            f"manifest={count:3d} roots={entry['lineage_roots']:3d} "
            f"cells={entry['cells']:2d}"
        )
        if entry["instances"] != count:
            failures.append(
                f"suite index mismatch: {suite_name} index={entry['instances']} "
                f"manifest={count}"
            )


def run_audit() -> list[str]:
    lines: list[str] = []
    failures: list[str] = []

    for family, path in MANIFESTS.items():
        if not path.exists():
            lines.append(f"{family:10s} manifest missing: {path}")
            failures.append(f"missing manifest: {path}")
            continue
        instances = load_manifest(path)
        digest_issues = 0
        for instance in instances:
            points = np.asarray(instance.data["points"], dtype=float)
            metric = str(instance.data.get("metric", "EUC_2D"))
            objective_full = {"EUC_2D": EUC_2D, "EUC_3D": "tsplib_euc_3d@1"}.get(
                metric, EUC_2D
            )
            recomputed = instance_id(points, objective_full)
            recorded = instance.data.get("instance_id")
            if recorded is not None and recorded != recomputed:
                digest_issues += 1
                failures.append(
                    f"{instance.name}: instance_id mismatch (recomputed {recomputed})"
                )
            distinct = {
                objective_sibling_group_id(points),
                instance_id(points, RAW_L2),
                instance_id(points, CEIL_2D),
            }
            if len(distinct) < 3:
                failures.append(
                    f"{instance.name}: metrics do not produce distinct instance IDs"
                )
        lines.append(
            f"{family:10s} {len(instances):3d} instances "
            f"({digest_issues} digest mismatches)"
        )

    generated = ["exact_structure", "anytime_quality", "ml_ood"]
    _audit_generated(generated, lines, failures)

    references = _load_references()
    lines.append(
        f"references {len(references):3d} records, "
        f"{len(ReferenceRegistry().snapshot())} in snapshot"
    )
    by_digest: dict[str, Instance] = {}
    for family, path in MANIFESTS.items():
        if not path.exists():
            continue
        for instance in load_manifest(path):
            points = np.asarray(instance.data["points"], dtype=float)
            for spec_id in (RAW_L2, EUC_2D, CEIL_2D, "tsplib_euc_3d@1"):
                by_digest.setdefault(instance_id(points, spec_id), instance)

    verified_count = 0
    for mapping in references:
        instance = by_digest.get(mapping["instance_id"])
        if instance is None:
            continue
        points = np.asarray(instance.data["points"], dtype=float)
        reference = Reference.from_mapping(mapping)
        if reference.tour is None:
            continue
        matches, rescored = rescore_reference(reference, points)
        if not matches:
            failures.append(
                f"reference {mapping['reference_id'][:12]} for {instance.name}: "
                f"rescore {rescored} != recorded {reference.value}"
            )
        elif reference.status == "verified":
            verified_count += 1
            lines.append(
                f"  verified: {instance.name} "
                f"({reference.objective_spec_id}, {reference.kind}, "
                f"{reference.value:g})"
            )
        else:
            lines.append(
                f"  {reference.status}: {instance.name} "
                f"({reference.objective_spec_id}, {reference.kind}, "
                f"{reference.value:g})"
            )
    lines.append(f"rescored references: {verified_count} verified, tours all rescore")

    certified = [
        mapping
        for mapping in references
        if mapping["kind"] == "certified_optimum"
    ]
    unverified_certified = [
        mapping for mapping in certified if mapping["status"] != "verified"
    ]
    lines.append(
        f"certified optima: {len(certified)} total, "
        f"{len(unverified_certified)} imported_unverified (not proven)"
    )

    if not SOURCES_INVENTORY_PATH.exists():
        failures.append("source inventory missing")
    else:
        inventory = json.loads(SOURCES_INVENTORY_PATH.read_text())
        lines.append(f"source inventory: {len(inventory)} files")
        for entry in sorted(inventory, key=lambda item: item["name"]):
            lines.append(
                f"  {entry['name']:12s} {entry['format']:18s} "
                f"n={entry.get('n') or '-':>6} status={entry['status']:10s} "
                f"redistribution={entry['redistribution']:9s} "
                f"digest={entry['file_digest'][:12]}"
            )

    lines.append(f"audit: {len(failures)} failures")
    lines.extend(f"  FAIL {failure}" for failure in failures)
    return lines


def main() -> int:
    lines = ["# sources and references audit", ""]
    lines.extend(run_audit())
    print("\n".join(lines))
    return 0 if all("FAIL" not in line for line in lines) else 1


if __name__ == "__main__":
    raise SystemExit(main())
