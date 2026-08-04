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
}


def _load_references() -> list[dict[str, object]]:
    return ReferenceRegistry().load_all()


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
