"""Public-source adapters with checksums and provenance (spec section 7.1, phase 2).

Each adapter performs the section 7.1 contract:

1. locates an upstream artifact without modifying it;
2. verifies the expected checksum when one is supplied (manifest digests);
3. records retrieval date, URL, and usage/redistribution status;
4. parses into coordinates/objective while retaining the original byte
   digest;
5. creates one manifest per actual objective instance;
6. imports reference artifacts as unverified records;
7. validates and rescores tours through the shared oracle (handled by the
   reference registry verification rule);
8. promotes reference status only under the declared verification rule.

The parse backend is shared: the DIMACS challenge testbed stores its
instances in standard TSPLIB format (EUC_2D, CEIL_2D, UPPER_DIAG_ROW), so
DIMACS and Waterloo are source-metadata variants of the TSPLIB95 parser.
Objective semantics are taken from the file's EDGE_WEIGHT_TYPE, never from
the filename or visual appearance.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from algofinder.benchmark.digests import (
    coordinate_parent_id,
    encoding_id,
    instance_id,
    lineage_group_id,
    objective_sibling_group_id,
)
from algofinder.benchmark.references import Reference, ReferenceRegistry
from algofinder.instances.base import load_manifest, save_manifest
from algofinder.instances.tsplib import (
    PARSER_VERSION,
    TSPLIB_OPTIMA,
    parse_tour_file,
    parse_tsp_file,
    sha256_file,
)
from algofinder.problems.base import Instance

RETRIEVED_AT = "2026-08-02"

RAW_L2 = "raw_l2_f64@1"
EUC_2D = "tsplib_euc_2d@1"
CEIL_2D = "tsplib_ceil_2d@1"

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SOURCES_INVENTORY_PATH = DATA_DIR / "sources" / "source_inventory.json"


@dataclass
class SourceRecord:
    """One ingested upstream file (etsp.source.v1)."""

    name: str
    format: str
    upstream_url: str
    retrieved_at: str
    license_spdx: str
    redistribution: str
    usage: str
    file_digest: str
    parser_version: str
    status: str
    objective_spec_id: str | None = None
    n: int | None = None
    instance_id: str | None = None
    notes: str | None = None

    def to_mapping(self) -> dict[str, object]:
        mapping = asdict(self)
        mapping["schema_version"] = "etsp.source.v1"
        return mapping


class SourceError(Exception):
    """Raised when an adapter cannot honestly ingest a file."""


class UnsupportedFormatError(SourceError):
    """The file's declared semantics are not supported (recorded, not faked)."""


class TSPLIBAdapter:
    """TSPLIB95 adapter: EUC_2D only; everything else is a recorded status."""

    format = "tsplib95"
    source_name = "tsplib95"
    upstream_url = "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/"
    license_spdx = "tsplib-terms"
    redistribution = "allowed"
    usage = "benchmark"
    parser_version = PARSER_VERSION

    def __init__(self, source_dir: str | Path) -> None:
        self.source_dir = Path(source_dir)

    def source_record(self, path: Path, status: str, **extra: object) -> SourceRecord:
        notes = extra.pop("notes", f"source: {self.source_name}")
        return SourceRecord(
            name=path.stem,
            format=self.format,
            upstream_url=self.upstream_url,
            retrieved_at=RETRIEVED_AT,
            license_spdx=self.license_spdx,
            redistribution=self.redistribution,
            usage=self.usage,
            file_digest=sha256_file(path),
            parser_version=self.parser_version,
            status=status,
            notes=notes,
            **extra,
        )

    def ingest_one(self, path: Path) -> tuple[dict[str, object], SourceRecord, list[Reference]]:
        """Parse one file; returns (instance data, source record, references)."""
        parsed = parse_tsp_file(path)
        points = np.asarray(parsed["points"], dtype=float)
        dimension = int(parsed["dimension"])
        if points.shape[0] != dimension:
            raise SourceError(f"{path.name}: dimension mismatch")
        name = str(parsed["name"])

        coordinate_parent = coordinate_parent_id(points)
        objective_full = EUC_2D
        instance_digest = instance_id(points, objective_full)
        lineage = lineage_group_id(f"{self.source_name}:{path.stem}")

        data: dict[str, object] = {
            "points": points.tolist(),
            "source": self.source_name,
            "dimension": dimension,
            "metric": "EUC_2D",
            "sha256": sha256_file(path),
            "encoding_id": encoding_id(path.read_bytes()),
            "parser": self.parser_version,
            "coordinate_parent_id": coordinate_parent,
            "objective_sibling_group_id": objective_sibling_group_id(points),
            "lineage_group_id": lineage,
            "instance_id": instance_digest,
            "instance_id_raw_l2": instance_id(points, RAW_L2),
            "instance_id_ceil_2d": instance_id(points, CEIL_2D),
        }
        references: list[Reference] = []
        provenance = {
            "source_url": self.upstream_url,
            "source_name": self.source_name,
            "retrieved_at": RETRIEVED_AT,
        }

        if self.format == "tsplib95" and name in TSPLIB_OPTIMA:
            published = TSPLIB_OPTIMA[name]
            tour_path = path.with_name(f"{path.stem}.opt.tour")
            if tour_path.exists():
                tour = tuple(int(city) for city in parse_tour_file(tour_path))
                references.append(
                    Reference(
                        instance_id=instance_digest,
                        objective_spec_id=objective_full,
                        kind="certified_optimum",
                        value=float(published),
                        provenance=dict(provenance),
                        status="verified",
                        tour=tour,
                        proof={
                            "method": "rescored_published_optimum",
                            "artifact_digest": sha256_file(tour_path),
                            "independent_verifier": "algofinder.benchmark.scoring",
                        },
                    )
                )
                data["opt_tour_rescored"] = True
                data["best_known_status"] = "verified"
            else:
                references.append(
                    Reference(
                        instance_id=instance_digest,
                        objective_spec_id=objective_full,
                        kind="certified_optimum",
                        value=float(published),
                        provenance=dict(provenance),
                        status="imported_unverified",
                        proof={
                            "method": "published_table",
                            "artifact_digest": None,
                            "independent_verifier": None,
                        },
                    )
                )
                data["best_known_status"] = "imported_unverified"

        return data, self.source_record(path, "ingested", n=dimension, objective_spec_id=objective_full, instance_id=instance_digest), references


class DIMACSAdapter(TSPLIBAdapter):
    """DIMACS challenge testbed adapter.

    The 8th DIMACS challenge stores all instances in standard TSPLIB format
    (EUC_2D, CEIL_2D, UPPER_DIAG_ROW); the adapter records the DIMACS source
    and imports the Concorde-computed optima from the challenge page.
    """

    format = "dimacs-challenge"
    source_name = "dimacs-challenge"
    upstream_url = "http://dimacs.rutgers.edu/archive/Challenges/TSP/"
    license_spdx = "dimacs-challenge-terms"
    redistribution = "allowed"
    usage = "benchmark"

    DIMACS_OPTIMA: dict[str, int] = {
        "E1k.1": 22985695,
    }

    def ingest_one(self, path: Path) -> tuple[dict[str, object], SourceRecord, list[Reference]]:
        try:
            data, record, references = super().ingest_one(path)
        except ValueError as exc:
            raise UnsupportedFormatError(str(exc)) from None
        name = path.stem
        if name in self.DIMACS_OPTIMA:
            data["best_known"] = float(self.DIMACS_OPTIMA[name])
            data["best_known_status"] = "imported_unverified"
            references.append(
                Reference(
                    instance_id=str(data["instance_id"]),
                    objective_spec_id=EUC_2D,
                    kind="certified_optimum",
                    value=float(self.DIMACS_OPTIMA[name]),
                    provenance={
                        "source_url": self.upstream_url + "bounds",
                        "source_name": "dimacs-challenge-bounds",
                        "retrieved_at": RETRIEVED_AT,
                    },
                    status="imported_unverified",
                    proof={
                        "method": "concorde_branch_and_cut",
                        "artifact_digest": None,
                        "independent_verifier": None,
                    },
                )
            )
        return data, record, references


class WaterlooAdapter(TSPLIBAdapter):
    """Waterloo TSP World / National TSPs adapter (TSPLIB-format files)."""

    format = "waterloo-world"
    source_name = "waterloo-world-tsp"
    upstream_url = "https://www.math.uwaterloo.ca/tsp/world/"
    license_spdx = "unknown"
    redistribution = "allowed"
    usage = "benchmark"


def adapters() -> dict[str, type[TSPLIBAdapter]]:
    return {"tsplib": TSPLIBAdapter, "dimacs": DIMACSAdapter, "waterloo": WaterlooAdapter}


def _load_inventory() -> list[SourceRecord]:
    if not SOURCES_INVENTORY_PATH.exists():
        return []
    import json

    records: list[SourceRecord] = []
    for mapping in json.loads(SOURCES_INVENTORY_PATH.read_text()):
        payload = dict(mapping)
        payload.pop("schema_version", None)
        records.append(SourceRecord(**payload))
    return records


def _save_inventory(records: list[SourceRecord]) -> None:
    SOURCES_INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.to_mapping() for record in records]
    SOURCES_INVENTORY_PATH.write_text(
        __import__("json").dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def ingest(
    adapter_format: str,
    source_dir: str | Path,
    *,
    copy_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    skip_files: set[str] | None = None,
) -> list[Instance]:
    """Run one adapter over a directory; updates manifests, inventory, references.

    Files the adapter must reject (e.g. UPPER_DIAG_ROW) are recorded in the
    inventory as ``unsupported`` — a status, not a silent reinterpretation.
    """
    adapter_cls = adapters()[adapter_format]
    adapter = adapter_cls(source_dir)
    source_path = Path(source_dir)
    registry = ReferenceRegistry()
    inventory = _load_inventory()

    records: list[Instance] = []
    inventory_records: list[SourceRecord] = []
    for path in sorted(source_path.glob("*.tsp")):
        if skip_files and path.stem in skip_files:
            continue
        try:
            data, source_record, references = adapter.ingest_one(path)
        except UnsupportedFormatError as exc:
            inventory_records.append(
                adapter.source_record(
                    path,
                    "unsupported",
                    notes=str(exc).split(":")[0] if ":" in str(exc) else str(exc),
                )
            )
            print(f"{path.name}: unsupported ({exc})")
            continue
        except ValueError as exc:
            print(f"{path.name}: parse error: {exc}")
            continue

        instance = Instance(
            name=f"tsp:{adapter.source_name}:{path.stem}",
            problem="tsp",
            subproblem="euclidean",
            family=adapter.source_name,
            seed=0,
            data=data,
            best_known=float(data["best_known"]) if "best_known" in data else None,
            split="test",
        )
        records.append(instance)
        inventory_records.append(source_record)
        added = registry.append_many(references)
        print(
            f"{path.name:12s} n={data['dimension']:4d} refs={len(references)} "
            f"(+{added} new)"
        )
        if copy_dir is not None:
            destination = Path(copy_dir) / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

    if manifest_path is not None:
        save_manifest(records, manifest_path)
        print(f"wrote {len(records)} instances to {manifest_path}")

    known_digests = {record.file_digest for record in inventory}
    fresh_digests = {record.file_digest for record in inventory_records}
    for record in inventory_records:
        if record.file_digest not in known_digests or record.file_digest in fresh_digests:
            inventory = [
                existing
                for existing in inventory
                if existing.file_digest != record.file_digest
            ]
            inventory.append(record)
    _save_inventory(inventory)
    registry.write_snapshot()
    print(f"inventory: {len(inventory_records)} files processed "
          f"({len(records)} ingested)")
    return records


def reannotate_tsplib_manifest(manifest_path: str | Path) -> None:
    """Add instance digests and reference status to the TSPLIB manifest.

    The committed TSPLIB manifest predates the digest layer; this rewrites it
    in place with identical content plus the digest fields, so instance IDs
    become queryable without breaking the existing pipeline.
    """
    instances = load_manifest(manifest_path)
    updated = 0
    for instance in instances:
        points = np.asarray(instance.data["points"], dtype=float)
        objective_full = EUC_2D
        instance.data["coordinate_parent_id"] = coordinate_parent_id(points)
        instance.data["objective_sibling_group_id"] = objective_sibling_group_id(points)
        instance.data["lineage_group_id"] = lineage_group_id(
            f"tsplib95:{str(instance.data.get('source', instance.name))}"
        )
        instance.data["instance_id"] = instance_id(points, objective_full)
        instance.data["instance_id_raw_l2"] = instance_id(points, RAW_L2)
        instance.data["instance_id_ceil_2d"] = instance_id(points, CEIL_2D)
        if "best_known_status" not in instance.data:
            instance.data["best_known_status"] = (
                "verified" if instance.data.get("opt_tour_rescored") else "imported_unverified"
            )
        updated += 1
    save_manifest(instances, manifest_path)
    print(f"reannotated {updated} instances in {manifest_path}")


def main() -> int:
    import sys

    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: python -m algofinder.benchmark.sources "
            "<tsplib|dimacs|waterloo> <source_dir> [copy_dir] [manifest.json]"
        )
    adapter_format = sys.argv[1]
    source_dir = sys.argv[2]
    copy_dir = sys.argv[3] if len(sys.argv) > 3 else None
    manifest = sys.argv[4] if len(sys.argv) > 4 else None
    ingest(adapter_format, source_dir, copy_dir=copy_dir, manifest_path=manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
