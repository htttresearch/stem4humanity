"""Indexer: rebuild the registry and the DuckDB query layer (tier 3).

The ``index`` pipeline step runs: (1) rescan canonical artifacts, (2)
regenerate the append-only registry logs (deterministic, byte-identical
on unchanged input), (3) rebuild the DuckDB file — views over the
canonical files plus regenerable Parquet caches — and optionally (4) run
the parity gate (SQL leaderboard vs the existing Python renderer).

Everything is idempotent; nothing is ever deleted or rewritten in place
in the canonical store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import duckdb

from algofinder.features.storage import find_feature_files
from algofinder.store.canonical import (
    SCHEMA_VERSION,
    find_instances_manifests,
    find_results_files,
    references_paths,
    run_digest,
    sha256_file,
)
from algofinder.store.queries import _list_literal, parity_check
from algofinder.store.registry import CAMPAIGN_KINDS, CORE_KINDS, Registry, write_kind


def _load_instances(instances_dir: str | Path) -> list[dict[str, Any]]:
    from algofinder.instances.base import load_manifest

    instances: list[dict[str, Any]] = []
    for path in find_instances_manifests(instances_dir):
        for instance in load_manifest(path):
            mapping = instance.to_mapping()
            instances.append(
                {
                    "id": run_digest(mapping),
                    "name": instance.name,
                    "problem": instance.problem,
                    "subproblem": instance.subproblem,
                    "family": instance.family,
                    "split": instance.split,
                    "best_known": instance.best_known,
                    "source": str(path),
                    "file_sha256": sha256_file(path),
                    "schema_version": SCHEMA_VERSION,
                }
            )
    return _dedupe(instances)


def _solver_entries() -> list[dict[str, Any]]:
    import algofinder.solvers.registry  # noqa: F401  (register all solvers)
    from algofinder.solvers.base import all_solvers

    entries: list[dict[str, Any]] = []
    for solver_id, cls in sorted(all_solvers().items()):
        instance = None
        try:
            instance = cls()
            capabilities = instance.capabilities().to_mapping()
        except Exception:
            capabilities = {"error": "instantiation failed"}
        entries.append(
            {
                "id": solver_id,
                "display": cls.display,
                "tags": sorted(cls.tags),
                "applies_to": sorted(cls.applies_to),
                "module": cls.__module__,
                "capabilities": capabilities,
                "manifest": instance.manifest().to_mapping() if instance is not None else None,
                "schema_version": SCHEMA_VERSION,
            }
        )
    return entries


def _run_entries(results_files: Iterable[str | Path]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(Path(item) for item in results_files):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        for mapping in record.get("runs", []):
            entries.append(
                {
                    "id": run_digest(mapping),
                    "instance": mapping.get("instance"),
                    "problem": mapping.get("problem"),
                    "subproblem": mapping.get("subproblem"),
                    "solver": mapping.get("solver"),
                    "status": mapping.get("status"),
                    "seed": mapping.get("seed"),
                    "environment_id": mapping.get("environment_id"),
                    "source": str(path),
                    "file_sha256": sha256_file(path),
                    "schema_version": SCHEMA_VERSION,
                }
            )
    return _dedupe(entries)


def _environment_entries(results_files: Iterable[str | Path]) -> list[dict[str, Any]]:
    from algofinder.harness.runenv import environment_id, environment_record

    entries: list[dict[str, Any]] = []
    observed: set[str] = set()
    for path in sorted(Path(item) for item in results_files):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        for mapping in record.get("runs", []):
            env_id = mapping.get("environment_id")
            if env_id:
                observed.add(env_id)
    for env_id in sorted(observed):
        if env_id == environment_id():
            fields: dict[str, Any] = {"record": environment_record()}
        else:
            fields = {"record": None, "note": "observed in run records only"}
        entries.append(
            {
                "id": env_id,
                "digest": env_id,
                **fields,
                "schema_version": SCHEMA_VERSION,
            }
        )
    return entries


def _feature_entries(
    instances_dir: str | Path,
) -> list[dict[str, Any]]:
    """Registry entries for canonical feature-record files."""
    entries: list[dict[str, Any]] = []
    for path in find_feature_files(instances_dir):
        records = _read_jsonl(path)
        for record in records:
            stable = {
                "instance_id": record.get("instance_id"),
                "feature_set_id": record.get("feature_set_id"),
                "extractor_digest": record.get("extractor_digest"),
                "values": record.get("values"),
                "status_by_feature": record.get("status_by_feature"),
            }
            entries.append(
                {
                    "id": run_digest(stable),
                    "instance_id": record.get("instance_id"),
                    "feature_set_id": record.get("feature_set_id"),
                    "tier": record.get("tier"),
                    "extractor_digest": record.get("extractor_digest"),
                    "status": sorted(record.get("status_by_feature", {})),
                    "source": str(path),
                    "file_sha256": sha256_file(path),
                    "schema_version": SCHEMA_VERSION,
                }
            )
    return _dedupe(entries)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def _dedupe(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for entry in entries:
        if entry["id"] in seen:
            continue
        seen.add(entry["id"])
        unique.append(entry)
    return unique


def build_registry(
    *,
    instances_dir: str | Path,
    results_files: Iterable[str | Path],
    registry_dir: str | Path,
) -> dict[str, int]:
    """Regenerate every registry log from the canonical files."""
    counts: dict[str, int] = {}
    for kind in CORE_KINDS:
        write_kind(registry_dir, kind, [])
        counts[kind] = 0
    for kind, entries in (
        ("instances", _load_instances(instances_dir)),
        ("solvers", _solver_entries()),
        ("runs", _run_entries(results_files)),
        ("environments", _environment_entries(results_files)),
        ("features", _feature_entries(instances_dir)),
    ):
        ordered = write_kind(registry_dir, kind, entries)
        counts[kind] = len(ordered)
    return counts


def verify_registry(
    registry_dir: str | Path,
    *,
    instances_dir: str | Path,
    results_files: Iterable[str | Path],
) -> list[str]:
    """Cross-check registry entries against the canonical files.

    For every entry with a ``file_sha256``, recompute the digest of the
    referenced canonical file and report drift.
    """
    registry = Registry(registry_dir)
    problems: list[str] = []
    for kind in ("instances", "runs"):
        for entry in registry.read(kind):
            source = entry.get("source")
            recorded = entry.get("file_sha256")
            if not source or not Path(source).exists():
                problems.append(f"{kind}/{entry['id']}: missing source {source!r}")
                continue
            if recorded != sha256_file(source):
                problems.append(f"{kind}/{entry['id']}: file digest mismatch on {source}")
    return problems


def _results_glob_paths(results_dir: str | Path, results_glob: str) -> list[str]:
    return [str(path) for path in find_results_files(results_dir, results_glob)]


def build_db(
    *,
    results_dir: str | Path,
    results_glob: str,
    instances_dir: str | Path,
    registry_dir: str | Path,
    db_path: str | Path,
    cache_dir: str | Path,
    learning_root: str | Path | None = None,
) -> dict[str, int]:
    """Rebuild the DuckDB file: views over canonical files + Parquet caches."""
    db_path = Path(db_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_paths = _results_glob_paths(results_dir, results_glob)
    manifest_paths = [str(path) for path in find_instances_manifests(instances_dir)]
    registry = Registry(registry_dir)

    con = duckdb.connect(str(db_path))
    try:
        if result_paths:
            con.execute(
                "CREATE OR REPLACE VIEW v_runs AS " + _results_view(result_paths)
            )
            con.execute(
                f"COPY (SELECT * FROM v_runs) TO '{_cache(cache_dir, 'runs')}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000, "
                "OVERWRITE_OR_IGNORE)"
            )
            con.execute(
                "CREATE OR REPLACE VIEW v_runs_cache AS "
                "SELECT * FROM read_parquet(" + _list_literal([str(_cache(cache_dir, "runs"))]) + ")"
            )
        if manifest_paths:
            con.execute(
                "CREATE OR REPLACE VIEW v_instances AS SELECT * FROM read_json_auto("
                + _list_literal(manifest_paths)
                + ")"
            )
            con.execute(
                f"COPY (SELECT * FROM v_instances) TO '{_cache(cache_dir, 'instances')}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000, "
                "OVERWRITE_OR_IGNORE)"
            )
            con.execute(
                "CREATE OR REPLACE VIEW v_instances_cache AS "
                "SELECT * FROM read_parquet(" + _list_literal([str(_cache(cache_dir, "instances"))]) + ")"
            )
        registry_path, snapshot_path = references_paths()
        if registry_path.exists():
            con.execute(
                "CREATE OR REPLACE VIEW v_references AS SELECT * FROM read_json_auto("
                + _list_literal([str(registry_path)])
                + ")"
            )
        feature_files = sorted(
            str(path) for path in find_feature_files(instances_dir)
        )
        if feature_files:
            con.execute(
                "CREATE OR REPLACE VIEW v_features AS SELECT * FROM read_json_auto("
                + _list_literal(feature_files)
                + ")"
            )
            con.execute(
                f"COPY (SELECT * FROM v_features) TO '{_cache(cache_dir, 'features')}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000, "
                "OVERWRITE_OR_IGNORE)"
            )
            con.execute(
                "CREATE OR REPLACE VIEW v_features_cache AS "
                "SELECT * FROM read_parquet(" + _list_literal([str(_cache(cache_dir, "features"))]) + ")"
            )
        for kind in ("solvers", "environments", *CAMPAIGN_KINDS):
            _create_registry_view(con, registry, kind)
        if learning_root is not None:
            _create_learning_views(con, Path(learning_root))
    finally:
        con.close()

    counts: dict[str, int] = {}
    with duckdb.connect(str(db_path)) as con:
        for name in (
            "v_runs", "v_instances", "v_references", "v_solvers", "v_environments",
            "v_features", *(f"v_{kind}" for kind in CAMPAIGN_KINDS),
            "v_distribution_profiles", "v_learning_runs", "v_agent_policies",
            "v_agent_evaluations", "v_agent_memories", "v_learning_datasets",
        ):
            try:
                counts[name] = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            except Exception:
                counts[name] = 0
    counts["cache_bytes"] = sum(
        path.stat().st_size for path in cache_dir.glob("*.parquet") if path.is_file()
    )
    return counts


def _results_view(paths: list[str]) -> str:
    return (
        "WITH runs AS (SELECT r.* FROM "
        f"(SELECT unnest(runs) AS r FROM read_json_auto({_list_literal(paths)})) x) "
        "SELECT * FROM runs"
    )


def _create_registry_view(
    con: duckdb.DuckDBPyConnection, registry: Registry, kind: str
) -> None:
    path = Path(registry.root) / f"{kind}.jsonl"
    if not path.exists():
        return
    con.execute(
        f"CREATE OR REPLACE VIEW v_{kind} AS SELECT * FROM read_json_auto("
        + _list_literal([str(path)])
        + ")"
    )


def _create_learning_views(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    """Expose immutable learning artifacts as rebuildable DuckDB views."""
    from algofinder.agents.learning.store import LearningArtifactStore

    store = LearningArtifactStore(root)
    sources = {
        "distribution_profiles": root / "distributions" / "*.json",
        "learning_runs": root / "runs" / "*.json",
        "agent_policies": root / "policies" / "*" / "policy.json",
        "agent_evaluations": root / "evaluations" / "*.json",
        "agent_memories": root / "memories" / "*.json",
        "learning_datasets": root / "datasets" / "*" / "manifest.json",
    }
    for name, pattern in sources.items():
        paths = sorted(str(path) for path in root.glob(str(pattern.relative_to(root))))
        if paths:
            for path in paths:
                store.read(Path(path).relative_to(root))
            con.execute(
                f"CREATE OR REPLACE VIEW v_{name} AS SELECT * FROM read_json_auto("
                + _list_literal(paths)
                + ")"
            )


def _cache(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name}.parquet"


def index_all(
    *,
    instances_dir: str | Path,
    results_dir: str | Path,
    results_glob: str,
    registry_dir: str | Path,
    db_path: str | Path,
    cache_dir: str | Path,
    learning_root: str | Path | None = None,
    parity: bool = True,
) -> dict[str, Any]:
    """Full index step: registry, DB, parity gate. Returns a summary."""
    results_files = _results_glob_paths(results_dir, results_glob)
    if not results_files:
        raise SystemExit(
            f"no result files found under {results_dir} matching {results_glob!r}; "
            "run 'benchmark' first"
        )
    registry_counts = build_registry(
        instances_dir=instances_dir,
        results_files=results_files,
        registry_dir=registry_dir,
    )
    drift = verify_registry(
        registry_dir, instances_dir=instances_dir, results_files=results_files
    )
    db_counts = build_db(
        results_dir=results_dir,
        results_glob=results_glob,
        instances_dir=instances_dir,
        registry_dir=registry_dir,
        db_path=db_path,
        cache_dir=cache_dir,
        learning_root=learning_root,
    )
    summary: dict[str, Any] = {
        "registry": registry_counts,
        "drift": drift,
        "db": db_counts,
    }
    if parity:
        from algofinder.harness.benchmark import BenchmarkRun
        from algofinder.harness.leaderboard import summarize

        runs = [BenchmarkRun(**mapping) for path in results_files for mapping in json.loads(Path(path).read_text())["runs"]]
        py_rows = summarize(runs)
        with duckdb.connect(str(db_path)) as con:
            diffs = parity_check(py_rows, con, results_files)
        summary["parity"] = {"rows": len(py_rows), "diffs": diffs}
    return summary


__all__ = [
    "build_db",
    "build_registry",
    "index_all",
    "verify_registry",
]
