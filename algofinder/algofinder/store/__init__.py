"""Canonical store, registry, and DuckDB query layer (Design A).

Three-tier discipline:

1. ``canonical`` — the immutable system of record: instance manifests,
   benchmark run records, reference registries (files under ``public/``
   and ``private/sessions/``). Nothing derived is ever written back.
2. ``registry`` — an append-only inventory (JSONL) of what exists:
   instances, solvers, runs, environments, with content digests.
3. ``indexer``/``queries`` — the query layer: a gitignored DuckDB file
   holding *views* over the canonical files plus regenerable Parquet
   caches. The DB never owns data; it references canonical bytes.
"""

from algofinder.store.canonical import (
    PROJECT_ROOT,
    SCHEMA_VERSION,
    find_instances_manifests,
    find_results_files,
    references_paths,
    run_digest,
    sha256_file,
)
from algofinder.store.registry import (
    Registry,
    read_kind,
    write_kind,
)
from algofinder.store.indexer import build_db, build_registry
from algofinder.store.queries import (
    leaderboard_rows,
    parity_compare,
    parity_check,
)

__all__ = [
    "PROJECT_ROOT",
    "Registry",
    "SCHEMA_VERSION",
    "build_db",
    "build_registry",
    "find_instances_manifests",
    "find_results_files",
    "leaderboard_rows",
    "parity_check",
    "parity_compare",
    "read_kind",
    "references_paths",
    "run_digest",
    "sha256_file",
    "write_kind",
]
