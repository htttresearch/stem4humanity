"""Build immutable, split-specific manifests for real agent experiments."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from algofinder.benchmark.compiler import compile_cell
from algofinder.benchmark.suites import _ml_ood_cells
from algofinder.instances.base import load_manifest, save_manifest
from algofinder.problems.base import Instance


HOLDOUT_SPLITS = ("test_iid", "test_family_ood", "test_parameter_ood")
DEVELOPMENT_SPLITS = ("validation",)


def prepare_three_model_manifests(
    *,
    source_manifest: Path,
    output_dir: Path,
    seed_offset: int = 8_220_000,
    suite_name: str = "ml_ood_three_model_confirmatory_20260822",
) -> dict[str, object]:
    """Freeze public development data and newly sampled held-out test data.

    Only the pre-existing validation split is copied into the agent-visible
    manifest.  Test cells are regenerated from the registered suite
    definitions with a preregistered seed offset and a new lineage namespace.
    """
    source_manifest = source_manifest.resolve()
    output_dir = output_dir.resolve()
    if seed_offset <= 0:
        raise ValueError("seed_offset must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)

    source = load_manifest(source_manifest)
    development = [instance for instance in source if instance.split == "validation"]
    if not development:
        raise RuntimeError("source manifest has no validation instances")

    holdout: list[Instance] = []
    selected_cells: list[dict[str, object]] = []
    for cell_index, cell in enumerate(_ml_ood_cells()):
        if cell.split not in HOLDOUT_SPLITS:
            continue
        shifted = replace(cell, seed_base=cell.seed_base + seed_offset)
        holdout.extend(compile_cell(shifted, cell_index, suite_name))
        selected_cells.append({
            "cell_index": cell_index,
            "generator": shifted.generator,
            "params": shifted.params,
            "count": shifted.count,
            "seed_base": shifted.seed_base,
            "split": shifted.split,
        })

    expected_counts = {"test_iid": 42, "test_family_ood": 36, "test_parameter_ood": 36}
    observed_counts = {
        split: sum(instance.split == split for instance in holdout)
        for split in HOLDOUT_SPLITS
    }
    if observed_counts != expected_counts:
        raise RuntimeError(f"unexpected held-out split counts: {observed_counts}")

    source_ids = {_instance_id(instance) for instance in source}
    development_ids = {_instance_id(instance) for instance in development}
    holdout_ids = {_instance_id(instance) for instance in holdout}
    if len(development_ids) != len(development):
        raise RuntimeError("development manifest contains duplicate canonical instances")
    if len(holdout_ids) != len(holdout):
        raise RuntimeError("fresh holdout contains duplicate canonical instances")
    if holdout_ids & source_ids:
        raise RuntimeError("fresh holdout overlaps a previously compiled source instance")
    development_lineages = {_lineage(instance) for instance in development}
    holdout_lineages = {_lineage(instance) for instance in holdout}
    if development_lineages & holdout_lineages:
        raise RuntimeError("development and fresh holdout lineage identifiers overlap")

    development_path = output_dir / "development-validation.json"
    holdout_path = output_dir / "confirmatory-holdout.json"
    save_manifest(development, development_path)
    save_manifest(holdout, holdout_path)
    metadata: dict[str, object] = {
        "schema_id": "algofinder.agents.experiment-manifests",
        "schema_version": "1",
        "source_manifest": str(source_manifest),
        "source_manifest_digest": _digest(source_manifest),
        "suite_name": suite_name,
        "seed_offset": seed_offset,
        "development": {
            "path": str(development_path),
            "digest": _digest(development_path),
            "splits": {"validation": len(development)},
        },
        "holdout": {
            "path": str(holdout_path),
            "digest": _digest(holdout_path),
            "splits": observed_counts,
            "overlap_with_source_instances": 0,
            "overlap_with_development_lineages": 0,
        },
        "cells": selected_cells,
    }
    metadata_path = output_dir / "manifest-provenance.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**metadata, "metadata_path": str(metadata_path)}


def prepare_fresh_learning_manifests(
    *,
    source_manifest: Path,
    output_dir: Path,
    seed_offset: int,
    suite_name: str,
) -> dict[str, object]:
    """Compile a fresh development/holdout pair for policy evaluation.

    Unlike :func:`prepare_three_model_manifests`, this function also
    regenerates the validation cells.  It is intended for prospective tests of
    a policy trained on the earlier three-model comparison: neither policy arm
    has seen any exact development or held-out instance in this pair.
    """
    source_manifest = source_manifest.resolve()
    output_dir = output_dir.resolve()
    if seed_offset <= 0:
        raise ValueError("seed_offset must be positive")
    if not suite_name.strip():
        raise ValueError("suite_name is required")
    output_dir.mkdir(parents=True, exist_ok=False)

    source = load_manifest(source_manifest)
    development: list[Instance] = []
    holdout: list[Instance] = []
    selected_cells: list[dict[str, object]] = []
    for cell_index, cell in enumerate(_ml_ood_cells()):
        if cell.split not in (*DEVELOPMENT_SPLITS, *HOLDOUT_SPLITS):
            continue
        shifted = replace(cell, seed_base=cell.seed_base + seed_offset)
        compiled = compile_cell(shifted, cell_index, suite_name)
        if shifted.split in DEVELOPMENT_SPLITS:
            development.extend(compiled)
        else:
            holdout.extend(compiled)
        selected_cells.append({
            "cell_index": cell_index,
            "generator": shifted.generator,
            "params": shifted.params,
            "count": shifted.count,
            "seed_base": shifted.seed_base,
            "split": shifted.split,
        })

    observed_development = {split: sum(item.split == split for item in development) for split in DEVELOPMENT_SPLITS}
    observed_holdout = {split: sum(item.split == split for item in holdout) for split in HOLDOUT_SPLITS}
    if observed_development != {"validation": 28}:
        raise RuntimeError(f"unexpected fresh development split counts: {observed_development}")
    expected_holdout = {"test_iid": 42, "test_family_ood": 36, "test_parameter_ood": 36}
    if observed_holdout != expected_holdout:
        raise RuntimeError(f"unexpected fresh holdout split counts: {observed_holdout}")

    source_ids = {_instance_id(item) for item in source}
    development_ids = {_instance_id(item) for item in development}
    holdout_ids = {_instance_id(item) for item in holdout}
    if len(development_ids) != len(development) or len(holdout_ids) != len(holdout):
        raise RuntimeError("fresh manifests contain duplicate canonical instances")
    if development_ids & source_ids or holdout_ids & source_ids:
        raise RuntimeError("fresh manifests overlap a previously compiled source instance")
    if development_ids & holdout_ids:
        raise RuntimeError("fresh development and holdout instances overlap")
    development_lineages = {_lineage(item) for item in development}
    holdout_lineages = {_lineage(item) for item in holdout}
    if development_lineages & holdout_lineages:
        raise RuntimeError("fresh development and holdout lineage identifiers overlap")

    development_path = output_dir / "development-validation.json"
    holdout_path = output_dir / "confirmatory-holdout.json"
    save_manifest(development, development_path)
    save_manifest(holdout, holdout_path)
    metadata: dict[str, object] = {
        "schema_id": "algofinder.agents.fresh-learning-manifests",
        "schema_version": "1",
        "source_manifest": str(source_manifest),
        "source_manifest_digest": _digest(source_manifest),
        "suite_name": suite_name,
        "seed_offset": seed_offset,
        "development": {
            "path": str(development_path),
            "digest": _digest(development_path),
            "splits": observed_development,
        },
        "holdout": {
            "path": str(holdout_path),
            "digest": _digest(holdout_path),
            "splits": observed_holdout,
        },
        "overlap": {
            "source_instances": 0,
            "development_holdout_instances": 0,
            "development_holdout_lineages": 0,
        },
        "cells": selected_cells,
    }
    metadata_path = output_dir / "manifest-provenance.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**metadata, "metadata_path": str(metadata_path)}


def _instance_id(instance: Instance) -> str:
    value = instance.data.get("instance_id")
    if not isinstance(value, str):
        raise RuntimeError(f"instance has no canonical instance_id: {instance.name}")
    return value


def _lineage(instance: Instance) -> str:
    value = instance.data.get("lineage_group_id")
    if not isinstance(value, str):
        raise RuntimeError(f"instance has no lineage_group_id: {instance.name}")
    return value


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


__all__ = [
    "DEVELOPMENT_SPLITS",
    "HOLDOUT_SPLITS",
    "prepare_fresh_learning_manifests",
    "prepare_three_model_manifests",
]
