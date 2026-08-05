"""ASlib scenario export for the benchmark corpus (spec section 13.6).

Writes an ASlib-format scenario under
``public/benchmarks/aslib/<scenario>/``::

    scenario.txt         metadata (algorithms, features, folds)
    description.txt      human-readable provenance
    algorithm_runs.arff  performance (minimized cost) per instance/algorithm
    feature_values.arff  flattened feature values per instance
    feature_costs.arff   feature computation cost (runtime, memory)
    cv.arff              fold assignments (leakage-safe vs instance split)

Fold assignment is split-derived so no fold mixes train and test
instances: train-split instances are fold 0 and test-split instances are
fold 1.  Missing values are encoded as ``?`` per ASlib convention.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from algofinder.features import FEATURE_SETS, instance_id
from algofinder.features.storage import read_feature_set
from algofinder.harness.benchmark import BenchmarkRun
from algofinder.instances.base import load_manifest
from algofinder.store.canonical import find_instances_manifests

SCENARIO_NAME = "algofinder"


def _arff(relation: str, attributes: list[tuple[str, str]], rows: list[list[Any]]) -> str:
    lines = [f"@relation {relation}", ""]
    for name, kind in attributes:
        lines.append(f"@attribute {name} {kind}")
    lines.append("")
    lines.append("@data")
    for row in rows:
        lines.append(",".join("?" if value is None else str(value) for value in row))
    return "\n".join(lines) + "\n"


def _flatten(values: dict[str, Any], prefix: str = "") -> list[tuple[str, float]]:
    flat: list[tuple[str, float]] = []
    for key, value in sorted(values.items()):
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.extend(_flatten(value, f"{name}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat.append((name, float(value)))
    return flat


def export_aslib(
    runs: Iterable[BenchmarkRun],
    instances_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    runs = list(runs)
    ok = [run for run in runs if run.status == "ok" and run.cost is not None]
    instance_names = sorted({run.instance for run in ok})
    algorithms = sorted({run.solver for run in ok})

    digest_by_name = {
        instance.name: instance_id(instance)
        for path in find_instances_manifests(instances_dir)
        for instance in load_manifest(path)
    }
    name_by_digest = {digest: name for name, digest in digest_by_name.items()}

    performance: dict[str, dict[str, float | None]] = {
        name: {solver: None for solver in algorithms} for name in instance_names
    }
    for run in ok:
        performance[run.instance][run.solver] = float(run.cost)

    feature_columns: list[tuple[str, str]] = []
    feature_values: dict[str, dict[str, float | None]] = {
        name: {} for name in instance_names
    }
    for feature_set in FEATURE_SETS:
        for record in read_feature_set(instances_dir, feature_set.id):
            if "error" in record.get("status_by_feature", {}):
                continue
            digest = record.get("instance_id")
            name = name_by_digest.get(digest)
            if name is None or name not in feature_values:
                continue
            for key, value in _flatten(record.get("values", {}), f"{feature_set.id}."):
                column = f"f_{key}"
                feature_columns.append((column, "NUMERIC"))
                feature_values[name][column] = value
    column_names = [name for name, _ in feature_columns]

    feature_cost_rows: list[list[Any]] = []
    cost_attrs: list[tuple[str, str]] = []
    for feature_set in FEATURE_SETS:
        records = read_feature_set(instances_dir, feature_set.id)
        timing = {
            name_by_digest.get(record["instance_id"]): record.get("timing_seconds")
            for record in records
            if record["instance_id"] in name_by_digest
        }
        rss = {
            name_by_digest.get(record["instance_id"]): record.get("peak_rss_bytes")
            for record in records
            if record["instance_id"] in name_by_digest
        }
        cost_attrs.append((f"runtime_{feature_set.id}", "NUMERIC"))
        cost_attrs.append((f"memory_{feature_set.id}", "NUMERIC"))
        feature_cost_rows.append((timing, rss))

    cost_rows: list[list[Any]] = []
    for name in instance_names:
        row: list[Any] = []
        for timing, rss in feature_cost_rows:
            row.append(timing.get(name))
            row.append(rss.get(name))
        cost_rows.append(row)

    fold_rows: list[list[Any]] = []
    folds: dict[str, int] = {}
    for name in instance_names:
        split = next((run.split for run in runs if run.instance == name), "test")
        folds[name] = 0 if split == "train" else 1
    fold_rows = [[name, folds[name]] for name in instance_names]

    algorithm_rows: list[list[Any]] = []
    for name in instance_names:
        algorithm_rows.append([performance[name][solver] for solver in algorithms])

    feature_rows: list[list[Any]] = []
    for name in instance_names:
        feature_rows.append([feature_values[name].get(key) for key in column_names])

    output = Path(output_dir) / SCENARIO_NAME
    output.mkdir(parents=True, exist_ok=True)
    (output / "scenario.txt").write_text(
        "\n".join(
            [
                f"algorithm_cutoff_time = 60.0",
                f"algorithm_cutoff_memory = 1073741824",
                f"extended_feature_set = true",
                f"scenario = {SCENARIO_NAME}",
                f"feature_steps = {','.join(feature_set.id for feature_set in FEATURE_SETS)}",
                f"performance_measure = minimized_solution_quality",
                f"number_of_instances = {len(instance_names)}",
                f"number_of_algorithms = {len(algorithms)}",
                f"number_of_features = {len(feature_columns)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "description.txt").write_text(
        "algofinder benchmark corpus: instance features (tier-0 params@1, "
        "tier-1 etsp-geometry@1) and minimized solution cost per algorithm. "
        "Folds are split-derived (0=train, 1=test) and never mix splits.\n",
        encoding="utf-8",
    )
    (output / "algorithm_runs.arff").write_text(
        _arff(
            f"{SCENARIO_NAME}_algorithm_runs",
            [("instance_id", "STRING")] + [(solver, "NUMERIC") for solver in algorithms],
            [[name] + row for name, row in zip(instance_names, algorithm_rows)],
        ),
        encoding="utf-8",
    )
    (output / "feature_values.arff").write_text(
        _arff(
            f"{SCENARIO_NAME}_feature_values",
            [("instance_id", "STRING")] + feature_columns,
            [[name] + row for name, row in zip(instance_names, feature_rows)],
        ),
        encoding="utf-8",
    )
    (output / "feature_costs.arff").write_text(
        _arff(
            f"{SCENARIO_NAME}_feature_costs",
            [("instance_id", "STRING")] + cost_attrs,
            [[name] + row for name, row in zip(instance_names, cost_rows)],
        ),
        encoding="utf-8",
    )
    (output / "cv.arff").write_text(
        _arff(
            f"{SCENARIO_NAME}_cv",
            [("instance_id", "STRING"), ("fold", "NUMERIC")],
            fold_rows,
        ),
        encoding="utf-8",
    )
    return {
        "scenario": SCENARIO_NAME,
        "directory": str(output),
        "instances": len(instance_names),
        "algorithms": len(algorithms),
        "features": len(feature_columns),
    }


__all__ = ["SCENARIO_NAME", "export_aslib"]
