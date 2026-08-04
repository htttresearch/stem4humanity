"""Suite compiler (benchmark spec section 7.3, phase 3).

Cells are declarative (generator, parameters, independent-sample count,
seed base, split role, weight, reference policy). Compiling a suite:

1. runs every cell deterministically (generator version + params + seed
   -> canonical coordinates + realized diagnostics);
2. records construction, lineage, and digest metadata on every instance;
3. verifies the section 7.3 checks: no missing/ambiguous objective spec,
   no duplicate canonical instance within a cell's independent samples,
   no lineage leakage across train/validation vs test;
4. produces references according to each cell's policy (exact-reference
   pipeline via held-karp / incremental-exact, or strong-ILS upper
   bounds), registered in the append-only registry and rescored through
   the shared objective oracle;
5. writes per-suite manifests plus a suite index, and prints a coverage
   summary (families, sizes, splits, weights).

Re-compiling the same suite definition must reproduce identical digests,
names, and weights (phase 3 exit criterion).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from algofinder.benchmark.digests import (
    coordinate_parent_id,
    instance_id,
    lineage_group_id,
    objective_sibling_group_id,
)
from algofinder.benchmark.references import Reference, ReferenceRegistry
from algofinder.benchmark.scoring import PARENT_OPS, score_tour
from algofinder.benchmark.sources import CEIL_2D, EUC_2D, RAW_L2
from algofinder.benchmark.strata import PRNG, STRATA_SCHEMA, get_generator
from algofinder.instances.base import save_manifest
from algofinder.problems.base import Instance
from algofinder.problems.tsp import EuclideanTravellingSalespersonProblem

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MANIFEST_DIR = DATA_DIR / "instances"
SUITE_INDEX_PATH = MANIFEST_DIR / "suite_index.json"

CLUSTERED_FAMILIES = frozenset({"gaussian_mixture", "disk_clusters"})
CREATED_AT = "2026-08-04"

FIT_ROLES = frozenset({"train", "validation"})
EVAL_SPLITS = frozenset(
    {"test_iid", "test_family_ood", "test_parameter_ood", "test_metric_ood"}
)


@dataclass(frozen=True)
class CellSpec:
    """One declarative suite cell (spec section 7.3)."""

    generator: str
    params: dict[str, object]
    count: int = 1
    seed_base: int = 0
    split: str = "test"
    weight: float = 1.0
    references: str = "none"  # none | exact | exact_or_ub | upper_bound
    ub_budget_seconds: float = 2.0
    transforms: tuple[str, ...] = ()
    family: str | None = None


@dataclass
class CompiledSuite:
    name: str
    cells: list[CellSpec]
    instances: list[Instance] = field(default_factory=list)
    lineage_roots: list[str] = field(default_factory=list)
    report: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def compile_cell(cell: CellSpec, cell_index: int, suite_name: str) -> list[Instance]:
    """Run one cell deterministically and build its manifest instances."""
    generator = get_generator(cell.generator)
    spec_id = f"{generator.family}@{generator.version}"
    params = dict(generator.default_params)
    params.update(cell.params)
    family = cell.family or generator.family
    subproblem = "clustered" if family in CLUSTERED_FAMILIES else "euclidean"
    tag = f"{family}-{cell_index}"
    lineage = lineage_group_id(f"strata:{suite_name}:{tag}")

    instances: list[Instance] = []
    for sample in range(cell.count):
        seed = cell.seed_base + sample
        points, realized = generator.run(params, seed)
        parent_id = coordinate_parent_id(points)
        data = _instance_data(
            suite_name, tag, points, params, seed, spec_id, realized,
            cell, cell_index, lineage, parent_id,
        )
        instances.append(_make_instance(subproblem, family, tag, sample, seed, cell, data))
        for operation in cell.transforms:
            sibling = np.asarray(PARENT_OPS[operation](points), dtype=float)
            sibling_data = _instance_data(
                suite_name, tag, sibling, params, seed, spec_id, realized,
                cell, cell_index, lineage, coordinate_parent_id(sibling),
                geometry_parent_id=parent_id, transform=operation,
            )
            instances.append(
                _make_instance(subproblem, family, tag, sample, seed, cell, sibling_data)
            )
    return instances


def _instance_data(
    suite_name: str,
    tag: str,
    points: NDArray[np.float64],
    params: dict[str, object],
    seed: int,
    spec_id: str,
    realized: dict[str, object],
    cell: CellSpec,
    cell_index: int,
    lineage: str,
    parent_id: str,
    geometry_parent_id: str | None = None,
    transform: str | None = None,
) -> dict[str, object]:
    return {
        "points": points.tolist(),
        "metric": "EUC_2D",
        "construction": {
            "generator": spec_id,
            "params": {key: value for key, value in params.items()},
            "seed": seed,
            "prng": PRNG,
            "strata_schema": STRATA_SCHEMA,
        },
        "realized": realized,
        "lineage_group": f"strata:{suite_name}:{tag}",
        "lineage_group_id": lineage,
        "coordinate_parent_id": parent_id,
        "geometry_parent_id": geometry_parent_id,
        "objective_sibling_group_id": objective_sibling_group_id(points),
        "instance_id": instance_id(points, EUC_2D),
        "instance_id_raw_l2": instance_id(points, RAW_L2),
        "instance_id_ceil_2d": instance_id(points, CEIL_2D),
        "suite": suite_name,
        "suite_cell_index": cell_index,
        "suite_weight": cell.weight,
        "split_role": cell.split,
        **({"transform": transform} if transform else {}),
    }


def _make_instance(
    subproblem: str,
    family: str,
    tag: str,
    sample: int,
    seed: int,
    cell: CellSpec,
    data: dict[str, object],
) -> Instance:
    operation = data.get("transform")
    suffix = f"-{operation}" if operation else ""
    name = f"tsp:{subproblem}:{family}:{tag}-s{seed}-i{sample}{suffix}"
    return Instance(
        name=name,
        problem="tsp",
        subproblem=subproblem,
        family=family,
        seed=seed,
        data=data,
        best_known=None,
        split=cell.split,
    )


def _check_lineage_leakage(suite: CompiledSuite) -> None:
    training = [i for i in suite.instances if (i.split or "test") in FIT_ROLES]
    evaluation = [i for i in suite.instances if (i.split or "test") in EVAL_SPLITS]
    train_lineages = {i.data["lineage_group_id"] for i in training}
    eval_lineages = {i.data["lineage_group_id"] for i in evaluation}
    leakage = train_lineages & eval_lineages
    if leakage:
        suite.failures.append(
            f"lineage leakage: {len(leakage)} lineage groups appear in both "
            f"train/validation and test splits"
        )


def _check_duplicate_samples(suite: CompiledSuite) -> None:
    seen: dict[str, str] = {}
    for instance in suite.instances:
        parent = str(instance.data["coordinate_parent_id"])
        if parent in seen:
            suite.failures.append(
                f"duplicate canonical instance: {seen[parent]} and {instance.name} "
                f"share coordinate_parent_id {parent[:12]}"
            )
        seen[parent] = instance.name


def _check_objective_specs(suite: CompiledSuite) -> None:
    for instance in suite.instances:
        recorded = instance.data.get("instance_id")
        expected = instance_id(
            np.asarray(instance.data["points"], dtype=float), EUC_2D
        )
        if recorded != expected:
            suite.failures.append(
                f"{instance.name}: recorded instance_id does not match "
                f"EUC_2D digest"
            )


def _exact_optimum(points: NDArray[np.float64], name: str) -> tuple[float, tuple[int, ...]] | None:
    """Exact tour via held-karp (n <= 18) or budgeted incremental-exact."""
    import algofinder.solvers.registry  # noqa: F401
    from algofinder.solvers.base import all_solvers

    problem = EuclideanTravellingSalespersonProblem(name, points)
    n = points.shape[0]
    for solver_id in ("held-karp", "incremental-exact"):
        solver_cls = all_solvers().get(solver_id)
        if solver_cls is None:
            continue
        solver = solver_cls()
        if n > getattr(solver, "max_cities", 0):
            continue
        budget = 18.0 if solver_id == "held-karp" else 60.0
        result = solver.solve(problem, budget_seconds=budget)
        if result.exact and problem.verify(result.solution):
            return float(result.cost), tuple(int(city) for city in result.solution)
    return None


def _upper_bound_tour(
    points: NDArray[np.float64], name: str, budget_seconds: float
) -> tuple[tuple[int, ...], float] | None:
    """Strong-ILS feasible tour; the bound is the RAW_L2 oracle score."""
    import algofinder.solvers.registry  # noqa: F401
    from algofinder.solvers.base import all_solvers

    problem = EuclideanTravellingSalespersonProblem(name, points)
    solver_cls = all_solvers().get("strong-ils-euclidean")
    if solver_cls is None:
        return None
    result = solver_cls().solve(problem, budget_seconds=budget_seconds)
    if not problem.verify(result.solution):
        return None
    tour = tuple(int(city) for city in result.solution)
    return tour, score_tour(points, tour, RAW_L2)


def _append_tour_references(
    registry: ReferenceRegistry,
    instance: Instance,
    points: NDArray[np.float64],
    tour: tuple[int, ...],
    provenance: dict[str, object],
    kind: str,
    status: str,
    proof: dict[str, object],
) -> None:
    for spec_id in (RAW_L2, EUC_2D):
        rescored = score_tour(points, list(tour), spec_id)
        registry.append(
            Reference(
                instance_id=instance_id(points, spec_id),
                objective_spec_id=spec_id,
                kind=kind,
                value=float(rescored),
                provenance=dict(provenance),
                status=status,
                tour=tour,
                proof=dict(proof),
                created_at=CREATED_AT,
            )
        )


def _reuse_record_value(
    index: dict[tuple[str, str], dict[str, object]],
    raw_id: str,
    kinds: tuple[str, ...],
) -> float | None:
    for kind in kinds:
        record = index.get((raw_id, kind))
        if record is not None and record["status"] not in ("superseded", "disputed"):
            return float(record["value"])
    return None


def _register_references(suite: CompiledSuite, registry: ReferenceRegistry) -> list[Instance]:
    """Fill best_known and append reference records; returns updated instances.

    Reference solving is cached: if the registry already holds an accepted
    record for an instance, the recorded value is reused and the solver is
    not run again.
    """
    all_records = registry.load_all()
    index = {
        (record["instance_id"], record["kind"]): record for record in all_records
    }
    updated: list[Instance] = []
    for instance in suite.instances:
        points = np.asarray(instance.data["points"], dtype=float)
        cell = suite.cells[int(instance.data["suite_cell_index"])]
        policy = cell.references
        if instance.data.get("transform") or policy == "none":
            updated.append(instance)
            continue
        canonical = instance_id(points, EUC_2D)
        raw_id = instance_id(points, RAW_L2)
        if policy == "upper_bound":
            cache_kinds = ("upper_bound",)
        elif policy == "exact":
            cache_kinds = ("certified_optimum",)
        else:
            cache_kinds = ("certified_optimum", "upper_bound")
        cached = _reuse_record_value(index, raw_id, cache_kinds)
        if cached is not None:
            updated.append(_with_best_known(instance, cached))
            continue
        provenance = {
            "source_url": f"generated:strata:{suite.name}",
            "source_name": f"strata:{suite.name}:{instance.family}",
            "retrieved_at": CREATED_AT,
        }
        exact = None
        if policy in ("exact", "exact_or_ub"):
            exact = _exact_optimum(points, instance.name)
            if exact is None and policy == "exact":
                suite.failures.append(
                    f"{instance.name}: exact reference required but infeasible"
                )
        if exact is not None:
            value, tour = exact
            _append_tour_references(
                registry, instance, points, tour, provenance,
                "certified_optimum", "verified",
                {"method": "held_karp_exact",
                 "independent_verifier": "algofinder.benchmark.scoring"},
            )
            updated.append(_with_best_known(instance, float(value)))
            continue
        ub = _upper_bound_tour(points, instance.name, cell.ub_budget_seconds)
        if ub is None:
            suite.failures.append(f"{instance.name}: upper bound tour failed")
            updated.append(instance)
            continue
        tour, raw_score = ub
        _append_tour_references(
            registry, instance, points, tour, provenance,
            "upper_bound", "imported_unverified",
            {"method": "strong_ils_euclidean",
             "budget_seconds": cell.ub_budget_seconds},
        )
        updated.append(_with_best_known(instance, float(raw_score)))
    return updated


def _with_best_known(instance: Instance, value: float) -> Instance:
    return Instance(
        name=instance.name,
        problem=instance.problem,
        subproblem=instance.subproblem,
        family=instance.family,
        seed=instance.seed,
        data=instance.data,
        best_known=value,
        split=instance.split,
    )


def compile_suite(name: str, cells: Iterable[CellSpec]) -> CompiledSuite:
    suite = CompiledSuite(name=name, cells=list(cells))
    for index, cell in enumerate(suite.cells):
        suite.instances.extend(compile_cell(cell, index, name))
    _check_lineage_leakage(suite)
    _check_duplicate_samples(suite)
    _check_objective_specs(suite)
    suite.lineage_roots = sorted({i.data["lineage_group_id"] for i in suite.instances})

    registry = ReferenceRegistry()
    suite.instances = _register_references(suite, registry)
    registry.write_snapshot()
    return suite


def suite_index(suites: list[CompiledSuite]) -> dict[str, object]:
    index: dict[str, object] = {"schema": "etsp.suite_index.v1"}
    for suite in suites:
        splits: dict[str, int] = {}
        weights: dict[str, float] = {}
        for instance in suite.instances:
            role = instance.split or "test"
            splits[role] = splits.get(role, 0) + 1
            weights[role] = weights.get(role, 0.0) + float(instance.data["suite_weight"])
        index[suite.name] = {
            "manifest": f"tsp_{suite.name}_manifests.json",
            "instances": len(suite.instances),
            "lineage_roots": len(suite.lineage_roots),
            "cells": len(suite.cells),
            "splits": splits,
            "split_weights": weights,
            "families": sorted({i.family for i in suite.instances}),
            "sizes": sorted({len(i.data["points"]) for i in suite.instances}),
        }
    return index


def _coverage_report(suite: CompiledSuite) -> list[str]:
    lines = [
        f"{suite.name}: {len(suite.instances)} instances, {len(suite.cells)} cells, "
        f"{len(suite.lineage_roots)} lineage roots"
    ]
    by_family: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for instance in suite.instances:
        by_family[instance.family] = by_family.get(instance.family, 0) + 1
        by_split[instance.split or "test"] = by_split.get(instance.split or "test", 0) + 1
    lines.append(
        "  splits: "
        + ", ".join(f"{role}={count}" for role, count in sorted(by_split.items()))
    )
    lines.append(
        "  families: "
        + ", ".join(f"{family}={count}" for family, count in sorted(by_family.items()))
    )
    return lines


def main() -> int:
    """Compile every pilot suite and write manifests plus the suite index."""
    from algofinder.benchmark.suites import ALL_SUITES

    compiled: list[CompiledSuite] = []
    lines: list[str] = ["# phase 3 suite compilation", ""]
    failures: list[str] = []
    for name, cells in ALL_SUITES.items():
        suite = compile_suite(name, cells)
        compiled.append(suite)
        failures.extend(suite.failures)
        lines.extend(_coverage_report(suite))
        lines.extend(f"  FAIL {failure}" for failure in suite.failures)

    for suite in compiled:
        save_manifest(
            suite.instances, MANIFEST_DIR / f"tsp_{suite.name}_manifests.json"
        )

    SUITE_INDEX_PATH.write_text(
        json.dumps(suite_index(compiled), indent=2, sort_keys=True) + "\n"
    )

    lines.append("")
    lines.append(f"suite index: {SUITE_INDEX_PATH}")
    lines.append(f"compilation: {len(failures)} failures")
    lines.extend(f"  FAIL {failure}" for failure in failures)
    print("\n".join(lines))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
