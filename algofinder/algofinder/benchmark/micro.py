"""``semantic_micro`` correctness corpus and validation report (catalog 3, spec Phase 1).

The corpus is intentionally easy to solve and difficult to parse incorrectly:
small instances covering degeneracy, ties, and objective-semantics traps, each
with metamorphic siblings (translation, reflection, rotation, scale, city
permutation) whose optima must relate to the parent in a known way.

Running this module::

    python -m algofinder.benchmark.micro

builds the corpus, computes exact references with Held-Karp under every
first-release objective spec, verifies every metamorphic relation, writes the
manifest, and exits non-zero if anything fails.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from algofinder.benchmark.objectives import (
    all_objectives,
    get_objective,
    verify_conformance,
)
from algofinder.benchmark.sources import CEIL_2D, EUC_2D, RAW_L2
from algofinder.benchmark.scoring import (
    apply_operation,
    check_relation,
    score_tour,
    verify_feasible,
)
from algofinder.instances.base import save_manifest
from algofinder.problems.base import Instance
from algofinder.problems.tsp import EuclideanTravellingSalespersonProblem

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "instances" / "tsp_micro_manifests.json"

# Lattice-preserving sibling operations (catalog 3.2: integer-coordinate
# rotations that do not preserve the lattice are NOT EUC_2D siblings).
EUCLIDEAN_SIBLING_OPS = ("identity", "translate_int", "reflect_x")

RAW_L2_SIBLING_OPS = ("identity", "translate", "reflect_x", "rotate_0.7", "scale_2.5")

CONFIGURATIONS: dict[str, list[tuple[float, float]]] = {
    "triangle": [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)],
    "square": [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
    "rectangle": [(0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (0.0, 1.0)],
    "regular_pentagon": [
        (np.cos(2 * np.pi * k / 5), np.sin(2 * np.pi * k / 5)) for k in range(5)
    ],
    "convex_random": [
        (0.1, 0.1), (0.8, 0.2), (0.9, 0.9), (0.3, 1.0), (0.0, 0.6), (0.05, 0.3)
    ],
    "line_horizontal": [(k * 1.0, 0.0) for k in range(5)],
    "line_vertical": [(0.0, k * 1.0) for k in range(5)],
    "all_coincident": [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
    "partial_duplicates": [(0.0, 0.0), (0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
    "collinear_run": [(k * 0.5, 0.25) for k in range(6)],
    "cocircular": [
        (2 * np.cos(2 * np.pi * k / 6), 2 * np.sin(2 * np.pi * k / 6))
        for k in range(6)
    ],
    "integer_grid_3x3": [(i, j) for i in range(3) for j in range(3)],
    "one_interior_point": [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (1.5, 1.5)],
    "nested_layers": [
        (0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0),
        (2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0),
    ],
    "extreme_aspect": [(0.0, 0.0), (10000.0, 0.0), (10000.0, 1.0), (0.0, 1.0)],
}

NOTES = {
    "triangle": "smallest nontrivial polygon",
    "square": "ties everywhere under L2",
    "rectangle": "many degenerate L2 alternatives",
    "regular_pentagon": "rotation symmetry",
    "convex_random": "general convex position",
    "line_horizontal": "zero-area, unique tour",
    "line_vertical": "zero-area, vertical",
    "all_coincident": "fully degenerate",
    "partial_duplicates": "exact duplicate cities",
    "collinear_run": "collinear points",
    "cocircular": "many optimal tours",
    "integer_grid_3x3": "repeated distances; modes differ (raw/euc/ceil)",
    "one_interior_point": "interior city",
    "nested_layers": "nested convex layers",
    "extreme_aspect": "extreme aspect ratio",
}


def _integer_points(points: NDArray[np.float64]) -> bool:
    return bool(np.all(np.asarray(points) == np.round(np.asarray(points))))


def _siblings(points: NDArray[np.float64], ops: tuple[str, ...]) -> dict[str, NDArray[np.float64]]:
    siblings: dict[str, NDArray[np.float64]] = {}
    for operation in ops:
        if operation == "translate_int":
            transformed = points + np.asarray([2.0, -1.0])
        elif operation == "translate":
            transformed = apply_operation(points, "translate")
        else:
            transformed = apply_operation(points, operation)
        siblings[operation] = np.asarray(transformed, dtype=float)
    return siblings


def _exact_optimum(
    points: NDArray[np.float64], spec_id: str
) -> tuple[float, list[int]] | None:
    """Held-Karp exact optimum scored under ``spec_id``; None if infeasible."""
    try:
        import algofinder.solvers.registry  # noqa: F401
        from algofinder.solvers.base import InapplicableError, all_solvers

        solver_cls = all_solvers()["held-karp"]
    except KeyError:
        raise SystemExit("held-karp solver not registered") from None
    if points.shape[0] > 12:
        return None
    problem = EuclideanTravellingSalespersonProblem("micro", points)
    try:
        result = solver_cls().solve(problem, budget_seconds=None)
    except InapplicableError:
        return None
    except Exception:
        return None
    if not result.exact or not problem.verify(result.solution):
        return None
    tour = [int(city) for city in result.solution]
    scored = score_tour(points, tour, spec_id)
    return scored, tour


def _validate() -> list[str]:
    """Run the full micro validation; returns report lines."""
    lines: list[str] = []
    failures: list[str] = []

    conformance = verify_conformance()
    lines.append(f"objective conformance: {len(all_objectives())} specs, "
                 f"{len(conformance)} failures")
    failures.extend(conformance)

    grid_refs: dict[str, float | None] = {}
    for name, raw_points in CONFIGURATIONS.items():
        points = np.asarray(raw_points, dtype=float)
        integer = _integer_points(points)

        line = f"  {name:20s} n={points.shape[0]:2d} note={NOTES[name]}"
        for spec_id in (RAW_L2, EUC_2D, CEIL_2D):
            if not integer and spec_id != RAW_L2:
                continue
            optimum = _exact_optimum(points, spec_id)
            if optimum is None:
                line += f" | {spec_id}: n/a"
                continue
            value, tour = optimum
            if not verify_feasible(tour, points.shape[0]):
                failures.append(f"{name}/{spec_id}: infeasible exact tour")
                continue
            line += f" | {spec_id}: {value:g}"
            if name == "integer_grid_3x3":
                grid_refs[spec_id] = value
            for operation, sibling in _siblings(
                points,
                EUCLIDEAN_SIBLING_OPS if integer else (),
            ).items():
                sibling_opt = _exact_optimum(sibling, spec_id)
                if sibling_opt is None:
                    failures.append(f"{name}/{operation}: no exact sibling optimum")
                    continue
                if not check_relation(value, sibling_opt[0], operation):
                    failures.append(
                        f"{name}/{operation}/{spec_id}: relation broken: "
                        f"parent={value:g} sibling={sibling_opt[0]:g}"
                    )
        for operation, sibling in _siblings(points, RAW_L2_SIBLING_OPS).items():
            parent_opt = _exact_optimum(points, RAW_L2)
            sibling_opt = _exact_optimum(sibling, RAW_L2)
            if parent_opt is None or sibling_opt is None:
                failures.append(f"{name}/{operation}: exact refs missing")
                continue
            if not check_relation(parent_opt[0], sibling_opt[0], operation):
                failures.append(
                    f"{name}/{operation}/raw_l2: relation broken: "
                    f"parent={parent_opt[0]:.9g} sibling={sibling_opt[0]:.9g}"
                )
        lines.append(line)

    if grid_refs.get(EUC_2D) is not None and grid_refs.get(CEIL_2D) is not None:
        if abs(grid_refs[EUC_2D] - grid_refs[CEIL_2D]) < 0.5:
            failures.append("integer_grid_3x3: EUC_2D and CEIL_2D optima must differ")
        lines.append(
            f"  integer_grid_3x3 modes distinct: "
            f"raw={grid_refs[RAW_L2]:.6g} euc2d={grid_refs[EUC_2D]:g} "
            f"ceil2d={grid_refs[CEIL_2D]:g}"
        )

    lines.append(f"validation: {len(failures)} failures")
    lines.extend(f"  FAIL {failure}" for failure in failures)
    return lines


def _write_manifest() -> int:
    """Write harness-compatible micro manifest; returns instance count."""
    from algofinder.benchmark.digests import (
        coordinate_parent_id,
        instance_id,
        lineage_group_id,
        objective_sibling_group_id,
    )
    from algofinder.benchmark.references import Reference, ReferenceRegistry

    records: list[Instance] = []
    registry = ReferenceRegistry()
    provenance = {
        "source_url": "generated:semantic_micro",
        "source_name": "semantic_micro",
        "retrieved_at": "2026-08-02",
    }
    for name, raw_points in CONFIGURATIONS.items():
        points = np.asarray(raw_points, dtype=float)
        integer = _integer_points(points)
        raw_opt = _exact_optimum(points, RAW_L2)
        euc_opt = _exact_optimum(points, EUC_2D) if integer else None
        ceil_opt = _exact_optimum(points, CEIL_2D) if integer else None
        data: dict[str, object] = {
            "points": points.tolist(),
            "note": NOTES[name],
            "lineage_group": f"micro:{name}",
            "coordinate_parent_id": coordinate_parent_id(points),
            "objective_sibling_group_id": objective_sibling_group_id(points),
            "lineage_group_id": lineage_group_id(f"micro:{name}"),
            "instance_id": instance_id(points, EUC_2D),
            "instance_id_raw_l2": instance_id(points, RAW_L2),
        }
        if euc_opt is not None:
            data["objective_refs"] = {
                RAW_L2: raw_opt[0] if raw_opt else None,
                EUC_2D: euc_opt[0],
                CEIL_2D: ceil_opt[0] if ceil_opt else None,
            }
        references: list[Reference] = []
        for spec_id, optimum in (
            (RAW_L2, raw_opt),
            (EUC_2D, euc_opt),
            (CEIL_2D, ceil_opt),
        ):
            if optimum is None:
                continue
            value, tour = optimum
            references.append(
                Reference(
                    instance_id=data["instance_id"]
                    if spec_id == EUC_2D
                    else instance_id(points, spec_id),
                    objective_spec_id=spec_id,
                    kind="certified_optimum",
                    value=float(value),
                    provenance=dict(provenance),
                    status="verified",
                    tour=tuple(tour),
                    proof={
                        "method": "held_karp_exact",
                        "artifact_digest": None,
                        "independent_verifier": "algofinder.benchmark.scoring",
                    },
                )
            )
        added = registry.append_many(references)
        records.append(
            Instance(
                name=f"tsp:micro:{name}",
                problem="tsp",
                subproblem="euclidean",
                family="micro",
                seed=0,
                data=data,
                best_known=float(raw_opt[0]) if raw_opt else None,
                split="test",
            )
        )
        if added:
            print(f"micro {name:20s} refs=+{added}")
    save_manifest(records, MANIFEST_PATH)
    registry.write_snapshot()
    return len(records)


def main() -> int:
    lines = ["# semantic_micro validation", ""]
    lines.extend(_validate())
    count = _write_manifest()
    lines.append("")
    lines.append(f"manifest: {MANIFEST_PATH} ({count} instances)")
    print("\n".join(lines))
    return 0 if all("FAIL" not in line for line in lines) else 1


if __name__ == "__main__":
    raise SystemExit(main())
