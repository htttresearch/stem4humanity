"""Deterministic 3D generator family: regular tetrahedra on a sphere.

Phase 2 item 4: tetrahedron acquisition/generation with construction
parameters. The family exercises the first 3D objective mode
(``tsplib_euc_3d@1``) in the semantic corpus.

Construction: the four vertices of a regular tetrahedron inscribed in a
sphere of radius ``radius``, then a fixed deterministic rotation and city
permutation drawn from ``numpy.default_rng(seed)`` (PRNG version recorded).

Reference semantics:

- regular tetrahedra: every tour visits four equal edges, so every tour is
  optimal under both ``raw_l2_f64`` and ``tsplib_euc_3d``; the held-karp
  tour certifies both modes;
- ``box5`` (seeded random points in a box): the held-karp tour certifies
  ``raw_l2_f64`` only; under ``tsplib_euc_3d`` the same tour is imported as
  an upper bound, never as an optimum, because rounding can shift the
  optimum.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from algofinder.benchmark.digests import (
    coordinate_parent_id,
    instance_id,
    lineage_group_id,
    objective_sibling_group_id,
)
from algofinder.benchmark.references import Reference, ReferenceRegistry
from algofinder.benchmark.scoring import score_tour
from algofinder.instances.base import save_manifest
from algofinder.problems.base import Instance
from algofinder.problems.tsp import TravellingSalespersonProblem

EUC_3D = "tsplib_euc_3d@1"
RAW_L2 = "raw_l2_f64@1"
PRNG = "numpy.default_rng(v1)"
GENERATOR_VERSION = "tetrahedron-regular-v1"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "instances" / "tsp_tetrahedra_manifests.json"


def regular_tetrahedron(radius: float, seed: int) -> NDArray[np.float64]:
    """Four vertices of a regular tetrahedron inscribed in a sphere."""
    base = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=float,
    )
    base = base / np.linalg.norm(base, axis=1, keepdims=True) * radius
    rng = np.random.default_rng(seed)
    rotation = rng.normal(size=(3, 3))
    rotation, _ = np.linalg.qr(rotation)
    rotated = base @ rotation.T
    permutation = rng.permutation(4)
    return np.asarray(rotated[permutation], dtype=float)


def box_points(count: int, seed: int) -> NDArray[np.float64]:
    """Seeded random points in the unit box (irregular 3D case)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(count, 3))


def _exact(raw_points: NDArray[np.float64]) -> tuple[float, list[int]] | None:
    """Exact optimum under raw L2 via held-karp on the distance matrix."""
    import algofinder.solvers.registry  # noqa: F401
    from algofinder.solvers.base import all_solvers

    distances = np.linalg.norm(
        raw_points[:, np.newaxis, :] - raw_points[np.newaxis, :, :], axis=2
    )
    problem = TravellingSalespersonProblem("tetra", distances)
    try:
        result = all_solvers()["held-karp"]().solve(problem, budget_seconds=None)
    except Exception:
        return None
    if not result.exact:
        return None
    return float(result.cost), [int(city) for city in result.solution]


def _references_for(
    points: NDArray[np.float64],
    instance_digest: str,
    *,
    all_tours_equal: bool,
) -> tuple[list[Reference], float | None]:
    """Build reference records; returns (records, best_known)."""
    exact = _exact(points)
    if exact is None:
        return [], None
    cost, tour = exact
    provenance = {
        "source_url": "generated:tetrahedron",
        "source_name": GENERATOR_VERSION,
        "retrieved_at": "2026-08-02",
    }
    records = [
        Reference(
            instance_id=instance_id(points, RAW_L2),
            objective_spec_id=RAW_L2,
            kind="certified_optimum",
            value=float(cost),
            provenance=dict(provenance),
            status="verified",
            tour=tuple(tour),
            proof={
                "method": "held_karp_exact",
                "artifact_digest": None,
                "independent_verifier": "algofinder.benchmark.scoring",
            },
        )
    ]
    euc_3d = score_tour(points, tour, EUC_3D)
    if all_tours_equal:
        records.append(
            Reference(
                instance_id=instance_digest,
                objective_spec_id=EUC_3D,
                kind="certified_optimum",
                value=float(euc_3d),
                provenance=dict(provenance),
                status="verified",
                tour=tuple(tour),
                proof={
                    "method": "held_karp_exact_plus_all_edges_equal",
                    "artifact_digest": None,
                    "independent_verifier": "algofinder.benchmark.scoring",
                },
            )
        )
        return records, float(euc_3d)
    records.append(
        Reference(
            instance_id=instance_digest,
            objective_spec_id=EUC_3D,
            kind="upper_bound",
            value=float(euc_3d),
            provenance=dict(provenance),
            status="imported_unverified",
            tour=tuple(tour),
            proof={"method": "raw_l2_optimal_tour", "artifact_digest": None, "independent_verifier": None},
        )
    )
    return records, float(cost)


def generate(output_path: str | Path = MANIFEST_PATH) -> list[Instance]:
    instances: list[Instance] = []
    registry = ReferenceRegistry()

    for radius in (1.0, 2.5, 0.5):
        points = regular_tetrahedron(radius, seed=7)
        assert np.allclose(
            np.linalg.norm(points, axis=1), radius
        ), "vertices must lie on the sphere"
        coordinate_parent = coordinate_parent_id(points)
        instance_digest = instance_id(points, EUC_3D)
        records, best_known = _references_for(points, instance_digest, all_tours_equal=True)
        instances.append(
            Instance(
                name=f"tsp:tetrahedron:{radius:.2f}",
                problem="tsp",
                subproblem="euclidean",
                family="tetrahedron",
                seed=7,
                data={
                    "points": points.tolist(),
                    "dimension": 3,
                    "metric": "EUC_3D",
                    "construction": {
                        "family": "tetrahedron_regular",
                        "radius": radius,
                        "seed": 7,
                        "prng": PRNG,
                        "generator_version": GENERATOR_VERSION,
                    },
                    "lineage_group_id": lineage_group_id(f"tetrahedron:{radius}"),
                    "coordinate_parent_id": coordinate_parent,
                    "objective_sibling_group_id": objective_sibling_group_id(points),
                    "instance_id": instance_digest,
                },
                best_known=best_known,
                split="test",
            )
        )
        added = registry.append_many(records)
        print(f"tetrahedron r={radius:.2f} opt3d={best_known} refs=+{added}")

    points = box_points(5, seed=11)
    instance_digest = instance_id(points, EUC_3D)
    records, best_known = _references_for(points, instance_digest, all_tours_equal=False)
    instances.append(
        Instance(
            name="tsp:tetrahedron:box5",
            problem="tsp",
            subproblem="euclidean",
            family="tetrahedron",
            seed=11,
            data={
                "points": points.tolist(),
                "dimension": 3,
                "metric": "EUC_3D",
                "construction": {
                    "family": "box_random",
                    "count": 5,
                    "seed": 11,
                    "prng": PRNG,
                    "generator_version": GENERATOR_VERSION,
                },
                "lineage_group_id": lineage_group_id("tetrahedron:box5"),
                "coordinate_parent_id": coordinate_parent_id(points),
                "objective_sibling_group_id": objective_sibling_group_id(points),
                "instance_id": instance_digest,
            },
            best_known=best_known,
            split="test",
        )
    )
    added = registry.append_many(records)
    print(f"tetrahedron box5 raw_l2_opt={best_known} euc3d=upper_bound refs=+{added}")

    save_manifest(instances, output_path)
    registry.write_snapshot()
    print(f"wrote {len(instances)} tetrahedron instances to {output_path}")
    return instances


if __name__ == "__main__":
    generate()
