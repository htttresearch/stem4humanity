"""Pilot suite definitions (benchmark spec section 13, phase 3).

Three suites exercising the built-in generator catalog:

- ``exact_structure``: small instances (n <= 25) across every structural
  family, every instance with a certified optimum from the exact-reference
  pipeline (held-karp / incremental-exact);
- ``anytime_quality``: mid-size instances (n = 100/300/500) with strong-ILS
  upper bounds for anytime-performance ranking;
- ``ml_ood``: train/validation/iid-test/family-ood/parameter-ood splits for
  the learned-solver track; lineage groups never cross fit/test roles.

Compile with ``python -m algofinder.benchmark.compiler``.
"""

from __future__ import annotations

from algofinder.benchmark.compiler import CellSpec

# ---------------------------------------------------------------------------
# exact_structure: certified optima via held-karp (n <= 18) and
# incremental-exact (n = 20/25); no lineage transforms in the exact suite.
# ---------------------------------------------------------------------------

EXACT_PARAMS: dict[str, dict[str, object]] = {
    "uniform_rect": {},
    "gaussian_mixture": {
        "cluster_count": 3,
        "within_std": 0.05,
        "center_min_separation": 0.3,
    },
    "grid_jitter": {"jitter": 0.02},
    "convex_k_inner": {"hull_shape": "square", "inner_scale": 0.8},
    "narrow_strip": {"width": 0.05},
    "rings": {"rings": 2, "radius_step": 1.0},
    "duplicates": {"duplicate_fraction": 0.2},
    "collinear_run": {"jitter": 0.02},
    "cocircular": {"radius": 1.0},
    "poisson_disc": {"min_separation": 0.08},
    "onion_layers": {},
    "parallel_lines": {"lines": 3, "spacing": 0.5},
    "line": {"jitter": 0.05},
    "spiral": {"turns": 3.0, "noise": 0.02},
    "small_integer": {"bound": 50},
    "quantized": {"bit_depth": 6},
}

EXACT_SIZES: dict[str, list[tuple[int, int]]] = {
    "uniform_rect": [(10, 3), (14, 3), (18, 2)],
    "gaussian_mixture": [(10, 3), (14, 3), (18, 2)],
    "grid_jitter": [(9, 3), (16, 2)],
    "convex_k_inner": [(10, 3), (14, 3), (18, 2)],
    "narrow_strip": [(10, 3), (14, 3), (18, 2)],
    "rings": [(10, 3), (14, 3), (18, 2)],
    "duplicates": [(12, 3), (16, 2)],
    "collinear_run": [(10, 3), (16, 2)],
    "cocircular": [(10, 3), (14, 3)],
    "poisson_disc": [(12, 3), (18, 2)],
    "onion_layers": [(12, 3), (16, 2)],
    "parallel_lines": [(12, 3), (16, 2)],
    "line": [(10, 3), (16, 2)],
    "spiral": [(12, 3), (18, 2)],
    "small_integer": [(12, 3), (16, 2)],
    "quantized": [(12, 3), (16, 2)],
}


def _exact_cells() -> list[CellSpec]:
    cells: list[CellSpec] = []
    seed_base = 1000
    for family, sizes in EXACT_SIZES.items():
        for index, (n, count) in enumerate(sizes):
            params = dict(EXACT_PARAMS[family])
            if family == "grid_jitter":
                import math

                cols = int(math.ceil(math.sqrt(n)))
                rows = cols - 1
                while rows * cols < n:
                    rows += 1
                params.update({"rows": rows, "cols": cols})
            elif family == "onion_layers":
                params.update({"layers": max(2, n // 6), "per_layer": n // max(2, n // 6)})
            elif family == "convex_k_inner":
                params.update({"count": n, "k_inner": n - 4})
            else:
                params.update({"count": n})
            cells.append(
                CellSpec(
                    generator=family,
                    params=params,
                    count=count,
                    seed_base=seed_base + index * 17,
                    split="test",
                    weight=1.0,
                    references="exact",
                )
            )
    return cells


# ---------------------------------------------------------------------------
# anytime_quality: upper bounds via strong-ILS at matched wall time.
# ---------------------------------------------------------------------------

ANYTIME_FAMILIES: tuple[tuple[str, dict[str, object]], ...] = (
    ("uniform_rect", {}),
    ("gaussian_mixture", {"cluster_count": 4, "within_std": 0.04, "center_min_separation": 0.35}),
    ("grid_jitter", {"jitter": 0.1}),
    ("narrow_strip", {"width": 0.2}),
    ("rings", {"rings": 2, "radius_step": 1.0}),
    ("spiral", {"turns": 3.0, "noise": 0.02}),
)

ANYTIME_SIZES: tuple[tuple[int, int, float], ...] = (
    (100, 2, 2.0),
    (300, 2, 3.0),
    (500, 1, 6.0),
)


def _anytime_cells() -> list[CellSpec]:
    cells: list[CellSpec] = []
    seed_base = 3000
    index = 0
    for family, base_params in ANYTIME_FAMILIES:
        for n, count, budget in ANYTIME_SIZES:
            params = dict(base_params)
            if family == "grid_jitter":
                import math

                cols = int(math.ceil(math.sqrt(n)))
                rows = cols
                params.update({"rows": rows, "cols": cols})
                n_realized = rows * cols
            else:
                params.update({"count": n})
                n_realized = n
            cells.append(
                CellSpec(
                    generator=family,
                    params=params,
                    count=count,
                    seed_base=seed_base + index * 31,
                    split="test",
                    weight=1.0,
                    references="upper_bound",
                    ub_budget_seconds=budget,
                )
            )
            index += 1
    return cells


# ---------------------------------------------------------------------------
# ml_ood: group-aware splits; every cell is one lineage root that never
# crosses fit/test roles.
# ---------------------------------------------------------------------------

ML_OOD_FIT_FAMILIES: tuple[tuple[str, dict[str, object]], ...] = (
    ("uniform_rect", {}),
    ("gaussian_mixture", {"cluster_count": 3, "within_std": 0.04, "center_min_separation": 0.3}),
    ("grid_jitter", {"jitter": 0.05}),
    ("convex_k_inner", {"hull_shape": "square", "inner_scale": 0.8}),
    ("narrow_strip", {"width": 0.2}),
    ("rings", {"rings": 2, "radius_step": 1.0}),
    ("poisson_disc", {"min_separation": 0.09}),
)

ML_OOD_FAMILY_OOD: tuple[tuple[str, dict[str, object]], ...] = (
    ("line", {"jitter": 0.1}),
    ("cocircular", {"radius": 1.0}),
    ("spiral", {"turns": 3.0, "noise": 0.02}),
    ("disk_clusters", {"cluster_count": 4, "cluster_radius": 0.15, "separation": 0.3}),
    ("small_integer", {"bound": 100}),
    ("collinear_run", {"jitter": 0.05}),
)

ML_OOD_PARAM_OOD: tuple[tuple[str, dict[str, object]], ...] = (
    ("narrow_strip", {"width": 0.02}),
    ("gaussian_mixture", {"cluster_count": 8, "within_std": 0.015, "center_min_separation": 0.22}),
    ("grid_jitter", {"jitter": 0.4}),
    ("uniform_rect", {"aspect_ratio": 6.0}),
    ("rings", {"rings": 5, "radius_step": 0.6}),
    ("convex_k_inner", {"hull_shape": "circle", "inner_scale": 0.5}),
)

ML_OOD_TRANSFORMS: dict[str, tuple[str, ...]] = {
    "gaussian_mixture": ("translate", "rotate_0.7"),
    "grid_jitter": ("reflect_x", "translate"),
}


def _ml_ood_cells() -> list[CellSpec]:
    cells: list[CellSpec] = []
    seed_base = 5000
    index = 0

    for family, base_params in ML_OOD_FIT_FAMILIES:
        for n in (50, 100):
            params = dict(base_params)
            if family == "grid_jitter":
                params.update({"rows": 7 if n == 50 else 10, "cols": 7 if n == 50 else 10})
            elif family == "convex_k_inner":
                params.update({"count": n, "k_inner": n - 4})
            elif family == "poisson_disc":
                params.update({"count": n, "min_separation": 0.09 if n == 50 else 0.07})
            else:
                params.update({"count": n})
            transforms = ML_OOD_TRANSFORMS.get(family, ()) if n == 50 else ()
            for split, count in (("train", 3), ("validation", 2), ("test_iid", 3)):
                cells.append(
                    CellSpec(
                        generator=family,
                        params=params,
                        count=count,
                        seed_base=seed_base + index * 37,
                        split=split,
                        weight=1.0,
                        references="none",
                        transforms=transforms if split == "train" else (),
                    )
                )
                index += 1

    for family, base_params in ML_OOD_FAMILY_OOD:
        for n in (50, 100):
            params = dict(base_params)
            params.update({"count": n})
            cells.append(
                CellSpec(
                    generator=family,
                    params=params,
                    count=3,
                    seed_base=seed_base + index * 37,
                    split="test_family_ood",
                    weight=1.0,
                    references="none",
                )
            )
            index += 1

    for family, base_params in ML_OOD_PARAM_OOD:
        for n in (50, 100):
            params = dict(base_params)
            if family == "grid_jitter":
                params.update({"rows": 7 if n == 50 else 10, "cols": 7 if n == 50 else 10})
            elif family == "convex_k_inner":
                params.update({"count": n, "k_inner": n - 8})
            else:
                params.update({"count": n})
            cells.append(
                CellSpec(
                    generator=family,
                    params=params,
                    count=3,
                    seed_base=seed_base + index * 37,
                    split="test_parameter_ood",
                    weight=1.0,
                    references="none",
                )
            )
            index += 1

    return cells


ALL_SUITES: dict[str, list[CellSpec]] = {
    "exact_structure": _exact_cells(),
    "anytime_quality": _anytime_cells(),
    "ml_ood": _ml_ood_cells(),
}
