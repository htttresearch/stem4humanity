"""Versioned controlled generator family (benchmark spec section 7.2, catalog section 4).

Every family is a pure function of ``(params, seed)`` returning coordinates
plus realized diagnostics. No wall clock, global RNG, host name, or mutable
defaults are read; re-running the same generator version, parameters, and
PRNG version yields identical canonical coordinates (determinism exit
criterion).

Realized-condition validation: when a draw is supposed to realize a specific
condition (e.g. exactly ``k`` inner points), the generator validates the
realized value and returns it in ``diagnostics``. The caller never silently
mislabels an intended condition that failed; ``realized`` always reports the
truth and ``intended`` stays visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

PRNG = "numpy.default_rng(v1)"
STRATA_SCHEMA = "etsp.strata.v1"

GeneratorFn = Callable[[dict[str, object], int], tuple[NDArray[np.float64], dict[str, object]]]


@dataclass(frozen=True)
class GeneratorSpec:
    """One versioned generator stage."""

    family: str
    version: str
    dimension: int
    function: GeneratorFn
    default_params: dict[str, object] = field(default_factory=dict)

    def run(
        self, params: dict[str, object] | None = None, seed: int = 0
    ) -> tuple[NDArray[np.float64], dict[str, object]]:
        merged = dict(self.default_params)
        if params:
            merged.update(params)
        points, diagnostics = self.function(merged, seed)
        if "count" in merged and points.shape[0] != int(merged["count"]):
            raise ValueError(
                f"{key}: realized {points.shape[0]} rows, intended "
                f"{int(merged['count'])}"
            )
        return points, diagnostics


_GENERATORS: dict[str, GeneratorSpec] = {}


def register_generator(spec: GeneratorSpec) -> GeneratorSpec:
    key = f"{spec.family}@{spec.version}"
    if key in _GENERATORS:
        raise ValueError(f"duplicate generator: {key}")
    _GENERATORS[key] = spec
    return spec


def get_generator(spec_id: str) -> GeneratorSpec:
    if "@" not in spec_id:
        matches = [spec for key, spec in _GENERATORS.items() if key.startswith(spec_id + "@")]
        if not matches:
            raise KeyError(f"unknown generator: {spec_id}")
        return max(matches, key=lambda spec: spec.version)
    try:
        return _GENERATORS[spec_id]
    except KeyError:
        raise KeyError(f"unknown generator: {spec_id}") from None


def all_generators() -> dict[str, GeneratorSpec]:
    return dict(_GENERATORS)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _finite(points: NDArray[np.float64]) -> bool:
    return bool(np.isfinite(points).all())


# ---------------------------------------------------------------------------
# 4.1 baseline densities
# ---------------------------------------------------------------------------


def _uniform_rect(params: dict[str, object], seed: int) -> tuple[NDArray[np.float64], dict[str, object]]:
    count = int(params["count"])
    dimension = int(params["dimension"])
    aspect = float(params["aspect_ratio"])
    scale = np.ones(dimension)
    scale[0] = aspect
    points = _rng(seed).uniform(size=(count, dimension)) * scale
    if not _finite(points):
        raise ValueError("uniform_rect produced non-finite coordinates")
    return points, {"count": count, "dimension": dimension, "aspect_ratio": aspect}


def _beta_density(params: dict[str, object], seed: int) -> tuple[NDArray[np.float64], dict[str, object]]:
    count = int(params["count"])
    dimension = int(params["dimension"])
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    points = _rng(seed).beta(alpha, beta, size=(count, dimension))
    return points, {"count": count, "alpha": alpha, "beta": beta}


def _poisson_disc(params: dict[str, object], seed: int) -> tuple[NDArray[np.float64], dict[str, object]]:
    count = int(params["count"])
    min_separation = float(params["min_separation"])
    max_tries = int(params.get("max_tries", 200))
    rng = _rng(seed)
    dimension = 2
    points: list[NDArray[np.float64]] = []
    first = rng.uniform(size=dimension)
    points.append(first)
    tries = 0
    while len(points) < count and tries < max_tries * count:
        candidate = rng.uniform(size=dimension)
        if all(np.linalg.norm(candidate - existing) >= min_separation for existing in points):
            points.append(candidate)
            tries = 0
        else:
            tries += 1
    realized = np.asarray(points, dtype=float)
    if len(realized) < count:
        raise ValueError(
            f"poisson_disc could not place {count} points at separation "
            f"{min_separation} (realized {len(realized)})"
        )
    return realized, {"count": count, "min_separation": min_separation, "max_tries": max_tries}


# ---------------------------------------------------------------------------
# 4.2 clustered and hierarchical families
# ---------------------------------------------------------------------------


def _gaussian_mixture(params: dict[str, object], seed: int) -> tuple[NDArray[np.float64], dict[str, object]]:
    count = int(params["count"])
    cluster_count = int(params["cluster_count"])
    within_std = float(params["within_std"])
    center_min_separation = float(params.get("center_min_separation", 0.0))
    imbalance_exponent = float(params.get("imbalance_exponent", 0.0))
    outlier_weight = float(params.get("outlier_weight", 0.0))
    rng = _rng(seed)

    centers: list[NDArray[np.float64]] = []
    tries = 0
    while len(centers) < cluster_count and tries < 500:
        candidate = rng.uniform(size=2)
        if all(np.linalg.norm(candidate - center) >= center_min_separation for center in centers):
            centers.append(candidate)
            tries = 0
        else:
            tries += 1
    if len(centers) < cluster_count:
        raise ValueError(
            f"gaussian_mixture could not separate {cluster_count} centers "
            f"at distance {center_min_separation}"
        )

    if imbalance_exponent:
        weights = np.asarray(
            [k ** imbalance_exponent for k in range(1, cluster_count + 1)], dtype=float
        )
        weights /= weights.sum()
    else:
        weights = np.full(cluster_count, 1.0 / cluster_count)

    outlier_count = int(outlier_weight * count)
    regular_count = count - outlier_count
    assignments = rng.choice(cluster_count, size=regular_count, p=weights)
    points = np.empty((count, 2), dtype=float)
    for cluster_index in range(cluster_count):
        mask = assignments == cluster_index
        count_in_cluster = int(mask.sum())
        if count_in_cluster:
            points[:regular_count][mask] = (
                centers[cluster_index] + rng.normal(scale=within_std, size=(count_in_cluster, 2))
            )
    if outlier_count:
        points[regular_count:] = rng.uniform(size=(outlier_count, 2))
    return points, {
        "cluster_count": cluster_count,
        "within_std": within_std,
        "center_min_separation": center_min_separation,
        "outlier_count": outlier_count,
    }


def _disk_clusters(params: dict[str, object], seed: int) -> tuple[NDArray[np.float64], dict[str, object]]:
    count = int(params["count"])
    cluster_count = int(params["cluster_count"])
    radius = float(params["cluster_radius"])
    separation = float(params["separation"])
    rng = _rng(seed)
    centers: list[NDArray[np.float64]] = []
    tries = 0
    while len(centers) < cluster_count and tries < 500:
        candidate = rng.uniform(size=2)
        if all(np.linalg.norm(candidate - center) >= separation for center in centers):
            centers.append(candidate)
            tries = 0
        else:
            tries += 1
    if len(centers) < cluster_count:
        raise ValueError(f"disk_clusters could not separate {cluster_count} centers")
    sizes = np.full(cluster_count, count // cluster_count)
    sizes[: count % cluster_count] += 1
    points: list[NDArray[np.float64]] = []
    for center, size in zip(centers, sizes):
        radiuses = rng.uniform(0.0, radius, size=size)
        angles = rng.uniform(0.0, 2 * np.pi, size=size)
        points.append(center + np.column_stack((radiuses * np.cos(angles), radiuses * np.sin(angles))))
    return np.concatenate(points, axis=0), {"cluster_count": cluster_count, "radius": radius, "separation": separation}


# ---------------------------------------------------------------------------
# 4.3 lattice and degeneracy families
# ---------------------------------------------------------------------------


def _grid_jitter(params: dict[str, object], seed: int) -> tuple[NDArray[np.float64], dict[str, object]]:
    rows = int(params["rows"])
    cols = int(params["cols"])
    jitter = float(params["jitter"])
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    base = np.column_stack((xs.ravel(), ys.ravel())).astype(float)
    if jitter > 0.0:
        base = base + _rng(seed).normal(scale=jitter, size=base.shape)
    return base, {"rows": rows, "cols": cols, "jitter": jitter, "count": base.shape[0]}


def _quantized(params: dict[str, object], seed: int) -> tuple[NDArray[np.float64], dict[str, object]]:
    count = int(params["count"])
    bit_depth = int(params["bit_depth"])
    scale = 2 ** bit_depth
    raw = _rng(seed).uniform(size=(count, 2))
    points = np.round(raw * scale) / scale
    return points, {"bit_depth": bit_depth, "count": count}


def _small_integer(params: dict[str, object], seed: int) -> tuple[NDArray[np.float64], dict[str, object]]:
    count = int(params["count"])
    bound = int(params["bound"])
    points = _rng(seed).integers(0, bound, size=(count, 2)).astype(float)
    return points, {"bound": bound, "count": count}


def _duplicates(params: dict[str, object], seed: int) -> tuple[NDArray[np.float64], dict[str, object]]:
    count = int(params["count"])
    duplicate_fraction = float(params["duplicate_fraction"])
    rng = _rng(seed)
    base_count = max(2, count - int(duplicate_fraction * count))
    base = rng.uniform(size=(base_count, 2))
    duplicates = rng.choice(base_count, size=count - base_count)
    points = np.concatenate([base, base[duplicates]], axis=0)
    return points, {"base_count": base_count, "duplicate_fraction": duplicate_fraction}


def _collinear_run(params: dict[str, object], seed: int) -> tuple[NDArray[np.float64], dict[str, object]]:
    count = int(params["count"])
    jitter = float(params.get("jitter", 0.0))
    xs = np.sort(_rng(seed).uniform(size=count))
    points = np.column_stack((xs, np.full(count, 0.5) + _rng(seed).normal(scale=jitter, size=count)))
    return points, {"count": count, "jitter": jitter}


def _cocircular(params: dict[str, object], seed: int) -> tuple[NDArray[np.float64], dict[str, object]]:
    count = int(params["count"])
    radius = float(params.get("radius", 1.0))
    angles = np.sort(_rng(seed).uniform(0.0, 2 * np.pi, size=count))
    points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    return points, {"count": count, "radius": radius}


# ---------------------------------------------------------------------------
# 4.4 boundary and layer families
# ---------------------------------------------------------------------------


def _convex_k_inner(params: dict[str, object], seed: int) -> tuple[NDArray[np.float64], dict[str, object]]:
    count = int(params["count"])
    intended_k = int(params["k_inner"])
    hull_shape = str(params.get("hull_shape", "square"))
    inner_scale = float(params.get("inner_scale", 0.8))
    rng = _rng(seed)

    if hull_shape == "square":
        hull = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
        hull = hull + rng.normal(scale=0.01, size=hull.shape)
    elif hull_shape == "circle":
        angles = np.linspace(0.0, 2 * np.pi, 8, endpoint=False) + rng.uniform(0.0, 0.2)
        hull = np.column_stack((np.cos(angles), np.sin(angles)))
        hull = (hull + 1.0) / 2.0
    else:
        raise ValueError(f"unknown hull_shape: {hull_shape}")

    inner = rng.uniform(low=(1 - inner_scale) / 2, high=(1 + inner_scale) / 2, size=(intended_k, 2))
    points = np.concatenate([hull, inner], axis=0)
    if points.shape[0] != count:
        raise ValueError(
            f"convex_k_inner: hull {hull.shape[0]} + inner {intended_k} != count {count}"
        )
    realized = len(np.unique(points, axis=0))
    return points, {"realized_count": points.shape[0], "realized_unique": realized}


def _onion_layers(params: dict[str, object], seed: int) -> tuple[NDArray[np.float64], dict[str, object]]:
    layers = int(params["layers"])
    per_layer = int(params["per_layer"])
    radius_step = float(params["radius_step"])
    phase = float(params.get("phase", 0.0))
    rng = _rng(seed)
    points: list[NDArray[np.float64]] = []
    for layer in range(1, layers + 1):
        radius = layer * radius_step
        angles = np.linspace(0.0, 2 * np.pi, per_layer, endpoint=False)
        angles = angles + phase + rng.uniform(0.0, 0.05, size=per_layer)
        points.append(np.column_stack((radius * np.cos(angles), radius * np.sin(angles))))
    return np.concatenate(points, axis=0), {"layers": layers, "per_layer": per_layer}


# ---------------------------------------------------------------------------
# 4.5 lines, strips, corridors, and manifolds
# ---------------------------------------------------------------------------


def _line_points(params: dict[str, object], seed: int) -> tuple[NDArray[np.float64], dict[str, object]]:
    count = int(params["count"])
    jitter = float(params.get("jitter", 0.0))
    xs = np.sort(_rng(seed).uniform(size=count))
    ys = _rng(seed).normal(scale=jitter, size=count)
    return np.column_stack((xs, ys)), {"count": count, "jitter": jitter}


def _parallel_lines(params: dict[str, object], seed: int) -> tuple[NDArray[np.float64], dict[str, object]]:
    count = int(params["count"])
    line_count = int(params["lines"])
    spacing = float(params["spacing"])
    rng = _rng(seed)
    per_line = count // line_count
    remainder = count - line_count * per_line
    points: list[NDArray[np.float64]] = []
    for line_index in range(line_count):
        size = per_line + (1 if line_index < remainder else 0)
        xs = np.sort(rng.uniform(size=size))
        y = line_index * spacing
        points.append(np.column_stack((xs, np.full(size, y))))
    return np.concatenate(points, axis=0), {"lines": line_count, "spacing": spacing, "count": count}


def _narrow_strip(params: dict[str, object], seed: int) -> tuple[NDArray[np.float64], dict[str, object]]:
    count = int(params["count"])
    width = float(params["width"])
    xs = np.sort(_rng(seed).uniform(size=count))
    ys = _rng(seed).uniform(-width / 2, width / 2, size=count)
    return np.column_stack((xs, ys)), {"count": count, "width": width}


def _rings(params: dict[str, object], seed: int) -> tuple[NDArray[np.float64], dict[str, object]]:
    count = int(params["count"])
    ring_count = int(params.get("rings", 1))
    radius_step = float(params.get("radius_step", 1.0))
    rng = _rng(seed)
    points: list[NDArray[np.float64]] = []
    per_ring = count // ring_count
    remainder = count - ring_count * per_ring
    for ring in range(ring_count):
        radius = (ring + 1) * radius_step
        size = per_ring + (1 if ring < remainder else 0)
        angles = np.linspace(0.0, 2 * np.pi, size, endpoint=False)
        angles = angles + rng.uniform(0.0, 0.05, size=size)
        points.append(np.column_stack((radius * np.cos(angles), radius * np.sin(angles))))
    return np.concatenate(points, axis=0), {"rings": ring_count, "radius_step": radius_step, "count": count}


def _spiral(params: dict[str, object], seed: int) -> tuple[NDArray[np.float64], dict[str, object]]:
    count = int(params["count"])
    turns = float(params.get("turns", 3.0))
    noise = float(params.get("noise", 0.02))
    angles = np.linspace(0.0, 2 * np.pi * turns, count)
    radius = np.linspace(0.05, 1.0, count)
    points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    points = points + _rng(seed).normal(scale=noise, size=points.shape)
    return points, {"count": count, "turns": turns, "noise": noise}


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

for _spec in (
    GeneratorSpec("uniform_rect", "v1", 2, _uniform_rect, {"count": 16, "dimension": 2, "aspect_ratio": 1.0}),
    GeneratorSpec("beta_density", "v1", 2, _beta_density, {"count": 16, "dimension": 2, "alpha": 2.0, "beta": 2.0}),
    GeneratorSpec("poisson_disc", "v1", 2, _poisson_disc, {"count": 16, "min_separation": 0.05}),
    GeneratorSpec("gaussian_mixture", "v1", 2, _gaussian_mixture, {"count": 16, "cluster_count": 3, "within_std": 0.05, "center_min_separation": 0.2}),
    GeneratorSpec("disk_clusters", "v1", 2, _disk_clusters, {"count": 16, "cluster_count": 3, "cluster_radius": 0.2, "separation": 0.4}),
    GeneratorSpec("grid_jitter", "v1", 2, _grid_jitter, {"rows": 4, "cols": 4, "jitter": 0.0}),
    GeneratorSpec("quantized", "v1", 2, _quantized, {"count": 16, "bit_depth": 8}),
    GeneratorSpec("small_integer", "v1", 2, _small_integer, {"count": 16, "bound": 100}),
    GeneratorSpec("duplicates", "v1", 2, _duplicates, {"count": 16, "duplicate_fraction": 0.1}),
    GeneratorSpec("collinear_run", "v1", 2, _collinear_run, {"count": 16, "jitter": 0.0}),
    GeneratorSpec("cocircular", "v1", 2, _cocircular, {"count": 16, "radius": 1.0}),
    GeneratorSpec("convex_k_inner", "v1", 2, _convex_k_inner, {"count": 16, "k_inner": 12, "hull_shape": "square", "inner_scale": 0.8}),
    GeneratorSpec("onion_layers", "v1", 2, _onion_layers, {"layers": 3, "per_layer": 8, "radius_step": 1.0}),
    GeneratorSpec("line", "v1", 2, _line_points, {"count": 16, "jitter": 0.0}),
    GeneratorSpec("parallel_lines", "v1", 2, _parallel_lines, {"count": 16, "lines": 4, "spacing": 0.6}),
    GeneratorSpec("narrow_strip", "v1", 2, _narrow_strip, {"count": 16, "width": 0.2}),
    GeneratorSpec("rings", "v1", 2, _rings, {"count": 16, "rings": 2, "radius_step": 1.0}),
    GeneratorSpec("spiral", "v1", 2, _spiral, {"count": 16, "turns": 3.0, "noise": 0.02}),
):
    register_generator(_spec)


__all__ = [
    "GeneratorSpec",
    "PRNG",
    "STRATA_SCHEMA",
    "all_generators",
    "get_generator",
    "register_generator",
]
