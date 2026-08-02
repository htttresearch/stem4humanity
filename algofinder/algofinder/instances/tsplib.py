"""TSPLIB95 instance support: spec-exact EUC_2D parsing and manifest emission.

Implements the parts of the TSPLIB95 specification (Reinelt, 1995) needed for
Euclidean symmetric TSP instances:

- ``EUC_2D`` distance function: each edge is the Euclidean distance rounded
  to the nearest integer (``floor(d + 0.5)``), so reported objectives are
  integer and directly comparable to the published TSPLIB optima.
- Optimal-tour files (``*.opt.tour``) are parsed and *independently rescored*
  under the same integer metric; the rescored value must match the official
  optimum before it is trusted (see the benchmarking protocol, section 2.1).
- Original file checksums (sha256) and parser version are recorded with every
  manifest instance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from algofinder.instances.base import save_manifest
from algofinder.problems.base import Instance

PARSER_VERSION = "tsplib95-euc2d-v1"

# Official TSPLIB optima (comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html).
# Instances with an archived optimal tour are rescored from the tour file and
# checked against this table; the table also supplies references for instances
# whose tours are not redistributed.
TSPLIB_OPTIMA: dict[str, int] = {
    "eil51": 426,
    "eil76": 538,
    "berlin52": 7542,
    "ch150": 6528,
    "kroA200": 29368,
    "pr299": 48191,
    "lin318": 42029,
    "u574": 36905,
}


def sha256_file(path: Path) -> str:
    """sha256 hex digest of the raw file bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_tsp_file(path: Path) -> dict[str, object]:
    """Parse a TSPLIB ``*.tsp`` file; returns name, dimension and coordinates.

    Only ``EUC_2D`` files are supported; any other ``EDGE_WEIGHT_TYPE`` is
    rejected explicitly so instances are never silently reinterpreted.
    """
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    header: dict[str, str] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        lower = line.lower()
        if ":" in line and not line[0].isdigit():
            key, _, value = line.partition(":")
            header[key.strip().upper()] = value.strip()
        if lower.startswith("node_coord_section"):
            index += 1
            break
        index += 1
    else:
        raise ValueError(f"{path.name}: missing NODE_COORD_SECTION")

    name = header.get("NAME", path.stem)
    try:
        dimension = int(header["DIMENSION"])
    except KeyError:
        raise ValueError(f"{path.name}: missing DIMENSION") from None
    edge_weight_type = header.get("EDGE_WEIGHT_TYPE", "").upper()
    if edge_weight_type != "EUC_2D":
        raise ValueError(
            f"{path.name}: unsupported EDGE_WEIGHT_TYPE {edge_weight_type!r} "
            "(only EUC_2D is supported)"
        )

    coordinates: list[tuple[float, float]] = []
    while index < len(lines) and len(coordinates) < dimension:
        parts = lines[index].split()
        if parts and parts[0].upper() == "EOF":
            break
        if len(parts) < 3:
            raise ValueError(f"{path.name}: malformed coordinate row: {line!r}")
        try:
            coordinates.append((float(parts[1]), float(parts[2])))
        except ValueError:
            raise ValueError(f"{path.name}: malformed coordinate row: {line!r}") from None
        index += 1

    if len(coordinates) != dimension:
        raise ValueError(
            f"{path.name}: expected {dimension} coordinates, found {len(coordinates)}"
        )
    return {
        "name": name,
        "dimension": dimension,
        "points": coordinates,
    }


def parse_tour_file(path: Path) -> list[int]:
    """Parse a TSPLIB ``*.opt.tour`` file into 0-based city indices."""
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    index = 0
    while index < len(lines):
        if lines[index].lower().startswith("tour_section"):
            index += 1
            break
        index += 1
    else:
        raise ValueError(f"{path.name}: missing TOUR_SECTION")

    tour: list[int] = []
    while index < len(lines):
        for token in lines[index].split():
            value = int(token)
            if value == -1:
                return tour
            tour.append(value - 1)
        index += 1
    return tour


def euc_2d_integer_cost(points: NDArray[np.float64], tour: list[int]) -> int:
    """TSPLIB EUC_2D integer tour length for a closed tour.

    Each edge contributes ``floor(sqrt(dx^2 + dy^2) + 0.5)``; the closure edge
    from the last city back to the first is included.
    """
    coords = np.asarray(points, dtype=float)
    indices = np.asarray(tour, dtype=int)
    if indices.size == 0:
        return 0
    delta = coords[indices] - coords[np.roll(indices, -1)]
    squared = np.einsum("ij,ij->i", delta, delta)
    return int(np.floor(np.sqrt(squared) + 0.5).sum())


def generate_tsplib_manifests(
    source_dir: str,
    output_path: str,
    *,
    optima: dict[str, int] | None = None,
) -> list[Instance]:
    """Emit manifest instances for every TSPLIB file found in ``source_dir``.

    ``source_dir`` must contain ``<name>.tsp`` files (EUC_2D) and, optionally,
    ``<name>.opt.tour`` files. Published optima come from the optional ``optima``
    table; when a tour file exists the rescored value must equal the table value.
    """
    optima = dict(TSPLIB_OPTIMA if optima is None else optima)
    records: list[Instance] = []
    for tsp_path in sorted(Path(source_dir).glob("*.tsp")):
        name = tsp_path.stem
        parsed = parse_tsp_file(tsp_path)
        dimension = int(parsed["dimension"])
        points = np.asarray(parsed["points"], dtype=float)
        if dimension != points.shape[0]:
            raise ValueError(f"{tsp_path.name}: dimension mismatch")
        if name not in optima:
            raise ValueError(f"{tsp_path.name}: no published optimum for {name!r}")

        tour_path = tsp_path.with_name(f"{name}.opt.tour")
        best_known = int(optima[name])
        rescored: int | None = None
        if tour_path.exists():
            tour = parse_tour_file(tour_path)
            if set(tour) != set(range(dimension)):
                raise ValueError(f"{tour_path.name}: tour does not visit every city")
            rescored = euc_2d_integer_cost(points, tour)
            if rescored != best_known:
                raise ValueError(
                    f"{tour_path.name}: rescored optimum {rescored} does not match "
                    f"published optimum {best_known}; investigate before publishing"
                )

        records.append(
            Instance(
                name=f"tsp:tsplib:{name}",
                problem="tsp",
                subproblem="euclidean",
                family="tsplib",
                seed=0,
                data={
                    "points": points.tolist(),
                    "source": name,
                    "dimension": dimension,
                    "metric": "EUC_2D",
                    "sha256": sha256_file(tsp_path),
                    "opt_tour_rescored": rescored is not None,
                    "parser": PARSER_VERSION,
                },
                best_known=float(best_known),
                split="test",
            )
        )
        print(
            f"tsplib: {name:10s} n={dimension:4d} opt={best_known:7d} "
            f"rescored={'yes' if rescored is not None else 'table'}"
        )

    save_manifest(records, output_path)
    print(f"wrote {len(records)} tsplib instances to {output_path}")
    return records


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m algofinder.instances.tsplib <source_dir> <output.json>")
    generate_tsplib_manifests(sys.argv[1], sys.argv[2])
