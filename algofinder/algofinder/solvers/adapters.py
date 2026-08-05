"""External solver adapters (benchmark spec 8.2, 8.3, 13 item 2).

These adapters translate a problem state to the solver's native format,
launch the external binary in an isolated process group with a hard
wall-clock cap and captured output, and parse the returned solution.
Every solution is re-scored by the harness, so the adapters only need to
produce *some* tour, never a claim about its quality.

Adapters are always registered so every (instance, solver) cell has a
row. When the binary cannot be resolved (no ``ALGOFINDER_<NAME>`` env
var, not on PATH), the cell raises :class:`UnsupportedError`, which the
harness records as ``unsupported`` — the row exists, the output does not
(spec 6.6 taxonomy).

Supported native formats:
  * LKH (``lkh``): TSPLIB ``EUC_2D`` (coordinates) or ``EXPLICIT``
    ``FULL_MATRIX``; parameter files with a hard TIME_LIMIT and SEED.
  * Concorde (``concorde``): TSPLIB ``EUC_2D`` only.
  * GA-EAX (``geax``): TSPLIB ``EUC_2D`` only.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from algofinder.problems.base import ProblemState
from algofinder.solvers.base import (
    InapplicableError,
    Solver,
    SolverCapabilities,
    SolverResult,
    UnsupportedError,
    register_solver,
)

BINARY_ENV = {"lkh": "ALGOFINDER_LKH", "concorde": "ALGOFINDER_CONCORDE", "geax": "ALGOFINDER_GAEAX"}


def _remaining(deadline: float | None) -> float | None:
    """Seconds left until the deadline, or ``None`` when unbounded."""
    if deadline is None:
        return None
    return max(0.0, deadline - perf_counter())


def _write_tsplib(state: ProblemState, path: Path, name: str) -> str:
    """Write a TSPLIB file; return the edge-weight type used."""
    points = getattr(state, "points", None)
    if points is not None:
        path.write_text(_euc2d_tsplib(np.asarray(points), name), encoding="utf-8")
        return "EUC_2D"
    distances = getattr(state, "distances", None)
    if distances is not None:
        path.write_text(_full_matrix_tsplib(np.asarray(distances), name), encoding="utf-8")
        return "EXPLICIT"
    raise InapplicableError("state has neither points nor a distance matrix")


def _euc2d_tsplib(points: np.ndarray, name: str) -> str:
    lines = [
        f"NAME: {name}",
        "TYPE: TSP",
        f"DIMENSION: {points.shape[0]}",
        "EDGE_WEIGHT_TYPE: EUC_2D",
        "NODE_COORD_SECTION",
    ]
    lines.extend(
        f"{i + 1} {x:.10f} {y:.10f}" for i, (x, y) in enumerate(points)
    )
    lines.append("EOF")
    return "\n".join(lines) + "\n"


def _full_matrix_tsplib(distances: np.ndarray, name: str) -> str:
    lines = [
        f"NAME: {name}",
        "TYPE: TSP",
        f"DIMENSION: {distances.shape[0]}",
        "EDGE_WEIGHT_TYPE: EXPLICIT",
        "EDGE_WEIGHT_FORMAT: FULL_MATRIX",
        "EDGE_WEIGHT_SECTION",
    ]
    lines.extend(
        " ".join(f"{value:.6f}" for value in row) for row in distances
    )
    lines.append("EOF")
    return "\n".join(lines) + "\n"


class ExternalAdapter(Solver):
    """Shared subprocess lifecycle for external binaries."""

    binary_name: str = ""
    display: str = ""

    def capabilities(self) -> SolverCapabilities:
        base = super().capabilities()
        base_dict = base.to_mapping()
        base_dict.update(
            {
                "interruptible": True,
                "streams_incumbents": False,
                "seed_control": "external",
                "memory_control": False,
                "native_work_counters": False,
                "binary_name": self.binary_name,
            }
        )
        return SolverCapabilities(**base_dict)

    def resolve_binary(self) -> str | None:
        env_key = BINARY_ENV.get(self.binary_name, "")
        override = os.environ.get(env_key)
        if override:
            candidate = Path(override)
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        return shutil.which(self.binary_name)

    def solve(
        self,
        state: ProblemState,
        *,
        budget_seconds: float | None = None,
        seed: int | None = None,
        context: Any | None = None,
    ) -> SolverResult:
        binary = self.resolve_binary()
        if binary is None:
            raise UnsupportedError(
                f"{self.id}: binary {self.binary_name!r} not found; "
                f"set {BINARY_ENV.get(self.binary_name, '')} or install it on PATH"
            )
        started = perf_counter()
        workdir = Path(tempfile.mkdtemp(prefix=f"algofinder-{self.binary_name}-"))
        deadline = (
            started + budget_seconds
            if budget_seconds is not None and budget_seconds > 0
            else None
        )
        completed: subprocess.CompletedProcess | None = None
        try:
            instance_path = workdir / "instance.tsp"
            edge_type = _write_tsplib(state, instance_path, "algofinder_instance")
            command = self.build_command(
                binary, instance_path, workdir, edge_type,
                budget_seconds=budget_seconds, seed=seed,
            )
            process = subprocess.Popen(
                command,
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=_remaining(deadline))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise InapplicableError(
                    f"{self.id}: external solver exceeded its time budget"
                ) from None
            completed = subprocess.CompletedProcess(
                command, process.returncode, stdout, stderr
            )
            if process.returncode != 0:
                raise ValueError(
                    f"exit {process.returncode}: {stderr[-512:]}"
                )
            tour = self.parse_solution(workdir, stdout, stderr)
        except (OSError, ValueError) as exc:
            raise InapplicableError(
                f"{self.id}: external run failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        return SolverResult(
            solution=tuple(int(city) for city in tour),
            cost=0.0,  # the harness re-scores; never trust the adapter
            exact=bool(getattr(self, "is_exact", False)),
            wall_seconds=perf_counter() - started,
            seed=seed,
            metadata={
                "adapter": self.id,
                "binary": binary,
                "stdout_bytes": len(completed.stdout or ""),
                "stderr_bytes": len(completed.stderr or ""),
                "edge_type": edge_type,
                "budget_seconds": budget_seconds,
            },
        )

    def build_command(
        self,
        binary: str,
        instance_path: Path,
        workdir: Path,
        edge_type: str,
        *,
        budget_seconds: float | None,
        seed: int | None,
    ) -> list[str]:
        raise NotImplementedError

    def parse_solution(self, workdir: Path, stdout: str, stderr: str) -> list[int]:
        raise NotImplementedError


@register_solver
class LkhAdapter(ExternalAdapter):
    """LKH 2.x (Helsgaun) via a TSPLIB instance and a parameter file."""

    id = "lkh"
    display = "LKH (Helsgaun)"
    tags = frozenset({"heuristic", "external"})
    applies_to = frozenset({"tsp:euclidean", "tsp:general"})
    binary_name = "lkh"
    is_exact = False

    def build_command(
        self,
        binary: str,
        instance_path: Path,
        workdir: Path,
        edge_type: str,
        *,
        budget_seconds: float | None,
        seed: int | None,
    ) -> list[str]:
        seed_value = seed if seed is not None else 1
        budget = budget_seconds if budget_seconds and budget_seconds > 0 else 30
        params = {
            "PROBLEM_FILE": str(instance_path),
            "TOUR_FILE": str(workdir / "tour.txt"),
            "RUNS": "1",
            "SEED": str(seed_value),
            "TIME_LIMIT": f"{int(budget)}",
        }
        param_path = workdir / "instance.par"
        param_path.write_text(
            "\n".join(f"{key} = {value}" for key, value in params.items()) + "\n",
            encoding="utf-8",
        )
        return [binary, str(param_path)]

    def parse_solution(self, workdir: Path, stdout: str, stderr: str) -> list[int]:
        tour_path = workdir / "tour.txt"
        if not tour_path.exists():
            raise ValueError(f"LKH wrote no TOUR_FILE: {stderr[-512:]}")
        lines = [
            line.split(":", 1)[-1].strip()
            for line in tour_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        body: list[str] = []
        in_body = False
        for line in lines:
            if line.upper().startswith("TOUR_SECTION"):
                in_body = True
                continue
            if not in_body:
                continue
            if line in ("-1", "EOF") or line.upper() == "EOF":
                break
            body.append(line)
        numbers = [int(item) for line in body for item in line.split()]
        tour = [city - 1 for city in numbers if city > 0]
        if not tour:
            raise ValueError("LKH TOUR_FILE contained no tour")
        return tour


@register_solver
class ConcordeAdapter(ExternalAdapter):
    """Concorde (Applegate et al.) via TSPLIB coordinates."""

    id = "concorde"
    display = "Concorde (Applegate et al.)"
    tags = frozenset({"exact", "external"})
    applies_to = frozenset({"tsp:euclidean"})
    binary_name = "concorde"
    is_exact = True

    def build_command(
        self,
        binary: str,
        instance_path: Path,
        workdir: Path,
        edge_type: str,
        *,
        budget_seconds: float | None,
        seed: int | None,
    ) -> list[str]:
        if edge_type != "EUC_2D":
            raise InapplicableError("concorde supports EUC_2D coordinates only")
        return [binary, "-o", str(workdir / "tour.sol"), str(instance_path)]

    def parse_solution(self, workdir: Path, stdout: str, stderr: str) -> list[int]:
        solution_path = workdir / "tour.sol"
        if not solution_path.exists():
            raise ValueError(f"Concorde wrote no .sol file: {stderr[-512:]}")
        numbers = solution_path.read_text(encoding="utf-8").split()
        return [int(city) for city in numbers]


@register_solver
class GaeaxAdapter(ExternalAdapter):
    """GA-EAX (Nagata) via TSPLIB coordinates."""

    id = "gaeax"
    display = "GA-EAX (Nagata)"
    tags = frozenset({"heuristic", "external"})
    applies_to = frozenset({"tsp:euclidean"})
    binary_name = "geax"
    is_exact = False

    def build_command(
        self,
        binary: str,
        instance_path: Path,
        workdir: Path,
        edge_type: str,
        *,
        budget_seconds: float | None,
        seed: int | None,
    ) -> list[str]:
        if edge_type != "EUC_2D":
            raise InapplicableError("GA-EAX supports EUC_2D coordinates only")
        return [binary, str(instance_path)]

    def parse_solution(self, workdir: Path, stdout: str, stderr: str) -> list[int]:
        solution_path = workdir / "tour.txt"
        if solution_path.exists():
            numbers = solution_path.read_text(encoding="utf-8").split()
            tour = [int(city) for city in numbers if city.isdigit()]
            if tour:
                return tour
        raise ValueError(f"GA-EAX wrote no tour: {stderr[-512:]}")
