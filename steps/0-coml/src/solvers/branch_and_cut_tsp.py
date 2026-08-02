"""Exact branch-and-cut solver for the symmetric complete TSP.

This implementation uses Gurobi's branch-and-cut engine with:

* an undirected degree-two formulation;
* lazy subtour-elimination constraints for integer incumbents;
* optional globally valid fractional subtour cuts separated by a
  Stoer--Wagner global minimum-cut algorithm; and
* a multi-start nearest-neighbour plus 2-opt MIP start.

The algorithm is exact when the optimizer terminates with OPTIMAL status.
Its worst-case running time remains exponential, as expected for general TSP.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations
from typing import Any

import numpy as np

from src.problems.travelling_salesperson_problem import (
    TravellingSalespersonProblem,
)
from src.solvers.base_solver import BaseSolver, SolverResult

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # Keep the module importable without the optional backend.
    gp = None
    GRB = None


Edge = tuple[int, int]


def _edge(first: int, second: int) -> Edge:
    """Return the canonical key for an undirected edge."""
    if first < second:
        return first, second
    return second, first


def _tour_cost(tour: Sequence[int], distances: np.ndarray) -> float:
    """Return the cost of a closed tour represented without a repeated start."""
    city_count = len(tour)
    return float(
        sum(
            distances[tour[position], tour[(position + 1) % city_count]]
            for position in range(city_count)
        )
    )


def _nearest_neighbour_tour(
    distances: np.ndarray,
    start: int,
) -> np.ndarray:
    """Construct a deterministic nearest-neighbour tour."""
    city_count = distances.shape[0]
    tour = np.empty(city_count, dtype=np.intp)
    unvisited = np.ones(city_count, dtype=np.bool_)

    current = start
    tour[0] = current
    unvisited[current] = False

    for position in range(1, city_count):
        candidates = np.flatnonzero(unvisited)
        candidate_costs = distances[current, candidates]
        next_city = int(candidates[int(np.argmin(candidate_costs))])

        tour[position] = next_city
        unvisited[next_city] = False
        current = next_city

    return tour


def _two_opt(
    initial_tour: np.ndarray,
    distances: np.ndarray,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Reach a deterministic 2-opt local optimum for a symmetric TSP.

    This is only an incumbent heuristic. It has no effect on exactness.
    """
    tour = initial_tour.copy()
    city_count = len(tour)

    if city_count < 4:
        return tour

    while True:
        best_delta = -tolerance
        best_first = -1
        best_last = -1

        for first in range(city_count - 1):
            # The two removed edges must be non-adjacent. When first == 0,
            # exclude the final edge because it is adjacent through the cycle.
            stop = city_count - 1 if first == 0 else city_count
            if first + 2 >= stop:
                continue

            last_positions = np.arange(first + 2, stop, dtype=np.intp)
            first_city = int(tour[first])
            second_city = int(tour[first + 1])
            third_cities = tour[last_positions]
            fourth_cities = tour[(last_positions + 1) % city_count]

            deltas = (
                distances[first_city, third_cities]
                + distances[second_city, fourth_cities]
                - distances[first_city, second_city]
                - distances[third_cities, fourth_cities]
            )
            offset = int(np.argmin(deltas))
            delta = float(deltas[offset])

            if delta < best_delta:
                best_delta = delta
                best_first = first + 1
                best_last = int(last_positions[offset])

        if best_first < 0:
            return tour

        tour[best_first : best_last + 1] = tour[
            best_first : best_last + 1
        ][::-1]


def _warm_start_tour(
    distances: np.ndarray,
    maximum_starts: int,
    apply_two_opt: bool,
) -> np.ndarray:
    """Build a good feasible tour for the MIP solver."""
    city_count = distances.shape[0]
    start_count = min(city_count, max(1, maximum_starts))

    if start_count == city_count:
        starts: Iterable[int] = range(city_count)
    else:
        # Evenly distributed deterministic starts make runs reproducible.
        starts = np.unique(
            np.linspace(0, city_count - 1, start_count, dtype=np.intp)
        )

    best_tour: np.ndarray | None = None
    best_cost = np.inf

    for start in starts:
        candidate = _nearest_neighbour_tour(distances, int(start))
        if apply_two_opt:
            candidate = _two_opt(candidate, distances)

        candidate_cost = _tour_cost(candidate, distances)
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_tour = candidate

    assert best_tour is not None
    return best_tour


def _connected_components(
    city_count: int,
    selected_edges: Iterable[Edge],
) -> list[list[int]]:
    """Return connected components of an undirected selected-edge graph."""
    adjacency: list[list[int]] = [[] for _ in range(city_count)]
    for first, second in selected_edges:
        adjacency[first].append(second)
        adjacency[second].append(first)

    components: list[list[int]] = []
    unseen = set(range(city_count))

    while unseen:
        root = min(unseen)
        unseen.remove(root)
        stack = [root]
        component: list[int] = []

        while stack:
            city = stack.pop()
            component.append(city)
            for neighbour in adjacency[city]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)

        components.append(component)

    return components


def _stoer_wagner_min_cut(
    capacities: np.ndarray,
) -> tuple[float, frozenset[int]]:
    """Compute a global minimum cut of a dense undirected graph.

    The implementation is O(n^3), deterministic, and dependency-free. The
    returned set is one side of the cut. Capacities must be non-negative.
    """
    city_count = capacities.shape[0]
    if city_count < 2:
        return np.inf, frozenset()

    merged_capacities = np.array(capacities, dtype=np.float64, copy=True)
    active = list(range(city_count))
    groups: list[set[int]] = [{city} for city in range(city_count)]

    best_value = np.inf
    best_side: frozenset[int] = frozenset()

    while len(active) > 1:
        added = np.zeros(city_count, dtype=np.bool_)
        connection_weights = np.zeros(city_count, dtype=np.float64)
        previous = -1

        for phase_position in range(len(active)):
            selected = -1
            selected_weight = -np.inf

            for city in active:
                if (
                    not added[city]
                    and connection_weights[city] > selected_weight
                ):
                    selected = city
                    selected_weight = float(connection_weights[city])

            if phase_position == len(active) - 1:
                cut_value = float(connection_weights[selected])
                if cut_value < best_value:
                    best_value = cut_value
                    best_side = frozenset(groups[selected])

                # Merge the final vertex of this phase into the previous one.
                for city in active:
                    if city == previous or city == selected:
                        continue
                    combined = (
                        merged_capacities[previous, city]
                        + merged_capacities[selected, city]
                    )
                    merged_capacities[previous, city] = combined
                    merged_capacities[city, previous] = combined

                groups[previous].update(groups[selected])
                active.remove(selected)
                break

            added[selected] = True
            previous = selected

            for city in active:
                if not added[city]:
                    connection_weights[city] += merged_capacities[
                        selected,
                        city,
                    ]

    return best_value, best_side


class _SubtourCutCallback:
    """Separate integer and optional fractional subtour constraints."""

    def __init__(
        self,
        variables: Any,
        city_count: int,
        separate_fractional_cuts: bool,
        fractional_cut_node_interval: int,
        fractional_cut_max_cities: int,
        cut_tolerance: float,
    ) -> None:
        self.variables = variables
        self.city_count = city_count
        self.all_cities = frozenset(range(city_count))
        self.separate_fractional_cuts = separate_fractional_cuts
        self.fractional_cut_node_interval = fractional_cut_node_interval
        self.fractional_cut_max_cities = fractional_cut_max_cities
        self.cut_tolerance = cut_tolerance

        self.lazy_cuts = 0
        self.fractional_cuts = 0
        self._known_lazy_cuts: set[frozenset[int]] = set()
        self._known_fractional_cuts: set[frozenset[int]] = set()

    def __call__(self, model: Any, where: int) -> None:
        if where == GRB.Callback.MIPSOL:
            self._separate_integer_solution(model)
        elif (
            where == GRB.Callback.MIPNODE
            and self.separate_fractional_cuts
            and self.city_count <= self.fractional_cut_max_cities
        ):
            self._separate_fractional_solution(model)

    def _canonical_side(
        self,
        side: Iterable[int],
    ) -> frozenset[int]:
        selected = frozenset(side)
        complement = self.all_cities - selected

        if len(complement) < len(selected):
            return complement
        if len(complement) > len(selected):
            return selected

        # Deterministic tie-breaking for equal-sized cut shores.
        return min(selected, complement, key=lambda values: tuple(sorted(values)))

    def _internal_edge_expression(self, side: frozenset[int]) -> Any:
        ordered = sorted(side)
        return gp.quicksum(
            self.variables[first, second]
            for first, second in combinations(ordered, 2)
        )

    def _separate_integer_solution(self, model: Any) -> None:
        solution = model.cbGetSolution(self.variables)
        selected_edges = [
            edge
            for edge, value in solution.items()
            if float(value) > 0.5
        ]
        components = _connected_components(
            self.city_count,
            selected_edges,
        )

        if len(components) == 1:
            return

        for component in components:
            side = self._canonical_side(component)
            if not side or len(side) == self.city_count:
                continue
            if side in self._known_lazy_cuts:
                continue

            model.cbLazy(
                self._internal_edge_expression(side) <= len(side) - 1
            )
            self._known_lazy_cuts.add(side)
            self.lazy_cuts += 1

    def _separate_fractional_solution(self, model: Any) -> None:
        status = int(model.cbGet(GRB.Callback.MIPNODE_STATUS))
        if status != GRB.OPTIMAL:
            return

        node_count = int(model.cbGet(GRB.Callback.MIPNODE_NODCNT))
        if (
            node_count > 0
            and node_count % self.fractional_cut_node_interval != 0
        ):
            return

        solution = model.cbGetNodeRel(self.variables)
        capacities = np.zeros(
            (self.city_count, self.city_count),
            dtype=np.float64,
        )

        for (first, second), value in solution.items():
            # Clamp tiny feasibility-tolerance excursions before min-cut.
            capacity = min(1.0, max(0.0, float(value)))
            capacities[first, second] = capacity
            capacities[second, first] = capacity

        cut_value, raw_side = _stoer_wagner_min_cut(capacities)
        if cut_value >= 2.0 - self.cut_tolerance:
            return

        side = self._canonical_side(raw_side)
        if len(side) <= 1 or side in self._known_fractional_cuts:
            return

        # Under the degree-two equations, this internal-edge form is
        # equivalent to x(delta(S)) >= 2 and usually has fewer nonzeros.
        model.cbCut(
            self._internal_edge_expression(side) <= len(side) - 1
        )
        self._known_fractional_cuts.add(side)
        self.fractional_cuts += 1


class BranchAndCutTSPSolver(BaseSolver[TravellingSalespersonProblem]):
    """Solve complete symmetric TSP instances exactly with branch-and-cut.

    This is usually far more scalable than Held--Karp because it does not
    explicitly enumerate every subset. Its worst case is still exponential.

    Gurobi is an optional dependency. Install ``gurobipy`` and configure a
    suitable Gurobi license before using this solver.
    """

    def __init__(
        self,
        *,
        time_limit: float | None = None,
        threads: int | None = None,
        log_output: bool = False,
        seed: int = 0,
        warm_start: bool = True,
        warm_start_maximum_starts: int = 16,
        warm_start_two_opt_max_cities: int = 500,
        separate_fractional_cuts: bool = True,
        fractional_cut_node_interval: int = 100,
        fractional_cut_max_cities: int = 250,
        cut_tolerance: float = 1e-6,
        symmetry_tolerance: float = 1e-10,
    ) -> None:
        if time_limit is not None and time_limit <= 0.0:
            raise ValueError("time_limit must be positive")
        if threads is not None and threads < 0:
            raise ValueError("threads cannot be negative")
        if warm_start_maximum_starts < 1:
            raise ValueError("warm_start_maximum_starts must be positive")
        if warm_start_two_opt_max_cities < 0:
            raise ValueError(
                "warm_start_two_opt_max_cities cannot be negative"
            )
        if fractional_cut_node_interval < 1:
            raise ValueError(
                "fractional_cut_node_interval must be positive"
            )
        if fractional_cut_max_cities < 2:
            raise ValueError(
                "fractional_cut_max_cities must be at least two"
            )
        if cut_tolerance <= 0.0:
            raise ValueError("cut_tolerance must be positive")
        if symmetry_tolerance < 0.0:
            raise ValueError("symmetry_tolerance cannot be negative")

        self.time_limit = time_limit
        self.threads = threads
        self.log_output = log_output
        self.seed = seed
        self.warm_start = warm_start
        self.warm_start_maximum_starts = warm_start_maximum_starts
        self.warm_start_two_opt_max_cities = (
            warm_start_two_opt_max_cities
        )
        self.separate_fractional_cuts = separate_fractional_cuts
        self.fractional_cut_node_interval = fractional_cut_node_interval
        self.fractional_cut_max_cities = fractional_cut_max_cities
        self.cut_tolerance = cut_tolerance
        self.symmetry_tolerance = symmetry_tolerance

    def solve(
        self,
        problem: TravellingSalespersonProblem,
    ) -> SolverResult:
        if gp is None or GRB is None:
            raise ImportError(
                "BranchAndCutTSPSolver requires gurobipy. Install it with "
                "`pip install gurobipy` and configure a Gurobi license."
            )

        city_count = problem.city_count
        distances = self._validated_distances(problem)

        if city_count == 1:
            return SolverResult(
                tour=(0,),
                cost=0.0,
                metadata={
                    "algorithm": "branch-and-cut",
                    "backend": "gurobi",
                    "exact": True,
                    "nodes": 0,
                },
            )

        if city_count == 2:
            return SolverResult(
                tour=(0, 1),
                cost=float(distances[0, 1] + distances[1, 0]),
                metadata={
                    "algorithm": "branch-and-cut",
                    "backend": "gurobi",
                    "exact": True,
                    "nodes": 0,
                },
            )

        model = gp.Model("symmetric_tsp")
        self._configure_model(model)

        edges = [
            (first, second)
            for first in range(city_count)
            for second in range(first + 1, city_count)
        ]
        variables = model.addVars(
            edges,
            obj={edge: float(distances[edge]) for edge in edges},
            vtype=GRB.BINARY,
            name="edge",
        )
        model.ModelSense = GRB.MINIMIZE

        for city in range(city_count):
            model.addConstr(
                gp.quicksum(
                    variables[_edge(city, other)]
                    for other in range(city_count)
                    if other != city
                )
                == 2,
                name=f"degree[{city}]",
            )

        initial_tour: np.ndarray | None = None
        initial_cost: float | None = None
        if self.warm_start:
            initial_tour = _warm_start_tour(
                distances,
                maximum_starts=self.warm_start_maximum_starts,
                apply_two_opt=(
                    city_count <= self.warm_start_two_opt_max_cities
                ),
            )
            initial_cost = _tour_cost(initial_tour, distances)
            self._set_mip_start(variables, initial_tour)

        callback = _SubtourCutCallback(
            variables=variables,
            city_count=city_count,
            separate_fractional_cuts=self.separate_fractional_cuts,
            fractional_cut_node_interval=(
                self.fractional_cut_node_interval
            ),
            fractional_cut_max_cities=self.fractional_cut_max_cities,
            cut_tolerance=self.cut_tolerance,
        )

        model.optimize(callback)

        if model.Status != GRB.OPTIMAL:
            self._raise_nonoptimal_status(model)

        solution = model.getAttr("X", variables)
        selected_edges = [
            edge
            for edge, value in solution.items()
            if float(value) > 0.5
        ]
        tour = self._reconstruct_tour(city_count, selected_edges)
        cost = _tour_cost(tour, distances)

        metadata: dict[str, Any] = {
            "algorithm": "branch-and-cut",
            "backend": "gurobi",
            "exact": True,
            "nodes": int(model.NodeCount),
            "runtime_seconds": float(model.Runtime),
            "lazy_subtour_cuts": callback.lazy_cuts,
            "fractional_subtour_cuts": callback.fractional_cuts,
            "objective_bound": float(model.ObjBound),
        }
        if initial_cost is not None:
            metadata["initial_tour_cost"] = initial_cost

        return SolverResult(
            tour=tour,
            cost=cost,
            metadata=metadata,
        )

    def _validated_distances(
        self,
        problem: TravellingSalespersonProblem,
    ) -> np.ndarray:
        city_count = problem.city_count
        if city_count < 1:
            raise ValueError("TSP requires at least one city")

        distances = np.ascontiguousarray(
            problem.distances,
            dtype=np.float64,
        )
        expected_shape = (city_count, city_count)

        if distances.shape != expected_shape:
            raise ValueError(
                "Distance matrix shape does not match city_count: "
                f"expected {expected_shape}, received {distances.shape}"
            )
        if not np.isfinite(distances).all():
            raise ValueError(
                "A complete TSP distance matrix must contain only "
                "finite values"
            )
        if not np.allclose(
            distances,
            distances.T,
            rtol=0.0,
            atol=self.symmetry_tolerance,
        ):
            maximum_asymmetry = float(
                np.max(np.abs(distances - distances.T))
            )
            raise ValueError(
                "BranchAndCutTSPSolver requires a symmetric distance "
                "matrix; maximum asymmetry is "
                f"{maximum_asymmetry:.3e}"
            )

        return distances

    def _configure_model(self, model: Any) -> None:
        model.Params.OutputFlag = int(self.log_output)
        model.Params.Seed = self.seed
        model.Params.MIPGap = 0.0
        model.Params.MIPGapAbs = 0.0
        model.Params.LazyConstraints = 1

        if self.separate_fractional_cuts:
            # Preserve original variable meanings for callback user cuts.
            model.Params.PreCrush = 1

        if self.time_limit is not None:
            model.Params.TimeLimit = self.time_limit
        if self.threads is not None:
            model.Params.Threads = self.threads

    @staticmethod
    def _set_mip_start(
        variables: Any,
        tour: Sequence[int],
    ) -> None:
        selected = {
            _edge(tour[position], tour[(position + 1) % len(tour)])
            for position in range(len(tour))
        }
        for edge, variable in variables.items():
            variable.Start = 1.0 if edge in selected else 0.0

    @staticmethod
    def _reconstruct_tour(
        city_count: int,
        selected_edges: Sequence[Edge],
    ) -> tuple[int, ...]:
        adjacency: list[list[int]] = [[] for _ in range(city_count)]
        for first, second in selected_edges:
            adjacency[first].append(second)
            adjacency[second].append(first)

        if any(len(neighbours) != 2 for neighbours in adjacency):
            raise RuntimeError(
                "Optimizer returned a solution that is not degree two"
            )

        # Choose the lexicographically smaller orientation for reproducibility.
        tour = [0]
        previous = -1
        current = 0
        next_city = min(adjacency[0])

        while next_city != 0:
            tour.append(next_city)
            previous, current = current, next_city
            first, second = adjacency[current]
            next_city = second if first == previous else first

            if len(tour) > city_count:
                raise RuntimeError("Invalid cycle returned by optimizer")

        if len(tour) != city_count:
            raise RuntimeError(
                "Optimizer returned a disconnected collection of subtours"
            )

        return tuple(tour)

    @staticmethod
    def _raise_nonoptimal_status(model: Any) -> None:
        status = int(model.Status)
        status_names = {
            GRB.LOADED: "LOADED",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.CUTOFF: "CUTOFF",
            GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
            GRB.NODE_LIMIT: "NODE_LIMIT",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.NUMERIC: "NUMERIC",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
        }
        status_name = status_names.get(status, str(status))

        if model.SolCount:
            raise RuntimeError(
                "TSP solve stopped without an optimality proof "
                f"(status={status_name}, incumbent={model.ObjVal}, "
                f"bound={model.ObjBound}). No approximate result was "
                "returned because this solver promises exact solutions."
            )

        raise RuntimeError(
            "TSP solve ended without a feasible optimal solution "
            f"(status={status_name})"
        )
