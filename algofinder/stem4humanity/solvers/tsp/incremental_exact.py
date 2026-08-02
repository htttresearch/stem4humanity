"""Fast self-contained exact solver for small symmetric complete TSPs.

The solver implements an assignment-relaxation branch-and-bound algorithm.
It requires no MILP/LP package, SciPy, NetworkX, Numba, or commercial solver;
only NumPy, already used by the project, is required.

The important optimization is incremental assignment reoptimization. Each
branch fixes a prefix of edges from the parent's optimal cycle cover and
forbids one matched edge. The remaining assignment is repaired with one
shortest augmenting-path step in O(n^2), instead of solving a new O(n^3)
Hungarian problem from scratch.

This is exact for finite symmetric complete distance matrices, subject to the
usual floating-point semantics of the supplied float64 coefficients. Worst-
case time remains exponential. The default 25-city limit is intentional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import itertools
import math
import time
from collections.abc import Iterable, Sequence

import numpy as np

from stem4humanity.problems.tsp import (
    TravellingSalespersonProblem,
)
from stem4humanity.solvers.base import Solver, SolverResult, register_solver, InapplicableError


_INF = float("inf")


@dataclass(order=True, slots=True)
class _QueueNode:
    """A search node ordered by assignment lower bound."""

    lower_bound: float
    serial: int
    forbidden_bits: int = field(compare=False)
    forced_out: tuple[int, ...] = field(compare=False)
    forced_in: tuple[int, ...] = field(compare=False)
    assignment: tuple[int, ...] = field(compare=False)
    cycles: tuple[tuple[int, ...], ...] = field(compare=False)
    dual_u: tuple[float, ...] = field(compare=False)
    dual_v: tuple[float, ...] = field(compare=False)
    depth: int = field(compare=False)


def _edge_code(city_count: int, source: int, target: int) -> int:
    return source * city_count + target


def _is_forbidden(forbidden_bits: int, code: int) -> bool:
    return bool((forbidden_bits >> code) & 1)


def _tour_cost(tour: Sequence[int], distances: np.ndarray) -> float:
    n = len(tour)
    total = 0.0
    previous = int(tour[-1])
    for current_value in tour:
        current = int(current_value)
        total += float(distances[previous, current])
        previous = current
    return total


def _canonical_tour(tour: Iterable[int]) -> tuple[int, ...]:
    """Rotate a tour to zero and choose one of its two orientations."""

    route = tuple(int(city) for city in tour)
    zero_position = route.index(0)
    route = route[zero_position:] + route[:zero_position]
    reverse = (0,) + tuple(reversed(route[1:]))
    return route if route <= reverse else reverse


def _assignment_cycles(
    successor: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Decompose a permutation into cycles, shortest first."""

    n = len(successor)
    seen = bytearray(n)
    cycles: list[tuple[int, ...]] = []

    for start in range(n):
        if seen[start]:
            continue
        cycle: list[int] = []
        city = start
        while not seen[city]:
            seen[city] = 1
            cycle.append(city)
            city = int(successor[city])
        cycles.append(tuple(cycle))

    cycles.sort(key=len)
    return tuple(cycles)


def _tour_from_successor(
    successor: Sequence[int],
    start: int = 0,
) -> tuple[int, ...]:
    tour = [start]
    current = start
    for _ in range(1, len(successor)):
        current = int(successor[current])
        tour.append(current)
    return tuple(tour)


def _hungarian_root(
    costs: np.ndarray,
) -> tuple[
    float,
    tuple[int, ...],
    tuple[float, ...],
    tuple[float, ...],
] | None:
    """Solve the root assignment and return reusable dual potentials.

    This is the shortest-augmenting-path Hungarian algorithm. Infinite entries
    are forbidden. Potentials satisfy u[i] + v[j] <= cost[i, j] for allowed
    entries, and selected assignment entries are tight up to rounding.
    """

    n = int(costs.shape[0])
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    column_row = [0] * (n + 1)
    previous_column = [0] * (n + 1)
    min_value = [_INF] * (n + 1)
    used = [False] * (n + 1)

    for row_one_based in range(1, n + 1):
        column_row[0] = row_one_based
        for column in range(n + 1):
            min_value[column] = _INF
            used[column] = False

        current_column = 0
        while True:
            used[current_column] = True
            row = column_row[current_column]
            row_costs = costs[row - 1]
            row_potential = u[row]
            delta = _INF
            next_column = -1

            for column in range(1, n + 1):
                if used[column]:
                    continue
                coefficient = float(row_costs[column - 1])
                if math.isfinite(coefficient):
                    reduced = coefficient - row_potential - v[column]
                    if reduced < min_value[column]:
                        min_value[column] = reduced
                        previous_column[column] = current_column
                if min_value[column] < delta:
                    delta = min_value[column]
                    next_column = column

            if next_column < 0 or not math.isfinite(delta):
                return None

            for column in range(n + 1):
                if used[column]:
                    u[column_row[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta

            current_column = next_column
            if column_row[current_column] == 0:
                break

        while True:
            prior = previous_column[current_column]
            column_row[current_column] = column_row[prior]
            current_column = prior
            if current_column == 0:
                break

    assignment = [-1] * n
    for column in range(1, n + 1):
        row = column_row[column] - 1
        if row < 0:
            return None
        assignment[row] = column - 1

    objective = 0.0
    for row, column in enumerate(assignment):
        coefficient = float(costs[row, column])
        if not math.isfinite(coefficient):
            return None
        objective += coefficient

    return (
        objective,
        tuple(assignment),
        tuple(u[1:]),
        tuple(v[1:]),
    )


def _repair_assignment(
    costs: np.ndarray,
    parent_assignment: Sequence[int],
    parent_u: Sequence[float],
    parent_v: Sequence[float],
    forbidden_bits: int,
    forced_out: Sequence[int],
    forced_in: Sequence[int],
    broken_row: int,
) -> tuple[
    float,
    tuple[int, ...],
    tuple[float, ...],
    tuple[float, ...],
] | None:
    """Reoptimize after forbidding one edge used by the parent assignment.

    Newly forced edges are all parent-assignment edges. Removing their rows and
    columns preserves optimality of the remaining parent matching. Forbidding
    one additional matched edge leaves one unmatched row and column, so one
    Hungarian shortest augmenting-path step restores an optimal assignment.
    """

    n = len(parent_assignment)
    assignment = list(parent_assignment)
    inverse = [-1] * n
    active_columns: list[int] = []

    for column in range(n):
        if forced_in[column] < 0:
            active_columns.append(column)

    for row in range(n):
        if forced_out[row] < 0:
            column = assignment[row]
            if column < 0 or forced_in[column] >= 0:
                return None
            inverse[column] = row

    broken_column = assignment[broken_row]
    if broken_column < 0 or forced_out[broken_row] >= 0:
        return None
    if forced_in[broken_column] >= 0:
        return None

    assignment[broken_row] = -1
    inverse[broken_column] = -1

    u = list(parent_u)
    v = list(parent_v)
    min_value = [_INF] * n
    predecessor_row = [-1] * n
    used_column = bytearray(n)

    # Alternating-tree rows consist of the unmatched row and rows matched to
    # columns already selected into the tree.
    current_row = broken_row

    while True:
        row_costs = costs[current_row]
        row_potential = u[current_row]
        delta = _INF
        next_column = -1

        for column in active_columns:
            if used_column[column]:
                continue
            code = _edge_code(n, current_row, column)
            if _is_forbidden(forbidden_bits, code):
                continue

            coefficient = float(row_costs[column])
            if not math.isfinite(coefficient):
                continue

            reduced = coefficient - row_potential - v[column]
            # Tiny negative reduced costs can result from accumulated roundoff.
            if reduced < 0.0 and reduced > -1e-12:
                reduced = 0.0
            if reduced < min_value[column]:
                min_value[column] = reduced
                predecessor_row[column] = current_row
            if min_value[column] < delta:
                delta = min_value[column]
                next_column = column

        if next_column < 0 or not math.isfinite(delta):
            return None

        # Dual update for all rows and columns currently in the alternating
        # tree, followed by the usual slack update outside the tree.
        u[broken_row] += delta
        for column in active_columns:
            if used_column[column]:
                matched_row = inverse[column]
                if matched_row < 0:
                    return None
                u[matched_row] += delta
                v[column] -= delta
            elif math.isfinite(min_value[column]):
                min_value[column] -= delta

        if inverse[next_column] < 0:
            terminal_column = next_column
            break

        used_column[next_column] = 1
        current_row = inverse[next_column]

    # Flip the alternating path. predecessor_row is stable while assignment
    # and inverse are mutated.
    column = terminal_column
    while True:
        row = predecessor_row[column]
        if row < 0:
            return None
        prior_column = assignment[row]
        assignment[row] = column
        inverse[column] = row
        if prior_column < 0:
            break
        column = prior_column

    objective = 0.0
    for row, column in enumerate(assignment):
        if column < 0:
            return None
        code = _edge_code(n, row, column)
        if _is_forbidden(forbidden_bits, code):
            return None
        if forced_out[row] >= 0 and forced_out[row] != column:
            return None
        if forced_in[column] >= 0 and forced_in[column] != row:
            return None
        coefficient = float(costs[row, column])
        if not math.isfinite(coefficient):
            return None
        objective += coefficient

    return objective, tuple(assignment), tuple(u), tuple(v)


def _nearest_neighbour_tour(
    distances: np.ndarray,
    start: int,
) -> tuple[int, ...]:
    n = int(distances.shape[0])
    unused_mask = ((1 << n) - 1) ^ (1 << start)
    route = [start]
    current = start

    while unused_mask:
        best_city = -1
        best_cost = _INF
        bits = unused_mask
        while bits:
            bit = bits & -bits
            city = bit.bit_length() - 1
            value = float(distances[current, city])
            if value < best_cost:
                best_cost = value
                best_city = city
            bits ^= bit
        route.append(best_city)
        unused_mask ^= 1 << best_city
        current = best_city

    return tuple(route)


def _cheapest_insertion_tour(
    distances: np.ndarray,
    start: int,
) -> tuple[int, ...]:
    """Construct a deterministic cheapest-insertion tour."""

    n = int(distances.shape[0])
    nearest = min(
        (city for city in range(n) if city != start),
        key=lambda city: (float(distances[start, city]), city),
    )
    route = [start, nearest]
    unused = set(range(n))
    unused.remove(start)
    unused.remove(nearest)

    while unused:
        best_delta = _INF
        best_city = -1
        best_position = -1
        route_length = len(route)
        for city in unused:
            for position in range(route_length):
                first = route[position]
                second = route[(position + 1) % route_length]
                delta = (
                    float(distances[first, city])
                    + float(distances[city, second])
                    - float(distances[first, second])
                )
                if delta < best_delta:
                    best_delta = delta
                    best_city = city
                    best_position = position + 1
        route.insert(best_position, best_city)
        unused.remove(best_city)

    return tuple(route)


def _two_opt(
    tour: Iterable[int],
    distances: np.ndarray,
    *,
    max_passes: int | None,
) -> tuple[tuple[int, ...], float]:
    """Deterministic best-improvement 2-opt for symmetric costs."""

    route = list(int(city) for city in tour)
    n = len(route)
    passes = 0

    while n >= 4 and (max_passes is None or passes < max_passes):
        passes += 1
        best_delta = 0.0
        best_first = -1
        best_last = -1

        for first in range(n - 1):
            a = route[first]
            b = route[(first + 1) % n]
            final_last = n - 1 if first > 0 else n - 2
            for last in range(first + 2, final_last + 1):
                c = route[last]
                d = route[(last + 1) % n]
                delta = (
                    float(distances[a, c])
                    + float(distances[b, d])
                    - float(distances[a, b])
                    - float(distances[c, d])
                )
                if delta < best_delta:
                    best_delta = delta
                    best_first = first + 1
                    best_last = last

        if best_first < 0:
            break
        route[best_first : best_last + 1] = reversed(
            route[best_first : best_last + 1]
        )

    canonical = _canonical_tour(route)
    return canonical, _tour_cost(canonical, distances)


def _greedy_multifragment_tour(
    distances: np.ndarray,
) -> tuple[int, ...] | None:
    """Construct a tour by Kruskal-style cheapest feasible edge insertion."""

    n = int(distances.shape[0])
    edges = [
        (float(distances[first, second]), first, second)
        for first in range(n)
        for second in range(first + 1, n)
    ]
    edges.sort()

    degree = [0] * n
    parent = list(range(n))
    component_size = [1] * n
    selected: list[tuple[int, int]] = []

    def find(city: int) -> int:
        root = city
        while parent[root] != root:
            root = parent[root]
        while parent[city] != city:
            next_city = parent[city]
            parent[city] = root
            city = next_city
        return root

    for _, first, second in edges:
        if degree[first] == 2 or degree[second] == 2:
            continue
        first_root = find(first)
        second_root = find(second)
        same_component = first_root == second_root
        if same_component and len(selected) != n - 1:
            continue

        selected.append((first, second))
        degree[first] += 1
        degree[second] += 1

        if not same_component:
            if component_size[first_root] < component_size[second_root]:
                first_root, second_root = second_root, first_root
            parent[second_root] = first_root
            component_size[first_root] += component_size[second_root]

        if len(selected) == n:
            break

    if len(selected) != n or any(value != 2 for value in degree):
        return None

    adjacency = [[] for _ in range(n)]
    for first, second in selected:
        adjacency[first].append(second)
        adjacency[second].append(first)

    route = [0]
    previous = -1
    current = 0
    for _ in range(1, n):
        left, right = adjacency[current]
        next_city = left if left != previous else right
        route.append(next_city)
        previous, current = current, next_city
    return tuple(route)


def _patch_cycle_cover(
    assignment: Sequence[int],
    cycles: Sequence[Sequence[int]],
    distances: np.ndarray,
) -> tuple[int, ...]:
    """Merge directed assignment cycles using cheapest two-arc exchanges."""

    successor = list(int(city) for city in assignment)
    groups = [set(int(city) for city in cycle) for cycle in cycles]

    while len(groups) > 1:
        best_delta = _INF
        best: tuple[int, int, int, int] | None = None

        for first_index in range(len(groups) - 1):
            first_group = groups[first_index]
            for second_index in range(first_index + 1, len(groups)):
                second_group = groups[second_index]
                for first in first_group:
                    first_next = successor[first]
                    old_first = float(distances[first, first_next])
                    for second in second_group:
                        second_next = successor[second]
                        delta = (
                            float(distances[first, second_next])
                            + float(distances[second, first_next])
                            - old_first
                            - float(distances[second, second_next])
                        )
                        if delta < best_delta:
                            best_delta = delta
                            best = (
                                first_index,
                                second_index,
                                first,
                                second,
                            )

        if best is None:
            raise RuntimeError("failed to patch assignment cycles")

        first_index, second_index, first, second = best
        successor[first], successor[second] = (
            successor[second],
            successor[first],
        )
        groups[first_index].update(groups[second_index])
        del groups[second_index]

    return _tour_from_successor(successor)


@register_solver
class ExactTSPSolver(Solver):
    id = "incremental-exact"
    display = "Incremental-assignment branch-and-bound (exact)"
    tags = frozenset({"exact"})
    applies_to = frozenset({"tsp:general", "tsp:euclidean", "tsp:clustered"})
    """Solve symmetric complete TSP exactly without an optimizer library.

    The implementation targets instances up to roughly 20--25 cities. Easy or
    structured instances may solve much faster; adversarial instances can still
    require exponential time.
    """

    def __init__(
        self,
        max_cities: int = 25,
        *,
        nearest_neighbour_starts: int = 25,
        insertion_starts: int = 8,
        root_two_opt_passes: int | None = None,
        patch_every_nodes: int = 32,
        node_two_opt_passes: int = 2,
        verify_symmetry: bool = True,
    ) -> None:
        if max_cities < 3:
            raise ValueError("max_cities must be at least three")
        if nearest_neighbour_starts < 1:
            raise ValueError("nearest_neighbour_starts must be positive")
        if insertion_starts < 0:
            raise ValueError("insertion_starts must be non-negative")
        if patch_every_nodes < 0:
            raise ValueError("patch_every_nodes must be non-negative")
        if node_two_opt_passes < 0:
            raise ValueError("node_two_opt_passes must be non-negative")

        self.max_cities = max_cities
        self.nearest_neighbour_starts = nearest_neighbour_starts
        self.insertion_starts = insertion_starts
        self.root_two_opt_passes = root_two_opt_passes
        self.patch_every_nodes = patch_every_nodes
        self.node_two_opt_passes = node_two_opt_passes
        self.verify_symmetry = verify_symmetry

    def _initial_upper_bound(
        self,
        distances: np.ndarray,
    ) -> tuple[tuple[int, ...], float]:
        n = int(distances.shape[0])
        best_tour: tuple[int, ...] | None = None
        best_cost = _INF

        multifragment = _greedy_multifragment_tour(distances)
        if multifragment is not None:
            best_tour, best_cost = _two_opt(
                multifragment,
                distances,
                max_passes=self.root_two_opt_passes,
            )

        nearest_count = min(n, self.nearest_neighbour_starts)
        if nearest_count == n:
            nearest_starts = range(n)
        else:
            nearest_starts = sorted(
                {
                    round(index * (n - 1) / max(1, nearest_count - 1))
                    for index in range(nearest_count)
                }
            )

        for start in nearest_starts:
            candidate, candidate_cost = _two_opt(
                _nearest_neighbour_tour(distances, int(start)),
                distances,
                max_passes=self.root_two_opt_passes,
            )
            if candidate_cost < best_cost:
                best_tour = candidate
                best_cost = candidate_cost

        insertion_count = min(n, self.insertion_starts)
        if insertion_count:
            if insertion_count == n:
                insertion_starts = range(n)
            else:
                insertion_starts = sorted(
                    {
                        round(index * (n - 1) / max(1, insertion_count - 1))
                        for index in range(insertion_count)
                    }
                )
            for start in insertion_starts:
                candidate, candidate_cost = _two_opt(
                    _cheapest_insertion_tour(distances, int(start)),
                    distances,
                    max_passes=self.root_two_opt_passes,
                )
                if candidate_cost < best_cost:
                    best_tour = candidate
                    best_cost = candidate_cost

        if best_tour is None:
            raise RuntimeError("failed to construct an initial tour")
        return best_tour, best_cost

    @staticmethod
    def _would_close_premature_cycle(
        forced_out: Sequence[int],
        source: int,
        target: int,
    ) -> bool:
        n = len(forced_out)
        current = target
        traversed = 1
        while current >= 0 and current != source and traversed <= n:
            current = int(forced_out[current])
            traversed += 1
        if current != source:
            return False
        return sum(value >= 0 for value in forced_out) < n

    @staticmethod
    def _include_edge(
        n: int,
        source: int,
        target: int,
        forbidden_bits: int,
        forced_out: list[int],
        forced_in: list[int],
    ) -> bool:
        if source == target:
            return False
        code = _edge_code(n, source, target)
        if _is_forbidden(forbidden_bits, code):
            return False

        existing_target = forced_out[source]
        existing_source = forced_in[target]
        if existing_target not in (-1, target):
            return False
        if existing_source not in (-1, source):
            return False
        if existing_target == target:
            return True

        forced_out[source] = target
        forced_in[target] = source
        if ExactTSPSolver._would_close_premature_cycle(
            forced_out,
            source,
            target,
        ):
            forced_out[source] = -1
            forced_in[target] = -1
            return False
        return True

    @staticmethod
    def _branch_penalty(
        node: _QueueNode,
        source: int,
        target: int,
        costs: np.ndarray,
    ) -> float:
        """Little-style reduced-cost penalty used only for branch ordering."""

        n = len(node.assignment)
        row_minimum = _INF
        column_minimum = _INF
        source_u = node.dual_u[source]
        target_v = node.dual_v[target]

        for column in range(n):
            if column == target or node.forced_in[column] >= 0:
                continue
            code = _edge_code(n, source, column)
            if _is_forbidden(node.forbidden_bits, code):
                continue
            coefficient = float(costs[source, column])
            if math.isfinite(coefficient):
                reduced = coefficient - source_u - node.dual_v[column]
                if reduced < row_minimum:
                    row_minimum = reduced

        for row in range(n):
            if row == source or node.forced_out[row] >= 0:
                continue
            code = _edge_code(n, row, target)
            if _is_forbidden(node.forbidden_bits, code):
                continue
            coefficient = float(costs[row, target])
            if math.isfinite(coefficient):
                reduced = coefficient - node.dual_u[row] - target_v
                if reduced < column_minimum:
                    column_minimum = reduced

        if not math.isfinite(row_minimum):
            row_minimum = 1e300
        if not math.isfinite(column_minimum):
            column_minimum = 1e300
        return row_minimum + column_minimum

    def solve(
        self,
        problem: TravellingSalespersonProblem,
        *,
        budget_seconds: float | None = None,
    ) -> SolverResult:
        started = time.perf_counter()
        n = int(problem.city_count)

        if n < 1:
            raise ValueError("TSP requires at least one city")
        if n > self.max_cities:
            raise InapplicableError(
                f"exact solver is limited to {self.max_cities} cities; "
                f"received {n}"
            )

        distances = np.ascontiguousarray(
            problem.distances,
            dtype=np.float64,
        )
        if distances.shape != (n, n):
            raise ValueError(
                f"expected a {(n, n)} distance matrix, "
                f"received {distances.shape}"
            )
        if not np.isfinite(distances).all():
            raise ValueError("complete TSP distances must all be finite")
        if self.verify_symmetry and not np.array_equal(
            distances,
            distances.T,
        ):
            raise ValueError("this solver requires an exactly symmetric matrix")

        if n == 1:
            return SolverResult(
                solution=(0,),
                cost=0.0,
                exact=True,
                wall_seconds=time.perf_counter() - started,
                metadata={
                    "algorithm": "incremental-assignment-branch-and-bound",
                    "nodes": 1,
                },
            )
        if n == 2:
            tour = (0, 1)
            return SolverResult(
                solution=tour,
                cost=_tour_cost(tour, distances),
                exact=True,
                wall_seconds=time.perf_counter() - started,
                metadata={
                    "algorithm": "incremental-assignment-branch-and-bound",
                    "nodes": 1,
                },
            )

        incumbent_tour, incumbent_cost = self._initial_upper_bound(distances)

        assignment_costs = distances.copy()
        np.fill_diagonal(assignment_costs, _INF)

        root_solution = _hungarian_root(assignment_costs)
        if root_solution is None:
            raise ValueError("the instance has no Hamiltonian tour")

        root_bound, root_assignment, root_u, root_v = root_solution
        root_cycles = _assignment_cycles(root_assignment)

        # Patching the root cycle cover is cheap and often gives a much better
        # initial incumbent for arbitrary non-geometric matrices.
        if len(root_cycles) > 1:
            patched, patched_cost = _two_opt(
                _patch_cycle_cover(
                    root_assignment,
                    root_cycles,
                    distances,
                ),
                distances,
                max_passes=self.node_two_opt_passes,
            )
            if patched_cost < incumbent_cost:
                incumbent_tour = patched
                incumbent_cost = patched_cost

        serial_counter = itertools.count()
        empty_forced = tuple([-1] * n)
        queue: list[_QueueNode] = [
            _QueueNode(
                lower_bound=root_bound,
                serial=next(serial_counter),
                forbidden_bits=0,
                forced_out=empty_forced,
                forced_in=empty_forced,
                assignment=root_assignment,
                cycles=root_cycles,
                dual_u=root_u,
                dual_v=root_v,
                depth=0,
            )
        ]

        nodes_popped = 0
        nodes_generated = 1
        nodes_pruned_bound = 0
        nodes_pruned_infeasible = 0
        incremental_repairs = 0
        patched_tours = 1 if len(root_cycles) > 1 else 0
        maximum_queue = 1
        maximum_depth = 0

        while queue:
            node = heapq.heappop(queue)
            nodes_popped += 1
            if node.depth > maximum_depth:
                maximum_depth = node.depth

            if node.lower_bound >= incumbent_cost:
                nodes_pruned_bound += 1
                continue

            if len(node.cycles) == 1:
                candidate = _canonical_tour(
                    _tour_from_successor(node.assignment)
                )
                candidate_cost = _tour_cost(candidate, distances)
                if candidate_cost < incumbent_cost:
                    incumbent_tour = candidate
                    incumbent_cost = candidate_cost
                continue

            if (
                self.patch_every_nodes > 0
                and nodes_popped % self.patch_every_nodes == 0
            ):
                patched, patched_cost = _two_opt(
                    _patch_cycle_cover(
                        node.assignment,
                        node.cycles,
                        distances,
                    ),
                    distances,
                    max_passes=self.node_two_opt_passes,
                )
                patched_tours += 1
                if patched_cost < incumbent_cost:
                    incumbent_tour = patched
                    incumbent_cost = patched_cost
                    if node.lower_bound >= incumbent_cost:
                        nodes_pruned_bound += 1
                        continue

            branch_cycle: tuple[int, ...] | None = None
            branch_edges: list[tuple[int, int]] = []
            for cycle in node.cycles:
                available = [
                    (source, int(node.assignment[source]))
                    for source in cycle
                    if node.forced_out[source] < 0
                ]
                if available:
                    branch_cycle = cycle
                    branch_edges = available
                    break

            if branch_cycle is None:
                nodes_pruned_infeasible += 1
                continue

            branch_edges.sort(
                key=lambda edge: self._branch_penalty(
                    node,
                    edge[0],
                    edge[1],
                    assignment_costs,
                ),
                reverse=True,
            )

            prefix_forbidden = node.forbidden_bits
            prefix_out = list(node.forced_out)
            prefix_in = list(node.forced_in)

            # Exact disjunction for a proper assignment subtour:
            #   exclude e0;
            #   include e0, exclude e1;
            #   include e0,e1, exclude e2; ...
            # The omitted all-included case is an infeasible proper subtour.
            for source, target in branch_edges:
                code = _edge_code(n, source, target)
                child_forbidden = prefix_forbidden | (1 << code)
                child_out = tuple(prefix_out)
                child_in = tuple(prefix_in)

                repaired = _repair_assignment(
                    assignment_costs,
                    node.assignment,
                    node.dual_u,
                    node.dual_v,
                    child_forbidden,
                    child_out,
                    child_in,
                    source,
                )
                incremental_repairs += 1

                if repaired is None:
                    nodes_pruned_infeasible += 1
                else:
                    bound, assignment, dual_u, dual_v = repaired
                    if bound >= incumbent_cost:
                        nodes_pruned_bound += 1
                    else:
                        cycles = _assignment_cycles(assignment)
                        heapq.heappush(
                            queue,
                            _QueueNode(
                                lower_bound=bound,
                                serial=next(serial_counter),
                                forbidden_bits=child_forbidden,
                                forced_out=child_out,
                                forced_in=child_in,
                                assignment=assignment,
                                cycles=cycles,
                                dual_u=dual_u,
                                dual_v=dual_v,
                                depth=node.depth + 1,
                            ),
                        )
                        nodes_generated += 1

                if not self._include_edge(
                    n,
                    source,
                    target,
                    prefix_forbidden,
                    prefix_out,
                    prefix_in,
                ):
                    break

            if len(queue) > maximum_queue:
                maximum_queue = len(queue)

        incumbent_tour = _canonical_tour(incumbent_tour)
        incumbent_cost = _tour_cost(incumbent_tour, distances)
        runtime = time.perf_counter() - started

        return SolverResult(
            solution=incumbent_tour,
            cost=incumbent_cost,
            exact=True,
            wall_seconds=runtime,
            metadata={
                "algorithm": "incremental-assignment-branch-and-bound",
                "max_cities": self.max_cities,
                "nodes": nodes_popped,
                "nodes_generated": nodes_generated,
                "nodes_pruned_bound": nodes_pruned_bound,
                "nodes_pruned_infeasible": nodes_pruned_infeasible,
                "incremental_assignment_repairs": incremental_repairs,
                "full_assignment_solves": 1,
                "patched_tours": patched_tours,
                "maximum_queue": maximum_queue,
                "maximum_depth": maximum_depth,
                "root_lower_bound": root_bound,
                "runtime_seconds": runtime,
            },
        )


IncrementalExactTSPSolver = ExactTSPSolver
