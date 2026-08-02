"""Self-contained exact solver for complete symmetric TSP instances.

The solver uses branch-and-bound over the assignment relaxation:

* A minimum-cost cycle cover is computed with an in-file Hungarian algorithm.
* A one-cycle assignment is a Hamiltonian tour.
* If the assignment contains subtours, a shortest subtour is selected and the
  node is partitioned by forcing/excluding its edges.
* Feasible tours from greedy construction, assignment-cycle patching, and
  exact 2-opt local search provide strong upper bounds.

No MILP/LP solver, SciPy, NetworkX, Numba, or commercial package is required.
Only NumPy, already used by the project, is needed.

Worst-case running time remains exponential: exact general TSP is NP-hard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import itertools
import math
import time
from typing import Iterable

import numpy as np

from src.problems.travelling_salesperson_problem import TravellingSalespersonProblem
from src.solvers.base_solver import BaseSolver, SolverResult


_INF = float("inf")


@dataclass(order=True, slots=True)
class _QueueNode:
    """A branch-and-bound node ordered by lower bound, then insertion order."""

    lower_bound: float
    serial: int
    forbidden: frozenset[int] = field(compare=False)
    forced_out: tuple[int, ...] = field(compare=False)
    forced_in: tuple[int, ...] = field(compare=False)
    assignment: tuple[int, ...] = field(compare=False)
    cycles: tuple[tuple[int, ...], ...] = field(compare=False)
    depth: int = field(compare=False)


def _hungarian(cost: np.ndarray) -> tuple[float, np.ndarray] | None:
    """Solve a square linear assignment problem in O(n^3).

    This is the shortest augmenting-path Hungarian algorithm. Infinite entries
    represent forbidden assignments. The returned objective is recomputed from
    the selected original coefficients rather than from the dual potentials.
    """

    n = int(cost.shape[0])
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int32)
    way = np.zeros(n + 1, dtype=np.int32)

    minv = np.empty(n + 1, dtype=np.float64)
    used = np.empty(n + 1, dtype=np.bool_)

    for i in range(1, n + 1):
        p[0] = i
        minv.fill(_INF)
        used.fill(False)
        j0 = 0

        while True:
            used[j0] = True
            i0 = int(p[j0])
            row = cost[i0 - 1]
            delta = _INF
            j1 = -1

            ui = float(u[i0])
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cij = float(row[j - 1])
                if math.isinf(cij):
                    continue
                cur = cij - ui - float(v[j])
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0

            for j in range(1, n + 1):
                if used[j]:
                    continue
                if minv[j] < delta:
                    delta = float(minv[j])
                    j1 = j

            if j1 < 0 or math.isinf(delta):
                return None

            for j in range(n + 1):
                if used[j]:
                    pj = int(p[j])
                    u[pj] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = np.empty(n, dtype=np.int32)
    for j in range(1, n + 1):
        row = int(p[j]) - 1
        if row < 0:
            return None
        assignment[row] = j - 1

    selected = cost[np.arange(n), assignment]
    if not np.isfinite(selected).all():
        return None
    return float(np.sum(selected, dtype=np.float64)), assignment


def _assignment_cycles(successor: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    """Decompose a permutation into directed cycles."""

    succ = tuple(int(x) for x in successor)
    n = len(succ)
    seen = bytearray(n)
    cycles: list[tuple[int, ...]] = []

    for start in range(n):
        if seen[start]:
            continue
        cycle: list[int] = []
        current = start
        while not seen[current]:
            seen[current] = 1
            cycle.append(current)
            current = succ[current]
        cycles.append(tuple(cycle))

    cycles.sort(key=len)
    return tuple(cycles)


def _tour_from_successor(successor: Iterable[int], start: int = 0) -> tuple[int, ...]:
    succ = tuple(int(x) for x in successor)
    tour = [start]
    current = start
    for _ in range(1, len(succ)):
        current = succ[current]
        tour.append(current)
    return tuple(tour)


def _tour_cost(tour: Iterable[int], distances: np.ndarray) -> float:
    route = tuple(int(x) for x in tour)
    n = len(route)
    return float(
        sum(
            float(distances[route[i], route[(i + 1) % n]])
            for i in range(n)
        )
    )


def _canonical_tour(tour: Iterable[int]) -> tuple[int, ...]:
    """Rotate to city zero and choose a deterministic orientation."""

    route = tuple(int(x) for x in tour)
    zero = route.index(0)
    route = route[zero:] + route[:zero]
    reverse = (0,) + tuple(reversed(route[1:]))
    return min(route, reverse)


def _two_opt(
    tour: Iterable[int],
    distances: np.ndarray,
    *,
    max_passes: int | None = None,
) -> tuple[tuple[int, ...], float]:
    """Run deterministic best-improvement 2-opt for symmetric costs."""

    route = list(int(x) for x in tour)
    n = len(route)
    if n < 4:
        canonical = _canonical_tour(route)
        return canonical, _tour_cost(canonical, distances)

    passes = 0
    while max_passes is None or passes < max_passes:
        passes += 1
        best_delta = 0.0
        best_i = -1
        best_j = -1

        for i in range(n - 1):
            a = route[i]
            b = route[(i + 1) % n]
            max_j = n - 1 if i > 0 else n - 2
            for j in range(i + 2, max_j + 1):
                c = route[j]
                d = route[(j + 1) % n]
                delta = (
                    float(distances[a, c])
                    + float(distances[b, d])
                    - float(distances[a, b])
                    - float(distances[c, d])
                )
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j

        if best_i < 0:
            break
        route[best_i + 1 : best_j + 1] = reversed(
            route[best_i + 1 : best_j + 1]
        )

    canonical = _canonical_tour(route)
    return canonical, _tour_cost(canonical, distances)


def _nearest_neighbour_tour(distances: np.ndarray, start: int) -> tuple[int, ...]:
    n = int(distances.shape[0])
    unused = set(range(n))
    unused.remove(start)
    route = [start]
    current = start

    while unused:
        next_city = min(unused, key=lambda city: (distances[current, city], city))
        unused.remove(next_city)
        route.append(next_city)
        current = next_city

    return tuple(route)


def _greedy_multifragment_tour(distances: np.ndarray) -> tuple[int, ...] | None:
    """Build a tour with the sorted-edge multi-fragment heuristic."""

    n = int(distances.shape[0])
    edges = [
        (float(distances[i, j]), i, j)
        for i in range(n)
        for j in range(i + 1, n)
    ]
    edges.sort()

    degree = [0] * n
    parent = list(range(n))
    size = [1] * n
    selected: list[tuple[int, int]] = []

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for _, a, b in edges:
        if degree[a] == 2 or degree[b] == 2:
            continue
        same = find(a) == find(b)
        if same and len(selected) != n - 1:
            continue
        selected.append((a, b))
        degree[a] += 1
        degree[b] += 1
        if not same:
            union(a, b)
        if len(selected) == n:
            break

    if len(selected) != n or any(value != 2 for value in degree):
        return None

    adjacency = [[] for _ in range(n)]
    for a, b in selected:
        adjacency[a].append(b)
        adjacency[b].append(a)

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
    assignment: Iterable[int],
    cycles: Iterable[Iterable[int]],
    distances: np.ndarray,
) -> tuple[int, ...]:
    """Merge assignment cycles by minimum-cost directed 2-edge exchanges."""

    successor = [int(x) for x in assignment]
    groups = [set(int(x) for x in cycle) for cycle in cycles]

    while len(groups) > 1:
        best_delta = _INF
        best: tuple[int, int, int, int] | None = None

        for first_index in range(len(groups) - 1):
            first = groups[first_index]
            for second_index in range(first_index + 1, len(groups)):
                second = groups[second_index]
                for a in first:
                    a_next = successor[a]
                    old_a = float(distances[a, a_next])
                    for b in second:
                        b_next = successor[b]
                        delta = (
                            float(distances[a, b_next])
                            + float(distances[b, a_next])
                            - old_a
                            - float(distances[b, b_next])
                        )
                        if delta < best_delta:
                            best_delta = delta
                            best = (first_index, second_index, a, b)

        if best is None:
            raise RuntimeError("Failed to patch assignment cycles")

        first_index, second_index, a, b = best
        successor[a], successor[b] = successor[b], successor[a]
        groups[first_index].update(groups[second_index])
        del groups[second_index]

    return _tour_from_successor(successor, 0)


class AssignmentBranchAndBoundTSPSolver(BaseSolver[TravellingSalespersonProblem]):
    """Solve complete symmetric TSP exactly without an external optimizer.

    This solver is normally much faster than Held--Karp on instances whose
    assignment relaxation is strong, while using polynomial memory. Its
    worst-case running time is still exponential and highly instance-dependent.
    """

    def __init__(
        self,
        *,
        nearest_neighbour_starts: int = 24,
        root_two_opt_passes: int | None = None,
        node_two_opt_passes: int = 4,
        patch_every_nodes: int = 1,
        verify_symmetry: bool = True,
    ) -> None:
        if nearest_neighbour_starts < 1:
            raise ValueError("nearest_neighbour_starts must be positive")
        if node_two_opt_passes < 0:
            raise ValueError("node_two_opt_passes must be non-negative")
        if patch_every_nodes < 1:
            raise ValueError("patch_every_nodes must be positive")

        self.nearest_neighbour_starts = nearest_neighbour_starts
        self.root_two_opt_passes = root_two_opt_passes
        self.node_two_opt_passes = node_two_opt_passes
        self.patch_every_nodes = patch_every_nodes
        self.verify_symmetry = verify_symmetry

    @staticmethod
    def _edge_code(city_count: int, source: int, target: int) -> int:
        return source * city_count + target

    @staticmethod
    def _decode_edge(city_count: int, code: int) -> tuple[int, int]:
        return divmod(code, city_count)

    def _initial_upper_bound(self, distances: np.ndarray) -> tuple[tuple[int, ...], float]:
        n = int(distances.shape[0])
        candidate_starts = min(n, self.nearest_neighbour_starts)

        if candidate_starts == n:
            starts = range(n)
        else:
            starts = sorted(
                {
                    int(round(index * (n - 1) / max(1, candidate_starts - 1)))
                    for index in range(candidate_starts)
                }
            )

        best_tour: tuple[int, ...] | None = None
        best_cost = _INF

        greedy = _greedy_multifragment_tour(distances)
        if greedy is not None:
            greedy, greedy_cost = _two_opt(
                greedy,
                distances,
                max_passes=self.root_two_opt_passes,
            )
            best_tour, best_cost = greedy, greedy_cost

        for start in starts:
            tour = _nearest_neighbour_tour(distances, int(start))
            tour, cost = _two_opt(
                tour,
                distances,
                max_passes=self.root_two_opt_passes,
            )
            if cost < best_cost:
                best_tour, best_cost = tour, cost

        if best_tour is None:
            raise RuntimeError("Failed to construct an initial tour")
        return best_tour, best_cost

    def _build_cost_matrix(
        self,
        base_cost: np.ndarray,
        forbidden: frozenset[int],
        forced_out: tuple[int, ...],
        forced_in: tuple[int, ...],
    ) -> np.ndarray | None:
        n = int(base_cost.shape[0])
        cost = base_cost.copy()

        for code in forbidden:
            source, target = self._decode_edge(n, code)
            cost[source, target] = _INF

        for source, target in enumerate(forced_out):
            if target >= 0:
                cost[source, :] = _INF
        for target, source in enumerate(forced_in):
            if source >= 0:
                cost[:, target] = _INF
        for source, target in enumerate(forced_out):
            if target >= 0:
                code = self._edge_code(n, source, target)
                if code in forbidden:
                    return None
                cost[source, target] = base_cost[source, target]

        if np.isinf(cost).all(axis=1).any() or np.isinf(cost).all(axis=0).any():
            return None
        return cost

    def _solve_relaxation(
        self,
        base_cost: np.ndarray,
        forbidden: frozenset[int],
        forced_out: tuple[int, ...],
        forced_in: tuple[int, ...],
    ) -> tuple[float, tuple[int, ...], tuple[tuple[int, ...], ...]] | None:
        cost = self._build_cost_matrix(base_cost, forbidden, forced_out, forced_in)
        if cost is None:
            return None
        solved = _hungarian(cost)
        if solved is None:
            return None
        lower_bound, assignment_array = solved
        assignment = tuple(int(x) for x in assignment_array)
        cycles = _assignment_cycles(assignment)
        return lower_bound, assignment, cycles

    @staticmethod
    def _would_close_premature_cycle(
        forced_out: list[int],
        source: int,
        target: int,
    ) -> bool:
        """Return true when source->target closes a forced cycle before n edges."""

        n = len(forced_out)
        current = target
        length = 1
        while current >= 0 and current != source and length <= n:
            current = forced_out[current]
            length += 1

        if current != source:
            return False

        forced_edge_count = sum(value >= 0 for value in forced_out)
        return forced_edge_count < n

    def _include_edge(
        self,
        n: int,
        source: int,
        target: int,
        forbidden: set[int],
        forced_out: list[int],
        forced_in: list[int],
    ) -> bool:
        code = self._edge_code(n, source, target)
        if code in forbidden or source == target:
            return False

        existing_target = forced_out[source]
        existing_source = forced_in[target]
        if existing_target not in (-1, target) or existing_source not in (-1, source):
            return False
        if existing_target == target:
            return True

        forced_out[source] = target
        forced_in[target] = source
        if self._would_close_premature_cycle(forced_out, source, target):
            forced_out[source] = -1
            forced_in[target] = -1
            return False
        return True

    def solve(self, problem: TravellingSalespersonProblem) -> SolverResult:
        started = time.perf_counter()
        n = int(problem.city_count)
        if n < 1:
            raise ValueError("TSP requires at least one city")

        distances = np.ascontiguousarray(problem.distances, dtype=np.float64)
        if distances.shape != (n, n):
            raise ValueError(
                f"expected a {(n, n)} distance matrix, received {distances.shape}"
            )
        if not np.isfinite(distances).all():
            raise ValueError("complete TSP distances must all be finite")
        if self.verify_symmetry and not np.array_equal(distances, distances.T):
            raise ValueError("this solver requires an exactly symmetric matrix")

        if n == 1:
            return SolverResult(
                tour=(0,),
                cost=0.0,
                metadata={
                    "algorithm": "assignment-branch-and-bound",
                    "exact": True,
                    "nodes": 1,
                    "runtime_seconds": time.perf_counter() - started,
                },
            )
        if n == 2:
            tour = (0, 1)
            return SolverResult(
                tour=tour,
                cost=_tour_cost(tour, distances),
                metadata={
                    "algorithm": "assignment-branch-and-bound",
                    "exact": True,
                    "nodes": 1,
                    "runtime_seconds": time.perf_counter() - started,
                },
            )

        incumbent_tour, incumbent_cost = self._initial_upper_bound(distances)

        base_cost = distances.copy()
        np.fill_diagonal(base_cost, _INF)

        empty_forced = tuple([-1] * n)
        root_relaxation = self._solve_relaxation(
            base_cost,
            frozenset(),
            empty_forced,
            empty_forced,
        )
        if root_relaxation is None:
            raise ValueError("the instance has no Hamiltonian tour")

        root_bound, root_assignment, root_cycles = root_relaxation
        serial_counter = itertools.count()
        queue: list[_QueueNode] = [
            _QueueNode(
                lower_bound=root_bound,
                serial=next(serial_counter),
                forbidden=frozenset(),
                forced_out=empty_forced,
                forced_in=empty_forced,
                assignment=root_assignment,
                cycles=root_cycles,
                depth=0,
            )
        ]

        nodes_popped = 0
        nodes_generated = 1
        nodes_pruned_bound = 0
        nodes_pruned_infeasible = 0
        assignment_solves = 1
        patched_tours = 0
        maximum_queue = 1
        maximum_depth = 0

        while queue:
            node = heapq.heappop(queue)
            nodes_popped += 1
            maximum_depth = max(maximum_depth, node.depth)

            if node.lower_bound >= incumbent_cost:
                nodes_pruned_bound += 1
                continue

            if len(node.cycles) == 1:
                tour = _canonical_tour(_tour_from_successor(node.assignment, 0))
                cost = _tour_cost(tour, distances)
                if cost < incumbent_cost:
                    incumbent_tour = tour
                    incumbent_cost = cost
                continue

            if nodes_popped % self.patch_every_nodes == 0:
                patched = _patch_cycle_cover(
                    node.assignment,
                    node.cycles,
                    distances,
                )
                if self.node_two_opt_passes:
                    patched, patched_cost = _two_opt(
                        patched,
                        distances,
                        max_passes=self.node_two_opt_passes,
                    )
                else:
                    patched = _canonical_tour(patched)
                    patched_cost = _tour_cost(patched, distances)
                patched_tours += 1
                if patched_cost < incumbent_cost:
                    incumbent_tour = patched
                    incumbent_cost = patched_cost
                    if node.lower_bound >= incumbent_cost:
                        nodes_pruned_bound += 1
                        continue

            branch_cycle = node.cycles[0]
            branch_edges = [
                (city, node.assignment[city])
                for city in branch_cycle
                if node.forced_out[city] < 0
            ]

            if not branch_edges:
                nodes_pruned_infeasible += 1
                continue

            def exclusion_penalty(edge: tuple[int, int]) -> float:
                source, target = edge
                row = base_cost[source]
                best = _INF
                for candidate in range(n):
                    if candidate == target:
                        continue
                    value = float(row[candidate])
                    if value < best:
                        best = value
                return best - float(base_cost[source, target])

            branch_edges.sort(key=exclusion_penalty, reverse=True)

            prefix_forbidden = set(node.forbidden)
            prefix_out = list(node.forced_out)
            prefix_in = list(node.forced_in)

            for source, target in branch_edges:
                edge_code = self._edge_code(n, source, target)

                child_forbidden = set(prefix_forbidden)
                child_forbidden.add(edge_code)
                child_forbidden_frozen = frozenset(child_forbidden)
                child_out = tuple(prefix_out)
                child_in = tuple(prefix_in)

                relaxation = self._solve_relaxation(
                    base_cost,
                    child_forbidden_frozen,
                    child_out,
                    child_in,
                )
                assignment_solves += 1
                if relaxation is None:
                    nodes_pruned_infeasible += 1
                else:
                    bound, assignment, cycles = relaxation
                    if bound >= incumbent_cost:
                        nodes_pruned_bound += 1
                    else:
                        heapq.heappush(
                            queue,
                            _QueueNode(
                                lower_bound=bound,
                                serial=next(serial_counter),
                                forbidden=child_forbidden_frozen,
                                forced_out=child_out,
                                forced_in=child_in,
                                assignment=assignment,
                                cycles=cycles,
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

            maximum_queue = max(maximum_queue, len(queue))

        incumbent_tour = _canonical_tour(incumbent_tour)
        incumbent_cost = _tour_cost(incumbent_tour, distances)
        runtime = time.perf_counter() - started

        return SolverResult(
            tour=incumbent_tour,
            cost=incumbent_cost,
            metadata={
                "algorithm": "assignment-branch-and-bound",
                "exact": True,
                "nodes": nodes_popped,
                "nodes_generated": nodes_generated,
                "nodes_pruned_bound": nodes_pruned_bound,
                "nodes_pruned_infeasible": nodes_pruned_infeasible,
                "assignment_solves": assignment_solves,
                "patched_tours": patched_tours,
                "maximum_queue": maximum_queue,
                "maximum_depth": maximum_depth,
                "root_lower_bound": root_bound,
                "runtime_seconds": runtime,
            },
        )


ExactTSPSolver = AssignmentBranchAndBoundTSPSolver

__all__ = [
    "AssignmentBranchAndBoundTSPSolver",
    "ExactTSPSolver",
]
