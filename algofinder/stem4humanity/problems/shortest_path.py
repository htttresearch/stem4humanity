"""Single-source shortest-path tree problem family.

The general class accepts any directed graph with non-negative arc
weights; the DAG subclass restricts graphs to be acyclic, where
topological relaxation is linear in the number of arcs. Solutions are
shortest-path trees stored as parent pointers (one entry per node,
``-1`` for the source and for unreachable nodes); the objective is the
sum of tree distances from the source to every reachable node.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


class ShortestPathState:
    """A single-source shortest-path instance on a directed graph."""

    def __init__(
        self,
        name: str,
        node_count: int,
        source: int,
        arcs: Sequence[Sequence[float]],
    ) -> None:
        arc_matrix = np.asarray(arcs, dtype=float)
        if arc_matrix.ndim != 2 or arc_matrix.shape[1] != 3:
            raise ValueError("arcs must have shape (arc_count, 3)")
        if not np.isfinite(arc_matrix).all():
            raise ValueError("arc weights must be finite")
        if np.any(arc_matrix[:, 2] < 0):
            raise ValueError("arc weights cannot be negative")
        node_count = int(node_count)
        if node_count < 2:
            raise ValueError("a graph needs at least two nodes")
        if not np.all(arc_matrix[:, :2] >= 0) or not np.all(
            arc_matrix[:, :2] < node_count
        ):
            raise ValueError("arc endpoints must be valid node indices")
        source = int(source)
        if not 0 <= source < node_count:
            raise ValueError("source must be a valid node index")

        self.name = name
        self._node_count = node_count
        self._source = source
        self._arcs: NDArray[np.float64] = arc_matrix.copy()
        self._arcs.setflags(write=False)
        self.nodes = tuple(range(node_count))
        self._out_arcs: tuple[tuple[tuple[int, float], ...], ...] = tuple(
            tuple(
                (int(head), float(weight))
                for tail, head, weight in self._arcs
                if int(tail) == node
            )
            for node in self.nodes
        )
        self._out_edges: tuple[tuple[int, ...], ...] = tuple(
            tuple(tail for tail, _ in self._out_arcs[node]) for node in self.nodes
        )

    @property
    def node_count(self) -> int:
        return self._node_count

    @property
    def source(self) -> int:
        return self._source

    @property
    def arcs(self) -> NDArray[np.float64]:
        """Read-only ``(arc_count, 3)`` array of (tail, head, weight)."""
        return self._arcs

    @property
    def arc_count(self) -> int:
        return len(self._arcs)

    def out_arcs(self, node: int) -> tuple[tuple[int, float], ...]:
        """Outgoing arcs of ``node`` as ``(head, weight)`` pairs."""
        return self._out_arcs[node]

    def out_edges(self, node: int) -> tuple[int, ...]:
        """Heads of the outgoing arcs of ``node``."""
        return self._out_edges[node]

    def has_arc(self, tail: int, head: int) -> bool:
        return head in self._out_edges[tail]

    def reachable_from_source(self) -> set[int]:
        """Nodes reachable from the source through directed arcs."""
        seen = {self._source}
        frontier = deque([self._source])
        while frontier:
            node = frontier.popleft()
            for neighbor in self._out_edges[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append(neighbor)
        return seen

    def _arc_weight(self, tail: int, head: int) -> float | None:
        """Cheapest weight among the parallel arcs between ``tail`` and ``head``.

        A parent-pointer tree does not record which parallel arc it uses,
        so the cost is interpreted with the cheapest arc, which any
        shortest-path tree would use.
        """
        best: float | None = None
        for candidate, weight in self._out_arcs[tail]:
            if candidate == head and (best is None or weight < best):
                best = weight
        return best

    def _walk_distance(self, parents: Sequence[int], node: int) -> float | None:
        distance = 0.0
        visited: set[int] = set()
        current = int(node)
        while current != self._source:
            if current in visited:
                return None
            visited.add(current)
            parent = int(parents[current])
            if parent < 0:
                return None
            weight = self._arc_weight(parent, current)
            if weight is None:
                return None
            distance += weight
            current = parent
        return distance

    def verify(self, solution: Any) -> bool:
        try:
            parents = [int(item) for item in solution]
        except (TypeError, ValueError):
            return False
        if len(parents) != self._node_count:
            return False
        if any(parent < -1 or parent >= self._node_count for parent in parents):
            return False
        if parents[self._source] != -1:
            return False
        for node in self.nodes:
            parent = parents[node]
            if node != self._source and parent == node:
                return False
            if parent >= 0 and not self.has_arc(parent, node):
                return False
        reachable = self.reachable_from_source()
        for node in self.nodes:
            if node in reachable:
                if self._walk_distance(parents, node) is None:
                    return False
            elif parents[node] != -1:
                return False
        return True

    def objective_value(self, solution: Any) -> float:
        if not self.verify(solution):
            raise ValueError("solution must be a valid shortest-path tree")
        parents = [int(item) for item in solution]
        total = 0.0
        for node in self.nodes:
            if node == self._source or parents[node] == -1:
                continue
            distance = self._walk_distance(parents, node)
            if distance is None:
                raise ValueError("unreachable node cannot carry a distance")
            total += distance
        return float(total)


__all__ = ["ShortestPathState"]
