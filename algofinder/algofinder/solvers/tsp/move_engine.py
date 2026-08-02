"""Tour representation and candidate-restricted move kernels.

A tour is a cyclic permutation stored as an ``order`` array plus a
``position`` array; segment reversals update both in vectorized form.
The 2-opt kernel uses an active-position queue (don't-look bits), the
Or-opt kernel relocates short segments, and the 3-opt kernel evaluates
the four non-2-opt reconnections with a configurable non-candidate
slack.  All kernels are deadline-aware and count their work.
"""

from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from algofinder.problems.tsp import (
    EuclideanTravellingSalespersonProblem,
)
from algofinder.solvers.tsp.candidates import CandidateGraph

EPSILON = 1e-12

# Orientation bitmasks (1 = reverse segment one, 2 = segment two, 4 = segment
# three) mapping to the added-edge slots over (t1, t2, t3, t4, t5, t6).
# The three 2-opt-equivalent variants (100, 010, 001) are omitted because the
# 2-opt kernel already covers them.
THREE_OPT_VARIANTS: dict[int, tuple[tuple[int, int], ...]] = {
    3: ((0, 2), (1, 4), (3, 5)),
    5: ((0, 4), (1, 3), (2, 5)),
    6: ((0, 3), (1, 5), (2, 4)),
    7: ((0, 3), (1, 4), (2, 5)),
}


class Tour:
    """Cyclic permutation of city ids with O(1) position lookups."""

    __slots__ = ("order", "pos", "cost")

    def __init__(
        self,
        order: tuple[int, ...] | NDArray[np.int64],
        distances: NDArray[np.float64],
    ) -> None:
        self.rebuild(np.asarray(order, dtype=int), distances)

    def rebuild(self, order: NDArray[np.int64], distances: NDArray[np.float64]) -> None:
        """Replace the tour with ``order`` and recompute positions and cost."""
        self.order = order
        self.pos = np.empty(order.size, dtype=int)
        self.pos[self.order] = np.arange(order.size)
        self.cost = float(distances[self.order, np.roll(self.order, -1)].sum())

    def reverse_segment(
        self,
        first: int,
        second: int,
        distances: NDArray[np.float64],
    ) -> None:
        """Reverse the inclusive segment ``order[first:second]`` (``first <= second``)."""
        segment = self.order[first : second + 1].copy()
        self.order[first : second + 1] = segment[::-1]
        self.pos[self.order[first : second + 1]] = np.arange(first, second + 1)
        self.cost = float(distances[self.order, np.roll(self.order, -1)].sum())


def _deadline_hit(deadline: float | None) -> bool:
    return deadline is not None and perf_counter() > deadline


def double_bridge(order: NDArray[np.int64], rng: np.random.Generator) -> NDArray[np.int64]:
    """Four-cut cyclic perturbation; returns a new order array."""
    city_count = order.size
    if city_count < 8:
        return order.copy()
    cuts = sorted(int(index) for index in rng.choice(city_count, size=4, replace=False))
    segment_starts = cuts + [cuts[0] + city_count]
    segments: list[list[int]] = []
    for index in range(4):
        start = segment_starts[index] + 1
        stop = segment_starts[index + 1]
        segments.append(
            [int(order[position % city_count]) for position in range(start, stop + 1)]
        )
    return np.asarray(segments[0] + segments[2] + segments[1] + segments[3], dtype=int)


def two_opt_improve(
    problem: EuclideanTravellingSalespersonProblem,
    tour: Tour,
    candidate_lists: CandidateGraph,
    *,
    max_moves: int = 0,
    deadline: float | None = None,
) -> dict[str, int]:
    """First-improvement candidate 2-opt with an active-position queue."""
    city_count = problem.city_count
    distances = problem.distances
    queue = deque(range(city_count))
    moves = 0
    evaluations = 0
    while queue:
        position = queue.popleft()
        t1 = int(tour.order[position])
        t2 = int(tour.order[(position + 1) % city_count])
        applied = False
        for t3 in candidate_lists[t1]:
            j = int(tour.pos[t3])
            if j <= position + 1:
                continue
            if position == 0 and j == city_count - 1:
                continue
            t4 = int(tour.order[(j + 1) % city_count])
            delta = (
                distances[t1, t3]
                + distances[t2, t4]
                - distances[t1, t2]
                - distances[t3, t4]
            )
            evaluations += 1
            if delta < -EPSILON:
                tour.reverse_segment(position + 1, j, distances)
                moves += 1
                applied = True
                for touched in (
                    position - 1,
                    position,
                    position + 1,
                    j - 1,
                    j,
                    j + 1,
                ):
                    queue.append(touched % city_count)
                break
        if max_moves and moves >= max_moves:
            break
        if evaluations % 512 == 0 and _deadline_hit(deadline):
            break
    return {"moves": moves, "evaluations": evaluations}


def _or_opt_apply(
    tour: Tour,
    start: int,
    length: int,
    position: int,
    distances: NDArray[np.float64],
) -> None:
    """Remove the segment at ``start`` (length ``length``) and reinsert it
    after ``position``; both coordinates are pre-move positions."""
    city_count = len(tour.order)
    listing = list(tour.order)
    segment = [listing[(start + offset) % city_count] for offset in range(length)]
    body = [
        listing[(start + length + offset) % city_count]
        for offset in range(city_count - length)
    ]
    body[body.index(listing[position]) + 1 : body.index(listing[position]) + 1] = segment
    shift = (city_count - start) % city_count
    tour.rebuild(np.asarray(body[shift:] + body[:shift], dtype=int), distances)


def or_opt_improve(
    problem: EuclideanTravellingSalespersonProblem,
    tour: Tour,
    candidate_lists: CandidateGraph,
    *,
    max_segment_length: int = 3,
    deadline: float | None = None,
) -> dict[str, int]:
    """First-improvement Or-opt: relocate short segments between candidate edges."""
    city_count = problem.city_count
    distances = problem.distances
    moves = 0
    evaluations = 0
    improved = True
    while improved:
        improved = False
        for length in range(1, max_segment_length + 1):
            if city_count - length < 2:
                continue
            for start in range(city_count):
                head = int(tour.order[start])
                tail = int(tour.order[(start + length - 1) % city_count])
                prev = int(tour.order[(start - 1) % city_count])
                nxt = int(tour.order[(start + length) % city_count])
                removed_cost = distances[prev, head] + distances[tail, nxt]
                added_base = distances[prev, nxt]
                segment_positions = {
                    (start + offset) % city_count for offset in range(length)
                }
                candidate_positions = set()
                for city in candidate_lists[head]:
                    candidate_positions.add(int(tour.pos[city]))
                for city in candidate_lists[tail]:
                    candidate_positions.add(int(tour.pos[city]))
                for position in sorted(candidate_positions):
                    if position in segment_positions:
                        continue
                    if (position + 1) % city_count in segment_positions:
                        continue
                    a = int(tour.order[position])
                    b = int(tour.order[(position + 1) % city_count])
                    delta = (
                        added_base
                        + distances[a, head]
                        + distances[tail, b]
                        - removed_cost
                        - distances[a, b]
                    )
                    evaluations += 1
                    if delta < -EPSILON:
                        _or_opt_apply(tour, start, length, position, distances)
                        moves += 1
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
            if evaluations % 1024 == 0 and _deadline_hit(deadline):
                return {"moves": moves, "evaluations": evaluations}
    return {"moves": moves, "evaluations": evaluations}


def _three_opt_apply(
    tour: Tour,
    first: int,
    second: int,
    third: int,
    orientations: int,
    distances: NDArray[np.float64],
) -> None:
    """Reconnect the three segments of ``(first, second, third)`` by the
    orientation bits ``orientations`` (1 = reversed)."""
    order = tour.order
    segment_one = list(order[first + 1 : second + 1])
    segment_two = list(order[second + 1 : third + 1])
    segment_three = list(order[third + 1 :]) + list(order[: first + 1])
    if orientations & 1:
        segment_one.reverse()
    if orientations & 2:
        segment_two.reverse()
    if orientations & 4:
        segment_three.reverse()
    tour.rebuild(
        np.asarray(segment_one + segment_two + segment_three, dtype=int),
        distances,
    )


def _three_opt_pass(
    problem: EuclideanTravellingSalespersonProblem,
    tour: Tour,
    candidate_lists: CandidateGraph,
    *,
    slack: int = 1,
    max_moves: int = 100,
    deadline: float | None = None,
) -> dict[str, int]:
    """One candidate-restricted 3-opt pass over the four non-2-opt variants."""
    city_count = problem.city_count
    distances = problem.distances
    moves = 0
    evaluations = 0
    for position in range(city_count):
        t1 = int(tour.order[position])
        t2 = int(tour.order[(position + 1) % city_count])
        j_set = set()
        for candidate in candidate_lists[t1]:
            j_set.add(int(tour.pos[candidate]))
            j_set.add((int(tour.pos[candidate]) - 1) % city_count)
        for j in sorted(index for index in j_set if position + 2 <= index <= city_count - 2):
            t3 = int(tour.order[j])
            t4 = int(tour.order[(j + 1) % city_count])
            k_set = set()
            for city in (t1, t2, t3):
                for candidate in candidate_lists[city]:
                    k_set.add(int(tour.pos[candidate]))
                    k_set.add((int(tour.pos[candidate]) - 1) % city_count)
            for k in sorted(index for index in k_set if j + 2 <= index <= city_count - 2):
                t5 = int(tour.order[k])
                t6 = int(tour.order[(k + 1) % city_count])
                endpoints = (t1, t2, t3, t4, t5, t6)
                removed_cost = (
                    distances[t1, t2] + distances[t3, t4] + distances[t5, t6]
                )
                for orientations, edge_slots in THREE_OPT_VARIANTS.items():
                    non_candidate = 0
                    added_cost = 0.0
                    for first_slot, second_slot in edge_slots:
                        u = endpoints[first_slot]
                        v = endpoints[second_slot]
                        if v not in candidate_lists[u]:
                            non_candidate += 1
                        added_cost += distances[u, v]
                    if non_candidate > slack:
                        continue
                    evaluations += 1
                    if added_cost - removed_cost < -EPSILON:
                        _three_opt_apply(
                            tour, position, j, k, orientations, distances
                        )
                        moves += 1
                        if moves >= max_moves:
                            return {"moves": moves, "evaluations": evaluations}
                        position = -1
                        j = city_count
                        k = city_count
                        break
                if k == city_count:
                    break
            if j == city_count:
                break
        if evaluations % 512 == 0 and _deadline_hit(deadline):
            break
    return {"moves": moves, "evaluations": evaluations}


def three_opt_improve(
    problem: EuclideanTravellingSalespersonProblem,
    tour: Tour,
    candidate_lists: CandidateGraph,
    *,
    slack: int = 1,
    max_passes: int = 2,
    max_moves: int = 100,
    deadline: float | None = None,
) -> dict[str, int]:
    """Run candidate-restricted 3-opt passes until improvement stalls."""
    if problem.city_count < 6:
        return {"moves": 0, "evaluations": 0}
    total_moves = 0
    total_evaluations = 0
    for _ in range(max_passes):
        stats = _three_opt_pass(
            problem,
            tour,
            candidate_lists,
            slack=slack,
            max_moves=max_moves,
            deadline=deadline,
        )
        total_moves += stats["moves"]
        total_evaluations += stats["evaluations"]
        if stats["moves"] == 0 or _deadline_hit(deadline):
            break
    return {"moves": total_moves, "evaluations": total_evaluations}


def improve_tour(
    problem: EuclideanTravellingSalespersonProblem,
    tour: Tour,
    candidate_lists: CandidateGraph,
    *,
    max_2opt_moves: int = 0,
    max_segment_length: int = 3,
    three_opt_passes: int = 2,
    three_opt_slack: int = 1,
    deep: bool = True,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Run 2-opt, Or-opt, 2-opt, then 3-opt to a local optimum.

    With ``deep=False`` the 3-opt phase is skipped (cheap passes used
    between kicks).  Returns a counter summary under the keys ``two_opt``,
    ``or_opt`` and ``three_opt``.  The tour is mutated in place.
    """
    phases = ("two_opt", "or_opt", "two_opt", "three_opt") if deep else (
        "two_opt",
        "or_opt",
        "two_opt",
    )
    counters: dict[str, Any] = {
        "two_opt": {"moves": 0, "evaluations": 0},
        "or_opt": {"moves": 0, "evaluations": 0},
        "three_opt": {"moves": 0, "evaluations": 0},
    }
    for phase in phases:
        if _deadline_hit(deadline):
            break
        if phase == "two_opt":
            stats = two_opt_improve(
                problem,
                tour,
                candidate_lists,
                max_moves=max_2opt_moves,
                deadline=deadline,
            )
        elif phase == "or_opt":
            stats = or_opt_improve(
                problem,
                tour,
                candidate_lists,
                max_segment_length=max_segment_length,
                deadline=deadline,
            )
        else:
            stats = three_opt_improve(
                problem,
                tour,
                candidate_lists,
                slack=three_opt_slack,
                max_passes=three_opt_passes,
                deadline=deadline,
            )
        for key, value in stats.items():
            counters[phase][key] += value
    return counters


__all__ = [
    "EPSILON",
    "Tour",
    "double_bridge",
    "improve_tour",
    "or_opt_improve",
    "three_opt_improve",
    "two_opt_improve",
]
