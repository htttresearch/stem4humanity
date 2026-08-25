"""Trusted, constrained candidate scaffolds for small local research models.

The scaffold owns Python boilerplate, solver identity, and the TSP execution
pipeline.  A model supplies only a short numeric edge-priority expression.
That expression is checked as a restricted AST before it is interpolated into
the candidate source.  It is intentionally a narrow *infrastructure* surface:
it makes experiments feasible for small models without giving them a way to
modify the evaluator, solver registry, or generic control plane.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping

from algofinder.agents.learning.normalization import (
    NormalizationError,
    normalize_formula,
)
from algofinder.agents.tsp_hir import (
    HIRError,
    TspGenome,
    apply_edit,
    default_genome,
    parse_edit_json,
)


class TemplateError(ValueError):
    """A template identifier, value, or rendered source is invalid."""


@dataclass(frozen=True)
class CandidateTemplate:
    template_id: str
    relative_path: str
    entrypoints: tuple[str, ...]
    value_names: tuple[str, ...]


TSP_EDGE_FORMULA_TEMPLATE = CandidateTemplate(
    template_id="tsp-edge-formula-v1",
    relative_path="algofinder/algofinder/solvers/tsp/campaign_candidate_formula.py",
    entrypoints=("algofinder.solvers.tsp.campaign_candidate_formula:TspFormulaCandidateSolver",),
    value_names=("priority_expression",),
)

TSP_HIR_TEMPLATE = CandidateTemplate(
    template_id="tsp-hir-v1",
    relative_path="algofinder/algofinder/solvers/tsp/campaign_candidate_hir.py",
    entrypoints=("algofinder.solvers.tsp.campaign_candidate_hir:TspHIRCandidateSolver",),
    value_names=("edit_json",),
)

_TEMPLATES = {
    TSP_EDGE_FORMULA_TEMPLATE.template_id: TSP_EDGE_FORMULA_TEMPLATE,
    TSP_HIR_TEMPLATE.template_id: TSP_HIR_TEMPLATE,
}
_FORMULA_MARKER = "{{PRIORITY_EXPRESSION}}"
_FORMULA_NAMES = frozenset(
    {"distance", "normalized_rank", "angular_offset", "mean_distance", "n"}
)
_FORMULA_CALLS = frozenset({"abs", "min", "max"})


def get_template(template_id: str) -> CandidateTemplate:
    try:
        return _TEMPLATES[template_id]
    except KeyError as exc:
        raise TemplateError(f"unknown candidate template {template_id!r}") from exc


def validate_values(template_id: str, values: Mapping[str, str]) -> dict[str, str]:
    template = get_template(template_id)
    if set(values) != set(template.value_names):
        raise TemplateError(
            f"{template_id} requires exactly {', '.join(template.value_names)}"
        )
    if template_id == TSP_HIR_TEMPLATE.template_id:
        try:
            return {"edit_json": parse_edit_json(values["edit_json"]).to_json()}
        except (HIRError, KeyError) as exc:
            raise TemplateError(str(exc)) from exc
    expression = values["priority_expression"]
    if not isinstance(expression, str):
        raise TemplateError("priority_expression must be a string")
    expression = expression.strip()
    if not expression or len(expression) > 240 or "\n" in expression or "\r" in expression:
        raise TemplateError("priority_expression must be one non-empty line of at most 240 characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise TemplateError(f"priority_expression is not valid Python: {exc.msg}") from exc
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (
                ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
                ast.IfExp, ast.Call, ast.Name, ast.Constant, ast.Load,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
                ast.Pow, ast.USub, ast.UAdd, ast.And, ast.Or,
                ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
            ),
        ):
            raise TemplateError(f"priority_expression uses unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in _FORMULA_NAMES | _FORMULA_CALLS:
            raise TemplateError(f"priority_expression uses unknown name {node.id!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FORMULA_CALLS:
                raise TemplateError("priority_expression may call only abs, min, and max")
            if node.keywords:
                raise TemplateError("priority_expression calls may not use keyword arguments")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise TemplateError("priority_expression constants must be numeric")
    try:
        canonical = normalize_formula(expression).exact_expression
    except NormalizationError as exc:
        raise TemplateError(str(exc)) from exc
    return {"priority_expression": canonical}


def render_template(template_id: str, values: Mapping[str, str] | None = None) -> str:
    get_template(template_id)
    if template_id == TSP_HIR_TEMPLATE.template_id:
        if values is not None:
            raise TemplateError("HIR templates are rendered from a genome, not directly from an edit")
        return render_hir_template(default_genome())
    if values is None:
        values = {"priority_expression": "distance"}
    checked = validate_values(template_id, values)
    source = _TSP_EDGE_FORMULA_SOURCE.replace(
        _FORMULA_MARKER, checked["priority_expression"]
    )
    if _FORMULA_MARKER in source:
        raise TemplateError("candidate template did not render completely")
    return source


def validate_rendered_template(template_id: str, source: str) -> dict[str, str]:
    """Accept only the exact trusted scaffold with one verified model gap."""
    if template_id == TSP_HIR_TEMPLATE.template_id:
        return {"genome_json": _extract_hir_genome(source).to_json()}
    marker = "            # AGENT_FORMULA\n            return float("
    start = source.find(marker)
    if start < 0:
        raise TemplateError("candidate template formula marker is missing")
    expression_start = start + len(marker)
    expression_end = source.find(")\n", expression_start)
    if expression_end < 0:
        raise TemplateError("candidate template formula return is malformed")
    expression = source[expression_start:expression_end]
    checked = validate_values(template_id, {"priority_expression": expression})
    if source != render_template(template_id, checked):
        raise TemplateError("candidate template contains changes outside the permitted formula")
    return checked


def describe_rendered_template(template_id: str, source: str) -> dict[str, Any]:
    """Return compact, JSON-safe genotype and behavior descriptors.

    The description is derived from the exact authority-rendered source after
    validation.  It lets later proposal turns see the selected parent's real
    formula/genome without asking a small model to reconstruct state from Git
    patches, and supplies non-constant axes for MAP-Elites.
    """
    if template_id == TSP_HIR_TEMPLATE.template_id:
        genome = _extract_hir_genome(source)
        mapping = genome.to_mapping()
        edge = mapping["edge_generator"]
        move = mapping["move_schedule"]
        diversification = mapping["diversification"]
        niche = "|".join((
            f"c:{mapping['constructor']['kind']}",
            f"e:{edge['kind']}:nn{int(edge['nearest_neighbors']) // 4}:a{int(edge['angular_sectors']) // 4}:k{int(edge['top_k']) // 2}",
            f"s:{_expression_shape(mapping['edge_score'])}",
            f"m:{move['kind']}:i{_magnitude_bucket(int(move['max_iterations']))}",
            f"a:{mapping['acceptance']['kind']}",
            f"d:{diversification['kind']}:r{int(diversification['restarts']) // 2}",
        ))
        return {
            "template_state": {"genome": mapping},
            "edge_score_features": _expression_features(mapping["edge_score"]),
            "hir_niche": niche,
            "hir_constructor": mapping["constructor"]["kind"],
            "hir_edge_generator": mapping["edge_generator"]["kind"],
            "hir_move_schedule": mapping["move_schedule"]["kind"],
            "hir_diversification": mapping["diversification"]["kind"],
        }

    expression = validate_rendered_template(template_id, source)["priority_expression"]
    identity = normalize_formula(expression)
    names = {
        node.id
        for node in ast.walk(ast.parse(expression, mode="eval"))
        if isinstance(node, ast.Name) and node.id in _FORMULA_NAMES
    }
    return {
        "template_state": {"priority_expression": expression},
        "formula_exact_digest": identity.exact_digest,
        "formula_structural_digest": identity.structural_digest,
        "formula_behavior_digest": identity.probe_behavior_digest,
        "formula_features": identity.features,
        "edge_score_features": "+".join(sorted(names)) or "constant",
        "distance_monotonicity": _formula_monotonicity(expression, "distance"),
        "rank_monotonicity": _formula_monotonicity(expression, "normalized_rank"),
        "uses_angular_offset": "angular_offset" in names,
    }


def _expression_features(value: Mapping[str, Any]) -> str:
    names: set[str] = set()

    def visit(node: object) -> None:
        if isinstance(node, Mapping):
            if node.get("kind") == "feature" and isinstance(node.get("name"), str):
                names.add(str(node["name"]))
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return "+".join(sorted(names)) or "constant"


def _expression_shape(value: Mapping[str, Any]) -> str:
    kind = str(value.get("kind", "unknown"))
    if kind == "feature":
        return f"feature:{value.get('name')}"
    if kind == "constant":
        return "constant"
    if kind == "unary":
        return f"{value.get('op')}({_expression_shape(value.get('value', {}))})"
    if kind == "binary":
        return (
            f"{value.get('op')}({_expression_shape(value.get('left', {}))},"
            f"{_expression_shape(value.get('right', {}))})"
        )
    if kind == "if":
        condition = value.get("condition", {})
        condition_shape = (
            f"{condition.get('op')}({_expression_shape(condition.get('left', {}))},"
            f"{_expression_shape(condition.get('right', {}))})"
            if isinstance(condition, Mapping) else "unknown"
        )
        return (
            f"if:{condition_shape}({_expression_shape(value.get('then', {}))},"
            f"{_expression_shape(value.get('otherwise', {}))})"
        )
    return kind


def _magnitude_bucket(value: int) -> str:
    if value <= 250:
        return "tiny"
    if value <= 1_000:
        return "small"
    if value <= 4_000:
        return "medium"
    return "large"


def _formula_monotonicity(expression: str, feature: str) -> str:
    code = compile(ast.parse(expression, mode="eval"), "<priority-expression>", "eval")
    base: dict[str, float | int] = {
        "distance": 1.0,
        "normalized_rank": 0.5,
        "angular_offset": 0.35,
        "mean_distance": 1.0,
        "n": 64,
    }
    probes = (0.1, 0.4, 0.8, 1.6) if feature == "distance" else (0.0, 0.33, 0.67, 1.0)
    try:
        values = []
        for probe in probes:
            variables = {**base, feature: probe}
            values.append(float(eval(code, {"__builtins__": {}, "abs": abs, "min": min, "max": max}, variables)))
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return "invalid-on-probe"
    deltas = [right - left for left, right in zip(values, values[1:])]
    positive = any(delta > 1e-10 for delta in deltas)
    negative = any(delta < -1e-10 for delta in deltas)
    if positive and not negative:
        return "increasing"
    if negative and not positive:
        return "decreasing"
    if not positive and not negative:
        return "flat"
    return "nonmonotone"


def apply_values_to_template(
    template_id: str,
    source: str,
    values: Mapping[str, str],
) -> str:
    """Apply a constrained model contribution to an already-rendered scaffold."""
    checked = validate_values(template_id, values)
    if template_id == TSP_HIR_TEMPLATE.template_id:
        try:
            return render_hir_template(apply_edit(_extract_hir_genome(source), parse_edit_json(checked["edit_json"])))
        except HIRError as exc:
            raise TemplateError(str(exc)) from exc
    validate_rendered_template(template_id, source)
    return render_template(template_id, checked)


def render_hir_template(genome: TspGenome) -> str:
    """Compile one type-checked HIR genome into the fixed candidate scaffold."""
    source = _TSP_HIR_SOURCE.replace("{{HIR_GENOME_JSON}}", repr(genome.to_json()))
    source = source.replace("{{HIR_SCORE_EXPRESSION}}", genome.score_python)
    if "{{" in source:
        raise TemplateError("HIR template did not render completely")
    return source


def _extract_hir_genome(source: str) -> TspGenome:
    try:
        tree = ast.parse(source, filename="tsp-hir-candidate.py")
    except SyntaxError as exc:
        raise TemplateError(f"HIR candidate source is not Python: {exc.msg}") from exc
    serialized: str | None = None
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "_GENOME_JSON" for target in statement.targets
        ):
            continue
        if isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            serialized = statement.value.value
            break
    if serialized is None:
        raise TemplateError("HIR candidate genome marker is missing")
    try:
        genome = TspGenome.from_json(serialized)
    except HIRError as exc:
        raise TemplateError(str(exc)) from exc
    if source != render_hir_template(genome):
        raise TemplateError("HIR candidate contains changes outside its compiled genome")
    return genome


def template_relative_path(template_id: str) -> str:
    return str(PurePosixPath(get_template(template_id).relative_path))


_TSP_EDGE_FORMULA_SOURCE = '''"""Generated trusted scaffold; only AGENT_FORMULA may vary."""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np

from algofinder.problems.tsp import EuclideanTravellingSalespersonProblem
from algofinder.solvers.base import Solver, SolverResult
from algofinder.solvers.tsp.euclidean_geometry import (
    build_geometric_candidate_lists,
    candidate_edge_count,
    regret_insertion_tour,
)


class TspFormulaCandidateSolver(Solver):
    id = "tsp-formula-candidate"
    display = "Template-driven Euclidean TSP candidate"
    tags = frozenset({"heuristic"})
    applies_to = frozenset({"tsp:euclidean", "tsp:clustered"})

    @staticmethod
    def _priority(
        distance: float,
        normalized_rank: float,
        angular_offset: float,
        mean_distance: float,
        n: int,
    ) -> float:
        try:
            # AGENT_FORMULA
            return float({{PRIORITY_EXPRESSION}})
        except (ArithmeticError, TypeError, ValueError):
            return distance

    @staticmethod
    def _improve(
        problem: EuclideanTravellingSalespersonProblem,
        initial_tour: tuple[int, ...],
        candidate_lists: list[set[int]],
        *,
        max_iterations: int,
    ) -> tuple[tuple[int, ...], int, int]:
        tour = list(initial_tour)
        accepted_moves = 0
        move_evaluations = 0
        n = problem.city_count
        while accepted_moves < max_iterations:
            positions = {city: position for position, city in enumerate(tour)}
            applied = False
            for first_edge, first_city in enumerate(tour):
                for candidate_city in sorted(candidate_lists[first_city]):
                    second_edge = positions[candidate_city]
                    left, right = sorted((first_edge, second_edge))
                    if right - left <= 1 or (left == 0 and right == n - 1):
                        continue
                    left_city = tour[left]
                    after_left = tour[left + 1]
                    right_city = tour[right]
                    after_right = tour[(right + 1) % n]
                    delta = (
                        problem.distances[left_city, right_city]
                        + problem.distances[after_left, after_right]
                        - problem.distances[left_city, after_left]
                        - problem.distances[right_city, after_right]
                    )
                    move_evaluations += 1
                    if delta < -1e-12:
                        tour[left + 1:right + 1] = reversed(tour[left + 1:right + 1])
                        accepted_moves += 1
                        applied = True
                        break
                if applied:
                    break
            if not applied:
                break
        return tuple(tour), accepted_moves, move_evaluations

    @staticmethod
    def _double_bridge(
        tour: tuple[int, ...], rng: np.random.Generator
    ) -> tuple[int, ...]:
        n = len(tour)
        if n < 8:
            return tour
        cuts = sorted(int(index) for index in rng.choice(n, size=4, replace=False))
        rotated = list(tour[cuts[0] + 1:]) + list(tour[:cuts[0] + 1])
        lengths = (
            cuts[1] - cuts[0], cuts[2] - cuts[1],
            cuts[3] - cuts[2], n - (cuts[3] - cuts[0]),
        )
        parts: list[list[int]] = []
        cursor = 0
        for length in lengths:
            parts.append(rotated[cursor:cursor + length])
            cursor += length
        return tuple(parts[0] + parts[2] + parts[1] + parts[3])

    def solve(
        self,
        problem: EuclideanTravellingSalespersonProblem,
        *,
        budget_seconds: float | None = None,
        context: object | None = None,
    ) -> SolverResult:
        started = perf_counter()
        raw_lists = build_geometric_candidate_lists(
            problem, nearest_neighbors=12, angular_sectors=8
        )
        n = problem.city_count
        nonzero = problem.distances[problem.distances > 0.0]
        mean_distance = float(np.mean(nonzero)) if nonzero.size else 1.0
        ranked_lists: list[set[int]] = []
        for city in range(n):
            neighbours = sorted(raw_lists[city], key=lambda other: (problem.distances[city, other], other))
            denominator = max(1, len(neighbours) - 1)
            def key(other: int) -> tuple[float, int]:
                offset = problem.points[other] - problem.points[city]
                angle = abs(math.atan2(float(offset[1]), float(offset[0]))) / math.pi
                priority = self._priority(
                    float(problem.distances[city, other]),
                    neighbours.index(other) / denominator,
                    angle,
                    mean_distance,
                    n,
                )
                return (priority if math.isfinite(priority) else float(problem.distances[city, other]), other)
            ranked_lists.append(set(sorted(neighbours, key=key)[:8]))
        initial_tour = regret_insertion_tour(problem)
        current, accepted_moves, move_evaluations = self._improve(
            problem, initial_tour, ranked_lists, max_iterations=2_000
        )
        current_cost = problem.objective_value(current)
        best, best_cost = current, current_cost
        rng = np.random.default_rng(0)
        for _ in range(8):
            perturbed = self._double_bridge(current, rng)
            candidate, extra_accepted, extra_evaluations = self._improve(
                problem, perturbed, ranked_lists, max_iterations=2_000
            )
            candidate_cost = problem.objective_value(candidate)
            accepted_moves += extra_accepted
            move_evaluations += extra_evaluations
            if candidate_cost <= current_cost:
                current, current_cost = candidate, candidate_cost
            if candidate_cost < best_cost:
                best, best_cost = candidate, candidate_cost
        return SolverResult(
            solution=best,
            cost=best_cost,
            exact=False,
            wall_seconds=perf_counter() - started,
            metadata={
                "algorithm": "tsp-edge-formula-v1",
                "candidate_edges": candidate_edge_count(ranked_lists),
                "accepted_moves": accepted_moves,
                "move_evaluations": move_evaluations,
                "restarts": 8,
            },
        )
'''


_TSP_HIR_SOURCE = '''"""Generated trusted TSP HIR scaffold; genome is authority-compiled."""

from __future__ import annotations

import json
import math
from time import perf_counter

import numpy as np

from algofinder.problems.tsp import EuclideanTravellingSalespersonProblem
from algofinder.solvers.base import Solver, SolverResult
from algofinder.solvers.tsp.euclidean_geometry import (
    build_geometric_candidate_lists,
    candidate_edge_count,
    regret_insertion_tour,
)


_GENOME_JSON = {{HIR_GENOME_JSON}}
_GENOME = json.loads(_GENOME_JSON)


class TspHIRCandidateSolver(Solver):
    id = "tsp-hir-candidate"
    display = "Compiled typed-HIR Euclidean TSP candidate"
    tags = frozenset({"heuristic"})
    applies_to = frozenset({"tsp:euclidean", "tsp:clustered"})

    @staticmethod
    def _priority(
        distance: float,
        normalized_rank: float,
        angular_offset: float,
        mean_distance: float,
        n: int,
    ) -> float:
        try:
            return float({{HIR_SCORE_EXPRESSION}})
        except (ArithmeticError, TypeError, ValueError):
            return distance

    @staticmethod
    def _nearest_neighbor(problem: EuclideanTravellingSalespersonProblem) -> tuple[int, ...]:
        remaining = set(problem.nodes)
        current = min(remaining)
        tour = [current]
        remaining.remove(current)
        while remaining:
            current = min(remaining, key=lambda city: (problem.distances[tour[-1], city], city))
            tour.append(current)
            remaining.remove(current)
        return tuple(tour)

    @staticmethod
    def _multi_fragment(problem: EuclideanTravellingSalespersonProblem) -> tuple[int, ...]:
        n = problem.city_count
        if n < 4:
            return tuple(problem.nodes)
        degree = [0] * n
        parent = list(range(n))
        def root(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node
        edges: list[tuple[float, int, int]] = []
        for left in range(n):
            for right in range(left + 1, n):
                edges.append((float(problem.distances[left, right]), left, right))
        chosen: list[tuple[int, int]] = []
        for _, left, right in sorted(edges):
            if degree[left] >= 2 or degree[right] >= 2:
                continue
            creates_cycle = root(left) == root(right)
            if creates_cycle and len(chosen) < n - 1:
                continue
            chosen.append((left, right))
            degree[left] += 1
            degree[right] += 1
            parent[root(left)] = root(right)
            if len(chosen) == n:
                break
        adjacency = {city: [] for city in range(n)}
        for left, right in chosen:
            adjacency[left].append(right)
            adjacency[right].append(left)
        tour = [0]
        previous = -1
        while len(tour) < n:
            options = [city for city in adjacency[tour[-1]] if city != previous and city not in tour]
            if not options:
                return TspHIRCandidateSolver._nearest_neighbor(problem)
            previous, current = tour[-1], min(options)
            tour.append(current)
        return tuple(tour)

    @staticmethod
    def _double_bridge(tour: tuple[int, ...], rng: np.random.Generator) -> tuple[int, ...]:
        n = len(tour)
        if n < 8:
            return tour
        cuts = sorted(int(index) for index in rng.choice(n, size=4, replace=False))
        rotated = list(tour[cuts[0] + 1:]) + list(tour[:cuts[0] + 1])
        lengths = (cuts[1] - cuts[0], cuts[2] - cuts[1], cuts[3] - cuts[2], n - (cuts[3] - cuts[0]))
        parts: list[list[int]] = []
        cursor = 0
        for length in lengths:
            parts.append(rotated[cursor:cursor + length])
            cursor += length
        return tuple(parts[0] + parts[2] + parts[1] + parts[3])

    @staticmethod
    def _ruin_recreate(
        problem: EuclideanTravellingSalespersonProblem,
        tour: tuple[int, ...],
        rng: np.random.Generator,
        fraction: float,
    ) -> tuple[int, ...]:
        n = len(tour)
        if n < 5:
            return tour
        remove_count = max(1, min(n - 3, int(round(n * fraction))))
        removed = {int(city) for city in rng.choice(n, size=remove_count, replace=False)}
        partial = [city for city in tour if city not in removed]
        while removed:
            best: tuple[float, int, int] | None = None
            for city in sorted(removed):
                for position, left in enumerate(partial):
                    right = partial[(position + 1) % len(partial)]
                    delta = float(
                        problem.distances[left, city]
                        + problem.distances[city, right]
                        - problem.distances[left, right]
                    )
                    option = (delta, city, position)
                    if best is None or option < best:
                        best = option
            assert best is not None
            _, city, position = best
            partial.insert(position + 1, city)
            removed.remove(city)
        return tuple(partial)

    @staticmethod
    def _restart_tour(
        problem: EuclideanTravellingSalespersonProblem,
        rng: np.random.Generator,
    ) -> tuple[int, ...]:
        remaining = set(problem.nodes)
        current = int(rng.integers(problem.city_count))
        tour = [current]
        remaining.remove(current)
        while remaining:
            current = min(
                remaining,
                key=lambda city: (problem.distances[tour[-1], city], city),
            )
            tour.append(current)
            remaining.remove(current)
        return tuple(tour)

    @staticmethod
    def _accept(delta: float, iteration: int, mean_distance: float) -> bool:
        policy = _GENOME["acceptance"]
        kind = policy["kind"]
        if kind in {"first", "best"}:
            return delta < -1e-12
        scale = float(policy["initial"]) * max(mean_distance, 1e-12) * float(policy["decay"]) ** iteration
        if kind == "threshold":
            return delta <= scale
        return delta <= scale * max(0.1, 1.0 - iteration / 10_000.0)

    @classmethod
    def _two_opt(
        cls,
        problem: EuclideanTravellingSalespersonProblem,
        initial_tour: tuple[int, ...],
        candidate_lists: list[set[int]],
        *,
        max_iterations: int,
        mean_distance: float,
    ) -> tuple[tuple[int, ...], int, int]:
        tour = list(initial_tour)
        accepted = evaluations = 0
        n = problem.city_count
        while accepted < max_iterations:
            positions = {city: position for position, city in enumerate(tour)}
            best: tuple[float, int, int] | None = None
            stop = False
            for first_edge, first_city in enumerate(tour):
                for candidate_city in sorted(candidate_lists[first_city]):
                    second_edge = positions[candidate_city]
                    left, right = sorted((first_edge, second_edge))
                    if right - left <= 1 or (left == 0 and right == n - 1):
                        continue
                    delta = (
                        problem.distances[tour[left], tour[right]]
                        + problem.distances[tour[left + 1], tour[(right + 1) % n]]
                        - problem.distances[tour[left], tour[left + 1]]
                        - problem.distances[tour[right], tour[(right + 1) % n]]
                    )
                    evaluations += 1
                    if not cls._accept(float(delta), accepted, mean_distance):
                        continue
                    if _GENOME["acceptance"]["kind"] != "best":
                        best = (float(delta), left, right)
                        stop = True
                        break
                    if best is None or delta < best[0]:
                        best = (float(delta), left, right)
                if stop:
                    break
            if best is None:
                break
            _, left, right = best
            tour[left + 1:right + 1] = reversed(tour[left + 1:right + 1])
            accepted += 1
        return tuple(tour), accepted, evaluations

    @classmethod
    def _or_opt(
        cls,
        problem: EuclideanTravellingSalespersonProblem,
        initial_tour: tuple[int, ...],
        *,
        max_iterations: int,
        mean_distance: float,
    ) -> tuple[tuple[int, ...], int, int]:
        tour = list(initial_tour)
        accepted = evaluations = 0
        segment_size = int(_GENOME["move_schedule"]["or_opt_segment"])
        while accepted < max_iterations:
            current_cost = problem.objective_value(tuple(tour))
            best: tuple[float, list[int]] | None = None
            for start in range(len(tour) - segment_size + 1):
                segment = tour[start:start + segment_size]
                remainder = tour[:start] + tour[start + segment_size:]
                for position in range(len(remainder) + 1):
                    candidate = remainder[:position] + segment + remainder[position:]
                    if candidate == tour:
                        continue
                    delta = problem.objective_value(tuple(candidate)) - current_cost
                    evaluations += 1
                    if not cls._accept(float(delta), accepted, mean_distance):
                        continue
                    if _GENOME["acceptance"]["kind"] != "best":
                        best = (float(delta), candidate)
                        break
                    if best is None or delta < best[0]:
                        best = (float(delta), candidate)
                if best is not None and _GENOME["acceptance"]["kind"] != "best":
                    break
            if best is None:
                break
            tour = best[1]
            accepted += 1
        return tuple(tour), accepted, evaluations

    @classmethod
    def _bounded_three_opt(
        cls,
        problem: EuclideanTravellingSalespersonProblem,
        initial_tour: tuple[int, ...],
        *,
        trials: int,
        mean_distance: float,
        rng: np.random.Generator,
    ) -> tuple[tuple[int, ...], int, int]:
        if len(initial_tour) < 6 or trials <= 0:
            return initial_tour, 0, 0
        tour = list(initial_tour)
        accepted = evaluations = 0
        best: tuple[float, list[int]] | None = None
        seen: set[tuple[int, int, int]] = set()
        while evaluations < trials and len(seen) < math.comb(len(tour), 3):
            i, j, k = sorted(int(index) for index in rng.choice(len(tour), size=3, replace=False))
            if i == j or j == k:
                continue
            cuts = (i, j, k)
            if cuts in seen:
                continue
            seen.add(cuts)
            a, b, c, d = tour[:i], tour[i:j], tour[j:k], tour[k:]
            variants = (
                a + list(reversed(b)) + c + d,
                a + b + list(reversed(c)) + d,
                a + c + b + d,
                a + list(reversed(c)) + b + d,
                a + c + list(reversed(b)) + d,
                a + list(reversed(b)) + list(reversed(c)) + d,
                a + list(reversed(c)) + list(reversed(b)) + d,
            )
            current_cost = problem.objective_value(tuple(tour))
            for candidate in variants:
                if evaluations >= trials:
                    break
                delta = problem.objective_value(tuple(candidate)) - current_cost
                evaluations += 1
                if not cls._accept(float(delta), accepted, mean_distance):
                    continue
                if _GENOME["acceptance"]["kind"] == "best":
                    if best is None or delta < best[0]:
                        best = (float(delta), candidate)
                else:
                    tour = candidate
                    accepted += 1
                    break
        if best is not None:
            tour = best[1]
            accepted += 1
        return tuple(tour), accepted, evaluations

    @classmethod
    def _local_search(
        cls,
        problem: EuclideanTravellingSalespersonProblem,
        initial_tour: tuple[int, ...],
        candidate_lists: list[set[int]],
        *,
        limit: int,
        mean_distance: float,
        rng: np.random.Generator,
    ) -> tuple[tuple[int, ...], int, int]:
        move = _GENOME["move_schedule"]
        if move["kind"] == "or_opt":
            return cls._or_opt(
                problem, initial_tour, max_iterations=limit, mean_distance=mean_distance
            )
        tour, accepted, evaluations = cls._two_opt(
            problem, initial_tour, candidate_lists,
            max_iterations=limit, mean_distance=mean_distance,
        )
        if move["kind"] == "mixed":
            tour, extra_accepted, extra_evaluations = cls._or_opt(
                problem, tour, max_iterations=max(1, limit // 8),
                mean_distance=mean_distance,
            )
            return tour, accepted + extra_accepted, evaluations + extra_evaluations
        if move["kind"] == "bounded_three_opt":
            tour, extra_accepted, extra_evaluations = cls._bounded_three_opt(
                problem, tour, trials=int(move["three_opt_trials"]),
                mean_distance=mean_distance, rng=rng,
            )
            return tour, accepted + extra_accepted, evaluations + extra_evaluations
        return tour, accepted, evaluations

    def _ranked_lists(self, problem: EuclideanTravellingSalespersonProblem, mean_distance: float) -> list[set[int]]:
        edge = _GENOME["edge_generator"]
        n = problem.city_count
        near = build_geometric_candidate_lists(problem, nearest_neighbors=edge["nearest_neighbors"], angular_sectors=1)
        angular = build_geometric_candidate_lists(problem, nearest_neighbors=2, angular_sectors=edge["angular_sectors"])
        if edge["kind"] == "k_nearest":
            raw = near
        elif edge["kind"] == "angular":
            raw = angular
        elif edge["kind"] == "reciprocal":
            raw = [set(other for other in near[city] if city in near[other]) for city in range(n)]
        else:
            raw = [near[city] | angular[city] for city in range(n)]
        ranked: list[set[int]] = []
        for city in range(n):
            neighbours = sorted(raw[city], key=lambda other: (problem.distances[city, other], other))
            denominator = max(1, len(neighbours) - 1)
            def key(other: int) -> tuple[float, int]:
                offset = problem.points[other] - problem.points[city]
                score = self._priority(
                    float(problem.distances[city, other]),
                    neighbours.index(other) / denominator,
                    abs(math.atan2(float(offset[1]), float(offset[0]))) / math.pi,
                    mean_distance,
                    n,
                )
                return (score if math.isfinite(score) else float(problem.distances[city, other]), other)
            ranked.append(set(sorted(neighbours, key=key)[:edge["top_k"]]))
        return ranked

    def solve(
        self,
        problem: EuclideanTravellingSalespersonProblem,
        *,
        budget_seconds: float | None = None,
        context: object | None = None,
    ) -> SolverResult:
        started = perf_counter()
        n = problem.city_count
        nonzero = problem.distances[problem.distances > 0.0]
        mean_distance = float(np.mean(nonzero)) if nonzero.size else 1.0
        constructor = _GENOME["constructor"]["kind"]
        if constructor == "nearest_neighbor":
            tour = self._nearest_neighbor(problem)
        elif constructor == "multi_fragment":
            tour = self._multi_fragment(problem)
        else:
            tour = regret_insertion_tour(problem)
        candidate_lists = self._ranked_lists(problem, mean_distance)
        move = _GENOME["move_schedule"]
        limit = max(20, int(move["max_iterations"] * _GENOME["adaptation"]["budget_share"]))
        rng = np.random.default_rng(0)
        tour, accepted, evaluations = self._local_search(
            problem, tour, candidate_lists, limit=limit,
            mean_distance=mean_distance, rng=rng,
        )
        best = tour
        best_cost = problem.objective_value(best)
        stagnation = 0
        diversification = _GENOME["diversification"]
        for _ in range(diversification["restarts"]):
            if diversification["kind"] == "none":
                break
            if diversification["kind"] == "double_bridge":
                perturbed = self._double_bridge(best, rng)
            elif diversification["kind"] == "ruin_recreate":
                perturbed = self._ruin_recreate(
                    problem, best, rng, float(diversification["ruin_fraction"])
                )
            else:
                perturbed = self._restart_tour(problem, rng)
            candidate, extra_accepted, extra_evaluations = self._local_search(
                problem, perturbed, candidate_lists, limit=max(20, limit // 4),
                mean_distance=mean_distance, rng=rng,
            )
            accepted += extra_accepted
            evaluations += extra_evaluations
            cost = problem.objective_value(candidate)
            if cost < best_cost:
                best, best_cost, stagnation = candidate, cost, 0
            else:
                stagnation += 1
            if stagnation >= _GENOME["adaptation"]["stagnation_limit"]:
                break
        return SolverResult(
            solution=best,
            cost=best_cost,
            exact=False,
            wall_seconds=perf_counter() - started,
            metadata={
                "algorithm": "tsp-hir-v1",
                "candidate_edges": candidate_edge_count(candidate_lists),
                "accepted_moves": accepted,
                "move_evaluations": evaluations,
                "genome": _GENOME,
            },
        )
'''


__all__ = [
    "CandidateTemplate", "TemplateError", "TSP_EDGE_FORMULA_TEMPLATE", "TSP_HIR_TEMPLATE",
    "apply_values_to_template", "get_template", "render_hir_template", "render_template",
    "template_relative_path", "validate_rendered_template", "validate_values",
]
