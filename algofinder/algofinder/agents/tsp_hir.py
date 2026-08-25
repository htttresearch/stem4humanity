"""Typed heuristic intermediate representation for trusted Euclidean-TSP scaffolds.

HIR is deliberately data, never generated Python.  It is expressive enough to
vary the decisions relevant to a local-search heuristic while allowing the
authority to compile a deterministic solver and reject malformed proposals
before a candidate worktree exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from typing import Any, Mapping

from algofinder.trace.serialize import canonical_dumps


class HIRError(ValueError):
    """Raised when a genome, expression, or edit is not type-safe HIR."""


_FEATURES = frozenset({"distance", "normalized_rank", "angular_offset", "mean_distance", "n"})
_CONSTRUCTORS = frozenset({"nearest_neighbor", "regret_insertion", "multi_fragment"})
_EDGE_GENERATORS = frozenset({"k_nearest", "angular", "reciprocal", "union"})
_NEIGHBORHOODS = frozenset({"two_opt", "or_opt", "mixed", "bounded_three_opt"})
_ACCEPTANCE = frozenset({"first", "best", "threshold", "annealed"})
_DIVERSIFICATION = frozenset({"none", "double_bridge", "ruin_recreate", "restart"})
_EDIT_OPERATORS = frozenset({"replace_component", "set_parameter", "replace_expression"})
_COMPONENTS = frozenset(
    {"constructor", "edge_generator", "move_schedule", "acceptance", "diversification", "adaptation"}
)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HIRError(f"{label} must be an object")
    return dict(value)


def _finite(value: object, label: str, *, low: float | None = None, high: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise HIRError(f"{label} must be a finite number")
    number = float(value)
    if low is not None and number < low or high is not None and number > high:
        raise HIRError(f"{label} must be in [{low}, {high}]")
    return number


def _integer(value: object, label: str, *, low: int, high: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise HIRError(f"{label} must be an integer in [{low}, {high}]")
    return value


def _kind(mapping: Mapping[str, Any], allowed: frozenset[str], label: str) -> str:
    value = mapping.get("kind")
    if not isinstance(value, str) or value not in allowed:
        raise HIRError(f"{label}.kind must be one of {sorted(allowed)}")
    return value


def default_expression() -> dict[str, Any]:
    return {"kind": "feature", "name": "distance"}


def validate_expression(value: object, *, depth: int = 0) -> dict[str, Any]:
    """Validate a numeric expression tree and return its canonical mapping."""
    if depth > 6:
        raise HIRError("edge_score expression exceeds maximum depth 6")
    node = _mapping(value, "edge_score expression")
    kind = node.get("kind")
    if kind == "feature":
        name = node.get("name")
        if name not in _FEATURES or set(node) != {"kind", "name"}:
            raise HIRError(f"edge_score feature must be one of {sorted(_FEATURES)}")
        return {"kind": "feature", "name": name}
    if kind == "constant":
        if set(node) != {"kind", "value"}:
            raise HIRError("edge_score constant must contain only kind and value")
        return {"kind": "constant", "value": _finite(node["value"], "edge_score constant", low=-10_000, high=10_000)}
    if kind == "unary":
        if set(node) != {"kind", "op", "value"} or node.get("op") not in {"neg", "abs"}:
            raise HIRError("edge_score unary expression requires neg or abs")
        return {"kind": "unary", "op": node["op"], "value": validate_expression(node["value"], depth=depth + 1)}
    if kind == "binary":
        if set(node) != {"kind", "op", "left", "right"} or node.get("op") not in {"add", "sub", "mul", "div", "min", "max"}:
            raise HIRError("edge_score binary expression has an unsupported operator")
        op = str(node["op"])
        left = validate_expression(node["left"], depth=depth + 1)
        right = validate_expression(node["right"], depth=depth + 1)
        if left == right and op in {"min", "max"}:
            return left
        if op == "sub" and left == right:
            return {"kind": "constant", "value": 0.0}
        if op in {"add", "sub"} and _constant_value(right) == 0.0:
            return left
        if op == "add" and _constant_value(left) == 0.0:
            return right
        if op == "mul" and _constant_value(right) == 1.0:
            return left
        if op == "mul" and _constant_value(left) == 1.0:
            return right
        if op == "mul" and (_constant_value(left) == 0.0 or _constant_value(right) == 0.0):
            return {"kind": "constant", "value": 0.0}
        return {"kind": "binary", "op": op, "left": left, "right": right}
    if kind == "if":
        if set(node) != {"kind", "condition", "then", "otherwise"}:
            raise HIRError("edge_score if expression has invalid fields")
        condition = validate_condition(node["condition"], depth=depth + 1)
        then = validate_expression(node["then"], depth=depth + 1)
        otherwise = validate_expression(node["otherwise"], depth=depth + 1)
        if then == otherwise:
            return then
        return {"kind": "if", "condition": condition, "then": then, "otherwise": otherwise}
    raise HIRError("edge_score expression kind must be feature, constant, unary, binary, or if")


def _constant_value(value: Mapping[str, Any]) -> float | None:
    if value.get("kind") != "constant":
        return None
    number = value.get("value")
    return float(number) if isinstance(number, (int, float)) and not isinstance(number, bool) else None


def validate_condition(value: object, *, depth: int = 0) -> dict[str, Any]:
    node = _mapping(value, "edge_score condition")
    if set(node) != {"op", "left", "right"} or node.get("op") not in {"lt", "lte", "gt", "gte"}:
        raise HIRError("edge_score condition requires lt, lte, gt, or gte")
    return {
        "op": node["op"],
        "left": validate_expression(node["left"], depth=depth + 1),
        "right": validate_expression(node["right"], depth=depth + 1),
    }


def compile_expression(value: Mapping[str, Any]) -> str:
    """Compile validated HIR to a parenthesized numeric expression string."""
    kind = value["kind"]
    if kind == "feature":
        return str(value["name"])
    if kind == "constant":
        return repr(float(value["value"]))
    if kind == "unary":
        operand = compile_expression(value["value"])
        return f"(-({operand}))" if value["op"] == "neg" else f"abs({operand})"
    if kind == "binary":
        left = compile_expression(value["left"])
        right = compile_expression(value["right"])
        symbols = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
        if value["op"] in symbols:
            if value["op"] == "div":
                return f"(({left}) / (1e-12 + abs({right})))"
            return f"(({left}) {symbols[value['op']]} ({right}))"
        return f"{value['op']}(({left}), ({right}))"
    if kind == "if":
        condition = compile_condition(value["condition"])
        return f"(({compile_expression(value['then'])}) if ({condition}) else ({compile_expression(value['otherwise'])}))"
    raise HIRError(f"cannot compile expression kind {kind!r}")


def compile_condition(value: Mapping[str, Any]) -> str:
    symbols = {"lt": "<", "lte": "<=", "gt": ">", "gte": ">="}
    return f"({compile_expression(value['left'])} {symbols[value['op']]} {compile_expression(value['right'])})"


@dataclass(frozen=True)
class TspGenome:
    """A fully typed, compilable Euclidean-TSP heuristic proposal."""

    constructor: dict[str, Any]
    edge_generator: dict[str, Any]
    edge_score: dict[str, Any]
    move_schedule: dict[str, Any]
    acceptance: dict[str, Any]
    diversification: dict[str, Any]
    adaptation: dict[str, Any]
    schema_version: str = "tsp-hir-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "tsp-hir-v1":
            raise HIRError("unsupported TSP HIR schema version")
        object.__setattr__(self, "constructor", _validate_constructor(self.constructor))
        object.__setattr__(self, "edge_generator", _validate_edge_generator(self.edge_generator))
        object.__setattr__(self, "edge_score", validate_expression(self.edge_score))
        object.__setattr__(self, "move_schedule", _validate_move_schedule(self.move_schedule))
        object.__setattr__(self, "acceptance", _validate_acceptance(self.acceptance))
        object.__setattr__(self, "diversification", _validate_diversification(self.diversification))
        object.__setattr__(self, "adaptation", _validate_adaptation(self.adaptation))

    @classmethod
    def from_mapping(cls, value: object) -> "TspGenome":
        mapping = _mapping(value, "TSP HIR genome")
        required = {
            "schema_version", "constructor", "edge_generator", "edge_score", "move_schedule",
            "acceptance", "diversification", "adaptation",
        }
        if set(mapping) != required:
            raise HIRError("TSP HIR genome has missing or unknown components")
        return cls(**mapping)

    @classmethod
    def from_json(cls, value: str) -> "TspGenome":
        try:
            return cls.from_mapping(json.loads(value))
        except json.JSONDecodeError as exc:
            raise HIRError(f"TSP HIR genome is not JSON: {exc.msg}") from exc

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "constructor": self.constructor,
            "edge_generator": self.edge_generator,
            "edge_score": self.edge_score,
            "move_schedule": self.move_schedule,
            "acceptance": self.acceptance,
            "diversification": self.diversification,
            "adaptation": self.adaptation,
        }

    def to_json(self) -> str:
        return canonical_dumps(self.to_mapping())

    @property
    def score_python(self) -> str:
        return compile_expression(self.edge_score)


def default_genome() -> TspGenome:
    return TspGenome(
        constructor={"kind": "regret_insertion"},
        edge_generator={"kind": "union", "nearest_neighbors": 12, "angular_sectors": 8, "top_k": 8},
        edge_score=default_expression(),
        move_schedule={"kind": "two_opt", "max_iterations": 2_000, "or_opt_segment": 1, "three_opt_trials": 0},
        acceptance={"kind": "first", "initial": 0.0, "decay": 1.0},
        diversification={"kind": "double_bridge", "restarts": 4, "ruin_fraction": 0.15},
        adaptation={"stagnation_limit": 24, "budget_share": 1.0},
    )


def _validate_constructor(value: object) -> dict[str, Any]:
    mapping = _mapping(value, "constructor")
    kind = _kind(mapping, _CONSTRUCTORS, "constructor")
    if set(mapping) != {"kind"}:
        raise HIRError("constructor accepts only kind")
    return {"kind": kind}


def _validate_edge_generator(value: object) -> dict[str, Any]:
    mapping = _mapping(value, "edge_generator")
    if set(mapping) != {"kind", "nearest_neighbors", "angular_sectors", "top_k"}:
        raise HIRError("edge_generator requires kind, nearest_neighbors, angular_sectors, and top_k")
    kind = _kind(mapping, _EDGE_GENERATORS, "edge_generator")
    result = {
        "kind": kind,
        "nearest_neighbors": _integer(mapping["nearest_neighbors"], "nearest_neighbors", low=2, high=48),
        "angular_sectors": _integer(mapping["angular_sectors"], "angular_sectors", low=1, high=32),
        "top_k": _integer(mapping["top_k"], "top_k", low=2, high=32),
    }
    if kind == "angular":
        result["nearest_neighbors"] = 2
    if kind in {"k_nearest", "reciprocal"}:
        result["angular_sectors"] = 1
    return result


def _validate_move_schedule(value: object) -> dict[str, Any]:
    mapping = _mapping(value, "move_schedule")
    if set(mapping) != {"kind", "max_iterations", "or_opt_segment", "three_opt_trials"}:
        raise HIRError("move_schedule requires kind, max_iterations, or_opt_segment, and three_opt_trials")
    kind = _kind(mapping, _NEIGHBORHOODS, "move_schedule")
    result = {
        "kind": kind,
        "max_iterations": _integer(mapping["max_iterations"], "max_iterations", low=20, high=20_000),
        "or_opt_segment": _integer(mapping["or_opt_segment"], "or_opt_segment", low=1, high=3),
        "three_opt_trials": _integer(mapping["three_opt_trials"], "three_opt_trials", low=0, high=200),
    }
    if kind not in {"or_opt", "mixed"}:
        result["or_opt_segment"] = 1
    if kind == "bounded_three_opt" and result["three_opt_trials"] < 1:
        raise HIRError("bounded_three_opt requires at least one three_opt_trial")
    if kind != "bounded_three_opt":
        result["three_opt_trials"] = 0
    return result


def _validate_acceptance(value: object) -> dict[str, Any]:
    mapping = _mapping(value, "acceptance")
    if set(mapping) != {"kind", "initial", "decay"}:
        raise HIRError("acceptance requires kind, initial, and decay")
    kind = _kind(mapping, _ACCEPTANCE, "acceptance")
    result = {
        "kind": kind,
        "initial": _finite(mapping["initial"], "acceptance.initial", low=0.0, high=2.0),
        "decay": _finite(mapping["decay"], "acceptance.decay", low=0.01, high=1.0),
    }
    if kind in {"first", "best"}:
        result["initial"] = 0.0
        result["decay"] = 1.0
    return result


def _validate_diversification(value: object) -> dict[str, Any]:
    mapping = _mapping(value, "diversification")
    if set(mapping) != {"kind", "restarts", "ruin_fraction"}:
        raise HIRError("diversification requires kind, restarts, and ruin_fraction")
    kind = _kind(mapping, _DIVERSIFICATION, "diversification")
    result = {
        "kind": kind,
        "restarts": _integer(mapping["restarts"], "restarts", low=0, high=48),
        "ruin_fraction": _finite(mapping["ruin_fraction"], "ruin_fraction", low=0.02, high=0.8),
    }
    if kind == "none":
        result["restarts"] = 0
    if kind != "ruin_recreate":
        result["ruin_fraction"] = 0.15
    return result


def _validate_adaptation(value: object) -> dict[str, Any]:
    mapping = _mapping(value, "adaptation")
    if set(mapping) != {"stagnation_limit", "budget_share"}:
        raise HIRError("adaptation requires stagnation_limit and budget_share")
    return {
        "stagnation_limit": _integer(mapping["stagnation_limit"], "stagnation_limit", low=1, high=1_000),
        "budget_share": _finite(mapping["budget_share"], "budget_share", low=0.05, high=1.0),
    }


@dataclass(frozen=True)
class HIREdit:
    operator: str
    target: str
    value: dict[str, Any]
    hypothesis: str

    def __post_init__(self) -> None:
        if self.operator not in _EDIT_OPERATORS:
            raise HIRError(f"HIR edit operator must be one of {sorted(_EDIT_OPERATORS)}")
        if not isinstance(self.target, str) or not self.target:
            raise HIRError("HIR edit target must be a non-empty string")
        if not isinstance(self.value, dict):
            raise HIRError("HIR edit value must be an object")
        if not isinstance(self.hypothesis, str) or not self.hypothesis.strip() or len(self.hypothesis) > 600:
            raise HIRError("HIR edit hypothesis must be a non-empty string of at most 600 characters")
        if self.operator == "replace_component" and self.target not in _COMPONENTS:
            raise HIRError("replace_component target must be a genome component")
        if self.operator == "replace_expression" and self.target != "edge_score":
            raise HIRError("replace_expression target must be edge_score")
        if self.operator == "set_parameter":
            component, separator, field = self.target.partition(".")
            if not separator or component not in _COMPONENTS or not field:
                raise HIRError("set_parameter target must use component.field")

    @classmethod
    def from_mapping(cls, value: object) -> "HIREdit":
        mapping = _mapping(value, "HIR edit")
        if set(mapping) != {"operator", "target", "value", "hypothesis"}:
            raise HIRError("HIR edit requires operator, target, value, and hypothesis")
        return cls(
            operator=mapping["operator"], target=mapping["target"],
            value=_mapping(mapping["value"], "HIR edit value"), hypothesis=mapping["hypothesis"],
        )

    def to_mapping(self) -> dict[str, Any]:
        return {"operator": self.operator, "target": self.target, "value": self.value, "hypothesis": self.hypothesis}

    def to_json(self) -> str:
        return canonical_dumps(self.to_mapping())


def apply_edit(genome: TspGenome, edit: HIREdit) -> TspGenome:
    """Apply one typed model proposal; validation happens on the whole result."""
    mapping = genome.to_mapping()
    if edit.operator == "replace_component":
        current = _mapping(mapping[edit.target], edit.target)
        mapping[edit.target] = {**current, **edit.value}
    elif edit.operator == "set_parameter":
        component, _, field = edit.target.partition(".")
        if set(edit.value) != {"value"}:
            raise HIRError("set_parameter value must contain exactly value")
        mapping[component] = {**_mapping(mapping[component], component), field: edit.value["value"]}
    else:
        mapping["edge_score"] = edit.value
    return TspGenome.from_mapping(mapping)


def mutate_genome(genome: TspGenome, *, seed: int) -> TspGenome:
    """One deterministic, type-safe non-LLM mutation for high-volume search."""
    rng = random.Random(seed)
    mapping = genome.to_mapping()
    choice = rng.randrange(7)
    if choice == 0:
        edge = dict(mapping["edge_generator"])
        field = rng.choice(("nearest_neighbors", "angular_sectors", "top_k"))
        bounds = {"nearest_neighbors": (2, 48), "angular_sectors": (1, 32), "top_k": (2, 32)}[field]
        edge[field] = max(bounds[0], min(bounds[1], int(edge[field]) + rng.choice((-4, -2, 2, 4))))
        mapping["edge_generator"] = edge
    elif choice == 1:
        acceptance = dict(mapping["acceptance"])
        acceptance["kind"] = rng.choice(sorted(_ACCEPTANCE - {str(acceptance["kind"])}))
        acceptance["initial"] = round(rng.uniform(0.0, 0.08), 4)
        acceptance["decay"] = round(rng.uniform(0.85, 1.0), 4)
        mapping["acceptance"] = acceptance
    elif choice == 2:
        moves = dict(mapping["move_schedule"])
        moves["kind"] = rng.choice(sorted(_NEIGHBORHOODS - {str(moves["kind"])}))
        moves["max_iterations"] = max(20, min(20_000, int(moves["max_iterations"]) + rng.choice((-400, -200, 200, 400))))
        moves["or_opt_segment"] = rng.randrange(1, 4)
        moves["three_opt_trials"] = rng.randrange(16, 97) if moves["kind"] == "bounded_three_opt" else 0
        mapping["move_schedule"] = moves
    elif choice == 3:
        diversify = dict(mapping["diversification"])
        diversify["kind"] = rng.choice(sorted(_DIVERSIFICATION - {str(diversify["kind"])}))
        diversify["restarts"] = 0 if diversify["kind"] == "none" else rng.randrange(1, 13)
        diversify["ruin_fraction"] = round(rng.uniform(0.05, 0.35), 3)
        mapping["diversification"] = diversify
    elif choice == 4:
        mapping["constructor"] = {
            "kind": rng.choice(sorted(_CONSTRUCTORS - {str(mapping["constructor"]["kind"])}))
        }
    elif choice == 5:
        mapping["edge_score"] = {
            "kind": "binary", "op": rng.choice(("add", "sub", "mul", "min", "max")),
            "left": mapping["edge_score"],
            "right": {"kind": "binary", "op": "mul", "left": {"kind": "constant", "value": round(rng.uniform(0.01, 0.3), 3)}, "right": {"kind": "feature", "name": rng.choice(sorted(_FEATURES - {"distance"}))}},
        }
    else:
        mapping["edge_score"] = {
            "kind": "if",
            "condition": {"op": "gt", "left": {"kind": "feature", "name": "normalized_rank"}, "right": {"kind": "constant", "value": 0.5}},
            "then": mapping["edge_score"],
            "otherwise": {
                "kind": "binary", "op": "mul",
                "left": {"kind": "feature", "name": "mean_distance"},
                "right": {"kind": "feature", "name": "normalized_rank"},
            },
        }
    mutated = TspGenome.from_mapping(mapping)
    if mutated.to_json() == genome.to_json():
        fallback = mutated.to_mapping()
        edge = dict(fallback["edge_generator"])
        edge["top_k"] = 9 if int(edge["top_k"]) == 8 else 8
        fallback["edge_generator"] = edge
        mutated = TspGenome.from_mapping(fallback)
    return mutated


def crossover_genomes(left: TspGenome, right: TspGenome, *, seed: int) -> TspGenome:
    """Type-compatible component crossover; both parents remain immutable."""
    rng = random.Random(seed)
    left_map, right_map = left.to_mapping(), right.to_mapping()
    result = {"schema_version": "tsp-hir-v1"}
    for component in (
        "constructor", "edge_generator", "edge_score", "move_schedule", "acceptance", "diversification", "adaptation",
    ):
        result[component] = (left_map if rng.random() < 0.5 else right_map)[component]
    return TspGenome.from_mapping(result)


def edit_between_genomes(
    parent: TspGenome,
    child: TspGenome,
    *,
    hypothesis: str,
) -> HIREdit:
    """Encode a one-component genome change as the exact typed edit applied."""
    parent_map = parent.to_mapping()
    child_map = child.to_mapping()
    changed = [
        component
        for component in (
            "constructor", "edge_generator", "edge_score", "move_schedule",
            "acceptance", "diversification", "adaptation",
        )
        if parent_map[component] != child_map[component]
    ]
    if len(changed) != 1:
        raise HIRError(
            f"one HIR proposal must change exactly one component, found {len(changed)}"
        )
    target = changed[0]
    return HIREdit(
        operator="replace_expression" if target == "edge_score" else "replace_component",
        target=target,
        value=dict(child_map[target]),
        hypothesis=hypothesis,
    )


def _json(value: str, label: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise HIRError(f"{label} is not JSON: {exc.msg}") from exc


def parse_edit_json(value: str) -> HIREdit:
    if not isinstance(value, str) or len(value.encode("utf-8")) > 8_000:
        raise HIRError("HIR edit JSON must be a string of at most 8 KiB")
    return HIREdit.from_mapping(_json(value, "HIR edit"))


__all__ = [
    "HIREdit", "HIRError", "TspGenome", "apply_edit", "compile_expression", "crossover_genomes",
    "default_expression", "default_genome", "edit_between_genomes", "mutate_genome",
    "parse_edit_json", "validate_expression",
]
