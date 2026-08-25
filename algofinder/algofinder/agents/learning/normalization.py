"""Stable formula and HIR identities for duplicate checks and features."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Iterable


class NormalizationError(ValueError):
    """Raised when a proposal is outside the trusted representation grammar."""


_FORMULA_VARIABLES = frozenset({"distance", "normalized_rank", "angular_offset", "mean_distance", "n"})
_FORMULA_CALLS = frozenset({"abs", "min", "max"})
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.IfExp, ast.Compare,
    ast.Name, ast.Load, ast.Constant, ast.Call, ast.Add, ast.Sub, ast.Mult,
    ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.And,
    ast.Or, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
)
_PROBE_CONTEXTS = (
    (
        {"mean_distance": 0.35, "n": 24.0},
        (
            (0.10, 0.00, 0.00), (0.12, 0.20, 0.85),
            (0.14, 0.40, 0.20), (0.20, 0.60, 0.70),
            (0.32, 0.80, 0.10), (0.48, 1.00, 1.00),
        ),
    ),
    (
        {"mean_distance": 0.80, "n": 72.0},
        (
            (0.12, 0.00, 1.00), (0.18, 0.20, 0.05),
            (0.30, 0.40, 0.55), (0.55, 0.60, 0.25),
            (0.90, 0.80, 0.80), (1.40, 1.00, 0.00),
        ),
    ),
    (
        {"mean_distance": 1.40, "n": 144.0},
        (
            (0.20, 0.00, 0.30), (0.35, 0.20, 0.95),
            (0.60, 0.40, 0.00), (1.00, 0.60, 0.65),
            (1.60, 0.80, 0.15), (2.30, 1.00, 0.75),
        ),
    ),
)


@dataclass(frozen=True)
class FormulaIdentity:
    exact_expression: str
    exact_ast: str
    structural_ast: str
    exact_digest: str
    structural_digest: str
    probe_values: tuple[float, ...]
    probe_behavior_digest: str
    features: dict[str, Any]


def normalize_formula(expression: str) -> FormulaIdentity:
    """Return canonical syntax, structure, and edge-order behavior identities.

    Formula programs are ranking functions, so their scientific identity is
    the edge order they induce—not whitespace or redundant operations.  The
    canonicalizer knows the representation's domain invariants (all five
    variables are non-negative and ``normalized_rank``/``angular_offset`` are
    in ``[0, 1]``), while the behavior digest compares rankings within several
    fixed synthetic instances.  Each probe's rank is monotone in distance,
    matching the trusted scaffold; treating those fields as independent would
    falsely label distance-monotone formulas as behaviorally novel.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise NormalizationError(f"invalid formula syntax: {exc.msg}") from exc
    _validate_formula(tree)
    tree = ast.fix_missing_locations(_DomainCanonicalizer().visit(tree))
    _validate_formula(tree)
    exact = ast.dump(tree, annotate_fields=True, include_attributes=False)
    structural_tree = _LiteralShape().visit(ast.fix_missing_locations(ast.parse(expression.strip(), mode="eval")))
    structural = ast.dump(structural_tree, annotate_fields=True, include_attributes=False)
    canonical = ast.unparse(tree)
    probes, behavior = _probe_behavior(tree)
    features = _formula_features(tree)
    return FormulaIdentity(
        exact_expression=canonical,
        exact_ast=exact,
        structural_ast=structural,
        exact_digest=_digest(exact),
        structural_digest=_digest(structural),
        probe_values=probes,
        probe_behavior_digest=_digest(json.dumps(behavior, separators=(",", ":"))),
        features=features,
    )


def normalize_hir(value: str | dict[str, Any]) -> dict[str, Any]:
    """Canonicalize a typed HIR JSON value without interpreting its fitness."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise NormalizationError("invalid HIR JSON") from exc
    if not isinstance(value, dict):
        raise NormalizationError("HIR must be a JSON object")
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return {"canonical_json": canonical, "digest": _digest(canonical), "keys": tuple(sorted(value))}


def action_identity(value: str | dict[str, Any], *, representation: str) -> dict[str, Any]:
    if representation == "formula":
        identity = normalize_formula(str(value))
        return {
            "representation": "formula",
            "exact_digest": identity.exact_digest,
            "structural_digest": identity.structural_digest,
            "probe_behavior_digest": identity.probe_behavior_digest,
            "features": identity.features,
        }
    if representation == "hir":
        identity = normalize_hir(value)
        return {"representation": "hir", "exact_digest": identity["digest"], "structural_digest": identity["digest"], "features": {"keys": identity["keys"]}}
    raise NormalizationError(f"unsupported normalized representation {representation!r}")


def _validate_formula(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise NormalizationError(f"formula uses unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in _FORMULA_VARIABLES | _FORMULA_CALLS:
            raise NormalizationError(f"formula uses unknown name {node.id!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FORMULA_CALLS:
                raise NormalizationError("formula may call only abs, min, or max")
            if node.keywords:
                raise NormalizationError("formula calls may not use keyword arguments")
        if isinstance(node, ast.Constant) and (not isinstance(node.value, (int, float)) or isinstance(node.value, bool) or not math.isfinite(float(node.value))):
            raise NormalizationError("formula literals must be finite numeric values")


class _LiteralShape(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.Constant:  # noqa: N802
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return ast.copy_location(ast.Constant(value="<number>"), node)
        return node


class _DomainCanonicalizer(ast.NodeTransformer):
    """Remove identities that are exact on the trusted formula domain."""

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        node = self.generic_visit(node)
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return node
        if node.func.id == "abs" and len(node.args) == 1 and _nonnegative(node.args[0]):
            return ast.copy_location(node.args[0], node)
        if node.func.id == "max" and len(node.args) == 2:
            left, right = node.args
            if _zero(left) and _nonnegative(right):
                return ast.copy_location(right, node)
            if _zero(right) and _nonnegative(left):
                return ast.copy_location(left, node)
        if node.func.id == "min" and len(node.args) == 2:
            left, right = node.args
            if _one(left) and _unit_interval(right):
                return ast.copy_location(right, node)
            if _one(right) and _unit_interval(left):
                return ast.copy_location(left, node)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:  # noqa: N802
        node = self.generic_visit(node)
        if not isinstance(node, ast.BinOp):
            return node
        if isinstance(node.op, ast.Add):
            if _zero(node.left):
                return ast.copy_location(node.right, node)
            if _zero(node.right):
                return ast.copy_location(node.left, node)
        if isinstance(node.op, ast.Sub) and _zero(node.right):
            return ast.copy_location(node.left, node)
        if isinstance(node.op, ast.Mult):
            if _one(node.left):
                return ast.copy_location(node.right, node)
            if _one(node.right):
                return ast.copy_location(node.left, node)
        if isinstance(node.op, ast.Div) and _one(node.right):
            return ast.copy_location(node.left, node)
        return node


def _zero(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and float(node.value) == 0.0


def _one(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and float(node.value) == 1.0


def _unit_interval(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id in {"normalized_rank", "angular_offset"}


def _nonnegative(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in _FORMULA_VARIABLES
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value) >= 0.0
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id in {"abs", "max"}
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
        return _nonnegative(node.left) and _nonnegative(node.right)
    return False


def _probe_behavior(tree: ast.Expression) -> tuple[tuple[float, ...], tuple[tuple[int, ...], ...]]:
    code = compile(tree, "<formula>", "eval")
    values: list[float] = []
    rankings: list[tuple[int, ...]] = []
    for context, edges in _PROBE_CONTEXTS:
        local: list[float] = []
        for distance, normalized_rank, angular_offset in edges:
            point = {
                **context,
                "distance": distance,
                "normalized_rank": normalized_rank,
                "angular_offset": angular_offset,
            }
            try:
                value = eval(code, {"__builtins__": {}, "abs": abs, "min": min, "max": max}, point)  # noqa: S307 - AST was strictly validated above.
            except (ArithmeticError, ValueError, TypeError) as exc:
                raise NormalizationError(f"formula failed a deterministic probe: {exc}") from exc
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise NormalizationError("formula produced a non-finite numeric probe result")
            local.append(round(float(value), 12))
        values.extend(local)
        rankings.append(tuple(sorted(range(len(local)), key=lambda index: (local[index], index))))
    return tuple(values), tuple(rankings)


def _formula_features(tree: ast.AST) -> dict[str, Any]:
    nodes = list(ast.walk(tree))
    binary = [type(node.op).__name__.lower() for node in nodes if isinstance(node, ast.BinOp)]
    calls = [node.func.id for node in nodes if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    variables = sorted({node.id for node in nodes if isinstance(node, ast.Name) and node.id in _FORMULA_VARIABLES})
    literals = [float(node.value) for node in nodes if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)]
    return {
        "ast_nodes": len(nodes),
        "ast_depth": _depth(tree),
        "binary_operators": binary,
        "calls": calls,
        "variables": variables,
        "conditional_count": sum(isinstance(node, ast.IfExp) for node in nodes),
        "literal_signs": [0 if value == 0 else (1 if value > 0 else -1) for value in literals],
        "literal_log_magnitude": [round(math.log10(max(abs(value), 1e-12)), 3) for value in literals],
    }


def _depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 + max((_depth(child) for child in children), default=0)


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


__all__ = ["FormulaIdentity", "NormalizationError", "action_identity", "normalize_formula", "normalize_hir"]
