"""Small structured proposal-pool helpers; authority validation remains mandatory."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Iterable

from algofinder.agents.learning.normalization import NormalizationError, normalize_formula


@dataclass(frozen=True)
class RankedProposal:
    expression: str
    origin: str
    exact_digest: str
    structural_digest: str
    rank_key: tuple[float, ...]


def deterministic_formula_repairs(expression: str) -> tuple[str, ...]:
    """Produce causal one-literal repairs, never source-code mutations."""
    identity = normalize_formula(expression)
    tree = ast.parse(identity.exact_expression, mode="eval")
    values = [node for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)]
    candidates: list[str] = []
    for index in range(len(values)):
        for factor in (0.5, 1.5):
            clone = ast.parse(identity.exact_expression, mode="eval")
            clone_values = [node for node in ast.walk(clone) if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)]
            clone_values[index].value = round(float(clone_values[index].value) * factor, 12)
            candidate = ast.unparse(ast.fix_missing_locations(clone))
            if candidate != identity.exact_expression:
                candidates.append(candidate)
    # No literals: introduce exactly one approved low-magnitude rank term.
    if not candidates:
        candidates.append(f"({identity.exact_expression}) + 0.05 * mean_distance * normalized_rank")
    unique: list[str] = []
    seen = {identity.exact_digest}
    for candidate in candidates:
        try:
            digest = normalize_formula(candidate).exact_digest
        except NormalizationError:
            continue
        if digest not in seen:
            seen.add(digest)
            unique.append(candidate)
    return tuple(unique)


def rank_formula_pool(
    expressions: Iterable[str],
    *,
    existing_exact_digests: set[str],
    existing_structural_digests: set[str],
    predicted_valid_novel_probability: dict[str, float] | None = None,
    distribution_case_support: dict[str, float] | None = None,
    expected_qd_contribution: dict[str, float] | None = None,
    predicted_cost: dict[str, float] | None = None,
) -> tuple[RankedProposal, ...]:
    """Apply the v1 lexicographic ranking; no score becomes an acceptance gate."""
    rows: list[RankedProposal] = []
    for expression in expressions:
        identity = normalize_formula(expression)
        key = identity.exact_digest
        exact_unique = float(identity.exact_digest not in existing_exact_digests)
        structural_unique = float(identity.structural_digest not in existing_structural_digests)
        valid_novel = (predicted_valid_novel_probability or {}).get(key, 0.0)
        support = (distribution_case_support or {}).get(key, 0.0)
        qd = (expected_qd_contribution or {}).get(key, 0.0)
        cost = (predicted_cost or {}).get(key, 0.0)
        rows.append(RankedProposal(
            expression=identity.exact_expression,
            origin="deterministic_ast_repair",
            exact_digest=identity.exact_digest,
            structural_digest=identity.structural_digest,
            rank_key=(-exact_unique, -structural_unique, -valid_novel, -support, -qd, cost),
        ))
    return tuple(sorted(rows, key=lambda item: (item.rank_key, item.exact_digest)))


__all__ = ["RankedProposal", "deterministic_formula_repairs", "rank_formula_pool"]
