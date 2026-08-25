"""Inspectable deterministic case memory with BM25 retrieval."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping


_TOKEN = re.compile(r"[a-z0-9_+-]+", re.IGNORECASE)


@dataclass(frozen=True)
class CaseDocument:
    case_id: str
    campaign_id: str
    representation: str
    evidence_role: str
    profile_id: str | None
    operator: str | None
    text: str
    action: dict[str, Any]
    outcome: dict[str, Any]
    confidence: float = 1.0

    def to_mapping(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "campaign_id": self.campaign_id,
            "representation": self.representation, "evidence_role": self.evidence_role,
            "profile_id": self.profile_id, "operator": self.operator,
            "text": self.text, "action": self.action, "outcome": self.outcome,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class RetrievedCase:
    document: CaseDocument
    score: float
    reasons: tuple[str, ...]


class CaseMemory:
    """Small in-process memory; raw prompts and hidden results never enter it."""

    def __init__(self, documents: Iterable[CaseDocument] = ()) -> None:
        self.documents = tuple(sorted(documents, key=lambda item: item.case_id))
        self._tokens = {item.case_id: _tokens(item.text) for item in self.documents}
        self._df = Counter(token for values in self._tokens.values() for token in set(values))
        self._avg_length = sum(len(values) for values in self._tokens.values()) / max(len(self.documents), 1)

    def retrieve(
        self,
        query: str,
        *,
        representation: str,
        evidence_roles: tuple[str, ...] = ("public_learning", "historical_training"),
        profile_id: str | None = None,
        operator: str | None = None,
        exclude_campaign_id: str | None = None,
        limit: int = 6,
    ) -> tuple[RetrievedCase, ...]:
        if limit < 1:
            return ()
        query_tokens = _tokens(query)
        rows: list[RetrievedCase] = []
        for document in self.documents:
            if document.representation != representation or document.evidence_role not in evidence_roles:
                continue
            if exclude_campaign_id is not None and document.campaign_id == exclude_campaign_id:
                continue
            score = self._bm25(query_tokens, self._tokens[document.case_id]) * document.confidence
            reasons: list[str] = ["bm25"]
            if profile_id is not None and profile_id == document.profile_id:
                score += 0.25
                reasons.append("profile_match")
            if operator is not None and operator == document.operator:
                score += 0.10
                reasons.append("operator_match")
            quality = _quality_value(document)
            if quality is not None:
                # Quality is a minimized regret.  A modest bounded adjustment
                # makes genuinely improving cases outrank merely gate-passing
                # cases without allowing one noisy score to dominate BM25.
                score += max(-0.5, min(0.5, -quality))
                reasons.append("generalization_quality" if document.outcome.get("generalization_objective_value") is not None else "development_quality")
            if score > 0.0:
                rows.append(RetrievedCase(document, score, tuple(reasons)))
        return tuple(sorted(rows, key=lambda item: (-item.score, item.document.case_id))[:limit])

    def to_mapping(self) -> dict[str, Any]:
        return {"method": "bm25@1", "documents": [item.to_mapping() for item in self.documents]}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CaseMemory":
        if value.get("method") != "bm25@1" or not isinstance(value.get("documents"), list):
            raise ValueError("unsupported or malformed case-memory snapshot")
        documents: list[CaseDocument] = []
        for raw in value["documents"]:
            if not isinstance(raw, Mapping):
                raise ValueError("case-memory documents must be objects")
            documents.append(CaseDocument(
                case_id=str(raw["case_id"]),
                campaign_id=str(raw["campaign_id"]),
                representation=str(raw["representation"]),
                evidence_role=str(raw["evidence_role"]),
                profile_id=str(raw["profile_id"]) if raw.get("profile_id") is not None else None,
                operator=str(raw["operator"]) if raw.get("operator") is not None else None,
                text=str(raw.get("text", "")),
                action=dict(raw.get("action", {})),
                outcome=dict(raw.get("outcome", {})),
                confidence=float(raw.get("confidence", 1.0)),
            ))
        return cls(documents)

    def retrieve_balanced(
        self,
        query: str,
        *,
        representation: str,
        profile_id: str | None = None,
        operator: str | None = None,
        exclude_campaign_id: str | None = None,
        positive_limit: int = 3,
        negative_limit: int = 3,
    ) -> tuple[RetrievedCase, ...]:
        """Return quality wins and counterexamples, deduplicated by behavior.

        Passing the correctness/integrity gate is feasibility, not positive
        scientific evidence.  A positive case must beat the paired control.
        """
        ranked = self.retrieve(
            query,
            representation=representation,
            profile_id=profile_id,
            operator=operator,
            exclude_campaign_id=exclude_campaign_id,
            limit=max(len(self.documents), 1),
        )
        positive = _unique_behavior(
            (item for item in ranked if (_quality_value(item.document) or 0.0) < 0.0),
            positive_limit,
        )
        negative = _unique_behavior(
            (
                item for item in ranked
                if not bool(item.document.outcome.get("valid"))
                or ((_quality_value(item.document) or 0.0) > 0.0)
            ),
            negative_limit,
        )
        return tuple(sorted((*positive, *negative), key=lambda item: (-item.score, item.document.case_id)))

    def _bm25(self, query: list[str], document: list[str]) -> float:
        if not query or not document:
            return 0.0
        k1, b = 1.2, 0.75
        frequencies = Counter(document)
        total = 0.0
        for token in query:
            if token not in frequencies:
                continue
            df = self._df.get(token, 0)
            idf = math.log(1.0 + (len(self.documents) - df + 0.5) / (df + 0.5))
            tf = frequencies[token]
            denominator = tf + k1 * (1.0 - b + b * len(document) / max(self._avg_length, 1.0))
            total += idf * tf * (k1 + 1.0) / denominator
        return total


def _tokens(text: str) -> list[str]:
    return [item.lower() for item in _TOKEN.findall(text)]


def _quality_value(document: CaseDocument) -> float | None:
    for name in ("generalization_objective_value", "objective_value"):
        value = document.outcome.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
    return None


def _unique_behavior(items: Iterable[RetrievedCase], limit: int) -> list[RetrievedCase]:
    selected: list[RetrievedCase] = []
    seen: set[str] = set()
    for item in items:
        identity = item.document.action.get("formula_behavior_digest")
        key = str(identity) if identity else item.document.case_id
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


__all__ = ["CaseDocument", "CaseMemory", "RetrievedCase"]
