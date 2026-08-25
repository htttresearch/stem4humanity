"""Feedback-zone redaction for the evaluator-to-agent boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from algofinder.agents.contracts import FeedbackZone
from algofinder.trace.serialize import to_json_value


class FeedbackError(ValueError):
    """Raised when feedback would expose a protected evaluator detail."""


_PRIVATE_KEYS = frozenset({
    "instance", "instance_id", "instance_name", "instances", "solution",
    "trace", "trace_rel", "trace_regions", "manifest_path", "seed",
    "run_id", "run_ids", "session_id", "session_ids", "raw_runs",
})


@dataclass(frozen=True)
class FeedbackPolicy:
    """Defines the only evaluator output an agent may observe per zone."""

    rounding: int = 4

    def __post_init__(self) -> None:
        if self.rounding < 0 or self.rounding > 12:
            raise FeedbackError("rounding must be between 0 and 12")

    def redact(self, value: dict[str, Any], zone: FeedbackZone) -> dict[str, Any]:
        """Return machine-readable, zone-appropriate feedback.

        Public feedback can include detail.  Validation retains only aggregate
        measurements.  Challenge feedback is deliberately reduced to a gate
        status and rounded aggregate metrics, preventing adaptive probing of
        individual hidden instances.
        """
        if zone == "public":
            return _round(to_json_value(value), self.rounding)
        if zone == "validation":
            return _redact_validation(value, self.rounding)
        if zone == "challenge":
            return _redact_challenge(value, self.rounding)
        raise FeedbackError(f"unknown feedback zone {zone!r}")


def _redact_validation(value: dict[str, Any], rounding: int) -> dict[str, Any]:
    allowed = {"gate", "status", "aggregate", "metric_vector", "summary", "reason", "stage"}
    return {
        key: _round(item, rounding)
        for key, item in value.items()
        if key in allowed and key not in _PRIVATE_KEYS
    }


def _redact_challenge(value: dict[str, Any], rounding: int) -> dict[str, Any]:
    gate = value.get("gate", value.get("status", "blocked"))
    result: dict[str, Any] = {"gate": gate}
    aggregate = value.get("aggregate") or value.get("metric_vector")
    if isinstance(aggregate, dict):
        result["aggregate"] = _round({
            key: item for key, item in aggregate.items()
            if key not in _PRIVATE_KEYS
        }, rounding)
    if value.get("reason") in {"policy", "build", "correctness", "resource"}:
        result["reason"] = value["reason"]
    return result


def _round(value: Any, digits: int) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {str(key): _round(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item, digits) for item in value]
    return value


__all__ = ["FeedbackError", "FeedbackPolicy"]
