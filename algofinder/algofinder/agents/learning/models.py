"""Low-capacity, JSON-serializable learning models for v1 policy advising."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping


@dataclass
class BetaBinomialTable:
    """Hierarchically queried success posteriors without opaque model files."""

    alpha: float = 1.0
    beta: float = 1.0
    counts: dict[str, tuple[int, int]] = field(default_factory=dict)

    def observe(self, key: str, success: bool) -> None:
        successes, trials = self.counts.get(key, (0, 0))
        self.counts[key] = (successes + int(success), trials + 1)

    def posterior(self, keys: Iterable[str]) -> tuple[float, float, int]:
        """Use the first observed compatible key; callers supply broad fallbacks."""
        for key in keys:
            if key in self.counts:
                successes, trials = self.counts[key]
                alpha, beta = self.alpha + successes, self.beta + trials - successes
                return alpha / (alpha + beta), math.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))), trials
        return self.alpha / (self.alpha + self.beta), math.sqrt(self.alpha * self.beta / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))), 0

    def to_mapping(self) -> dict[str, Any]:
        return {"kind": "beta_binomial@1", "alpha": self.alpha, "beta": self.beta, "counts": {key: list(value) for key, value in sorted(self.counts.items())}}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BetaBinomialTable":
        counts = {
            str(key): (int(item[0]), int(item[1]))
            for key, item in dict(value.get("counts", {})).items()
            if isinstance(item, (list, tuple)) and len(item) == 2
        }
        return cls(float(value.get("alpha", 1.0)), float(value.get("beta", 1.0)), counts)


@dataclass
class ContinuousStatsTable:
    """Inspectable running moments for minimized continuous objectives.

    Estimates are shrunk toward the control value (zero gap) and expose an
    uncertainty term.  Runtime selection uses the conservative upper bound
    ``mean + uncertainty`` rather than rewarding poorly sampled programs.
    """

    prior_mean: float = 0.0
    prior_strength: float = 2.0
    prior_scale: float = 0.10
    stats: dict[str, tuple[int, float, float]] = field(default_factory=dict)

    def observe(self, key: str, value: float) -> None:
        if not math.isfinite(float(value)):
            return
        count, total, squares = self.stats.get(key, (0, 0.0, 0.0))
        numeric = float(value)
        self.stats[key] = (count + 1, total + numeric, squares + numeric * numeric)

    def estimate(self, keys: Iterable[str]) -> tuple[float, float, int]:
        for key in keys:
            if key not in self.stats:
                continue
            count, total, squares = self.stats[key]
            denominator = count + self.prior_strength
            posterior_mean = (total + self.prior_strength * self.prior_mean) / denominator
            sample_mean = total / count
            variance = (
                max(0.0, (squares - count * sample_mean * sample_mean) / (count - 1))
                if count > 1 else self.prior_scale * self.prior_scale
            )
            uncertainty = math.sqrt(
                (variance + self.prior_strength * self.prior_scale * self.prior_scale)
                / denominator
            )
            return posterior_mean, uncertainty, count
        return self.prior_mean, self.prior_scale, 0

    def to_mapping(self) -> dict[str, Any]:
        return {
            "kind": "continuous-moments@1",
            "prior_mean": self.prior_mean,
            "prior_strength": self.prior_strength,
            "prior_scale": self.prior_scale,
            "stats": {key: list(value) for key, value in sorted(self.stats.items())},
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ContinuousStatsTable":
        if value.get("kind") != "continuous-moments@1":
            raise ValueError("unsupported continuous statistics table")
        stats = {
            str(key): (int(item[0]), float(item[1]), float(item[2]))
            for key, item in dict(value.get("stats", {})).items()
            if isinstance(item, (list, tuple)) and len(item) == 3
        }
        return cls(
            prior_mean=float(value.get("prior_mean", 0.0)),
            prior_strength=float(value.get("prior_strength", 2.0)),
            prior_scale=float(value.get("prior_scale", 0.10)),
            stats=stats,
        )


@dataclass(frozen=True)
class LinearProbabilityModel:
    """Inspectable logistic model trained with deterministic batch descent."""

    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    regularization: float

    def predict(self, row: Mapping[str, float]) -> float:
        score = self.intercept + sum(coefficient * float(row.get(name, 0.0)) for name, coefficient in zip(self.feature_names, self.coefficients))
        return 1.0 / (1.0 + math.exp(-max(min(score, 40.0), -40.0)))

    def to_mapping(self) -> dict[str, Any]:
        return {"kind": "logistic@1", "feature_names": list(self.feature_names), "coefficients": list(self.coefficients), "intercept": self.intercept, "regularization": self.regularization}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LinearProbabilityModel":
        if value.get("kind") != "logistic@1":
            raise ValueError("unsupported probability model")
        names = tuple(str(item) for item in value.get("feature_names", ()))
        coefficients = tuple(float(item) for item in value.get("coefficients", ()))
        if len(names) != len(coefficients):
            raise ValueError("probability-model feature and coefficient counts differ")
        return cls(names, coefficients, float(value.get("intercept", 0.0)), float(value.get("regularization", 1.0)))


def fit_logistic(
    rows: Iterable[Mapping[str, float]], labels: Iterable[bool], *, feature_names: tuple[str, ...], regularization: float = 1.0, iterations: int = 300, learning_rate: float = 0.1,
) -> LinearProbabilityModel:
    data, targets = list(rows), [1.0 if item else 0.0 for item in labels]
    if len(data) != len(targets):
        raise ValueError("row and label counts differ")
    coefficients = [0.0] * len(feature_names)
    intercept = 0.0
    if not data:
        return LinearProbabilityModel(feature_names, tuple(coefficients), intercept, regularization)
    for _ in range(iterations):
        gradients = [0.0] * len(feature_names)
        intercept_gradient = 0.0
        for row, target in zip(data, targets):
            score = intercept + sum(coefficients[index] * float(row.get(name, 0.0)) for index, name in enumerate(feature_names))
            predicted = 1.0 / (1.0 + math.exp(-max(min(score, 40.0), -40.0)))
            error = predicted - target
            intercept_gradient += error
            for index, name in enumerate(feature_names):
                gradients[index] += error * float(row.get(name, 0.0))
        scale = 1.0 / len(data)
        intercept -= learning_rate * intercept_gradient * scale
        for index in range(len(coefficients)):
            coefficients[index] -= learning_rate * (gradients[index] * scale + regularization * coefficients[index] * scale)
    return LinearProbabilityModel(feature_names, tuple(coefficients), intercept, regularization)


__all__ = ["BetaBinomialTable", "ContinuousStatsTable", "LinearProbabilityModel", "fit_logistic"]
