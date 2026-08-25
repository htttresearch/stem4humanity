"""Quality-diversity search policy over evaluator-authored campaign evidence.

This module is deliberately a *policy*, not an evaluator.  It obtains candidate
metadata and redacted evaluation summaries through :class:`AgentTools`, builds
a deterministic MAP-Elites archive, and recommends a parent/operator tuple to
the research runner.  It never imports benchmark code and cannot write scores.

The implementation combines two useful ideas for program search:

* MAP-Elites preserves the best candidate in each behavioural niche instead of
  collapsing the population to one globally best program.
* A UCB-style operator policy allocates trials to operators that have produced
  good descendants while retaining an explicit exploration bonus.

The only state not reconstructible from the campaign ledger is the number of
times this policy instance selected each archive cell.  That state affects
exploration order, never recorded fitness or archive membership.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from bisect import bisect_left
from hashlib import sha256
import json
import math
import random
from typing import Any, Iterable, Mapping, TYPE_CHECKING

if TYPE_CHECKING:  # Avoid making this policy a second runtime facade.
    from algofinder.agents.tools import AgentTools


_STAGE_RANK = {f"gate_{number}": number for number in range(7)}
_DEFAULT_OPERATORS = ("invent", "mutate", "recombine", "repair", "tune", "distill")


@dataclass(frozen=True)
class DescriptorAxis:
    """One MAP-Elites axis.

    Numeric values are discretised by ``boundaries``.  Other values form
    categorical cells.  An axis first looks in ``candidate[\"descriptors\"]``
    and then at the candidate's top-level fields, allowing such axes as
    ``generation_operator`` without duplicating ledger data.
    """

    name: str
    boundaries: tuple[float, ...] = ()
    missing_label: str = "<missing>"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("descriptor axis name must not be empty")
        if any(not _finite_number(value) for value in self.boundaries):
            raise ValueError("descriptor boundaries must be finite numbers")
        if tuple(sorted(set(self.boundaries))) != self.boundaries:
            raise ValueError("descriptor boundaries must be strictly increasing")

    def bucket(self, candidate: Mapping[str, Any]) -> str:
        descriptors = candidate.get("descriptors", {})
        value = descriptors.get(self.name) if isinstance(descriptors, Mapping) else None
        if value is None:
            value = candidate.get(self.name)
        if value is None:
            return self.missing_label
        if self.boundaries and _finite_number(value):
            index = bisect_left(self.boundaries, float(value))
            if index == 0:
                return f"(-inf,{_number_label(self.boundaries[0])}]"
            if index == len(self.boundaries):
                return f"({_number_label(self.boundaries[-1])},+inf)"
            return (
                f"({_number_label(self.boundaries[index - 1])},"
                f"{_number_label(self.boundaries[index])}]"
            )
        return _stable_label(value)


@dataclass(frozen=True)
class ArchiveConfig:
    """Fixed, reproducible configuration for the QD search policy."""

    metric: str = "paired_gap_percent"
    metric_aliases: tuple[str, ...] = ("mean_gap_percent",)
    minimize: bool = True
    axes: tuple[DescriptorAxis, ...] = (
        DescriptorAxis("algorithm_family"),
        DescriptorAxis("runtime_class"),
    )
    eligible_zones: tuple[str, ...] = ("public", "validation")
    max_stage: str = "gate_5"
    operators: tuple[str, ...] = _DEFAULT_OPERATORS
    cell_exploration: float = 0.35
    operator_exploration: float = 0.7

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ValueError("objective metric must not be empty")
        if any(not alias.strip() for alias in self.metric_aliases):
            raise ValueError("objective metric aliases must not be empty")
        if self.metric in self.metric_aliases or len(set(self.metric_aliases)) != len(
            self.metric_aliases
        ):
            raise ValueError("objective metric aliases must be unique and differ from metric")
        if self.max_stage not in _STAGE_RANK:
            raise ValueError(f"unknown maximum stage {self.max_stage!r}")
        if not self.axes or len({axis.name for axis in self.axes}) != len(self.axes):
            raise ValueError("archive axes must be non-empty and unique")
        if not self.operators or len(set(self.operators)) != len(self.operators):
            raise ValueError("operators must be non-empty and unique")
        if set(self.operators) - set(_DEFAULT_OPERATORS):
            raise ValueError("operators must use Candidate generation operators")
        if self.cell_exploration < 0 or self.operator_exploration < 0:
            raise ValueError("exploration coefficients must be non-negative")


@dataclass(frozen=True)
class CandidateObservation:
    candidate_id: str
    record: Mapping[str, Any]
    score: float | None
    stage: str | None
    zone: str | None
    evaluation_count: int
    failed_evaluations: int


@dataclass(frozen=True)
class Elite:
    candidate_id: str
    cell: tuple[str, ...]
    score: float
    stage: str
    zone: str
    evaluation_count: int
    descriptors: Mapping[str, Any]
    tie_key: tuple[int, str]


@dataclass(frozen=True)
class ArchiveSnapshot:
    """Immutable reconstruction of the archive at one ledger observation."""

    elites: tuple[Elite, ...]
    observations: tuple[CandidateObservation, ...]
    candidates_seen: int
    candidates_scored: int
    evaluations_seen: int
    failed_evaluations: int

    def elite_by_candidate(self) -> dict[str, Elite]:
        return {elite.candidate_id: elite for elite in self.elites}


@dataclass(frozen=True)
class OperatorEstimate:
    operator: str
    trials: int
    mean_reward: float
    exploration_bonus: float
    utility: float


@dataclass(frozen=True)
class ResearchStep:
    """A runner-consumable search decision; it performs no code mutation."""

    operator: str
    parent_ids: tuple[str, ...]
    target_cell: tuple[str, ...] | None
    policy_version: str
    rationale: str
    operator_estimates: tuple[OperatorEstimate, ...] = field(default_factory=tuple)
    behavior_policy: Mapping[str, Any] = field(default_factory=dict)


class QualityDiversitySearch:
    """Reconstruct a MAP-Elites archive and recommend research operations."""

    policy_version = "qd-map-elites-ucb@3"

    def __init__(
        self,
        tools: "AgentTools",
        *,
        config: ArchiveConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.tools = tools
        self.config = config or ArchiveConfig()
        self._rng = random.Random(seed)
        self._cell_visits: dict[tuple[str, ...], int] = {}
        self._validation_failures: dict[str, int] = {}
        self._last_behavior_policy: dict[str, Any] = {}

    @property
    def last_behavior_policy(self) -> dict[str, Any]:
        """Copy of the exact propensity metadata for the last recommendation."""
        return dict(self._last_behavior_policy)

    def record_validation_failure(self, operator: str) -> None:
        """Penalize an operator whose proposals all failed the build/novelty gate.

        Invalid proposals never become candidates, so they cannot be recovered
        from the candidate/evaluation ledger. The live policy still needs this
        signal or UCB treats an impossible operator as forever untried and keeps
        selecting it. Campaign summaries retain every rejected research step.
        """
        if operator not in self.config.operators:
            raise ValueError(f"unknown search operator {operator!r}")
        self._validation_failures[operator] = self._validation_failures.get(operator, 0) + 1

    def refresh(self) -> ArchiveSnapshot:
        """Rebuild the archive solely from records visible through AgentTools."""
        raw_candidates = self.tools.archive_query("candidate")
        candidates = sorted(
            (item for item in raw_candidates if isinstance(item, Mapping) and item.get("id")),
            key=_candidate_tie_key,
        )
        candidate_ids = [str(item["id"]) for item in candidates]
        rows: list[Mapping[str, Any]] = []
        if candidate_ids:
            comparison = self.tools.runs_compare(candidate_ids)
            raw_rows = comparison.get("evaluations", []) if isinstance(comparison, Mapping) else []
            rows = [item for item in raw_rows if isinstance(item, Mapping)]

        grouped: dict[str, list[Mapping[str, Any]]] = {item: [] for item in candidate_ids}
        for row in rows:
            candidate_id = row.get("candidate_id")
            if candidate_id in grouped:
                grouped[str(candidate_id)].append(row)

        observations = tuple(
            self._observe(candidate, grouped[str(candidate["id"])]) for candidate in candidates
        )
        elites_by_cell: dict[tuple[str, ...], Elite] = {}
        for observation in observations:
            if observation.score is None or observation.stage is None or observation.zone is None:
                continue
            cell = self._cell(observation.record)
            descriptors = observation.record.get("descriptors", {})
            elite = Elite(
                candidate_id=observation.candidate_id,
                cell=cell,
                score=observation.score,
                stage=observation.stage,
                zone=observation.zone,
                evaluation_count=observation.evaluation_count,
                descriptors=dict(descriptors) if isinstance(descriptors, Mapping) else {},
                tie_key=_candidate_tie_key(observation.record),
            )
            incumbent = elites_by_cell.get(cell)
            if incumbent is None or self._elite_key(elite) < self._elite_key(incumbent):
                elites_by_cell[cell] = elite

        return ArchiveSnapshot(
            elites=tuple(elites_by_cell[cell] for cell in sorted(elites_by_cell)),
            observations=observations,
            candidates_seen=len(candidates),
            candidates_scored=sum(item.score is not None for item in observations),
            evaluations_seen=len(rows),
            failed_evaluations=sum(item.failed_evaluations for item in observations),
        )

    def feasible_operators(self, snapshot: ArchiveSnapshot) -> tuple[str, ...]:
        """Return the current action support without consuming RNG state."""
        feasible = list(self.config.operators)
        if not snapshot.elites:
            has_failed_parent = any(item.failed_evaluations for item in snapshot.observations)
            if has_failed_parent and "repair" in feasible:
                feasible = ["repair"]
            else:
                feasible = [operator for operator in feasible if operator == "invent"]
        elif len(snapshot.elites) < 2:
            feasible = [operator for operator in feasible if operator != "recombine"]
        if snapshot.elites and not any(item.failed_evaluations for item in snapshot.observations):
            feasible = [operator for operator in feasible if operator != "repair"]
        if not feasible:
            raise RuntimeError("no generation operator is feasible for the current archive")
        return tuple(feasible)

    def recommend(
        self,
        snapshot: ArchiveSnapshot | None = None,
        *,
        operator_override: str | None = None,
        behavior_probabilities: Mapping[str, float] | None = None,
        behavior_policy_version: str | None = None,
    ) -> ResearchStep:
        """Choose an operator and parents with deterministic quality/exploration UCB."""
        snapshot = snapshot or self.refresh()
        feasible = list(self.feasible_operators(snapshot))

        rng_state_digest = sha256(repr(self._rng.getstate()).encode("utf-8")).hexdigest()
        estimates = self.operator_estimates(snapshot, operators=feasible)
        best_utility = max(item.utility for item in estimates)
        tied = sorted(
            item.operator for item in estimates if math.isclose(item.utility, best_utility)
        )
        if operator_override is not None:
            if operator_override not in feasible:
                raise ValueError(f"operator override is not feasible: {operator_override!r}")
            operator = operator_override
            supplied = dict(behavior_probabilities or {})
            if set(supplied) - set(feasible) or any(not math.isfinite(value) or value < 0 for value in supplied.values()):
                raise ValueError("behavior probabilities must be finite non-negative feasible actions")
            operator_probabilities = {candidate: float(supplied.get(candidate, 0.0)) for candidate in feasible}
            if not math.isclose(sum(operator_probabilities.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("behavior probabilities must sum to one over feasible actions")
            if operator_probabilities[operator] <= 0:
                raise ValueError("chosen operator must have positive logged propensity")
        else:
            operator = self._rng.choice(tied)
            operator_probabilities = {
                item.operator: (1.0 / len(tied) if item.operator in tied else 0.0)
                for item in estimates
            }

        selected, cell_probability = self._select_elite(snapshot)
        parents: tuple[str, ...] = ()
        parent_probability = 1.0
        target_cell: tuple[str, ...] | None = selected.cell if selected else None
        if operator == "repair":
            failed = [item for item in snapshot.observations if item.failed_evaluations]
            if failed:
                failed.sort(key=lambda item: (-item.failed_evaluations, _candidate_tie_key(item.record)))
                parents = (failed[0].candidate_id,)
                target_cell = self._cell(failed[0].record)
            elif selected:
                parents = (selected.candidate_id,)
        elif operator != "invent" and selected:
            parents = (selected.candidate_id,)
            if operator == "recombine":
                alternatives = [
                    elite for elite in snapshot.elites if elite.candidate_id != selected.candidate_id
                ]
                alternatives.sort(key=lambda elite: (elite.cell, elite.tie_key))
                second = self._rng.choice(alternatives)
                parents = (selected.candidate_id, second.candidate_id)
                parent_probability = 1.0 / len(alternatives)

        rationale = (
            f"{self.policy_version}: operator={operator}; parents={list(parents)}; "
            f"target_cell={list(target_cell) if target_cell else None}; "
            f"archive_cells={len(snapshot.elites)}; metric={self.config.metric}"
        )
        behavior_policy = {
            "policy_version": behavior_policy_version or self.policy_version,
            "behavior_mode": "stochastic_tiebreak",
            "feasible_operators": list(feasible),
            "operator_probabilities": operator_probabilities,
            "selection_probability": operator_probabilities[operator] * cell_probability * parent_probability,
            "operator_probability": operator_probabilities[operator],
            "cell_selection_probability": cell_probability,
            "parent_selection_probability": parent_probability,
            "rng_state_digest": rng_state_digest,
        }
        self._last_behavior_policy = behavior_policy
        return ResearchStep(
            operator=operator,
            parent_ids=parents,
            target_cell=target_cell,
            policy_version=self.policy_version,
            rationale=rationale,
            operator_estimates=estimates,
            behavior_policy=behavior_policy,
        )

    def operator_estimates(
        self,
        snapshot: ArchiveSnapshot,
        *,
        operators: Iterable[str] | None = None,
    ) -> tuple[OperatorEstimate, ...]:
        """Estimate operator utility from ledger outcomes plus a UCB bonus."""
        selected_operators = tuple(operators or self.config.operators)
        quality = self._quality_percentiles(snapshot.observations)
        by_id = {item.candidate_id: item for item in snapshot.observations}
        rewards: dict[str, list[float]] = {operator: [] for operator in selected_operators}
        for operator in selected_operators:
            rewards[operator].extend([-1.0] * self._validation_failures.get(operator, 0))

        for observation in snapshot.observations:
            operator = str(observation.record.get("generation_operator", "invent"))
            if operator not in rewards:
                continue
            if observation.score is None:
                rewards[operator].append(-1.0)
                continue
            child_quality = quality[observation.candidate_id]
            parent_qualities = [
                quality[parent_id]
                for parent_id in observation.record.get("parent_ids", ())
                if parent_id in by_id and by_id[parent_id].score is not None
            ]
            if parent_qualities:
                improvement = child_quality - max(parent_qualities)
                rewards[operator].append(0.5 * child_quality + 0.5 * improvement)
            else:
                rewards[operator].append(child_quality)

        total_trials = sum(len(items) for items in rewards.values())
        log_term = math.log(total_trials + len(selected_operators) + 1.0)
        estimates = []
        for operator in selected_operators:
            items = rewards[operator]
            mean = sum(items) / len(items) if items else 0.0
            if items:
                bonus = self.config.operator_exploration * math.sqrt(
                    log_term / (len(items) + 1.0)
                )
            else:
                # Force one trial of every feasible operator before ordinary
                # UCB competition; otherwise a lucky first invention can
                # permanently dominate small campaign budgets.
                bonus = 1.0 + self.config.operator_exploration
            estimates.append(
                OperatorEstimate(
                    operator=operator,
                    trials=len(items),
                    mean_reward=mean,
                    exploration_bonus=bonus,
                    utility=mean + bonus,
                )
            )
        return tuple(sorted(estimates, key=lambda item: item.operator))

    def nominate_elites(self, snapshot: ArchiveSnapshot | None = None) -> tuple[Any, ...]:
        """Nominate each new elite through AgentTools, never by editing the ledger."""
        snapshot = snapshot or self.refresh()
        decisions = self.tools.archive_query("decision")
        already_archived = {
            str(item.get("target_id"))
            for item in decisions
            if isinstance(item, Mapping)
            and item.get("decision_kind", item.get("kind")) == "archive"
        }
        nominations = []
        for elite in snapshot.elites:
            if elite.candidate_id in already_archived:
                continue
            descriptors = {
                "archive_cell": {
                    axis.name: value for axis, value in zip(self.config.axes, elite.cell)
                },
                "objective_metric": self.config.metric,
                "objective_score": elite.score,
                "evidence_stage": elite.stage,
                "evidence_zone": elite.zone,
                "search_policy_version": self.policy_version,
            }
            nominations.append(
                self.tools.archive_nominate(
                    elite.candidate_id,
                    descriptors=descriptors,
                    rationale=(
                        f"MAP-Elites incumbent for cell {elite.cell!r}; "
                        f"{self.config.metric}={elite.score:g} at {elite.stage}/{elite.zone}"
                    ),
                )
            )
        return tuple(nominations)

    def _observe(
        self,
        candidate: Mapping[str, Any],
        rows: list[Mapping[str, Any]],
    ) -> CandidateObservation:
        eligible = [
            row
            for row in rows
            if row.get("zone") in self.config.eligible_zones
            and row.get("stage") in _STAGE_RANK
            and _STAGE_RANK[str(row["stage"])] <= _STAGE_RANK[self.config.max_stage]
        ]
        failed = sum(row.get("gate") != "pass" for row in eligible)
        if not eligible:
            return CandidateObservation(str(candidate["id"]), candidate, None, None, None, 0, 0)

        strongest_rank = max(_STAGE_RANK[str(row["stage"])] for row in eligible)
        strongest = [row for row in eligible if _STAGE_RANK[str(row["stage"])] == strongest_rank]
        stage = str(strongest[0]["stage"])
        zone = str(strongest[0]["zone"])
        # A failed hard gate at the strongest attempted stage invalidates the
        # candidate even if an earlier public evaluation carried a good score.
        if any(row.get("gate") != "pass" for row in strongest):
            return CandidateObservation(
                str(candidate["id"]), candidate, None, stage, zone, len(strongest), failed
            )
        metric_names = (self.config.metric, *self.config.metric_aliases)
        values = []
        for row in strongest:
            for metric_name in metric_names:
                value = _metric_value(row, metric_name)
                if value is not None:
                    values.append(value)
                    break
        score = sum(values) / len(values) if values else None
        return CandidateObservation(
            str(candidate["id"]), candidate, score, stage, zone, len(strongest), failed
        )

    def _cell(self, candidate: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(axis.bucket(candidate) for axis in self.config.axes)

    def _elite_key(self, elite: Elite) -> tuple[float, int, tuple[int, str]]:
        quality = elite.score if self.config.minimize else -elite.score
        return (quality, -_STAGE_RANK[elite.stage], elite.tie_key)

    def _select_elite(self, snapshot: ArchiveSnapshot) -> tuple[Elite | None, float]:
        if not snapshot.elites:
            return None, 1.0
        quality = self._elite_percentiles(snapshot.elites)
        total_visits = sum(self._cell_visits.values())
        utilities: dict[tuple[str, ...], float] = {}
        for elite in snapshot.elites:
            visits = self._cell_visits.get(elite.cell, 0)
            bonus = self.config.cell_exploration * math.sqrt(
                math.log(total_visits + len(snapshot.elites) + 1.0) / (visits + 1.0)
            )
            utilities[elite.cell] = quality[elite.candidate_id] + bonus
        best = max(utilities.values())
        cells = sorted(cell for cell, utility in utilities.items() if math.isclose(utility, best))
        selected_cell = self._rng.choice(cells)
        self._cell_visits[selected_cell] = self._cell_visits.get(selected_cell, 0) + 1
        return (
            next(elite for elite in snapshot.elites if elite.cell == selected_cell),
            1.0 / len(cells),
        )

    def _elite_percentiles(self, elites: Iterable[Elite]) -> dict[str, float]:
        ordered = sorted(elites, key=self._elite_key)
        return _tie_aware_percentiles(
            ordered,
            identity=lambda elite: elite.candidate_id,
            quality=lambda elite: elite.score if self.config.minimize else -elite.score,
        )

    def _quality_percentiles(
        self, observations: Iterable[CandidateObservation]
    ) -> dict[str, float]:
        scored = [item for item in observations if item.score is not None]
        scored.sort(
            key=lambda item: (
                item.score if self.config.minimize else -item.score,
                _candidate_tie_key(item.record),
            )
        )
        return _tie_aware_percentiles(
            scored,
            identity=lambda item: item.candidate_id,
            quality=lambda item: float(item.score) if self.config.minimize else -float(item.score),
        )


def _tie_aware_percentiles(
    ordered: list[Any],
    *,
    identity: Any,
    quality: Any,
) -> dict[str, float]:
    """Assign equal average-rank reward to equal evaluator outcomes."""
    if not ordered:
        return {}
    if len(ordered) == 1:
        return {str(identity(ordered[0])): 1.0}
    result: dict[str, float] = {}
    denominator = len(ordered) - 1
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and math.isclose(
            float(quality(ordered[start])),
            float(quality(ordered[end])),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            end += 1
        average_rank = (start + end - 1) / 2.0
        reward = 1.0 - average_rank / denominator
        for item in ordered[start:end]:
            result[str(identity(item))] = reward
        start = end
    return result


def _candidate_tie_key(candidate: Mapping[str, Any]) -> tuple[int, str]:
    descriptors = candidate.get("descriptors", {})
    ordinal = descriptors.get("proposal_ordinal") if isinstance(descriptors, Mapping) else None
    stable_ordinal = int(ordinal) if isinstance(ordinal, int) and not isinstance(ordinal, bool) else 10**9
    digest = candidate.get("patch_digest") or candidate.get("build_digest") or candidate.get("id") or ""
    return stable_ordinal, str(digest)


def _metric_value(row: Mapping[str, Any], metric: str) -> float | None:
    for container_name in ("metric_vector", "paired_baselines"):
        container = row.get(container_name, {})
        value: Any = container
        for part in metric.split("."):
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        if _finite_number(value):
            return float(value)
    return None


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number_label(value: float) -> str:
    return f"{value:g}"


def _stable_label(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError):
        return repr(value)


__all__ = [
    "ArchiveConfig",
    "ArchiveSnapshot",
    "CandidateObservation",
    "DescriptorAxis",
    "Elite",
    "OperatorEstimate",
    "QualityDiversitySearch",
    "ResearchStep",
]
