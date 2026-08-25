from __future__ import annotations

from typing import Any

from algofinder.agents.quality_diversity import (
    ArchiveConfig,
    DescriptorAxis,
    QualityDiversitySearch,
)


class FakeAgentTools:
    """Small fake exposing only the AgentTools methods used by the policy."""

    def __init__(self, candidates: list[dict[str, Any]], evaluations: list[dict[str, Any]]) -> None:
        self.candidates = candidates
        self.evaluations = evaluations
        self.decisions: list[dict[str, Any]] = []
        self.calls: list[tuple[str, Any]] = []

    def archive_query(self, kind: str, **filters: Any) -> list[dict[str, Any]]:
        self.calls.append(("archive_query", kind))
        records = {
            "candidate": self.candidates,
            "evaluation": self.evaluations,
            "decision": self.decisions,
        }[kind]
        return [
            item for item in records
            if all(item.get(key) == value for key, value in filters.items())
        ]

    def runs_compare(self, candidate_ids: list[str]) -> dict[str, Any]:
        wanted = set(candidate_ids)
        self.calls.append(("runs_compare", tuple(sorted(wanted))))
        return {
            "candidates": sorted(wanted),
            "evaluations": [
                item for item in self.evaluations if item.get("candidate_id") in wanted
            ],
        }

    def archive_nominate(
        self, candidate_id: str, *, descriptors: dict[str, Any], rationale: str
    ) -> dict[str, Any]:
        self.calls.append(("archive_nominate", candidate_id))
        decision = {
            "id": f"decision-{candidate_id}",
            "decision_kind": "archive",
            "target_id": candidate_id,
            "descriptors": descriptors,
            "rationale": rationale,
        }
        self.decisions.append(decision)
        return decision


def candidate(
    candidate_id: str,
    family: str,
    runtime: str,
    operator: str = "mutate",
    parents: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "generation_operator": operator,
        "parent_ids": list(parents),
        "descriptors": {"algorithm_family": family, "runtime_class": runtime},
    }


def evaluation(
    candidate_id: str,
    score: Any,
    *,
    stage: str = "gate_1",
    zone: str = "public",
    gate: str = "pass",
    paired: bool = False,
) -> dict[str, Any]:
    metrics = {} if paired else {"paired_gap_percent": score}
    baselines = {"paired_gap_percent": score} if paired else {}
    return {
        "candidate_id": candidate_id,
        "stage": stage,
        "zone": zone,
        "gate": gate,
        "metric_vector": metrics,
        "paired_baselines": baselines,
    }


def populated_tools() -> FakeAgentTools:
    candidates = [
        candidate("candidate-a", "local-search", "fast", "invent"),
        candidate("candidate-b", "local-search", "fast", "mutate", ("candidate-a",)),
        candidate("candidate-c", "population", "slow", "tune", ("candidate-b",)),
        candidate("candidate-d", "population", "slow", "repair", ("candidate-c",)),
        candidate("candidate-e", "constructive", "fast", "distill", ("candidate-b",)),
        candidate("candidate-f", "oracle-like", "slow", "recombine", ("candidate-b", "candidate-c")),
    ]
    evaluations = [
        evaluation("candidate-a", 10.0),
        evaluation("candidate-b", 8.0),
        evaluation("candidate-b", 10.0),
        evaluation("candidate-c", 20.0, stage="gate_4", zone="validation", paired=True),
        evaluation("candidate-d", 1.0),
        evaluation("candidate-d", 0.1, stage="gate_4", zone="validation", gate="fail"),
        evaluation("candidate-e", float("nan")),
        evaluation("candidate-f", 0.0, stage="gate_6", zone="challenge"),
    ]
    return FakeAgentTools(candidates, evaluations)


def test_refresh_keeps_one_elite_per_cell_and_respects_authority_gates() -> None:
    tools = populated_tools()
    search = QualityDiversitySearch(tools, seed=17)

    snapshot = search.refresh()

    assert snapshot.candidates_seen == 6
    assert snapshot.candidates_scored == 3
    assert snapshot.evaluations_seen == 8
    assert snapshot.failed_evaluations == 1
    assert [(elite.cell, elite.candidate_id, elite.score) for elite in snapshot.elites] == [
        (("local-search", "fast"), "candidate-b", 9.0),
        (("population", "slow"), "candidate-c", 20.0),
    ]
    # The failed stronger gate, missing metric, and sealed challenge row cannot
    # become archive fitness even though each has an apparently attractive value.
    assert {elite.candidate_id for elite in snapshot.elites}.isdisjoint(
        {"candidate-d", "candidate-e", "candidate-f"}
    )
    assert [name for name, _ in tools.calls] == ["archive_query", "runs_compare"]


def test_recommendations_are_seeded_and_expose_operator_exploration() -> None:
    first = QualityDiversitySearch(populated_tools(), seed=75348)
    second = QualityDiversitySearch(populated_tools(), seed=75348)

    first_snapshot = first.refresh()
    second_snapshot = second.refresh()
    first_steps = [first.recommend(first_snapshot) for _ in range(5)]
    second_steps = [second.recommend(second_snapshot) for _ in range(5)]

    assert first_steps == second_steps
    assert all(step.policy_version == "qd-map-elites-ucb@3" for step in first_steps)
    candidate_ids = {item.candidate_id for item in first_snapshot.observations}
    assert all(set(step.parent_ids) <= candidate_ids for step in first_steps)
    estimates = {item.operator: item for item in first_steps[0].operator_estimates}
    assert estimates["repair"].mean_reward < estimates["mutate"].mean_reward
    assert estimates["repair"].exploration_bonus > 0


def test_nomination_uses_agent_tools_and_is_idempotent() -> None:
    tools = populated_tools()
    search = QualityDiversitySearch(tools, seed=2)
    snapshot = search.refresh()

    first = search.nominate_elites(snapshot)
    second = search.nominate_elites(snapshot)

    assert [item["target_id"] for item in first] == ["candidate-b", "candidate-c"]
    assert second == ()
    assert all(
        item["descriptors"]["search_policy_version"] == "qd-map-elites-ucb@3"
        for item in first
    )
    assert [call for call in tools.calls if call[0] == "archive_nominate"] == [
        ("archive_nominate", "candidate-b"),
        ("archive_nominate", "candidate-c"),
    ]


def test_descriptor_axis_bins_numbers_and_stably_labels_categories() -> None:
    size = DescriptorAxis("size", boundaries=(10.0, 100.0))
    strategy = DescriptorAxis("strategy")

    assert size.bucket({"descriptors": {"size": 10}}) == "(-inf,10]"
    assert size.bucket({"descriptors": {"size": 11}}) == "(10,100]"
    assert size.bucket({"descriptors": {"size": 101}}) == "(100,+inf)"
    assert strategy.bucket({"descriptors": {"strategy": {"b": 2, "a": 1}}}) == '{"a":1,"b":2}'


def test_empty_archive_falls_back_to_invention_without_scoring() -> None:
    tools = FakeAgentTools(
        [candidate("candidate-empty", "unknown", "unknown")],
        [evaluation("candidate-empty", None)],
    )
    search = QualityDiversitySearch(
        tools,
        config=ArchiveConfig(operators=("invent", "mutate")),
        seed=1,
    )

    step = search.recommend()

    assert step.operator == "invent"
    assert step.parent_ids == ()
    assert step.target_cell is None


def test_public_metric_alias_uses_recorded_value_without_deriving_a_score() -> None:
    tools = FakeAgentTools(
        [candidate("candidate-alias", "local-search", "fast", "invent")],
        [{
            "candidate_id": "candidate-alias",
            "stage": "gate_1",
            "zone": "public",
            "gate": "pass",
            "metric_vector": {"mean_gap_percent": 3.25},
            "paired_baselines": {},
        }],
    )

    snapshot = QualityDiversitySearch(tools).refresh()

    assert len(snapshot.elites) == 1
    assert snapshot.elites[0].score == 3.25


def test_failed_candidate_can_be_selected_for_repair_without_an_elite() -> None:
    tools = FakeAgentTools(
        [candidate("candidate-broken", "local-search", "fast", "repair")],
        [evaluation("candidate-broken", None, gate="fail")],
    )
    search = QualityDiversitySearch(tools, seed=3)

    step = search.recommend()

    assert step.operator == "repair"
    assert step.parent_ids == ("candidate-broken",)
    assert step.target_cell == ("local-search", "fast")


def test_validation_failure_prevents_impossible_operator_from_remaining_untried() -> None:
    tools = FakeAgentTools(
        [candidate("candidate-seed", "local-search", "fast", "invent")],
        [evaluation("candidate-seed", 0.0)],
    )
    search = QualityDiversitySearch(
        tools,
        config=ArchiveConfig(operators=("invent", "mutate")),
        seed=7,
    )
    snapshot = search.refresh()

    assert search.recommend(snapshot).operator == "mutate"
    search.record_validation_failure("mutate")
    next_step = search.recommend(snapshot)

    assert next_step.operator == "invent"
    estimates = {item.operator: item for item in next_step.operator_estimates}
    assert estimates["mutate"].trials == 1
    assert estimates["mutate"].mean_reward == -1.0
