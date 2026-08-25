from __future__ import annotations

from pathlib import Path

import pytest

from algofinder.agents.campaign import Campaign
from algofinder.agents.contracts import AgentSpec, BudgetLimits, CampaignSpec, SuiteBinding
from algofinder.agents.learning.contracts import DistributionProfile
from algofinder.agents.learning.dataset import DatasetError, build_dataset
from algofinder.agents.learning.features import FEATURE_SCHEMA_DIGEST, project_attempt
from algofinder.agents.learning.normalization import NormalizationError, normalize_formula
from algofinder.agents.learning.outer_evaluation import finalize_agent_evaluation, prepare_agent_evaluation
from algofinder.agents.learning.policy import load_learned_policy
from algofinder.agents.learning.recorder import EpisodeRecorder, RecorderError
from algofinder.agents.learning.store import LearningArtifactStore, LearningStoreError
from algofinder.agents.learning.trainer import train_policy_from_rows


def _campaign(tmp_path: Path) -> Campaign:
    agent = AgentSpec("agent-test", "fake", "fake:1", "0" * 64, "tools@1", "test")
    spec = CampaignSpec(
        campaign_id="campaign-test",
        base_commit="0" * 40,
        allowed_paths=("algofinder/solvers/**",),
        suites=(SuiteBinding("suite-test", "gate_0", "public", "1" * 64),),
        budgets=BudgetLimits(),
        agent_spec_ids=(agent.agent_spec_id,),
        search_policy_version="test@1",
        objective={"primary": "gap"},
        goal="test campaign",
    )
    return Campaign.create(tmp_path, spec, (agent,))


def test_native_recorder_keeps_rejected_attempt_after_workspace_cleanup(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    recorder = EpisodeRecorder(campaign.ledger)
    episode = recorder.start(
        agent_spec_id="agent-test",
        campaign_snapshot=campaign.snapshot(agent_visible=True),
        model_identity={"model": "fake"},
    )
    attempt = recorder.begin_attempt(
        attempt_number=1, operator="invent", parent_ids=(), state_before={"visible": "before"},
    )
    transition = recorder.record_transition(
        attempt,
        context={"visible": "before"},
        action={"operator": "invent"},
        behavior_policy={"selection_probability": 1.0},
        validation={"passed": False, "errors": ["duplicate"]},
        terminal_status="validation_rejected",
        rejection_code="candidate_validation_rejected",
    )
    recorder.finish_attempt(attempt, terminal_status="validation_rejected", state_after={"visible": "after"})

    attempts = campaign.ledger.list("research_attempt")
    assert attempts[0]["transition_ids"] == [transition.transition_id]
    assert campaign.ledger.list("agent_episode")[0]["id"] == episode.episode_id
    assert campaign.ledger.read_blob(transition.context_blob_digest)


def test_recorder_rejects_secret_and_challenge_path_blobs(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    recorder = EpisodeRecorder(campaign.ledger)
    recorder.start(
        agent_spec_id="agent-test",
        campaign_snapshot=campaign.snapshot(agent_visible=True),
        model_identity={"model": "fake"},
    )
    attempt = recorder.begin_attempt(
        attempt_number=1, operator="invent", parent_ids=(), state_before={"visible": "before"},
    )
    for context in ({"text": "Authorization: Bearer sk-supersecret-token"}, {"path": "/challenge/hidden.json"}):
        try:
            recorder.record_transition(
                attempt, context=context, action={"operator": "invent"},
                behavior_policy={"selection_probability": 1.0}, validation={}, terminal_status="rejected",
            )
        except RecorderError:
            pass
        else:  # pragma: no cover - this is the assertion's failure branch.
            raise AssertionError("unsafe blob was accepted")


def test_formula_normalization_separates_exact_and_structural_identity() -> None:
    first = normalize_formula("distance + 0.05 * mean_distance * normalized_rank")
    second = normalize_formula("distance + 0.10 * mean_distance * normalized_rank")
    angular = normalize_formula("distance + 0.10 * mean_distance * angular_offset")
    assert first.exact_digest != second.exact_digest
    assert first.structural_digest == second.structural_digest
    assert first.probe_behavior_digest == second.probe_behavior_digest
    assert first.probe_behavior_digest != angular.probe_behavior_digest


def test_learning_store_is_idempotent_but_refuses_id_collision(tmp_path: Path) -> None:
    store = LearningArtifactStore(tmp_path / "learning").create()
    profile = DistributionProfile(
        profile_id="profile-test", family="tsp", version="v1", parameter_schema={"x": 1},
        seed_set=(1,), split_rule="campaign",
    )
    first = store.write(profile)
    assert store.write(profile) == first
    conflict = DistributionProfile(
        profile_id="profile-test", family="tsp", version="v2", parameter_schema={"x": 2},
        seed_set=(1,), split_rule="campaign",
    )
    try:
        store.write(conflict)
    except LearningStoreError:
        pass
    else:  # pragma: no cover
        raise AssertionError("immutable learning artifact id collision was accepted")


def test_historical_projection_restores_success_model_and_formula() -> None:
    row = project_attempt({
        "campaign_id": "campaign-one",
        "historical_attempt_id": "attempt-one",
        "outer_attempt_number": 1,
        "operator": "invent",
        "terminal_status": "archived",
        "candidate_id": "candidate-one",
        "model": "local:9b",
        "representation": "formula",
        "evidence_role": "historical_training",
        "policy_decision": {"template_values": {"priority_expression": "distance + normalized_rank"}},
        "reward": {"all_requested_gates_passed": True, "public_metric_vector": {"paired_gap_percent": -0.2}},
    })
    assert row["passed"] is True
    assert row["quality_improved"] is True
    assert row["model"] == "local:9b"
    assert row["priority_expression"] == "distance + normalized_rank"
    assert row["attempt_id"] == "attempt-one"


def test_policy_training_excludes_dev_test_and_is_loadable(tmp_path: Path) -> None:
    store = LearningArtifactStore(tmp_path / "learning").create()
    base = {
        "representation": "formula", "model": "local:9b", "operator": "invent",
        "parent_count": 0, "transition_count": 1, "attempt_number": 1,
        "valid": True, "passed": True, "evidence_role": "historical_training",
        "backfill_confidence": 1.0, "terminal_status": "archived",
    }
    rows = [
        {**base, "campaign_id": "campaign-train", "attempt_id": "attempt-train", "split": "train", "priority_expression": "distance"},
        {**base, "campaign_id": "campaign-dev", "attempt_id": "attempt-dev", "split": "dev", "priority_expression": "normalized_rank"},
        {**base, "campaign_id": "campaign-test", "attempt_id": "attempt-test", "split": "test", "priority_expression": "angular_offset"},
        {**base, "campaign_id": "campaign-sealed", "attempt_id": "attempt-sealed", "split": "train", "evidence_role": "sealed"},
    ]
    result = train_policy_from_rows(
        store=store, rows=rows, dataset_digest="a" * 64,
        representation="formula", feature_schema_digest=FEATURE_SCHEMA_DIGEST,
        policy_id="policy-learning-test", seed=7,
    )

    class Baseline:
        pass

    policy = load_learned_policy(
        store=store, policy_id=result["policy_id"], baseline=Baseline(),  # type: ignore[arg-type]
        model="local:9b", representation="formula",
    )
    assert [item.case_id for item in policy.memory.documents] == ["attempt-train"]
    assert policy.advisor.posterior.posterior(("local:9b|formula|invent",))[2] == 1
    assert policy.advisor.quality_posterior.posterior(("local:9b|formula|invent",))[2] == 1


def test_outer_validation_is_achievable_and_rejects_arm_drift(tmp_path: Path) -> None:
    store = LearningArtifactStore(tmp_path / "learning").create()
    rows = [{
        "campaign_id": "campaign-train", "attempt_id": "attempt-train", "split": "train",
        "representation": "formula", "model": "local:9b", "operator": "invent",
        "parent_count": 0, "transition_count": 1, "attempt_number": 1,
        "valid": True, "passed": True, "evidence_role": "historical_training",
        "backfill_confidence": 1.0, "terminal_status": "archived",
    }]
    trained = train_policy_from_rows(
        store=store, rows=rows, dataset_digest="b" * 64,
        representation="formula", feature_schema_digest=FEATURE_SCHEMA_DIGEST,
        policy_id="policy-outer-test", seed=9,
    )
    profile_ids = ("profile-one", "profile-two", "profile-three", "profile-four")
    for profile_id in profile_ids:
        store.write(DistributionProfile(
            profile_id=profile_id, family="tsp", version="v1",
            parameter_schema={"visibility": "outer_validation"},
            seed_set=(1,), split_rule="campaign",
        ))
    plan = prepare_agent_evaluation(
        store=store, learned_policy_id=trained["policy_id"],
        profile_ids=profile_ids,
        models=("model:7b", "model:9b"), phase="outer-validation",
        evaluation_id="agent-evaluation-plan",
    )
    outcomes = []
    paired_seeds: dict[tuple[str, str], set[int]] = {}
    for arm in plan["arms"]:
        paired_seeds.setdefault((arm["task_profile_id"], arm["model"]), set()).add(arm["seed"])
        learned = arm["policy_id"] == trained["policy_id"]
        outcomes.append({
            "campaign_key": arm["campaign_key"],
            "task_profile_id": arm["task_profile_id"],
            "model": arm["model"],
            "policy_id": arm["policy_id"],
            "integrity_passed": True,
            "instance_rows_as_replicates": False,
            "accepted_normalized_novel": 1 if learned else 0,
            "duplicate_rate_percent": 40.0 if learned else 50.0,
            "champion_holdout_gap_percent": 1.05 if learned else 1.0,
            "tokens": 110 if learned else 100,
            "wall_seconds": 110.0 if learned else 100.0,
            "contract_failure_rate": 0.0,
        })
    assert all(len(values) == 1 for values in paired_seeds.values())
    result = finalize_agent_evaluation(
        store=store, planned_evaluation_id=plan["evaluation_id"], campaign_summaries=outcomes,
    )
    assert set(result["decision"]["by_model"].values()) == {"experimental"}
    drifted = [dict(item) for item in outcomes]
    drifted[0]["model"] = "wrong:model"
    with pytest.raises(ValueError, match="changed frozen model"):
        finalize_agent_evaluation(
            store=store, planned_evaluation_id=plan["evaluation_id"], campaign_summaries=drifted,
        )


def test_recorder_uses_episode_wide_attempt_ordinals(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    recorder = EpisodeRecorder(campaign.ledger)
    recorder.start(
        agent_spec_id="agent-test", campaign_snapshot=campaign.snapshot(agent_visible=True),
        model_identity={"model": "fake"},
    )
    first = recorder.begin_attempt(operator="invent", parent_ids=(), state_before={"step": 0})
    recorder.record_transition(
        first, context={"step": 0}, action={"operator": "invent"}, behavior_policy={},
        validation={}, terminal_status="rejected",
    )
    recorder.finish_attempt(first, terminal_status="rejected", state_after={"step": 1})
    second = recorder.begin_attempt(operator="invent", parent_ids=(), state_before={"step": 1})
    assert (first.attempt_number, second.attempt_number) == (1, 2)


def test_formula_probe_errors_are_contract_errors() -> None:
    with pytest.raises(NormalizationError, match="deterministic probe"):
        normalize_formula("1 / (distance - distance)")


def test_dataset_refuses_sealed_evidence(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="sealed evidence"):
        build_dataset(
            store=LearningArtifactStore(tmp_path / "learning").create(),
            dataset_id="dataset-leak-test",
            rows=[{
                "campaign_id": "campaign-sealed", "attempt_id": "attempt-sealed",
                "split_group_id": "group-sealed", "evidence_role": "sealed",
            }],
        )
