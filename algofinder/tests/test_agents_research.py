from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from algofinder.agents.contracts import Candidate, Evaluation, Hypothesis
from algofinder.agents.evaluation import GateOutcome
from algofinder.agents.research import (
    AIDETreeResearchAgent,
    AnalysisDraft,
    AuthorityEvaluator,
    CandidateProposal,
    EvaluationBatch,
    LineageTreeConfig,
    LineageTreeSearch,
    PublicEvaluationPlan,
    QualityDiversityResearchAgent,
)
from algofinder.agents.workspace import CandidateWorkspace


def candidate(candidate_id: str, parents: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"id": candidate_id, "parent_ids": list(parents), "created_at": candidate_id}


def evaluation(
    candidate_id: str,
    evaluation_id: str,
    *,
    gate: str = "pass",
    gap: float | None = 10.0,
) -> dict[str, Any]:
    return {
        "id": evaluation_id,
        "candidate_id": candidate_id,
        "stage": "gate_1",
        "gate": gate,
        "metric_vector": {} if gap is None else {"mean_gap_percent": gap},
        "paired_baselines": {},
        "created_at": evaluation_id,
    }


class ArchiveTools:
    def __init__(self, candidates: list[dict[str, Any]], evaluations: list[dict[str, Any]]) -> None:
        self.candidates = candidates
        self.evaluations = evaluations

    def archive_query(self, kind: str, **filters: Any) -> list[dict[str, Any]]:
        records = self.candidates if kind == "candidate" else self.evaluations
        return [item for item in records if all(item.get(key) == value for key, value in filters.items())]

    def archive_lineage(self, candidate_id: str) -> list[dict[str, Any]]:
        by_id = {item["id"]: item for item in self.candidates}
        result: list[dict[str, Any]] = []

        def walk(node_id: str) -> None:
            for parent_id in by_id[node_id].get("parent_ids", []):
                walk(parent_id)
            result.append(by_id[node_id])

        walk(candidate_id)
        return result

    def runs_compare(self, candidate_ids: Any) -> dict[str, Any]:
        wanted = set(candidate_ids)
        return {
            "candidates": sorted(wanted),
            "evaluations": [
                item for item in self.evaluations if item.get("candidate_id") in wanted
            ],
        }

    def campaign_snapshot(self) -> dict[str, Any]:
        return {
            "campaign": {
                "id": "campaign-one",
                "objective": {"primary": "paired_gap_percent", "direction": "minimize"},
            },
            "campaign_state": "active",
            "budget": {"remaining": {"evaluation_seconds": 100.0}},
        }

    def catalog_problems(self) -> list[dict[str, Any]]:
        return [{"id": "tsp"}]

    def catalog_solvers(self) -> list[dict[str, Any]]:
        return [{"id": "distance-ranked-2opt"}]


def test_tree_search_invents_when_archive_is_empty() -> None:
    policy = LineageTreeSearch(ArchiveTools([], []), LineageTreeConfig(seed=7))  # type: ignore[arg-type]
    directive = policy.select()
    assert directive.operator == "invent"
    assert directive.parent_ids == ()


def test_tree_search_balances_quality_and_branch_exploration() -> None:
    candidates = [
        candidate("candidate-best"),
        candidate("candidate-best-child-1", ("candidate-best",)),
        candidate("candidate-best-child-2", ("candidate-best",)),
        candidate("candidate-open"),
    ]
    evaluations = [
        evaluation("candidate-best", "evaluation-best", gap=1.0),
        evaluation("candidate-best-child-1", "evaluation-child-1", gap=4.0),
        evaluation("candidate-best-child-2", "evaluation-child-2", gap=4.0),
        evaluation("candidate-open", "evaluation-open", gap=2.0),
    ]
    policy = LineageTreeSearch(
        ArchiveTools(candidates, evaluations),  # type: ignore[arg-type]
        LineageTreeConfig(exploration_weight=1.0, depth_penalty=0.0, seed=2),
    )
    directive = policy.select()
    assert directive.parent_ids == ("candidate-open",)
    assert directive.operator == "mutate"


def test_tree_search_repairs_the_only_failed_branch() -> None:
    tools = ArchiveTools(
        [candidate("candidate-failed")],
        [evaluation("candidate-failed", "evaluation-failed", gate="fail", gap=None)],
    )
    directive = LineageTreeSearch(tools).select()  # type: ignore[arg-type]
    assert directive.operator == "repair"
    assert directive.parent_ids == ("candidate-failed",)


class FakeResearchModel:
    def propose(self, context: Any, directive: Any) -> CandidateProposal:
        return CandidateProposal(
            mechanism="restrict 2-opt moves to a geometric candidate set",
            affected_strata=("tsp:euclidean",),
            predicted_tradeoffs="less work, with risk on sparse neighborhoods",
            falsification_test="reject when verified public gap worsens",
            patch="diff --git a/algofinder/solvers/tsp/new.py b/algofinder/solvers/tsp/new.py\n",
            solver_entrypoints=("algofinder.solvers.tsp.new:CandidateSolver",),
            expected_result="lower paired gap without correctness failures",
            descriptors={"family": "local-search"},
        )

    def analyze(self, context: Any, directive: Any, proposal: Any, feedback: Any) -> AnalysisDraft:
        return AnalysisDraft(
            observation="both authority-owned public gates passed",
            interpretation="the mechanism remains viable on the smoke distribution",
            confidence=0.6,
            next_test="run a larger paired rotating-train race",
        )


class FakeAgentTools(ArchiveTools):
    def __init__(self) -> None:
        super().__init__([], [])
        self.calls: list[str] = []
        self.workspace = CandidateWorkspace(
            "candidate-one", Path("/tmp/candidate-one"), "753d48d", (), "hypothesis-one", "invent"
        )

    def hypothesis_create(self, **kwargs: Any) -> Hypothesis:
        self.calls.append("hypothesis_create")
        return Hypothesis(
            "hypothesis-one", "campaign-one", kwargs["mechanism"],
            tuple(kwargs["affected_strata"]), kwargs["predicted_tradeoffs"], kwargs["falsification_test"],
        )

    def candidate_create(self, **kwargs: Any) -> CandidateWorkspace:
        self.calls.append("candidate_create")
        return self.workspace

    def candidate_apply_patch(self, workspace: CandidateWorkspace, patch: str) -> None:
        self.calls.append("candidate_apply_patch")

    def candidate_validate(
        self, workspace: CandidateWorkspace, *, solver_entrypoints: tuple[str, ...] = ()
    ) -> dict[str, Any]:
        self.calls.append("candidate_validate")
        return {"passed": True, "changed_paths": ["algofinder/solvers/tsp/new.py"]}

    def candidate_freeze(self, workspace: CandidateWorkspace, **kwargs: Any) -> Candidate:
        self.calls.append("candidate_freeze")
        return Candidate(
            "candidate-one", "campaign-one", "hypothesis-one", (), "invent",
            tuple(kwargs["solver_entrypoints"]), descriptors=dict(kwargs["descriptors"]),
        )

    def analysis_record(self, **kwargs: Any) -> None:
        self.calls.append("analysis_record")

    def archive_nominate(self, candidate_id: str, **kwargs: Any) -> None:
        self.calls.append("archive_nominate")


class FakeEvaluator:
    def evaluate(self, **kwargs: Any) -> EvaluationBatch:
        record = Evaluation(
            "evaluation-one", "campaign-one", "experiment-one", "candidate-one",
            "gate_1", "public", "pass", metric_vector={"mean_gap_percent": 1.25},
        )
        return EvaluationBatch((GateOutcome(record, {"gate": "pass"}),))


def test_research_agent_materializes_through_agent_tools() -> None:
    tools = FakeAgentTools()
    agent = AIDETreeResearchAgent(
        tools=tools, model=FakeResearchModel(), evaluator=FakeEvaluator()  # type: ignore[arg-type]
    )
    result = agent.run_step()
    assert result.status == "archived"
    assert result.candidate_id == "candidate-one"
    assert tools.calls == [
        "hypothesis_create", "candidate_create", "candidate_apply_patch", "candidate_validate",
        "candidate_freeze", "analysis_record", "archive_nominate",
    ]


def test_quality_diversity_agent_uses_the_same_guarded_execution_loop() -> None:
    tools = FakeAgentTools()
    agent = QualityDiversityResearchAgent(
        tools=tools, model=FakeResearchModel(), evaluator=FakeEvaluator()  # type: ignore[arg-type]
    )

    result = agent.run_step()

    assert result.status == "archived"
    assert result.directive.operator == "invent"
    assert "qd-map-elites-ucb@3" in result.directive.rationale


def gate_outcome(evaluation_id: str, stage: str) -> GateOutcome:
    record = Evaluation(
        evaluation_id, "campaign-one", f"experiment-{stage.replace('_', '-')}", "candidate-one",
        stage, "public", "pass", metric_vector={"mean_gap_percent": 2.0},  # type: ignore[arg-type]
    )
    return GateOutcome(record, {"gate": "pass"})


class FakeAuthority:
    def __init__(self) -> None:
        self.campaign = SimpleNamespace(campaign_id="campaign-one")
        self.workspaces = SimpleNamespace(source_root=Path("/repo"))
        self.calls: list[str] = []

    def gate_0(self, workspace: CandidateWorkspace, experiment: Any) -> GateOutcome:
        self.calls.append("gate_0")
        return gate_outcome("evaluation-gate-zero", "gate_0")

    def gate_1_public(self, **kwargs: Any) -> GateOutcome:
        self.calls.append("gate_1_public")
        assert kwargs["manifest_path"] == "/repo/algofinder/data/instances/tsp_manifests.json"
        return gate_outcome("evaluation-gate-one", "gate_1")


class FakePublicAdapter:
    def reserved_evaluation_seconds(self, **kwargs: Any) -> float:
        return 5.0


class AuthorityTools:
    def __init__(self) -> None:
        self.campaign = SimpleNamespace(campaign_id="campaign-one")
        self.experiments: list[dict[str, Any]] = []

    def campaign_snapshot(self) -> dict[str, Any]:
        return {"campaign": {"suites": [
            {"name": "tsp-public-build", "stage": "gate_0", "zone": "public"},
            {"name": "tsp-public-smoke", "stage": "gate_1", "zone": "public",
             "manifest_path": "algofinder/data/instances/tsp_manifests.json"},
        ]}}

    def experiment_propose(self, **kwargs: Any) -> Any:
        self.experiments.append(kwargs)
        return SimpleNamespace(stage=kwargs["stage"])


def test_authority_evaluator_uses_authority_for_both_public_gates() -> None:
    authority = FakeAuthority()
    tools = AuthorityTools()
    evaluator = AuthorityEvaluator(
        authority=authority, public_adapter=FakePublicAdapter(),  # type: ignore[arg-type]
        plan=PublicEvaluationPlan(gate_1_budget_seconds=5.0, gate_1_timeout_seconds=10.0),
    )
    proposal = FakeResearchModel().propose(None, None)
    candidate_record = Candidate(
        "candidate-one", "campaign-one", "hypothesis-one", (), "invent",
        ("algofinder.solvers.tsp.new:CandidateSolver",),
    )
    workspace = CandidateWorkspace(
        "candidate-one", Path("/tmp/candidate-one"), "753d48d", (), "hypothesis-one", "invent"
    )
    batch = evaluator.evaluate(
        tools=tools, workspace=workspace, candidate=candidate_record, proposal=proposal  # type: ignore[arg-type]
    )
    assert batch.passed
    assert authority.calls == ["gate_0", "gate_1_public"]
    assert [item["stage"] for item in tools.experiments] == ["gate_0", "gate_1"]
