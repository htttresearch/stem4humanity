"""Run a small, real local-model solver-research campaign.

This is an executable assay, not a unit-test fixture.  It generates source
patches with a local model, freezes them in Git worktrees, evaluates them only
through EvaluationAuthority, and leaves an append-only campaign ledger behind.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter

from algofinder.agents.benchmark_adapter import PublicBenchmarkAdapter
from algofinder.agents.campaign import Campaign
from algofinder.agents.contracts import AgentSpec, BudgetLimits, CampaignSpec, Candidate, SuiteBinding
from algofinder.agents.evaluation import EvaluationAuthority
from algofinder.agents.ollama_model import (
    OllamaModelConfig,
    OllamaResearchModel,
    ollama_model_identity,
)
from algofinder.agents.research import (
    AuthorityEvaluator,
    DEFAULT_METHOD_PROMPT_DIGEST,
    PublicEvaluationPlan,
    QualityDiversityResearchAgent,
    TEMPLATE_METHOD_PROMPT_DIGEST,
)
from algofinder.agents.hir_evolution import EvolutionaryHIRModel
from algofinder.agents.learning.recorder import EpisodeRecorder
from algofinder.agents.learning.policy import load_learned_policy
from algofinder.agents.learning.store import LearningArtifactStore
from algofinder.agents.quality_diversity import ArchiveConfig, DescriptorAxis, QualityDiversitySearch
from algofinder.agents.research import QualityDiversityPolicy
from algofinder.agents.tools import AgentTools
from algofinder.agents.workspace import PatchPolicy, WorkspaceService


SOURCE_PATHS: tuple[str, ...] = ()
REPRESENTATIONS = {
    "formula": "tsp-edge-formula-v1",
    "hir": "tsp-hir-v1",
    "full-code": None,
}
FULL_CODE_SOURCE_PATHS = (
    "algofinder/algofinder/solvers/base.py",
    "algofinder/algofinder/solvers/tsp/euclidean_geometry.py",
    "algofinder/algofinder/solvers/tsp/euclidean_local_search.py",
    "algofinder/algofinder/solvers/tsp/distance_ranked.py",
)

FORBIDDEN_SOLVER_IMPORTS = (
    "algofinder.solvers.registry",
    "algofinder.solvers.tsp.candidates",
    "algofinder.solvers.tsp.constructors",
    "algofinder.solvers.tsp.general_two_opt",
    "algofinder.solvers.tsp.held_karp",
    "algofinder.solvers.tsp.incremental_exact",
    "algofinder.solvers.tsp.learned_candidate",
    "algofinder.solvers.tsp.move_engine",
    "algofinder.solvers.tsp.strong_ils",
)


def _archive_config(representation: str) -> ArchiveConfig:
    if representation == "formula":
        axes = (
            DescriptorAxis("rank_monotonicity"),
            DescriptorAxis("distance_monotonicity"),
            DescriptorAxis("uses_angular_offset"),
        )
    elif representation == "hir":
        axes = (DescriptorAxis("hir_niche"),)
    else:
        axes = (
            DescriptorAxis("algorithm_family"),
            DescriptorAxis("runtime_class"),
            DescriptorAxis("generation_operator"),
        )
    return ArchiveConfig(axes=axes)


def run_pilot(
    *,
    repo_root: Path,
    campaigns_root: Path,
    manifest_path: Path,
    model_name: str,
    endpoint: str,
    target_candidates: int,
    per_cell_budget_seconds: float,
    representation: str = "formula",
    backend: str = "model",
    model_seed: int = 75348,
    holdout_splits: tuple[str, ...] = (),
    holdout_manifest_path: Path | None = None,
    defer_holdout: bool = False,
    learning_root: Path | None = None,
    learned_policy_id: str | None = None,
    distribution_profile_id: str | None = None,
    evidence_role: str = "public_learning",
    evidence_partition_id: str | None = None,
) -> dict[str, object]:
    campaign_started = perf_counter()
    repo_root = repo_root.resolve()
    manifest_path = manifest_path.resolve()
    if holdout_splits:
        holdout_manifest_path = (holdout_manifest_path or manifest_path).resolve()
    elif holdout_manifest_path is not None:
        raise ValueError("holdout_manifest_path requires at least one holdout split")
    if defer_holdout and not holdout_splits:
        raise ValueError("defer_holdout requires at least one holdout split")
    if learned_policy_id is not None and learning_root is None:
        raise ValueError("learned_policy_id requires learning_root")
    if distribution_profile_id is not None and learning_root is None:
        raise ValueError("distribution_profile_id requires learning_root")
    if evidence_role not in {"public_learning", "outer_validation", "sealed"}:
        raise ValueError("invalid pilot evidence_role")
    campaigns_root.mkdir(parents=True, exist_ok=True)
    try:
        candidate_template = REPRESENTATIONS[representation]
    except KeyError as exc:
        raise ValueError(f"unknown pilot representation {representation!r}") from exc
    if backend not in {"model", "evolution"}:
        raise ValueError(f"unknown pilot backend {backend!r}")
    if backend == "evolution" and representation != "hir":
        raise ValueError("the evolution backend requires representation='hir'")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    campaign_id = f"tsp-{representation}-pilot-{stamp}"
    agent_id = f"{representation}-research-{stamp}"
    digest = sha256(manifest_path.read_bytes()).hexdigest()
    holdout_digest = (
        sha256(holdout_manifest_path.read_bytes()).hexdigest()
        if holdout_manifest_path is not None else None
    )
    base_commit = _git_head(repo_root)
    if base_commit != "753d48d93fbbd773f75cefb602b4a046dc715585":
        raise RuntimeError(f"pilot requires commit 753d48d, found {base_commit}")
    source_provenance = _experiment_source_provenance(repo_root)
    model_identity = (
        {"backend": "evolution", "implementation": "typed-hir-evolution-v2"}
        if backend == "evolution"
        else ollama_model_identity(endpoint=endpoint, model=model_name)
    )
    recorded_model_identity = {
        **model_identity,
        "model": "typed-hir-evolution-v1" if backend == "evolution" else model_name,
        "representation": representation,
        "backend": backend,
    }
    learning_store = LearningArtifactStore(learning_root).create() if learning_root is not None else None
    learned_policy_record = (
        learning_store.read(f"policies/{learned_policy_id}/policy.json")
        if learning_store is not None and learned_policy_id is not None else None
    )
    if learned_policy_record is not None and (
        learned_policy_record.get("kind") != "agent_policy"
        or learned_policy_record.get("policy_kind") != "learned"
    ):
        raise ValueError("learned_policy_id does not identify a frozen learned policy")
    if learned_policy_record is not None and learned_policy_record.get("representation") != representation:
        raise ValueError("learned policy representation does not match the pilot representation")
    learned_policy_digest = (
        str(learned_policy_record["content_digest"]) if learned_policy_record is not None else None
    )
    distribution_profile_record = (
        learning_store.read(f"distributions/{distribution_profile_id}.json")
        if learning_store is not None and distribution_profile_id is not None else None
    )
    distribution_profile_digest = (
        str(distribution_profile_record["content_digest"])
        if distribution_profile_record is not None else None
    )
    if distribution_profile_record is not None:
        profile_parameters = distribution_profile_record.get("parameter_schema", {})
        if (
            distribution_profile_record.get("kind") != "distribution_profile"
            or distribution_profile_record.get("id") != distribution_profile_id
            or not isinstance(profile_parameters, dict)
            or profile_parameters.get("visibility") != evidence_role
        ):
            raise ValueError("distribution profile does not match the pilot evidence role")
    search_policy_version = (
        str(learned_policy_record["policy_version"])
        if learned_policy_record is not None else QualityDiversitySearch.policy_version
    )

    suites = (
        SuiteBinding(
            name="pilot-public-build", stage="gate_0", zone="public",
            manifest_digest=digest, manifest_path=str(manifest_path),
            splits=("validation",),
        ),
        SuiteBinding(
            name="pilot-public-paired", stage="gate_1", zone="public",
            manifest_digest=digest, manifest_path=str(manifest_path),
            splits=("validation",),
        ),
    ) + (() if not holdout_splits else (
        SuiteBinding(
            name="pilot-nominated-holdout", stage="gate_2", zone="validation",
            manifest_digest=str(holdout_digest), manifest_path=None,
            splits=holdout_splits,
        ),
    ))
    spec = CampaignSpec(
        campaign_id=campaign_id,
        base_commit=base_commit,
        allowed_paths=("algofinder/algofinder/solvers/tsp/campaign_candidate_*.py",),
        suites=suites,
        budgets=BudgetLimits(
            # Template candidates are cheap to evaluate but weak local models
            # can need several structured generations to produce one novel,
            # valid state. This is a reservation ceiling; measured tokens are
            # recorded separately and remain a comparison outcome.
            tokens=max(300_000, target_candidates * 8 * 3 * 8_000),
            wall_seconds=14_400,
            cpu_seconds=7_200,
            evaluation_seconds=3_600,
            validation_submissions=1 if holdout_splits else 0,
        ),
        agent_spec_ids=(agent_id,),
        search_policy_version=search_policy_version,
        objective={
            "primary": "paired_gap_percent",
            "direction": "minimize",
            "secondary": ["mean_wall_seconds", "ok_rate"],
            "archive_axes": [axis.name for axis in _archive_config(representation).axes],
            "experiment_source_digest": source_provenance["source_digest"],
        },
        goal=(
            f"Improve Euclidean TSP quality with the {representation} representation "
            "against distance-ranked-2opt without importing later solver code."
        ),
        problem_scope=("tsp:euclidean", "tsp:clustered"),
        network_policy="deny",
        distribution_profile_id=distribution_profile_id,
        distribution_profile_digest=distribution_profile_digest,
        evidence_partition_id=evidence_partition_id or f"partition-{evidence_role.replace('_', '-')}",
    )
    agent_spec = AgentSpec(
        agent_spec_id=agent_id,
        provider="evolution" if backend == "evolution" else "ollama",
        model="typed-hir-evolution-v1" if backend == "evolution" else model_name,
        prompt_digest=(
            TEMPLATE_METHOD_PROMPT_DIGEST if candidate_template is not None else DEFAULT_METHOD_PROMPT_DIGEST
        ),
        tool_api_version=AgentTools.api_version,
        context_policy="parent-state+evaluator-feedback+immutable-ledger@2",
        sampling={
            "temperature": 0.2,
            "seed": model_seed,
            "max_output_tokens": 6_000 if representation == "full-code" else (1_200 if representation == "hir" else 512),
            "representation": representation,
            "backend": backend,
            "model_identity": recorded_model_identity,
            "experiment_source_digest": source_provenance["source_digest"],
        },
        permissions=(AgentTools.api_version,),
        policy_id=learned_policy_id,
        policy_digest=learned_policy_digest,
    )
    campaign = Campaign.create(campaigns_root, spec, (agent_spec,))
    _write_experiment_source_snapshot(
        repo_root=repo_root,
        campaign_root=campaign.root,
        provenance=source_provenance,
    )
    campaign.transition("active", reason="local four-candidate real-world assay")
    workspaces = WorkspaceService(
        source_root=repo_root,
        package_root=repo_root / "algofinder",
        ledger=campaign.ledger,
        policy=PatchPolicy(
            allowed_paths=spec.allowed_paths,
            forbidden_imports=FORBIDDEN_SOLVER_IMPORTS,
        ),
    )
    tools = AgentTools(campaign, workspaces)
    authority = EvaluationAuthority(campaign, workspaces)
    benchmark = PublicBenchmarkAdapter(
        authority_package_root=repo_root / "algofinder",
        baseline_solver_ids=("distance-ranked-2opt",),
        cell_timeout_seconds=per_cell_budget_seconds,
    )
    evaluator = AuthorityEvaluator(
        authority=authority,
        public_adapter=benchmark,
        plan=PublicEvaluationPlan(
            seeds=(0,),
            gate_1_budget_seconds=per_cell_budget_seconds,
            gate_1_timeout_seconds=240.0,
        ),
    )
    model = (
        EvolutionaryHIRModel(seed=model_seed)
        if backend == "evolution"
        else OllamaResearchModel(
            OllamaModelConfig(
                model=model_name,
                endpoint=endpoint,
                seed=model_seed,
                context_tokens=16_384 if representation == "full-code" else 4096,
                temperature=0.2,
                proposal_tokens=6_000 if representation == "full-code" else (1_200 if representation == "hir" else 512),
                analysis_tokens=450,
            ),
            campaign=campaign,
        )
    )
    recorder = EpisodeRecorder(campaign.ledger)
    episode = recorder.start(
        agent_spec_id=agent_id,
        campaign_snapshot=tools.campaign_snapshot(),
        model_identity=recorded_model_identity,
        policy_id=learned_policy_id,
        policy_digest=learned_policy_digest,
        distribution_profile_id=distribution_profile_id,
        evidence_role=evidence_role,
        rng_seed=model_seed,
    )
    search = QualityDiversitySearch(
        tools, config=_archive_config(representation), seed=model_seed
    )
    baseline_policy = QualityDiversityPolicy(
        tools,
        search=search,
        max_context_candidates=24,
        source_paths=FULL_CODE_SOURCE_PATHS if representation == "full-code" else SOURCE_PATHS,
        max_source_bytes=70_000,
    )
    runtime_policy = (
        load_learned_policy(
            store=learning_store,
            policy_id=learned_policy_id,
            baseline=baseline_policy,
            model=recorded_model_identity["model"],
            representation=representation,
            runtime_seed=model_seed,
        )
        if learning_store is not None and learned_policy_id is not None else baseline_policy
    )
    agent = QualityDiversityResearchAgent(
        tools=tools,
        model=model,
        evaluator=evaluator,
        search=search,
        policy=runtime_policy,
        source_paths=FULL_CODE_SOURCE_PATHS if representation == "full-code" else SOURCE_PATHS,
        max_source_bytes=70_000,
        max_validation_attempts=3,
        producing_episode_id=episode.episode_id,
        candidate_template=candidate_template,
        record_model_analysis=True,
        recorder=recorder,
    )

    steps: list[dict[str, object]] = []
    evaluated = 0
    attempts = 0
    max_attempts = target_candidates * 8
    while evaluated < target_candidates and attempts < max_attempts:
        attempts += 1
        step = agent.run_step()
        row = {
            "attempt": attempts,
            "candidate_id": step.candidate_id,
            "operator": step.directive.operator,
            "status": step.status,
            "validation_errors": step.validation.get("errors", []),
            "feedback": [
                {key: value for key, value in item.items() if key != "raw_runs"}
                for item in step.feedback
            ],
        }
        steps.append(row)
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
        if step.candidate_id is not None:
            evaluated += 1

    model_usage = getattr(model, "usage", [])
    prompt_tokens = sum(
        int(item.get("prompt_tokens", 0)) for item in model_usage if isinstance(item, dict)
    )
    completion_tokens = sum(
        int(item.get("completion_tokens", 0)) for item in model_usage if isinstance(item, dict)
    )
    agent_run = recorder.finish_run(
        inputs={
            "target_candidates": target_candidates,
            "representation": representation,
            "backend": backend,
            "model_seed": model_seed,
        },
        produced_candidate_ids=tuple(
            str(row["candidate_id"]) for row in steps if row.get("candidate_id") is not None
        ),
        token_cost=prompt_tokens + completion_tokens,
        terminal_outcome="completed",
        completion_reason=(
            "target_candidates_reached" if evaluated >= target_candidates
            else "proposal_round_budget_reached"
        ),
        cost_vector={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens": prompt_tokens + completion_tokens,
        },
    )

    holdout: dict[str, object] | None = None
    if holdout_splits and not defer_holdout:
        assert holdout_manifest_path is not None
        try:
            nominated, development_gap = _development_champion(campaign)
        except RuntimeError as exc:
            holdout = {
                "selection_rule": "lowest gate_1 paired_gap_percent; proposal ordinal then patch digest tie break",
                "gate": "not_run",
                "reason": str(exc),
                "splits": list(holdout_splits),
                "evidence_role": "reusable_test_not_sealed_challenge",
            }
        else:
            holdout_experiment = tools.experiment_propose(
                candidate_id=nominated.candidate_id,
                baseline_candidate_ids=(),
                suite_names=("pilot-nominated-holdout",),
                stage="gate_2",
                seeds=(0,),
                budget_seconds=per_cell_budget_seconds,
                stopping_rule="evaluate exactly one development-selected nomination on the frozen holdout suite",
                expected_result="a paired authority measurement on the untouched test splits",
                decision_rule="record the held-out result without feeding it back into model search",
            )
            holdout_outcome = authority.gate_2_validation(
                workspace=workspaces.load(nominated.candidate_id),
                candidate=nominated,
                experiment=holdout_experiment,
                adapter=benchmark,
                manifest_path=str(holdout_manifest_path),
                timeout_seconds=240.0,
            )
            holdout = {
                "selection_rule": "lowest gate_1 paired_gap_percent; proposal ordinal then patch digest tie break",
                "selected_candidate_id": nominated.candidate_id,
                "development_paired_gap_percent": development_gap,
                "splits": list(holdout_splits),
                "evidence_role": "reusable_test_not_sealed_challenge",
                "evaluation_id": holdout_outcome.evaluation.evaluation_id,
                "gate": holdout_outcome.evaluation.gate,
                "metric_vector": holdout_outcome.evaluation.metric_vector,
                "paired_baselines": holdout_outcome.evaluation.paired_baselines,
            }
    elif holdout_splits:
        holdout = {
            "gate": "deferred",
            "selection_rule": "lowest gate_1 paired_gap_percent; proposal ordinal then patch digest tie break",
            "splits": list(holdout_splits),
            "evidence_role": "fresh_confirmatory_test_held_until_all_searches_complete",
        }

    campaign.record_usage(
        purpose="measured-campaign-wall",
        amounts={"wall_seconds": perf_counter() - campaign_started},
    )
    if not defer_holdout:
        campaign.transition(
            "review",
            reason="planned candidate budget and optional single holdout nomination completed",
        )
    result: dict[str, object] = {
        "campaign_id": campaign_id,
        "campaign_root": str(campaign.root),
        "base_commit": base_commit,
        "model": model_name,
        "representation": representation,
        "backend": backend,
        "model_seed": model_seed,
        "model_identity": recorded_model_identity,
        "learned_policy_id": learned_policy_id,
        "learned_policy_digest": learned_policy_digest,
        "distribution_profile_id": distribution_profile_id,
        "distribution_profile_digest": distribution_profile_digest,
        "evidence_role": evidence_role,
        "episode_id": episode.episode_id,
        "agent_run_id": agent_run.agent_run_id,
        "provenance": source_provenance,
        "campaign_state": campaign.state,
        "target_candidates": target_candidates,
        "max_proposal_rounds": max_attempts,
        "evaluated_candidates": evaluated,
        "model_usage": getattr(model, "usage", []),
        "budget": campaign.budget_status().to_mapping(),
        "steps": steps,
        "holdout": holdout,
    }
    summary_path = campaign.root / "pilot-summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary_path": str(summary_path)}, ensure_ascii=False), flush=True)
    return result


def finalize_deferred_holdout(
    *,
    repo_root: Path,
    campaigns_root: Path,
    campaign_id: str,
    holdout_manifest_path: Path,
    per_cell_budget_seconds: float,
) -> dict[str, object]:
    """Open exactly one frozen nomination after every search arm has finished."""
    started = perf_counter()
    repo_root = repo_root.resolve()
    holdout_manifest_path = holdout_manifest_path.resolve()
    campaign = Campaign.open(campaigns_root.resolve(), campaign_id)
    campaign.require_active()
    expected_source_digest = campaign.spec.get("objective", {}).get("experiment_source_digest")
    actual_source_digest = _experiment_source_provenance(repo_root)["source_digest"]
    if actual_source_digest != expected_source_digest:
        raise RuntimeError("experiment control-plane source changed before deferred holdout")
    holdout_suite = next(
        (
            suite for suite in campaign.spec.get("suites", ())
            if suite.get("name") == "pilot-nominated-holdout"
        ),
        None,
    )
    if not isinstance(holdout_suite, dict):
        raise RuntimeError("campaign has no frozen deferred-holdout suite")
    actual_manifest_digest = sha256(holdout_manifest_path.read_bytes()).hexdigest()
    if actual_manifest_digest != holdout_suite.get("manifest_digest"):
        raise RuntimeError("deferred holdout manifest does not match the campaign's frozen digest")

    workspaces = WorkspaceService(
        source_root=repo_root,
        package_root=repo_root / "algofinder",
        ledger=campaign.ledger,
        policy=PatchPolicy(
            allowed_paths=tuple(campaign.spec["allowed_paths"]),
            forbidden_imports=FORBIDDEN_SOLVER_IMPORTS,
        ),
    )
    tools = AgentTools(campaign, workspaces)
    authority = EvaluationAuthority(campaign, workspaces)
    benchmark = PublicBenchmarkAdapter(
        authority_package_root=repo_root / "algofinder",
        baseline_solver_ids=("distance-ranked-2opt",),
        cell_timeout_seconds=per_cell_budget_seconds,
    )
    splits = tuple(str(item) for item in holdout_suite.get("splits", ()))
    try:
        nominated, development_gap = _development_champion(campaign)
    except RuntimeError as exc:
        holdout: dict[str, object] = {
            "selection_rule": "lowest gate_1 paired_gap_percent; proposal ordinal then patch digest tie break",
            "gate": "not_run",
            "reason": str(exc),
            "splits": list(splits),
            "evidence_role": "fresh_confirmatory_test",
        }
    else:
        holdout_experiment = tools.experiment_propose(
            candidate_id=nominated.candidate_id,
            baseline_candidate_ids=(),
            suite_names=("pilot-nominated-holdout",),
            stage="gate_2",
            seeds=(0,),
            budget_seconds=per_cell_budget_seconds,
            stopping_rule="evaluate exactly one development-selected nomination on the frozen holdout suite",
            expected_result="a paired authority measurement on the untouched test splits",
            decision_rule="record the held-out result without feeding it back into model search",
        )
        outcome = authority.gate_2_validation(
            workspace=workspaces.load(nominated.candidate_id),
            candidate=nominated,
            experiment=holdout_experiment,
            adapter=benchmark,
            manifest_path=str(holdout_manifest_path),
            timeout_seconds=240.0,
        )
        holdout = {
            "selection_rule": "lowest gate_1 paired_gap_percent; proposal ordinal then patch digest tie break",
            "selected_candidate_id": nominated.candidate_id,
            "development_paired_gap_percent": development_gap,
            "splits": list(splits),
            "evidence_role": "fresh_confirmatory_test",
            "evaluation_id": outcome.evaluation.evaluation_id,
            "gate": outcome.evaluation.gate,
            "metric_vector": outcome.evaluation.metric_vector,
            "paired_baselines": outcome.evaluation.paired_baselines,
        }
    campaign.record_usage(
        purpose="measured-deferred-holdout-wall",
        amounts={"wall_seconds": perf_counter() - started},
    )
    campaign.transition(
        "review",
        reason="deferred holdout opened after all preregistered searches completed",
    )
    summary_path = campaign.root / "pilot-summary.json"
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    result["holdout"] = holdout
    result["campaign_state"] = campaign.state
    result["budget"] = campaign.budget_status().to_mapping()
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repo_root,
        text=True, capture_output=True, check=True,
    )
    return result.stdout.strip()


def _experiment_source_provenance(repo_root: Path) -> dict[str, object]:
    """Content-address the dirty or committed control plane actually executed."""
    package = repo_root / "algofinder" / "algofinder"
    files = sorted((package / "agents").rglob("*.py"))
    files.extend(
        package / relative
        for relative in (
            "harness/benchmark.py",
            "solvers/tsp/distance_ranked.py",
            "solvers/tsp/euclidean_geometry.py",
            "solvers/tsp/euclidean_local_search.py",
        )
    )
    digests = {
        str(path.relative_to(repo_root)): sha256(path.read_bytes()).hexdigest()
        for path in files if path.is_file()
    }
    encoded = json.dumps(digests, sort_keys=True, separators=(",", ":")).encode("utf-8")
    status = subprocess.run(
        (
            "git", "status", "--porcelain=v1", "--untracked-files=all", "--",
            "algofinder/algofinder/agents",
            "algofinder/algofinder/harness/benchmark.py",
            "algofinder/algofinder/solvers/tsp",
        ),
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    return {
        "git_head": _git_head(repo_root),
        "worktree_clean_for_experiment_sources": not status,
        "dirty_source_paths": [line[3:] for line in status if len(line) > 3],
        "source_digest": sha256(encoded).hexdigest(),
        "source_snapshot": "provenance/experiment-sources.json",
        "file_digests": digests,
    }


def _write_experiment_source_snapshot(
    *,
    repo_root: Path,
    campaign_root: Path,
    provenance: dict[str, object],
) -> None:
    """Persist the actual dirty-or-clean source bytes needed for reconstruction."""
    file_digests = provenance.get("file_digests", {})
    if not isinstance(file_digests, dict):
        raise RuntimeError("experiment provenance has no file digest mapping")
    files: dict[str, str] = {}
    for relative, expected in sorted(file_digests.items()):
        path = repo_root / str(relative)
        content = path.read_text(encoding="utf-8")
        if sha256(content.encode("utf-8")).hexdigest() != expected:
            raise RuntimeError(f"experiment source changed while campaign was being created: {relative}")
        files[str(relative)] = content
    target = campaign_root / str(provenance["source_snapshot"])
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_text(
        json.dumps(
            {
                "schema_id": "algofinder.agents.experiment-source-snapshot",
                "schema_version": "1",
                "source_digest": provenance["source_digest"],
                "files": files,
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n",
        encoding="utf-8",
    )


def _development_champion(campaign: Campaign) -> tuple[Candidate, float]:
    """Select one nomination using only gate-1 evidence recorded during search."""
    scored: list[tuple[float, int, str, str]] = []
    for evaluation in campaign.ledger.list("evaluation"):
        if evaluation.get("stage") != "gate_1" or evaluation.get("gate") != "pass":
            continue
        metrics = evaluation.get("metric_vector", {})
        value = metrics.get("paired_gap_percent") if isinstance(metrics, dict) else None
        if isinstance(value, (int, float)):
            candidate_id = str(evaluation["candidate_id"])
            candidate = campaign.ledger.read("candidate", candidate_id)
            descriptors = candidate.get("descriptors", {})
            ordinal = descriptors.get("proposal_ordinal") if isinstance(descriptors, dict) else None
            scored.append((
                float(value),
                int(ordinal) if isinstance(ordinal, int) else 10**9,
                str(candidate.get("patch_digest") or candidate.get("build_digest") or candidate_id),
                candidate_id,
            ))
    if not scored:
        raise RuntimeError("cannot nominate a holdout candidate: no passing gate_1 score")
    development_gap, _, _, candidate_id = min(scored)
    record = campaign.ledger.read("candidate", candidate_id)
    return (
        Candidate(
            candidate_id=str(record["id"]),
            campaign_id=str(record["campaign_id"]),
            hypothesis_id=str(record["hypothesis_id"]),
            parent_ids=tuple(str(item) for item in record["parent_ids"]),
            generation_operator=str(record["generation_operator"]),  # type: ignore[arg-type]
            solver_entrypoints=tuple(str(item) for item in record.get("solver_entrypoints", ())),
            producing_agent_run_id=(
                str(record["producing_agent_run_id"])
                if record.get("producing_agent_run_id") is not None else None
            ),
            producing_episode_id=(
                str(record["producing_episode_id"])
                if record.get("producing_episode_id") is not None else None
            ),
            patch_digest=str(record["patch_digest"]) if record.get("patch_digest") else None,
            build_digest=str(record["build_digest"]) if record.get("build_digest") else None,
            descriptors=dict(record.get("descriptors", {})),
            created_at=str(record["created_at"]),
        ),
        development_gap,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--campaigns-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--representation", choices=tuple(REPRESENTATIONS), default="formula")
    parser.add_argument("--backend", choices=("model", "evolution"), default="model")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--budget-seconds", type=float, default=0.20)
    parser.add_argument("--model-seed", type=int, default=75348)
    parser.add_argument("--holdout-split", action="append", default=[])
    parser.add_argument("--holdout-manifest", type=Path)
    parser.add_argument("--defer-holdout", action="store_true")
    parser.add_argument("--learning-root", type=Path)
    parser.add_argument("--learned-policy-id")
    parser.add_argument("--distribution-profile-id")
    parser.add_argument("--evidence-role", choices=("public_learning", "outer_validation", "sealed"), default="public_learning")
    parser.add_argument("--evidence-partition-id")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.candidates < 1 or args.budget_seconds <= 0:
        raise SystemExit("candidates and budget-seconds must be positive")
    run_pilot(
        repo_root=args.repo_root,
        campaigns_root=args.campaigns_root,
        manifest_path=args.manifest,
        model_name=args.model,
        endpoint=args.endpoint,
        target_candidates=args.candidates,
        per_cell_budget_seconds=args.budget_seconds,
        representation=args.representation,
        backend=args.backend,
        model_seed=args.model_seed,
        holdout_splits=tuple(args.holdout_split),
        holdout_manifest_path=args.holdout_manifest,
        defer_holdout=args.defer_holdout,
        learning_root=args.learning_root,
        learned_policy_id=args.learned_policy_id,
        distribution_profile_id=args.distribution_profile_id,
        evidence_role=args.evidence_role,
        evidence_partition_id=args.evidence_partition_id,
    )


if __name__ == "__main__":
    main()


__all__ = ["finalize_deferred_holdout", "run_pilot"]
