"""CLI for campaign control-plane operations (not a solver-search agent)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from algofinder.agents.campaign import Campaign
from algofinder.agents.contracts import AgentSpec, BudgetLimits, CampaignSpec, SuiteBinding, utc_now
from algofinder.agents.workspace import PatchPolicy, WorkspaceService
from algofinder.trace.serialize import strict_dumps


def _read_json(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object in {path}")
    return value


def _campaign_spec(value: dict[str, Any]) -> CampaignSpec:
    return CampaignSpec(
        campaign_id=value.get("campaign_id", value.get("id")),
        base_commit=value["base_commit"],
        allowed_paths=tuple(value["allowed_paths"]),
        suites=tuple(SuiteBinding(**suite) for suite in value["suites"]),
        budgets=BudgetLimits(**value["budgets"]),
        agent_spec_ids=tuple(value["agent_spec_ids"]),
        search_policy_version=value["search_policy_version"],
        objective=dict(value["objective"]),
        goal=value["goal"],
        problem_scope=tuple(value.get("problem_scope", ())),
        feedback_rounding=value.get("feedback_rounding", 4),
        network_policy=value.get("network_policy", "deny"),
        human_approval_stages=tuple(value.get("human_approval_stages", ("gate_6",))),
        distribution_profile_id=value.get("distribution_profile_id"),
        distribution_profile_digest=value.get("distribution_profile_digest"),
        evidence_partition_id=value.get("evidence_partition_id"),
        created_at=value.get("created_at") or utc_now(),
    )


def _agent_spec(value: dict[str, Any]) -> AgentSpec:
    return AgentSpec(
        agent_spec_id=value.get("agent_spec_id", value.get("id")),
        provider=value["provider"], model=value["model"], prompt_digest=value["prompt_digest"],
        tool_api_version=value["tool_api_version"], context_policy=value["context_policy"],
        sampling=dict(value.get("sampling", {})), permissions=tuple(value.get("permissions", ())),
        secret_references=tuple(value.get("secret_references", ())),
        policy_id=value.get("policy_id"), policy_digest=value.get("policy_digest"),
        created_at=value.get("created_at") or utc_now(),
    )


def _campaigns_root(value: str | None) -> Path:
    return Path(value) if value else Path.cwd() / "private" / "campaigns"


def command_init(args: argparse.Namespace) -> None:
    spec = _campaign_spec(_read_json(args.spec))
    agents = [_agent_spec(_read_json(path)) for path in args.agent_spec]
    campaign = Campaign.create(_campaigns_root(args.campaigns_root), spec, agents)
    print(strict_dumps({"campaign_id": campaign.campaign_id, "state": campaign.state, "root": str(campaign.root)}))


def command_transition(args: argparse.Namespace) -> None:
    campaign = Campaign.open(_campaigns_root(args.campaigns_root), args.campaign_id)
    campaign.transition(args.target, reason=args.reason, actor=args.actor)
    print(strict_dumps({"campaign_id": campaign.campaign_id, "state": campaign.state}))


def command_status(args: argparse.Namespace) -> None:
    campaign = Campaign.open(_campaigns_root(args.campaigns_root), args.campaign_id)
    print(strict_dumps(campaign.snapshot(agent_visible=args.agent_visible)))


def command_validate(args: argparse.Namespace) -> None:
    campaign = Campaign.open(_campaigns_root(args.campaigns_root), args.campaign_id)
    service = WorkspaceService(
        source_root=args.source_root,
        package_root=args.package_root,
        ledger=campaign.ledger,
        policy=PatchPolicy(tuple(campaign.spec["allowed_paths"])),
    )
    workspace = service.create(
        hypothesis_id=args.hypothesis_id,
        parent_ids=tuple(args.parent_id),
        generation_operator=args.generation_operator,
        candidate_id=args.candidate_id,
    )
    print(strict_dumps(service.validate(workspace).to_mapping()))


def command_index(args: argparse.Namespace) -> None:
    from algofinder.store.campaigns import build_campaign_registry

    counts = build_campaign_registry(
        campaigns_root=_campaigns_root(args.campaigns_root), registry_dir=args.registry_dir,
    )
    print(strict_dumps({"campaign_registry": counts}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algofinder.agents.cli")
    parser.add_argument("--campaigns-root", help="defaults to ./private/campaigns")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create an immutable campaign ledger")
    init.add_argument("--spec", required=True, help="CampaignSpec JSON")
    init.add_argument("--agent-spec", action="append", required=True, help="AgentSpec JSON (repeatable)")
    init.set_defaults(handler=command_init)

    for name, target in (("start", "active"), ("stop", "stopped"), ("review", "review"), ("close", "closed")):
        command = subparsers.add_parser(name, help=f"transition campaign to {target}")
        command.add_argument("campaign_id")
        command.add_argument("--reason", required=True)
        command.add_argument("--actor", default="cli")
        command.set_defaults(handler=command_transition, target=target)

    status = subparsers.add_parser("status", help="read campaign state and budgets")
    status.add_argument("campaign_id")
    status.add_argument("--agent-visible", action="store_true")
    status.set_defaults(handler=command_status)

    validate = subparsers.add_parser("candidate-validate", help="create/resume and statically validate a workspace")
    validate.add_argument("campaign_id")
    validate.add_argument("candidate_id")
    validate.add_argument("hypothesis_id")
    validate.add_argument("--source-root", required=True)
    validate.add_argument("--package-root", required=True)
    validate.add_argument("--parent-id", action="append", default=[])
    validate.add_argument("--generation-operator", default="invent")
    validate.set_defaults(handler=command_validate)

    index = subparsers.add_parser("index", help="rebuild derived campaign registry JSONL files")
    index.add_argument("--registry-dir", required=True)
    index.set_defaults(handler=command_index)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":  # pragma: no cover
    main()
