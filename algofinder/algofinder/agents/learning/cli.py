"""Command surface for provenance-safe AlgoFinder agent learning.

Commands materialize records and preparations only.  ``run-agent-evaluation``
prints the frozen schedule for an explicit campaign runner; it does not start a
multi-hour Ollama assay implicitly.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from algofinder.agents.learning.backfill import backfill_pilot_summary
from algofinder.agents.learning.dataset import build_dataset, historical_backfill_rows, native_attempt_rows
from algofinder.agents.learning.distributions import build_distribution_profile, profile_feature_digest
from algofinder.agents.learning.eligibility import assess_campaign, registered_campaign_ids
from algofinder.agents.learning.features import FEATURE_SCHEMA_DIGEST
from algofinder.agents.learning.outer_evaluation import finalize_agent_evaluation, prepare_agent_evaluation
from algofinder.agents.learning.store import LearningArtifactStore
from algofinder.agents.learning.trainer import train_policy_from_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    backfill = commands.add_parser("backfill", help="append conservative projections of registered historical campaigns")
    backfill.add_argument("--campaigns-root", type=Path, required=True)
    backfill.add_argument("--comparison-summary", type=Path, required=True)
    backfill.add_argument("--learning-root", type=Path, required=True)

    dataset = commands.add_parser("build-dataset", help="build a grouped-split Parquet attempt dataset")
    dataset.add_argument("--campaigns-root", type=Path)
    dataset.add_argument("--campaign-id", action="append", default=[])
    dataset.add_argument("--learning-root", type=Path, required=True)
    dataset.add_argument("--dataset-id", required=True)
    dataset.add_argument("--split-seed", type=int, default=75348)

    train = commands.add_parser("train", help="fit and freeze an inspectable policy advisor")
    train.add_argument("--learning-root", type=Path, required=True)
    train.add_argument("--dataset-id", required=True)
    train.add_argument("--representation", default="formula")
    train.add_argument("--seed", type=int, default=75348)
    train.add_argument("--policy-id")

    distribution = commands.add_parser("register-distribution", help="freeze a score-free task-distribution declaration")
    distribution.add_argument("--learning-root", type=Path, required=True)
    distribution.add_argument("--declaration", type=Path, required=True)

    prepare = commands.add_parser("prepare-agent-evaluation", help="freeze a balanced outer-evaluation schedule")
    prepare.add_argument("--learning-root", type=Path, required=True)
    prepare.add_argument("--policy-id", required=True)
    prepare.add_argument("--profile-id", action="append", required=True)
    prepare.add_argument("--models", nargs=2, required=True)
    prepare.add_argument("--phase", choices=("outer-validation", "sealed-test"), required=True)
    prepare.add_argument("--evaluation-id")

    run = commands.add_parser("run-agent-evaluation", help="display a frozen schedule for an explicit campaign runner")
    run.add_argument("--learning-root", type=Path, required=True)
    run.add_argument("--evaluation-id", required=True)

    finalize = commands.add_parser("finalize-agent-evaluation", help="apply admission gates to campaign-level outcome summaries")
    finalize.add_argument("--learning-root", type=Path, required=True)
    finalize.add_argument("--evaluation-id", required=True)
    finalize.add_argument("--campaign-outcomes", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "backfill":
        _backfill(args)
    elif args.command == "build-dataset":
        _dataset(args)
    elif args.command == "train":
        _train(args)
    elif args.command == "register-distribution":
        _distribution(args)
    elif args.command == "prepare-agent-evaluation":
        _prepare(args)
    elif args.command == "run-agent-evaluation":
        _run(args)
    elif args.command == "finalize-agent-evaluation":
        _finalize(args)


def _backfill(args: argparse.Namespace) -> None:
    store = LearningArtifactStore(args.learning_root).create()
    registered = registered_campaign_ids(args.comparison_summary)
    results = []
    for campaign_id in sorted(registered):
        report = assess_campaign(args.campaigns_root, campaign_id, registered_final_campaign_ids=registered)
        if report.tier == "D":
            results.append({"campaign_id": campaign_id, "tier": report.tier, "reasons": list(report.reasons)})
            continue
        projection = backfill_pilot_summary(campaigns_root=args.campaigns_root, campaign_id=campaign_id, eligibility=report, learning_store=store)
        results.append({"campaign_id": campaign_id, "tier": report.tier, "backfill_id": projection["backfill_id"]})
    _print({"registered_campaigns": len(registered), "results": results})


def _dataset(args: argparse.Namespace) -> None:
    store = LearningArtifactStore(args.learning_root).create()
    rows = historical_backfill_rows(store)
    if args.campaign_id:
        if args.campaigns_root is None:
            raise SystemExit("--campaigns-root is required with --campaign-id")
        rows.extend(native_attempt_rows(args.campaigns_root, args.campaign_id))
    _print(build_dataset(store=store, rows=rows, dataset_id=args.dataset_id, split_seed=args.split_seed))


def _train(args: argparse.Namespace) -> None:
    store = LearningArtifactStore(args.learning_root).create()
    manifest = store.read(f"datasets/{args.dataset_id}/manifest.json")
    if manifest.get("feature_schema_digest") != FEATURE_SCHEMA_DIGEST:
        raise SystemExit("dataset feature schema is incompatible with this trainer")
    parquet = store.root / "datasets" / args.dataset_id / "attempts.parquet"
    _verify_dataset_table(store, manifest, parquet)
    rows = _read_dataset_rows(parquet)
    expected_rows = next(
        (item.get("rows") for item in manifest.get("tables", ()) if isinstance(item, dict) and item.get("name") == "attempts"),
        None,
    )
    if expected_rows != len(rows):
        raise SystemExit("dataset row count does not match its immutable manifest")
    _print(train_policy_from_rows(
        store=store, rows=rows, dataset_digest=str(manifest["content_digest"]),
        representation=args.representation, feature_schema_digest=FEATURE_SCHEMA_DIGEST,
        seed=args.seed, policy_id=args.policy_id,
    ))


def _prepare(args: argparse.Namespace) -> None:
    store = LearningArtifactStore(args.learning_root).create()
    _print(prepare_agent_evaluation(
        store=store, learned_policy_id=args.policy_id, profile_ids=tuple(args.profile_id),
        models=tuple(args.models), phase=args.phase, evaluation_id=args.evaluation_id,
    ))


def _distribution(args: argparse.Namespace) -> None:
    try:
        declaration = json.loads(args.declaration.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("distribution declaration is not valid JSON") from exc
    if not isinstance(declaration, dict):
        raise SystemExit("distribution declaration must be a JSON object")
    required = {
        "family", "version", "manifest_digests", "lineage_namespace", "generators",
        "feature_summary", "seed_set", "split_rule", "deployment_intent", "visibility",
    }
    optional = {
        "profile_id", "allowed_specialization_axes", "meaning_preserving_transformations",
        "diagnostic_transformations",
    }
    missing = required - set(declaration)
    extra = set(declaration) - required - optional
    if missing or extra:
        raise SystemExit(f"distribution declaration keys differ; missing={sorted(missing)}, extra={sorted(extra)}")
    try:
        summary = build_distribution_profile(
            family=str(declaration["family"]),
            version=str(declaration["version"]),
            manifest_digests=tuple(str(item) for item in declaration["manifest_digests"]),
            lineage_namespace=str(declaration["lineage_namespace"]),
            generators=dict(declaration["generators"]),
            feature_summary={str(key): float(value) for key, value in dict(declaration["feature_summary"]).items()},
            seed_set=tuple(int(item) for item in declaration["seed_set"]),
            split_rule=str(declaration["split_rule"]),
            deployment_intent=str(declaration["deployment_intent"]),
            allowed_specialization_axes=tuple(str(item) for item in declaration.get("allowed_specialization_axes", ())),
            meaning_preserving_transformations=tuple(str(item) for item in declaration.get("meaning_preserving_transformations", ())),
            diagnostic_transformations=tuple(str(item) for item in declaration.get("diagnostic_transformations", ())),
            visibility=str(declaration["visibility"]),
            profile_id=str(declaration["profile_id"]) if declaration.get("profile_id") else None,
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"invalid distribution declaration: {exc}") from exc
    store = LearningArtifactStore(args.learning_root).create()
    digest = store.write(summary.profile)
    _print({
        "profile_id": summary.profile.profile_id,
        "content_digest": digest,
        "feature_digest": profile_feature_digest(summary),
        "visibility": summary.visibility,
    })


def _run(args: argparse.Namespace) -> None:
    store = LearningArtifactStore(args.learning_root).create()
    plan = store.read(f"evaluations/{args.evaluation_id}.json")
    _print({
        "evaluation_id": args.evaluation_id,
        "phase": plan["phase"],
        "campaign_arms": plan.get("metrics", {}).get("campaign_arms", []),
        "status": "prepared_only",
        "message": "Use the emitted schedule with an explicit campaign runner; this command never starts the long GPU assay implicitly.",
    })


def _finalize(args: argparse.Namespace) -> None:
    store = LearningArtifactStore(args.learning_root).create()
    payload = json.loads(args.campaign_outcomes.read_text(encoding="utf-8"))
    outcomes = payload.get("campaign_outcomes", payload) if isinstance(payload, dict) else payload
    if not isinstance(outcomes, list):
        raise SystemExit("campaign outcomes must be a JSON array or an object with campaign_outcomes")
    _print(finalize_agent_evaluation(store=store, planned_evaluation_id=args.evaluation_id, campaign_summaries=outcomes))


def _read_dataset_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("DuckDB is required to train from a Parquet dataset") from exc
    with duckdb.connect() as connection:
        cursor = connection.execute("SELECT * FROM read_parquet(?)", [str(path)])
        columns = [item[0] for item in cursor.description]
        return [dict(zip(columns, values)) for values in cursor.fetchall()]


def _verify_dataset_table(
    store: LearningArtifactStore,
    manifest: dict[str, Any],
    path: Path,
) -> None:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(store.root)
    except (OSError, ValueError) as exc:
        raise SystemExit("dataset table is missing or escapes the learning store") from exc
    tables = manifest.get("tables", [])
    table = next(
        (item for item in tables if isinstance(item, dict) and item.get("name") == "attempts"),
        None,
    )
    if table is None or table.get("path") != path.name:
        raise SystemExit("dataset manifest has no canonical attempts table")
    actual = sha256(path.read_bytes()).hexdigest()
    if table.get("sha256") != actual:
        raise SystemExit("dataset Parquet digest does not match its immutable manifest")


def _print(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "main"]
