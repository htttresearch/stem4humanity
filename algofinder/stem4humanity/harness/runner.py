"""Command-line entry point for the stem4humanity running env.

Usage::

    python -m stem4humanity.harness.runner generate          # build manifests
    python -m stem4humanity.harness.runner train             # train the ML ranker
    python -m stem4humanity.harness.runner benchmark         # run all solvers
    python -m stem4humanity.harness.runner report            # render leaderboard
    python -m stem4humanity.harness.runner all               # generate + train + benchmark + report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INSTANCES_DIR = DATA_DIR / "instances"
RESULTS_DIR = DATA_DIR / "results"
MODEL_PATH = DATA_DIR / "models" / "candidate_ranker.joblib"

MANIFEST_GENERATORS = [
    ("stem4humanity.instances.bin_packing", "generate_bin_packing_manifests"),
    ("stem4humanity.instances.knapsack", "generate_knapsack_manifests"),
    ("stem4humanity.instances.euclidean_tsp", "generate_tsp_manifests"),
    ("stem4humanity.instances.general_tsp", "generate_general_tsp_manifests"),
    ("stem4humanity.instances.parallel_scheduling", "generate_parallel_scheduling_manifests"),
    ("stem4humanity.instances.shortest_path", "generate_shortest_path_manifests"),
    ("stem4humanity.instances.scheduling", "generate_scheduling_manifests"),
    ("stem4humanity.instances.unit_commitment", "generate_unit_commitment_manifests"),
]


def _load_instances(manifest_paths: list[str]) -> list[object]:
    import stem4humanity.solvers.registry  # noqa: F401
    from stem4humanity.instances.base import load_manifest

    paths = [Path(path) for path in manifest_paths]
    if not paths:
        if not INSTANCES_DIR.exists():
            raise SystemExit(
                f"no manifests found under {INSTANCES_DIR}; "
                "run 'python -m stem4humanity.harness.runner generate' first"
            )
        paths = sorted(INSTANCES_DIR.glob("*.json"))
    instances: list[object] = []
    for path in paths:
        instances.extend(load_manifest(path))
    return instances


def command_generate(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    for module_name, function_name in MANIFEST_GENERATORS:
        module = __import__(module_name, fromlist=[function_name])
        getattr(module, function_name)(str(output_dir))
        print(f"generated: {module_name}.{function_name} -> {output_dir}")
    print(f"manifests written to {output_dir}")


def command_benchmark(args: argparse.Namespace) -> None:
    from stem4humanity.harness.benchmark import run_benchmark
    from stem4humanity.trace.session import DevSession

    session: DevSession | None = None
    if args.mode == "dev":
        session = DevSession(profile=args.profile)
        session.__enter__()
        print(f"dev mode: session {session.session_id} at {session.dir}")
        print(f"dev mode: trace profile {args.profile}")
    instances = _load_instances(args.manifest)
    print(
        f"benchmarking {len(instances)} instances "
        f"(budget={args.budget_seconds}s, timeout={args.timeout_seconds}s, "
        f"mode={args.mode})"
    )
    try:
        runs = run_benchmark(
            instances,
            budget_seconds=args.budget_seconds,
            timeout_seconds=args.timeout_seconds,
            session=session,
        )
    finally:
        if session is not None:
            session.close()
    record = {
        "config": {
            "budget_seconds": args.budget_seconds,
            "timeout_seconds": args.timeout_seconds,
            "mode": args.mode,
            "profile": args.profile,
            "session": session.session_id if session else None,
            "manifests": [str(path) for path in _manifest_paths(args)],
        },
        "runs": [run.to_mapping() for run in runs],
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {len(runs)} runs to {output}")


def _manifest_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path) for path in args.manifest]
    if not paths and INSTANCES_DIR.exists():
        paths = sorted(INSTANCES_DIR.glob("*.json"))
    return paths


def command_report(args: argparse.Namespace) -> None:
    from stem4humanity.harness.benchmark import BenchmarkRun
    from stem4humanity.harness.leaderboard import render_leaderboard

    record = json.loads(Path(args.runs).read_text())
    runs = [BenchmarkRun(**mapping) for mapping in record["runs"]]
    leaderboard = render_leaderboard(runs)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(leaderboard + "\n")
    print(leaderboard)
    print(f"leaderboard written to {output}")


def command_train(args: argparse.Namespace) -> None:
    from stem4humanity.ml.train import (
        train_bp_packer,
        train_ps_prioritizer,
        train_ranker,
        train_sched_comparator,
        train_sp_pruner,
        train_uc_committer,
        train_uc_storage,
    )

    instances = _load_instances(args.manifest)
    train_instances = [instance for instance in instances if instance.split == "train"]
    if not train_instances:
        raise SystemExit(
            f"no train-split instances in {_manifest_paths(args)}; "
            "run 'python -m stem4humanity.harness.runner generate' first"
        )
    print(f"training on {len(train_instances)} train-split instances")
    train_ranker(
        train_instances,
        output_path=args.output,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    train_sp_pruner(
        train_instances,
        output_path=DATA_DIR / "models" / "sp_pruner.joblib",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    train_sched_comparator(
        train_instances,
        output_path=DATA_DIR / "models" / "sched_comparator.joblib",
        random_state=args.random_state,
    )
    train_bp_packer(
        train_instances,
        output_path=DATA_DIR / "models" / "bp_packer.joblib",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    train_ps_prioritizer(
        train_instances,
        output_path=DATA_DIR / "models" / "ps_prioritizer.joblib",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    train_uc_committer(
        train_instances,
        output_path=DATA_DIR / "models" / "uc_committer.joblib",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    train_uc_storage(
        train_instances,
        output_path=DATA_DIR / "models" / "uc_storage.joblib",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )


def command_all(args: argparse.Namespace) -> None:
    command_generate(args)
    command_train(args)
    command_benchmark(args)
    command_report(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stem4humanity-harness",
        description="stem4humanity running env: generate instances, run "
        "benchmarks, render the leaderboard.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="generate all instance manifests"
    )
    generate.add_argument(
        "--output-dir",
        type=str,
        default=str(INSTANCES_DIR),
        help=f"manifest output directory (default: {INSTANCES_DIR})",
    )
    generate.set_defaults(func=command_generate)

    benchmark = subparsers.add_parser(
        "benchmark", help="run every applicable solver on every instance"
    )
    benchmark.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="manifest file (repeatable; defaults to all under data/instances)",
    )
    benchmark.add_argument(
        "--budget-seconds", type=float, default=None, help="per-solver time budget"
    )
    benchmark.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="hard wall-clock cap per (instance, solver) cell (default: 30)",
    )
    benchmark.add_argument(
        "--mode",
        type=str,
        choices=["prod", "dev"],
        default="prod",
        help="prod: no tracing, run as fast as possible; "
        "dev: record solver traces in a session under data/sessions "
        "(default: prod)",
    )
    benchmark.add_argument(
        "--profile",
        type=str,
        choices=["full", "decisions", "summary"],
        default="full",
        help="dev-mode trace profile: full = every atomic step; "
        "decisions = state-changing steps; summary = lifecycle only "
        "(default: full)",
    )
    benchmark.add_argument(
        "--out",
        type=str,
        default=str(RESULTS_DIR / "benchmark.json"),
        help=f"results file (default: {RESULTS_DIR / 'benchmark.json'})",
    )
    benchmark.set_defaults(func=command_benchmark)

    report = subparsers.add_parser(
        "report", help="render the leaderboard from a benchmark run"
    )
    report.add_argument(
        "--runs",
        type=str,
        default=str(RESULTS_DIR / "benchmark.json"),
        help=f"benchmark results file (default: {RESULTS_DIR / 'benchmark.json'})",
    )
    report.add_argument(
        "--out",
        type=str,
        default=str(RESULTS_DIR / "leaderboard.md"),
        help=f"leaderboard output (default: {RESULTS_DIR / 'leaderboard.md'})",
    )
    report.set_defaults(func=command_report)

    train = subparsers.add_parser(
        "train", help="train and persist the candidate-edge ranker"
    )
    train.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="manifest file (repeatable; defaults to all under data/instances)",
    )
    train.add_argument(
        "--output",
        type=str,
        default=str(MODEL_PATH),
        help=f"model output (default: {MODEL_PATH})",
    )
    train.add_argument("--n-estimators", type=int, default=250)
    train.add_argument("--max-depth", type=int, default=10)
    train.add_argument("--random-state", type=int, default=0)
    train.set_defaults(func=command_train)

    all_command = subparsers.add_parser(
        "all", help="generate manifests, train, benchmark, and render the leaderboard"
    )
    all_command.add_argument(
        "--output-dir",
        type=str,
        default=str(INSTANCES_DIR),
        help=f"manifest output directory (default: {INSTANCES_DIR})",
    )
    all_command.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="manifest file (repeatable; defaults to all under data/instances)",
    )
    all_command.add_argument(
        "--output",
        type=str,
        default=str(MODEL_PATH),
        help=f"model output (default: {MODEL_PATH})",
    )
    all_command.add_argument(
        "--runs",
        type=str,
        default=str(RESULTS_DIR / "benchmark.json"),
        help=f"benchmark results file (default: {RESULTS_DIR / 'benchmark.json'})",
    )
    all_command.add_argument(
        "--budget-seconds", type=float, default=None, help="per-solver time budget"
    )
    all_command.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="hard wall-clock cap per (instance, solver) cell (default: 30)",
    )
    all_command.add_argument(
        "--mode",
        type=str,
        choices=["prod", "dev"],
        default="prod",
        help="prod: no tracing, run as fast as possible; "
        "dev: record solver traces in a session under data/sessions "
        "(default: prod)",
    )
    all_command.add_argument(
        "--profile",
        type=str,
        choices=["full", "decisions", "summary"],
        default="full",
        help="dev-mode trace profile (default: full)",
    )
    all_command.add_argument(
        "--out",
        type=str,
        default=str(RESULTS_DIR / "benchmark.json"),
        help=f"results file (default: {RESULTS_DIR / 'benchmark.json'})",
    )
    all_command.add_argument("--n-estimators", type=int, default=250)
    all_command.add_argument("--max-depth", type=int, default=10)
    all_command.add_argument("--random-state", type=int, default=0)
    all_command.set_defaults(func=command_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
