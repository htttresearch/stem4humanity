"""Command-line entry point for the algofinder running env.

Usage::

    python -m algofinder.harness.runner --config config/config.toml generate   # build manifests
    python -m algofinder.harness.runner --config config/config.toml train      # train the ML solvers
    python -m algofinder.harness.runner --config config/config.toml benchmark  # run all solvers
    python -m algofinder.harness.runner --config config/config.toml features   # compute per-instance features
    python -m algofinder.harness.runner --config config/config.toml index      # rebuild registry + DuckDB query layer
    python -m algofinder.harness.runner --config config/config.toml report     # render leaderboard
    python -m algofinder.harness.runner --config config/config.toml all        # generate + train + benchmark + features + index + report

``--config`` may appear before or after the subcommand.  The config
file (see ``config/config.toml``) supplies every default; explicit
flags always win over it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from algofinder.config import (
    Config,
    ConfigError,
    DEFAULT_CONFIG_PATH,
    SPLITS,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_GENERATORS = [
    ("algofinder.instances.bin_packing", "generate_bin_packing_manifests"),
    ("algofinder.instances.knapsack", "generate_knapsack_manifests"),
    ("algofinder.instances.euclidean_tsp", "generate_tsp_manifests"),
    ("algofinder.instances.general_tsp", "generate_general_tsp_manifests"),
    ("algofinder.instances.parallel_scheduling", "generate_parallel_scheduling_manifests"),
    ("algofinder.instances.shortest_path", "generate_shortest_path_manifests"),
    ("algofinder.instances.scheduling", "generate_scheduling_manifests"),
    ("algofinder.instances.unit_commitment", "generate_unit_commitment_manifests"),
]


def _log(cfg: Config, message: str) -> None:
    if cfg.verbose:
        print(message)


def _load_instances(manifest_paths: list[str], instances_dir: Path) -> list[object]:
    import algofinder.solvers.registry  # noqa: F401
    from algofinder.instances.base import load_manifest

    paths = [Path(path) for path in manifest_paths]
    if not paths:
        if not instances_dir.exists():
            raise SystemExit(
                f"no manifests found under {instances_dir}; "
                "run 'python -m algofinder.harness.runner generate' first"
            )
        paths = sorted(instances_dir.glob("*.json"))
    instances: list[object] = []
    for path in paths:
        instances.extend(load_manifest(path))
    return instances


def _manifest_paths(args: argparse.Namespace, instances_dir: Path) -> list[Path]:
    paths = [Path(path) for path in args.manifest]
    if not paths and instances_dir.exists():
        paths = sorted(instances_dir.glob("*.json"))
    return paths


def command_generate(args: argparse.Namespace, cfg: Config) -> None:
    output_dir = Path(args.output_dir)
    for module_name, function_name in MANIFEST_GENERATORS:
        module = __import__(module_name, fromlist=[function_name])
        getattr(module, function_name)(
            str(output_dir),
            best_known_budget_seconds=cfg.best_known_budget_seconds,
        )
        _log(cfg, f"generated: {module_name}.{function_name} -> {output_dir}")
    if getattr(args, "fill_features", False):
        from algofinder.features.fill import generate_fill_instances

        summary = generate_fill_instances(output_dir)
        _log(
            cfg,
            f"feature-fill: {summary['instances']} instances from "
            f"{summary['bins']} bins -> {summary['manifest']} "
            f"(skipped {summary['skipped_duplicates']} duplicates)",
        )
    _log(cfg, f"manifests written to {output_dir}")


def command_benchmark(args: argparse.Namespace, cfg: Config) -> None:
    from algofinder.harness.benchmark import run_benchmark
    from algofinder.trace.session import DevSession

    session: DevSession | None = None
    trace_profile = getattr(args, "profile", None) or cfg.trace_profile
    mode = getattr(args, "mode", None) or cfg.mode
    if mode == "dev":
        session = DevSession(
            cfg.sessions_dir,
            profile=trace_profile,
            event_limit=cfg.trace_event_limit,
            byte_limit=cfg.trace_byte_limit,
            fail_on_limit=cfg.trace_fail_on_limit,
            label=cfg.session_label,
        )
        session.__enter__()
        _log(cfg, f"dev mode: session {session.session_id} at {session.dir}")
        _log(cfg, f"dev mode: trace profile {trace_profile}")
    instances = _load_instances(args.manifest, cfg.instances_dir)
    splits = tuple(args.split) if getattr(args, "split", None) else cfg.splits
    solver_include = (
        tuple(args.solver_include) if getattr(args, "solver_include", None)
        else cfg.solver_include
    )
    solver_exclude = (
        tuple(args.solver_exclude) if getattr(args, "solver_exclude", None)
        else cfg.solver_exclude
    )
    _log(
        cfg,
        f"benchmarking {len(instances)} instances "
        f"(budget={args.budget_seconds}s, timeout={args.timeout_seconds}s, "
        f"seed={args.seed}, memory={args.memory_bytes} bytes, "
        f"mode={cfg.mode}, splits={splits}, include={solver_include}, exclude={solver_exclude})",
    )
    try:
        runs = run_benchmark(
            instances,
            budget_seconds=args.budget_seconds,
            timeout_seconds=args.timeout_seconds,
            splits=splits,
            session=session,
            solver_include=solver_include,
            solver_exclude=solver_exclude,
            seed=args.seed,
            memory_bytes=args.memory_bytes,
        )
    finally:
        if session is not None:
            session.close()
    record = {
        "config": {
            "profile": cfg.profile,
            "budget_seconds": args.budget_seconds,
            "timeout_seconds": args.timeout_seconds,
            "seed": args.seed,
            "memory_bytes": args.memory_bytes,
            "mode": cfg.mode,
            "splits": list(splits),
            "solver_include": list(solver_include),
            "solver_exclude": list(solver_exclude),
            "trace_profile": trace_profile,
            "session": session.session_id if session else None,
            "manifests": [str(path) for path in _manifest_paths(args, cfg.instances_dir)],
        },
        "runs": [run.to_mapping() for run in runs],
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    _log(cfg, f"wrote {len(runs)} runs to {output}")


def command_features(args: argparse.Namespace, cfg: Config) -> None:
    from algofinder.features import enabled_feature_sets
    from algofinder.features.storage import features_dir, write_feature_set

    instances = _load_instances(args.manifest, cfg.instances_dir)
    feature_sets = enabled_feature_sets(
        {"enabled": cfg.feature_sets} if cfg.feature_sets else None
    )
    if not feature_sets:
        raise SystemExit("no feature sets enabled; add [features] enabled to the config")
    _log(
        cfg,
        f"computing features for {len(instances)} instances "
        f"(sets={[feature_set.id for feature_set in feature_sets]})",
    )
    for feature_set in feature_sets:
        path = write_feature_set(cfg.instances_dir, feature_set, instances)
        records = path.read_text(encoding="utf-8").count("\n")
        _log(cfg, f"wrote {records} records to {path}")
    _log(cfg, f"features written to {features_dir(cfg.instances_dir)}")


def command_audit(args: argparse.Namespace, cfg: Config) -> None:
    import json as _json

    from algofinder.report.audit import audit_corpus, render_audit

    summary = audit_corpus(cfg.instances_dir)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_audit(summary) + "\n")
    json_path = output.with_suffix(".json")
    json_path.write_text(_json.dumps(summary, indent=2) + "\n")
    _log(cfg, f"audit written to {output} and {json_path}")
    _log(cfg, render_audit(summary))


def command_portfolio(args: argparse.Namespace, cfg: Config) -> None:
    import json as _json

    from algofinder.harness.benchmark import BenchmarkRun
    from algofinder.report.portfolio import portfolio, render_portfolio

    instances = _load_instances(args.manifest, cfg.instances_dir)
    record = _json.loads(Path(args.runs).read_text())
    runs = [BenchmarkRun(**mapping) for mapping in record["runs"]]
    summary = portfolio(runs, instances, cfg.instances_dir)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_portfolio(summary) + "\n")
    _log(cfg, f"portfolio written to {output}")
    _log(cfg, render_portfolio(summary))


def command_export_aslib(args: argparse.Namespace, cfg: Config) -> None:
    import json as _json

    from algofinder.harness.benchmark import BenchmarkRun
    from algofinder.report.aslib import export_aslib

    record = _json.loads(Path(args.runs).read_text())
    runs = [BenchmarkRun(**mapping) for mapping in record["runs"]]
    summary = export_aslib(runs, cfg.instances_dir, cfg.aslib_dir)
    _log(cfg, "aslib scenario: " + ", ".join(f"{k}={v}" for k, v in summary.items()))


def command_report(args: argparse.Namespace, cfg: Config) -> None:
    from algofinder.harness.benchmark import BenchmarkRun
    from algofinder.harness.leaderboard import render_leaderboard

    record = json.loads(Path(args.runs).read_text())
    runs = [BenchmarkRun(**mapping) for mapping in record["runs"]]
    leaderboard = render_leaderboard(runs)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(leaderboard + "\n")
    _log(cfg, leaderboard)
    _log(cfg, f"leaderboard written to {output}")


def command_index(args: argparse.Namespace, cfg: Config) -> None:
    from algofinder.store.indexer import index_all

    summary = index_all(
        instances_dir=cfg.instances_dir,
        results_dir=cfg.results_dir,
        results_glob=args.results_glob,
        registry_dir=cfg.registry_dir,
        db_path=args.db_out,
        cache_dir=cfg.cache_dir,
        parity=args.parity,
    )
    counts = summary["registry"]
    _log(
        cfg,
        f"registry: instances={counts['instances']} solvers={counts['solvers']} "
        f"runs={counts['runs']} environments={counts['environments']} "
        f"features={counts.get('features', 0)}",
    )
    db = summary["db"]
    _log(
        cfg,
        "db: " + " ".join(f"{name}={db[name]}" for name in db if name != "cache_bytes")
        + f" cache_bytes={db['cache_bytes']}",
    )
    if summary["drift"]:
        for problem in summary["drift"]:
            print(f"registry drift: {problem}", file=sys.stderr)
        raise SystemExit(1)
    if args.parity:
        parity = summary["parity"]
        if parity["diffs"]:
            for diff in parity["diffs"]:
                print(f"parity mismatch: {diff}", file=sys.stderr)
            raise SystemExit(1)
        _log(cfg, f"parity ok: {parity['rows']} leaderboard rows identical")


def command_train(args: argparse.Namespace, cfg: Config) -> None:
    from algofinder.ml.train import (
        train_bp_packer,
        train_ps_prioritizer,
        train_ranker,
        train_sched_comparator,
        train_sp_pruner,
        train_uc_committer,
        train_uc_storage,
    )

    instances = _load_instances(args.manifest, cfg.instances_dir)
    train_instances = [instance for instance in instances if instance.split == "train"]
    if not train_instances:
        raise SystemExit(
            f"no train-split instances in {_manifest_paths(args, cfg.instances_dir)}; "
            "run 'python -m algofinder.harness.runner generate' first"
        )
    _log(cfg, f"training on {len(train_instances)} train-split instances")
    train_ranker(
        train_instances,
        output_path=Path(args.output),
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    train_sp_pruner(
        train_instances,
        output_path=cfg.models_dir / "sp_pruner.joblib",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    train_sched_comparator(
        train_instances,
        output_path=cfg.models_dir / "sched_comparator.joblib",
        random_state=args.random_state,
    )
    train_bp_packer(
        train_instances,
        output_path=cfg.models_dir / "bp_packer.joblib",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    train_ps_prioritizer(
        train_instances,
        output_path=cfg.models_dir / "ps_prioritizer.joblib",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    train_uc_committer(
        train_instances,
        output_path=cfg.models_dir / "uc_committer.joblib",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    train_uc_storage(
        train_instances,
        output_path=cfg.models_dir / "uc_storage.joblib",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )


def command_all(args: argparse.Namespace, cfg: Config) -> None:
    steps = {
        "generate": lambda: command_generate(args, cfg),
        "train": lambda: command_train(args, cfg),
        "benchmark": lambda: command_benchmark(args, cfg),
        "features": lambda: command_features(args, cfg),
        "index": lambda: command_index(args, cfg),
    }
    for step in cfg.pipeline_steps:
        _log(cfg, f"[pipeline] {step}")
        if step == "report":
            report_args = argparse.Namespace(**vars(args))
            report_args.runs = getattr(args, "out", None) or str(cfg.benchmark_out)
            report_args.out = str(cfg.leaderboard_out)
            command_report(report_args, cfg)
        else:
            steps[step]()


def build_parser(cfg: Config) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="algofinder-harness",
        description="algofinder running env: generate instances, run "
        "benchmarks, render the leaderboard.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"path to the TOML config file (default: {DEFAULT_CONFIG_PATH})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _config_flags(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--config", default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )

    generate = subparsers.add_parser(
        "generate", help="generate all instance manifests"
    )
    _config_flags(generate)
    generate.add_argument(
        "--output-dir",
        type=str,
        default=str(cfg.instances_dir),
        help=f"manifest output directory (default: {cfg.instances_dir})",
    )
    generate.add_argument(
        "--fill-features",
        action="store_true",
        default=False,
        help="also generate deterministic feature-fill instances "
        "(train-split only, never final-evaluated)",
    )
    generate.set_defaults(func=lambda args: command_generate(args, cfg))

    benchmark = subparsers.add_parser(
        "benchmark", help="run every applicable solver on every instance"
    )
    _config_flags(benchmark)
    benchmark.add_argument(
        "--manifest",
        action="append",
        default=list(cfg.manifests),
        help="manifest file (repeatable; defaults to config 'manifests' "
        "or all under the instances dir)",
    )
    benchmark.add_argument(
        "--split",
        action="append",
        choices=SPLITS,
        default=None,
        help="instance split to benchmark (repeatable; defaults to config 'splits')",
    )
    benchmark.add_argument(
        "--solver-include",
        action="append",
        default=None,
        help="solver id to include (repeatable; defaults to config selection)",
    )
    benchmark.add_argument(
        "--solver-exclude",
        action="append",
        default=None,
        help="solver id to exclude (repeatable; defaults to config selection)",
    )
    benchmark.add_argument(
        "--budget-seconds", type=float, default=cfg.budget_seconds,
        help="per-solver time budget (config default: "
        f"{cfg.budget_seconds})",
    )
    benchmark.add_argument(
        "--timeout-seconds",
        type=float,
        default=cfg.timeout_seconds,
        help="hard wall-clock cap per (instance, solver) cell (config "
        f"default: {cfg.timeout_seconds})",
    )
    benchmark.add_argument(
        "--mode",
        type=str,
        choices=["prod", "dev"],
        default=cfg.mode,
        help="prod: no tracing, run as fast as possible; "
        "dev: record solver traces in a session under the sessions dir "
        f"(config default: {cfg.mode})",
    )
    benchmark.add_argument(
        "--seed",
        type=int,
        default=cfg.seed,
        help="solver random seed for adapters that declare seed control "
        f"(config default: {cfg.seed})",
    )
    benchmark.add_argument(
        "--memory-bytes",
        type=int,
        default=cfg.memory_bytes,
        help="self-reported process-tree RSS cap in bytes; runs above it "
        "are recorded as memory_limit with no accepted solution "
        f"(config default: {cfg.memory_bytes})",
    )
    benchmark.add_argument(
        "--profile",
        type=str,
        choices=["full", "decisions", "summary"],
        default=cfg.trace_profile,
        help="dev-mode trace profile: full = every atomic step; "
        "decisions = state-changing steps; summary = lifecycle only "
        f"(config default: {cfg.trace_profile})",
    )
    benchmark.add_argument(
        "--out",
        type=str,
        default=str(cfg.benchmark_out),
        help=f"results file (config default: {cfg.benchmark_out})",
    )
    benchmark.set_defaults(func=lambda args: command_benchmark(args, cfg))

    report = subparsers.add_parser(
        "report", help="render the leaderboard from a benchmark run"
    )
    _config_flags(report)
    report.add_argument(
        "--runs",
        type=str,
        default=str(cfg.benchmark_out),
        help=f"benchmark results file (config default: {cfg.benchmark_out})",
    )
    report.add_argument(
        "--out",
        type=str,
        default=str(cfg.leaderboard_out),
        help=f"leaderboard output (config default: {cfg.leaderboard_out})",
    )
    report.set_defaults(func=lambda args: command_report(args, cfg))

    index = subparsers.add_parser(
        "index", help="rebuild the registry and DuckDB query layer, "
        "then verify SQL vs Python leaderboard parity"
    )
    _config_flags(index)
    index.add_argument(
        "--db-out",
        type=str,
        default=str(cfg.db_path),
        help=f"DuckDB file (config default: {cfg.db_path})",
    )
    index.add_argument(
        "--results-glob",
        type=str,
        default=cfg.results_glob,
        help=f"canonical results glob under the results dir (config default: {cfg.results_glob})",
    )
    index.add_argument(
        "--no-parity",
        action="store_false",
        dest="parity",
        default=cfg.parity,
        help="skip the SQL-vs-Python leaderboard parity gate",
    )
    index.set_defaults(func=lambda args: command_index(args, cfg))

    train = subparsers.add_parser(
        "train", help="train and persist the learned solvers"
    )
    _config_flags(train)
    train.add_argument(
        "--manifest",
        action="append",
        default=list(cfg.manifests),
        help="manifest file (repeatable; defaults to config 'manifests' "
        "or all under the instances dir)",
    )
    train.add_argument(
        "--output",
        type=str,
        default=str(cfg.model_out),
        help=f"primary model output (config default: {cfg.model_out})",
    )
    train.add_argument(
        "--n-estimators", type=int, default=cfg.n_estimators,
        help=f"trees in tree-based learned solvers (config default: {cfg.n_estimators})",
    )
    train.add_argument(
        "--max-depth", type=int, default=cfg.max_depth,
        help=f"max tree depth (config default: {cfg.max_depth})",
    )
    train.add_argument(
        "--random-state", type=int, default=cfg.random_state,
        help=f"training seed (config default: {cfg.random_state})",
    )
    train.set_defaults(func=lambda args: command_train(args, cfg))

    features = subparsers.add_parser(
        "features", help="compute and store per-instance feature records"
    )
    _config_flags(features)
    features.add_argument(
        "--manifest",
        action="append",
        default=list(cfg.manifests),
        help="manifest file (repeatable; defaults to config 'manifests' "
        "or all under the instances dir)",
    )
    features.set_defaults(func=lambda args: command_features(args, cfg))

    audit = subparsers.add_parser(
        "audit", help="coverage audit of instances and feature sets"
    )
    _config_flags(audit)
    audit.add_argument(
        "--out",
        type=str,
        default=str(cfg.audit_out),
        help=f"audit markdown output (config default: {cfg.audit_out})",
    )
    audit.set_defaults(func=lambda args: command_audit(args, cfg))

    portfolio = subparsers.add_parser(
        "portfolio", help="virtual-best and regret report over benchmark runs"
    )
    _config_flags(portfolio)
    portfolio.add_argument(
        "--manifest",
        action="append",
        default=list(cfg.manifests),
        help="manifest file (repeatable; defaults to config 'manifests' "
        "or all under the instances dir)",
    )
    portfolio.add_argument(
        "--runs",
        type=str,
        default=str(cfg.benchmark_out),
        help=f"benchmark results file (config default: {cfg.benchmark_out})",
    )
    portfolio.add_argument(
        "--out",
        type=str,
        default=str(cfg.portfolio_out),
        help=f"portfolio output (config default: {cfg.portfolio_out})",
    )
    portfolio.set_defaults(func=lambda args: command_portfolio(args, cfg))

    export_aslib = subparsers.add_parser(
        "export-aslib", help="export the corpus as an ASlib scenario"
    )
    _config_flags(export_aslib)
    export_aslib.add_argument(
        "--runs",
        type=str,
        default=str(cfg.benchmark_out),
        help=f"benchmark results file (config default: {cfg.benchmark_out})",
    )
    export_aslib.set_defaults(func=lambda args: command_export_aslib(args, cfg))

    all_command = subparsers.add_parser(
        "all", help="run the configured pipeline steps "
        "(generate, train, benchmark, features, index, report)"
    )
    _config_flags(all_command)
    all_command.add_argument(
        "--output-dir",
        type=str,
        default=str(cfg.instances_dir),
        help=f"manifest output directory (default: {cfg.instances_dir})",
    )
    all_command.add_argument(
        "--manifest",
        action="append",
        default=list(cfg.manifests),
        help="manifest file (repeatable; defaults to config 'manifests' "
        "or all under the instances dir)",
    )
    all_command.add_argument(
        "--output",
        type=str,
        default=str(cfg.model_out),
        help=f"primary model output (config default: {cfg.model_out})",
    )
    all_command.add_argument(
        "--budget-seconds", type=float, default=cfg.budget_seconds,
        help="per-solver time budget",
    )
    all_command.add_argument(
        "--timeout-seconds",
        type=float,
        default=cfg.timeout_seconds,
        help="hard wall-clock cap per (instance, solver) cell",
    )
    all_command.add_argument(
        "--seed",
        type=int,
        default=cfg.seed,
        help="solver random seed (config default: {})".format(cfg.seed),
    )
    all_command.add_argument(
        "--memory-bytes",
        type=int,
        default=cfg.memory_bytes,
        help="process-tree RSS cap in bytes (config default: "
        f"{cfg.memory_bytes})",
    )
    all_command.add_argument(
        "--results-glob",
        type=str,
        default=cfg.results_glob,
        help="canonical results glob for the index step",
    )
    all_command.add_argument(
        "--db-out",
        type=str,
        default=str(cfg.db_path),
        help=f"DuckDB file for the index step (config default: {cfg.db_path})",
    )
    all_command.add_argument(
        "--no-parity",
        action="store_false",
        dest="parity",
        default=cfg.parity,
        help="skip the SQL-vs-Python leaderboard parity gate",
    )
    all_command.add_argument(
        "--out",
        type=str,
        default=str(cfg.benchmark_out),
        help=f"results file (config default: {cfg.benchmark_out})",
    )
    all_command.add_argument(
        "--n-estimators", type=int, default=cfg.n_estimators
    )
    all_command.add_argument("--max-depth", type=int, default=cfg.max_depth)
    all_command.add_argument("--random-state", type=int, default=cfg.random_state)
    all_command.set_defaults(func=lambda args: command_all(args, cfg))
    return parser


def _pre_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH),
        help="path to the TOML config file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = _pre_parser()
    pre_ns, _ = pre.parse_known_args(argv)
    try:
        cfg = load_config(pre_ns.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser = build_parser(cfg)
    args = parser.parse_args(argv)
    args.config = cfg
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
