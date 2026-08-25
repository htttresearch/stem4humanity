"""Command-line access to the rebuildable AlgoFinder evidence warehouse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from algofinder.store.warehouse import (
    DEFAULT_CORPUS_RELATIVE_PATH,
    DEFAULT_DB_RELATIVE_PATH,
    WarehouseError,
    discovery_report,
    ingest,
    overview,
    reproject,
    verify,
)


def _print(value: object) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, default=str))


def _roots(args: argparse.Namespace) -> list[Path]:
    values = args.experiments_root or []
    if not values:
        raise SystemExit("at least one --experiments-root is required")
    return [Path(value).resolve() for value in values]


def _db_path(args: argparse.Namespace) -> Path:
    return Path(args.db_path or DEFAULT_DB_RELATIVE_PATH).resolve()


def _corpus_path(args: argparse.Namespace) -> Path:
    return Path(args.corpus_root or DEFAULT_CORPUS_RELATIVE_PATH).resolve()


def command_discover(args: argparse.Namespace) -> None:
    _print(discovery_report(_roots(args)))


def command_ingest(args: argparse.Namespace) -> None:
    _print(ingest(db_path=_db_path(args), corpus_root=_corpus_path(args), experiments_roots=_roots(args), mode=args.mode, stop_after=args.stop_after))


def command_verify(args: argparse.Namespace) -> None:
    _print(verify(db_path=_db_path(args)))


def command_overview(args: argparse.Namespace) -> None:
    _print(overview(db_path=_db_path(args)))


def command_reproject(args: argparse.Namespace) -> None:
    _print(reproject(db_path=_db_path(args)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algofinder.data", description="Catalog immutable AlgoFinder experiment evidence into a rebuildable DuckDB warehouse.")
    parser.add_argument("--db-path", help=f"defaults to {DEFAULT_DB_RELATIVE_PATH}")
    parser.add_argument("--corpus-root", help=f"defaults to {DEFAULT_CORPUS_RELATIVE_PATH}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    discover_parser = subparsers.add_parser("discover", help="inventory sources without writing")
    discover_parser.add_argument("--experiments-root", action="append", required=True, help="recorded experiment parent/root (repeatable)")
    discover_parser.set_defaults(handler=command_discover)
    ingest_parser = subparsers.add_parser("ingest", help="idempotently import immutable evidence")
    ingest_parser.add_argument("--experiments-root", action="append", required=True, help="recorded experiment parent/root (repeatable)")
    ingest_parser.add_argument("--mode", choices=("dev", "prod"), default="dev")
    ingest_parser.add_argument("--stop-after", type=int, help=argparse.SUPPRESS)
    ingest_parser.set_defaults(handler=command_ingest)
    verify_parser = subparsers.add_parser("verify", help="rehash cataloged source locations")
    verify_parser.set_defaults(handler=command_verify)
    reproject_parser = subparsers.add_parser("reproject", help="rebuild typed tables from the catalog without rescanning sources")
    reproject_parser.set_defaults(handler=command_reproject)
    overview_parser = subparsers.add_parser("overview", help="show compact operational inventory")
    overview_parser.set_defaults(handler=command_overview)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except WarehouseError as exc:
        raise SystemExit(f"data: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover
    main()
