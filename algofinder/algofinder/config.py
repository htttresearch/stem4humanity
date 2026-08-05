"""Configuration and control layer for the algofinder harness.

A single TOML file (``config/config.toml``) selects the active profile
(``dev`` or ``prod``) and controls every runtime knob: trace behaviour,
paths, benchmark budgets, splits and solver selection, ML training
hyperparameters, and pipeline steps.  Every choice is explained inline
in the file itself.

Precedence: built-in defaults < config file < command-line flags.  The
runner installs the resolved config values as argparse defaults, so any
explicit flag overrides the file without extra plumbing.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on python < 3.11
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.toml"

MODES = ("dev", "prod")
TRACE_PROFILES = ("full", "decisions", "summary")
SPLITS = ("train", "test")
PIPELINE_STEPS = ("generate", "train", "benchmark", "index", "report")

DEFAULT_PATHS = {
    "instances_dir": "public/data",
    "results_dir": "public/results",
    "models_dir": "public/models",
    "sessions_dir": "private/sessions",
    "benchmark_out": "public/results/benchmark.json",
    "leaderboard_out": "public/results/leaderboard.md",
    "model_out": "public/models/candidate_ranker.joblib",
    "registry_dir": "private/registry",
    "db_dir": "private/db",
    "db_path": "private/db/analytics.duckdb",
    "cache_dir": "private/db/cache",
}

DEFAULT_PROFILES = {
    "dev": {"mode": "dev", "trace_profile": "full"},
    "prod": {"mode": "prod", "trace_profile": "summary"},
}


class ConfigError(ValueError):
    """Raised when the config file is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class Config:
    profile: str
    mode: str
    trace_profile: str
    trace_event_limit: int | None
    trace_byte_limit: int | None
    trace_fail_on_limit: bool
    session_label: str
    instances_dir: Path
    results_dir: Path
    models_dir: Path
    sessions_dir: Path
    benchmark_out: Path
    leaderboard_out: Path
    model_out: Path
    registry_dir: Path
    db_dir: Path
    db_path: Path
    cache_dir: Path
    results_glob: str
    parity: bool
    budget_seconds: float | None
    timeout_seconds: float
    seed: int | None
    memory_bytes: int | None
    manifests: tuple[str, ...]
    splits: tuple[str, ...]
    solver_include: tuple[str, ...]
    solver_exclude: tuple[str, ...]
    best_known_budget_seconds: float | None
    verbose: bool
    pipeline_steps: tuple[str, ...]
    n_estimators: int
    max_depth: int
    random_state: int


def _missing(key: str) -> ConfigError:
    return ConfigError(f"config: missing required key {key!r}")


def _bad(key: str, expected: str, value: Any) -> ConfigError:
    return ConfigError(f"config: {key!r} must be {expected}, got {value!r}")


def _take(
    table: dict[str, Any],
    key: str,
    *,
    required: bool = False,
    default: Any = None,
    kind: type | None = None,
    choices: tuple[str, ...] | None = None,
) -> Any:
    if key not in table:
        if required:
            raise _missing(key)
        return default
    value = table[key]
    if kind is not None:
        if kind is int and (not isinstance(value, int) or isinstance(value, bool)):
            raise _bad(key, "an integer", value)
        elif kind is float and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            raise _bad(key, "a number", value)
        elif kind is bool and not isinstance(value, bool):
            raise _bad(key, "true or false", value)
        elif kind is str and not isinstance(value, str):
            raise _bad(key, "a string", value)
        elif kind is list and not isinstance(value, list):
            raise _bad(key, "a list", value)
    if choices is not None and value not in choices:
        raise ConfigError(f"config: {key!r} must be one of {choices}, got {value!r}")
    return value


def _take_str_list(
    table: dict[str, Any], key: str, *, default: tuple[str, ...] = ()
) -> tuple[str, ...]:
    value = _take(table, key, default=list(default), kind=list)
    for item in value:
        if not isinstance(item, str):
            raise _bad(key, "a list of strings", value)
    return tuple(value)


def _take_nonnegative_int(table: dict[str, Any], key: str, *, default: int) -> int | None:
    value = _take(table, key, default=default, kind=int)
    if value < 0:
        raise ConfigError(f"config: {key!r} must be >= 0, got {value!r}")
    return value if value > 0 else None


def _rel(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _require_table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise _bad(key, "a table (e.g. [{}])".format(key), value)
    return value


def resolve(
    raw: dict[str, Any],
    base: Path,
    profile_override: str | None = None,
) -> Config:
    """Resolve a parsed TOML document into a validated :class:`Config`."""
    allowed_top = {"profile", "profiles", "paths", "instances", "benchmark", "pipeline", "runtime", "ml", "store"}
    unknown = set(raw) - allowed_top
    if unknown:
        raise ConfigError(f"config: unknown section(s) {sorted(unknown)}")

    profiles = _require_table(raw, "profiles")
    profile = _take(raw, "profile", required=True, kind=str)
    if profile_override is not None:
        profile = profile_override
    if profile not in profiles:
        raise ConfigError(
            f"config: profile {profile!r} not found in [profiles] "
            f"(available: {sorted(profiles)})"
        )
    selected = profiles[profile]
    if not isinstance(selected, dict):
        raise _bad(profile, "a table", selected)

    merged = {**DEFAULT_PROFILES.get(profile, {}), **selected}
    mode = _take(merged, "mode", required=True, kind=str, choices=MODES)
    trace_profile = _take(
        merged, "trace_profile", required=True, kind=str, choices=TRACE_PROFILES
    )
    trace_event_limit = _take_nonnegative_int(
        merged, "trace_event_limit", default=0
    )
    trace_byte_limit = _take_nonnegative_int(merged, "trace_byte_limit", default=0)
    trace_fail_on_limit = _take(merged, "trace_fail_on_limit", default=True, kind=bool)
    session_label = _take(merged, "session_label", default=mode, kind=str)

    paths = {**DEFAULT_PATHS, **_require_table(raw, "paths")}
    for key in DEFAULT_PATHS:
        paths[key] = _rel(base, _take(paths, key, kind=str))

    benchmark = _require_table(raw, "benchmark")
    budget_seconds = _take(benchmark, "budget_seconds", default=0, kind=float)
    if budget_seconds < 0:
        raise ConfigError(f"config: 'budget_seconds' must be >= 0, got {budget_seconds!r}")
    budget_seconds = budget_seconds or None
    timeout_seconds = _take(benchmark, "timeout_seconds", default=30.0, kind=float)
    if timeout_seconds <= 0:
        raise ConfigError(f"config: 'timeout_seconds' must be > 0, got {timeout_seconds!r}")
    seed = _take_nonnegative_int(benchmark, "seed", default=0)
    memory_bytes = _take_nonnegative_int(benchmark, "memory_bytes", default=0)
    manifests = tuple(
        str(_rel(base, item))
        for item in _take_str_list(benchmark, "manifests")
    )
    splits = _take_str_list(benchmark, "splits", default=("test",))
    for split in splits:
        if split not in SPLITS:
            raise ConfigError(f"config: 'splits' entries must be one of {SPLITS}, got {split!r}")
    solver_include = _take_str_list(benchmark, "solver_include")
    solver_exclude = _take_str_list(benchmark, "solver_exclude")

    instances = _require_table(raw, "instances")
    best_known_budget_seconds = _take(
        instances, "best_known_budget_seconds", default=60.0, kind=float
    )
    if best_known_budget_seconds < 0:
        raise ConfigError(
            f"config: 'best_known_budget_seconds' must be >= 0, "
            f"got {best_known_budget_seconds!r}"
        )
    best_known_budget_seconds = best_known_budget_seconds or None

    pipeline = _require_table(raw, "pipeline")
    pipeline_steps = _take_str_list(
        pipeline, "steps", default=PIPELINE_STEPS
    )
    for step in pipeline_steps:
        if step not in PIPELINE_STEPS:
            raise ConfigError(
                f"config: 'pipeline.steps' entries must be one of {PIPELINE_STEPS}, "
                f"got {step!r}"
            )

    runtime = _require_table(raw, "runtime")
    verbose = _take(runtime, "verbose", default=True, kind=bool)

    store = _require_table(raw, "store")
    results_glob = _take(store, "results_glob", default="*.json", kind=str)
    parity = _take(store, "parity", default=True, kind=bool)

    ml = _require_table(raw, "ml")
    n_estimators = _take(ml, "n_estimators", default=250, kind=int)
    if n_estimators <= 0:
        raise ConfigError(f"config: 'n_estimators' must be > 0, got {n_estimators!r}")
    max_depth = _take(ml, "max_depth", default=10, kind=int)
    random_state = _take(ml, "random_state", default=0, kind=int)

    return Config(
        profile=profile,
        mode=mode,
        trace_profile=trace_profile,
        trace_event_limit=trace_event_limit,
        trace_byte_limit=trace_byte_limit,
        trace_fail_on_limit=trace_fail_on_limit,
        session_label=session_label,
        instances_dir=paths["instances_dir"],
        results_dir=paths["results_dir"],
        models_dir=paths["models_dir"],
        sessions_dir=paths["sessions_dir"],
        benchmark_out=paths["benchmark_out"],
        leaderboard_out=paths["leaderboard_out"],
        model_out=paths["model_out"],
        registry_dir=paths["registry_dir"],
        db_dir=paths["db_dir"],
        db_path=paths["db_path"],
        cache_dir=paths["cache_dir"],
        results_glob=results_glob,
        parity=parity,
        budget_seconds=budget_seconds,
        timeout_seconds=timeout_seconds,
        seed=seed,
        memory_bytes=memory_bytes,
        manifests=manifests,
        splits=splits,
        solver_include=solver_include,
        solver_exclude=solver_exclude,
        best_known_budget_seconds=best_known_budget_seconds,
        verbose=verbose,
        pipeline_steps=pipeline_steps,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )


def load_config(path: str | Path | None = None, *, profile: str | None = None) -> Config:
    """Load and validate a TOML config file.

    ``path`` defaults to ``config/config.toml`` next to the project root.
    Relative paths inside the file resolve against the project root when
    the default config file is used, and against the config file's own
    directory for any other config file (so scratch configs can keep
    their outputs next to themselves).  ``profile`` overrides the
    ``profile`` key of the file.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config: invalid TOML in {config_path}: {exc}") from exc
    base = (
        PROJECT_ROOT
        if config_path.resolve() == DEFAULT_CONFIG_PATH.resolve()
        else config_path.parent
    )
    return resolve(raw, base, profile_override=profile)


__all__ = [
    "Config",
    "ConfigError",
    "DEFAULT_CONFIG_PATH",
    "MODES",
    "PIPELINE_STEPS",
    "SPLITS",
    "TRACE_PROFILES",
    "load_config",
    "resolve",
]
