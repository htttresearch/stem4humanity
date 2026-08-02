"""Benchmark harness: run every applicable solver on every instance.

This is the "running env" of the plan: each algorithm is run against
instances of its problem, both the general class and the subclasses,
with feasibility verification, per-solver time budgets, and a hard
wall-clock timeout, producing deterministic machine-readable records.

The harness owns the cell lifecycle. In dev mode a
:class:`DevSession` reserves run IDs and writes ``invocation.json``
before the worker starts and the authoritative ``outcome.json`` after
it returns or is killed. The worker only appends its event stream.
"""

from __future__ import annotations

import inspect
import json
import multiprocessing as mp
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Literal

from stem4humanity.problems.base import Instance
from stem4humanity.problems.registry import get_problem
from stem4humanity.solvers.base import InapplicableError, all_solvers, solvers_for
from stem4humanity.trace.artifacts import ArtifactStore
from stem4humanity.trace.contract import SolveContext
from stem4humanity.trace.model import TraceConfig, TraceSummary
from stem4humanity.trace.recorder import TraceRecorder
from stem4humanity.trace.session import DevSession
from stem4humanity.trace.serialize import sha256_hex, strict_dumps

Status = Literal["ok", "skipped", "error", "invalid", "timeout"]


@dataclass
class BenchmarkRun:
    """One (instance, solver) execution record, JSON-safe."""

    instance: str
    problem: str
    subproblem: str
    family: str
    split: str
    solver: str
    solver_display: str
    tags: list[str]
    status: Status
    cost: float | None = None
    best_known: float | None = None
    gap_percent: float | None = None
    exact: bool | None = None
    wall_seconds: float | None = None
    error: str | None = None
    solution: list[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    run_id: str | None = None
    trace_rel: str | None = None
    trace_summary: dict[str, Any] | None = None

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def _sanitize(value: Any) -> Any:
    """Convert numpy scalars/arrays to JSON-native Python values."""
    if hasattr(value, "item") and hasattr(value, "shape") and value.shape == ():
        return value.item()
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def _default_config(solver_cls: type) -> dict[str, Any]:
    """Constructor defaults from the class signature (no instantiation)."""
    try:
        signature = inspect.signature(solver_cls.__init__)
    except (TypeError, ValueError):
        return {}
    config: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name in ("self", "args", "kwargs"):
            continue
        if parameter.default is not inspect.Parameter.empty:
            config[name] = parameter.default
    return config


def _run_cell(
    solver_id: str,
    state: Any,
    budget_seconds: float | None,
    trace_config: TraceConfig,
    session_dir: str | None,
    run_id: str,
    session_id: str | None,
    instance_sha: str,
    state_sha: str,
    problem_id: str,
) -> dict[str, Any]:
    """Solve one instance state with one solver; return a JSON-safe outcome.

    Designed to be pickled into a worker process: ``state`` is any
    ``ProblemState`` (numpy-backed classes pickle cleanly) and the
    returned mapping contains only plain Python types. Tracing is
    configured explicitly; nothing is inherited from a parent process.
    """
    import stem4humanity.solvers.registry  # noqa: F401  (re-register in workers)

    started = perf_counter()
    outcome: dict[str, Any] = {
        "status": None,
        "cost": None,
        "exact": None,
        "wall_seconds": None,
        "error": None,
        "solution": None,
        "metadata": {},
        "trace": None,
    }
    recorder: TraceRecorder | None = None
    store: ArtifactStore | None = None
    try:
        if trace_config.enabled and session_dir:
            store = ArtifactStore(Path(session_dir))
            recorder = TraceRecorder(
                Path(session_dir) / "runs" / run_id / "events.jsonl.partial",
                session_id=session_id,
                run_id=run_id,
                config=trace_config,
                store=store,
            )
    except Exception as exc:
        outcome["status"] = "error"
        outcome["error"] = f"trace setup failed: {type(exc).__name__}: {exc}"
        return outcome

    solver = all_solvers()[solver_id]()
    context = SolveContext(
        run_id=run_id, trace=recorder, budget_seconds=budget_seconds
    )
    trace = context.trace
    contract = getattr(solver, "trace_contract", None)
    coverage_scope = (
        contract.coverage_scope if contract is not None else None
    )
    try:
        if trace.enabled:
            trace.span(
                "run.start",
                solver_config=solver.config() or _default_config(type(solver)),
                state_sha=state_sha,
                instance_sha=instance_sha,
            )
            store.publish_json("problem-states", _state_encoding(problem_id, state))
            trace.span("problem.snapshot", state_sha=state_sha)

        solve_kwargs: dict[str, Any] = {"budget_seconds": budget_seconds}
        if "context" in inspect.signature(solver.solve).parameters:
            solve_kwargs["context"] = context
        result = solver.solve(state, **solve_kwargs)
        if trace.enabled:
            trace.span(
                "solve.end",
                solver_cost=float(result.cost),
                exact=bool(result.exact),
            )
    except InapplicableError as exc:
        if trace.enabled:
            trace.event(
                "lifecycle", "run.skipped", outcome={"reason": str(exc) or exc.__class__.__name__}
            )
        outcome["status"] = "skipped"
        outcome["error"] = str(exc) or exc.__class__.__name__
        return _finish(outcome, recorder, complete=True, coverage_scope=coverage_scope)
    except Exception as exc:  # solver crashed on this instance
        if trace.enabled:
            trace.event(
                "lifecycle",
                "run.error",
                outcome={
                    "class": type(exc).__name__,
                    "message": str(exc),
                    "phase": "solve",
                },
            )
        outcome["status"] = "error"
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        return _finish(outcome, recorder, complete=True, coverage_scope=coverage_scope)

    outcome["wall_seconds"] = perf_counter() - started
    if trace.enabled:
        trace.span("verify.start")
    if not state.verify(result.solution):
        if trace.enabled:
            trace.event("lifecycle", "run.invalid", outcome={"reason": "infeasible"})
        outcome["status"] = "invalid"
        outcome["error"] = "solver returned an infeasible solution"
        return _finish(outcome, recorder, complete=True, coverage_scope=coverage_scope)
    try:
        cost = float(state.objective_value(result.solution))
    except Exception as exc:
        if trace.enabled:
            trace.event(
                "lifecycle",
                "run.invalid",
                outcome={"reason": f"objective failed: {exc}"},
            )
        outcome["status"] = "invalid"
        outcome["error"] = f"objective computation failed: {exc}"
        return _finish(outcome, recorder, complete=True, coverage_scope=coverage_scope)

    outcome["status"] = "ok"
    outcome["cost"] = cost
    outcome["cost_solver"] = float(result.cost)
    outcome["exact"] = bool(result.exact)
    outcome["solution"] = _sanitize(result.solution)
    outcome["metadata"] = _sanitize(result.metadata)
    if trace.enabled:
        trace.span(
            "verify.end",
            harness_cost=cost,
            solver_cost=float(result.cost),
            exact=bool(result.exact),
            wall_seconds=outcome["wall_seconds"],
        )
        trace.span("run.result", status="ok", cost=cost, exact=outcome["exact"])
    return _finish(outcome, recorder, complete=True, coverage_scope=coverage_scope)


def _state_encoding(problem_id: str, state: Any) -> dict[str, Any]:
    """Immutable solver-visible state snapshot via the problem codec."""
    from stem4humanity.trace.codec import get_codec

    return get_codec(problem_id).encode_state(state)


def _finish(
    outcome: dict[str, Any],
    recorder: TraceRecorder | None,
    *,
    complete: bool,
    coverage_scope: str | None = None,
) -> dict[str, Any]:
    """Close the recorder (if any) and attach the trace summary."""
    if recorder is not None:
        summary = recorder.close(complete=complete).to_mapping()
        if coverage_scope is not None:
            summary["coverage_scope"] = coverage_scope
        outcome["trace"] = summary
    return outcome


def _instance_sha(instance: Instance) -> str:
    return sha256_hex(instance.to_mapping())


def _build_run(
    instance: Instance,
    solver_cls: type,
    outcome: dict[str, Any],
    invocation: Any = None,
) -> BenchmarkRun:
    problem = get_problem(instance.problem)
    run = BenchmarkRun(
        instance=instance.name,
        problem=instance.problem,
        subproblem=instance.subproblem,
        family=instance.family,
        split=instance.split,
        solver=solver_cls.id,
        solver_display=solver_cls.display,
        tags=sorted(solver_cls.tags),
        status=outcome["status"],
        best_known=instance.best_known,
        error=outcome["error"],
        metadata=outcome["metadata"],
        session_id=invocation.session_id if invocation else None,
        run_id=invocation.run_id if invocation else None,
        trace_rel=f"runs/{invocation.run_id}/events.jsonl" if invocation else None,
        trace_summary=outcome["trace"],
    )
    if outcome["cost"] is not None:
        run.cost = outcome["cost"]
        run.exact = outcome["exact"]
        run.wall_seconds = outcome["wall_seconds"]
        run.solution = outcome["solution"]
        if instance.best_known is not None:
            run.gap_percent = problem.gap_percent(
                run.cost, float(instance.best_known)
            )
    return run


def summarize_partial(events_path: Path) -> TraceSummary:
    """Parent-side summary of a possibly-killed worker's event stream.

    Reads only complete lines; the worker may have died mid-line. The
    authoritative terminal status comes from the parent-written outcome.
    """
    count = 0
    last_seq = -1
    bytes_read = 0
    with open(events_path, "rb") as handle:
        for raw in handle:
            if not raw.endswith(b"\n"):
                break
            count += 1
            bytes_read += len(raw)
            try:
                event = json.loads(raw)
                last_seq = max(last_seq, int(event.get("seq", -1)))
            except (json.JSONDecodeError, ValueError):
                continue
    return TraceSummary(
        requested_profile="full",
        achieved_profile="partial",
        complete=False,
        event_count=count,
        bytes=bytes_read,
        last_seq=last_seq,
        termination="hard-timeout",
        error="worker terminated; stream left partial",
    )


def run_benchmark(
    instances: Iterable[Instance],
    *,
    budget_seconds: float | None = None,
    timeout_seconds: float | None = None,
    splits: tuple[str, ...] = ("test",),
    session: DevSession | None = None,
) -> list[BenchmarkRun]:
    """Run every applicable solver on every requested instance.

    ``budget_seconds`` is handed to each solver (cooperative early
    termination where implemented); ``timeout_seconds`` is a hard
    wall-clock cap per (instance, solver) cell, enforced by running the
    cell in a worker process that is terminated when it fires.

    With a ``session`` (dev mode), every cell gets a reserved run ID,
    an invocation written before the worker starts, and an authoritative
    outcome written after it returns or is killed. Without one (prod
    mode) no trace infrastructure is touched.
    """
    runs: list[BenchmarkRun] = []
    pool: mp.pool.Pool | None = None
    if timeout_seconds is not None and timeout_seconds > 0:
        pool = mp.get_context("fork").Pool(1)

    trace_config = session.trace_config() if session else TraceConfig(enabled=False)
    session_root = str(session.dir) if session else None

    try:
        for instance in instances:
            if instance.split not in splits:
                continue
            problem = get_problem(instance.problem)
            state = problem.build_state(instance)
            instance_sha = _instance_sha(instance)
            state_sha = _state_fingerprint(instance, state)
            if session:
                session.store.publish_json("instances", instance.to_mapping())
            for solver_cls in solvers_for(instance.problem, instance.subproblem):
                invocation = None
                if session:
                    invocation = session.start_run(
                        instance=instance,
                        instance_sha=instance_sha,
                        state_sha=state_sha,
                        solver_id=solver_cls.id,
                        solver_display=solver_cls.display,
                        solver_config=_default_config(solver_cls),
                        budget_seconds=budget_seconds,
                    )
                if pool is None:
                    outcome = _run_cell(
                        solver_cls.id, state, budget_seconds,
                        trace_config, session_root,
                        invocation.run_id if invocation else "",
                        session.session_id if session else None,
                        instance_sha, state_sha, instance.problem,
                    )
                else:
                    future = pool.apply_async(
                        _run_cell,
                        (
                            solver_cls.id, state, budget_seconds,
                            trace_config, session_root,
                            invocation.run_id if invocation else "",
                            session.session_id if session else None,
                            instance_sha, state_sha, instance.problem,
                        ),
                    )
                    try:
                        outcome = future.get(timeout=timeout_seconds)
                    except mp.TimeoutError:
                        pool.terminate()
                        pool = mp.get_context("fork").Pool(1)
                        outcome = {
                            "status": "timeout",
                            "cost": None,
                            "exact": None,
                            "wall_seconds": float(timeout_seconds),
                            "error": (
                                f"exceeded the {timeout_seconds:g}s "
                                "wall-clock timeout"
                            ),
                            "solution": None,
                            "metadata": {},
                            "trace": None,
                        }
                        if session and invocation is not None:
                            partial = session.run_dir(invocation.run_id) / "events.jsonl.partial"
                            timeout_contract = getattr(
                                all_solvers()[solver_cls.id], "trace_contract", None
                            )
                            outcome["trace"] = (
                                summarize_partial(partial).to_mapping()
                                if partial.exists()
                                else TraceSummary(
                                    requested_profile="full",
                                    achieved_profile="none",
                                    complete=False,
                                    termination="hard-timeout",
                                    error="worker killed before writing events",
                                ).to_mapping()
                            )
                            if timeout_contract is not None:
                                outcome["trace"]["coverage_scope"] = (
                                    timeout_contract.coverage_scope
                                )
                if session and invocation is not None:
                    session.finish_run(
                        invocation,
                        status=outcome["status"],
                        cost_solver=outcome.get("cost_solver"),
                        cost_harness=outcome.get("cost"),
                        exact=outcome.get("exact"),
                        wall_seconds=outcome.get("wall_seconds"),
                        error=outcome.get("error"),
                        solution=outcome.get("solution"),
                        metadata=outcome.get("metadata"),
                        trace=TraceSummary(**outcome["trace"])
                        if outcome.get("trace")
                        else None,
                    )
                runs.append(_build_run(instance, solver_cls, outcome, invocation))
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    return runs


def _state_fingerprint(instance: Instance, state: Any) -> str:
    from stem4humanity.trace.codec import get_codec

    return get_codec(instance.problem).fingerprint_state(state)
