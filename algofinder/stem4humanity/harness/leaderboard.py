"""Leaderboard rendering: aggregate benchmark runs into markdown tables."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable

from stem4humanity.harness.benchmark import BenchmarkRun


def _group_key(run: BenchmarkRun) -> tuple[str, str]:
    return (run.problem, run.subproblem)


def summarize(runs: Iterable[BenchmarkRun]) -> list[dict[str, object]]:
    """Aggregate runs into one leaderboard row per (problem, subproblem, solver)."""
    groups: dict[tuple[str, str], dict[str, dict[str, list[BenchmarkRun]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for run in runs:
        problem_key = _group_key(run)
        groups[problem_key][run.solver][run.status].append(run)

    rows: list[dict[str, object]] = []
    for problem_key in sorted(groups):
        problem, subproblem = problem_key
        solver_groups = groups[problem_key]
        for solver in sorted(solver_groups):
            by_status = solver_groups[solver]
            ok_runs = by_status.get("ok", [])
            gaps = [run.gap_percent for run in ok_runs if run.gap_percent is not None]
            walls = [run.wall_seconds for run in ok_runs if run.wall_seconds]
            best_instances = {
                run.instance for run in ok_runs if run.gap_percent is not None and run.gap_percent <= 0.0
            }
            exact = any(run.exact for run in ok_runs)
            tags = sorted({tag for run in ok_runs for tag in run.tags})
            rows.append(
                {
                    "problem": problem,
                    "subproblem": subproblem,
                    "solver": solver,
                    "solver_display": ok_runs[0].solver_display if ok_runs else solver,
                    "tags": tags,
                    "exact": exact,
                    "ok": len(ok_runs),
                    "skipped": len(by_status.get("skipped", [])),
                    "error": len(by_status.get("error", [])),
                    "invalid": len(by_status.get("invalid", [])),
                    "timeout": len(by_status.get("timeout", [])),
                    "mean_gap_percent": mean(gaps) if gaps else None,
                    "mean_wall_seconds": mean(walls) if walls else None,
                    "best_on": len(best_instances),
                }
            )
    return rows


def render_leaderboard(runs: Iterable[BenchmarkRun]) -> str:
    """Render aggregate rows as markdown, one table per problem:subproblem."""
    rows = summarize(runs)
    by_problem: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_problem[(row["problem"], row["subproblem"])].append(row)

    sections: list[str] = ["# Leaderboard", ""]
    for (problem, subproblem) in sorted(by_problem):
        section_rows = by_problem[(problem, subproblem)]
        section_rows.sort(key=lambda row: (row["mean_gap_percent"] is None, row["mean_gap_percent"]))
        sections.append(f"## {problem}:{subproblem}")
        sections.append("")
        sections.append(
            "| Solver | OK | Mean gap % | Best on | Exact | Mean s | Skipped | Error | Timeout | Invalid |"
        )
        sections.append("|---|---|---|---|---|---|---|---|---|---|")
        for row in section_rows:
            gap = (
                f"{row['mean_gap_percent']:.2f}"
                if row["mean_gap_percent"] is not None
                else "-"
            )
            wall = (
                f"{row['mean_wall_seconds']:.3f}"
                if row["mean_wall_seconds"] is not None
                else "-"
            )
            exact = "yes" if row["exact"] else "no"
            sections.append(
                f"| {row['solver']} | {row['ok']} | {gap} | {row['best_on']} "
                f"| {exact} | {wall} | {row['skipped']} | {row['error']} "
                f"| {row['timeout']} | {row['invalid']} |"
            )
        sections.append("")

    summary = render_summary(rows)
    sections.extend(["## Summary", "", summary, ""])
    return "\n".join(sections)


def render_summary(rows: list[dict[str, object]]) -> str:
    """Per-problem totals: instances, solver-runs, and status counts."""
    total_ok = sum(row["ok"] for row in rows)
    total_cells = sum(
        row["ok"] + row["skipped"] + row["error"] + row["invalid"] + row["timeout"]
        for row in rows
    )
    lines = [
        f"- problems with runs: {len({(r['problem'], r['subproblem']) for r in rows})}",
        f"- solver cells: {total_cells}",
        f"- ok: {total_ok}",
        f"- skipped: {sum(row['skipped'] for row in rows)}",
        f"- error: {sum(row['error'] for row in rows)}",
        f"- invalid: {sum(row['invalid'] for row in rows)}",
        f"- timeout: {sum(row['timeout'] for row in rows)}",
    ]
    return "\n".join(lines)
