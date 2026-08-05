"""Query layer helpers: SQL leaderboard and parity checks (tier 3).

``leaderboard_rows`` reproduces ``harness.leaderboard.summarize`` in SQL
over the ``v_runs`` view, so the existing report and the DB can be
cross-checked: the parity gate for the storage layer.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import duckdb

_RUNS_CTE = """
WITH runs AS (
  SELECT r.*
  FROM (SELECT unnest(runs) AS r FROM read_json_auto({})) x
)
""".strip()

LEADERBOARD_SQL = _RUNS_CTE + """
SELECT
  problem,
  subproblem,
  solver,
  COUNT(*) FILTER (WHERE status = 'ok')                         AS ok,
  COUNT(*) FILTER (WHERE status = 'skipped')                    AS skipped,
  COUNT(*) FILTER (WHERE status = 'unsupported')                AS unsupported,
  COUNT(*) FILTER (WHERE status = 'error')                      AS error,
  COUNT(*) FILTER (WHERE status = 'invalid')                    AS invalid,
  COUNT(*) FILTER (WHERE status = 'timeout')                    AS timeout,
  COUNT(*) FILTER (WHERE status = 'memory_limit')               AS memory_limit,
  AVG(gap_percent) FILTER (WHERE status = 'ok'
                           AND gap_percent IS NOT NULL)         AS mean_gap_percent,
  AVG(wall_seconds) FILTER (WHERE status = 'ok'
                            AND wall_seconds > 0)               AS mean_wall_seconds,
  COUNT(*) FILTER (WHERE status = 'ok' AND gap_percent IS NOT NULL
                   AND gap_percent <= 0.0)                      AS best_on,
  COALESCE(BOOL_OR(exact) FILTER (WHERE status = 'ok'), false)  AS exact
FROM runs
GROUP BY problem, subproblem, solver
ORDER BY problem, subproblem, solver
"""


def _list_literal(paths: Iterable[str]) -> str:
    """SQL array literal of quoted paths."""
    inner = ", ".join(
        f"'{str(path).replace(chr(39), chr(39) * 2)}'" for path in paths
    )
    return f"[{inner}]"


def runs_sql(paths: Iterable[str]) -> str:
    """SQL source that flattens run records out of results files."""
    return _RUNS_CTE.format(_list_literal(paths))


def leaderboard_rows(con: duckdb.DuckDBPyConnection, paths: Iterable[str]) -> list[dict[str, Any]]:
    """Leaderboard rows computed in SQL over the canonical results files."""
    return con.execute(LEADERBOARD_SQL.format(_list_literal(paths))).fetchall()


LEADERBOARD_FIELDS = (
    "problem",
    "subproblem",
    "solver",
    "ok",
    "skipped",
    "unsupported",
    "error",
    "invalid",
    "timeout",
    "memory_limit",
    "mean_gap_percent",
    "mean_wall_seconds",
    "best_on",
    "exact",
)

_FLOAT_FIELDS = ("mean_gap_percent", "mean_wall_seconds")


def parity_compare(
    py_rows: list[dict[str, Any]],
    sql_rows: list[dict[str, Any]],
) -> list[str]:
    """Diff Python-summarized leaderboard rows against SQL rows.

    Counts must match exactly; float means must match within 1e-9.
    Returns a list of human-readable differences (empty = parity).
    """
    by_key = {
        (row["problem"], row["subproblem"], row["solver"]): row for row in py_rows
    }
    sql_by_key = {
        (row["problem"], row["subproblem"], row["solver"]): row for row in sql_rows
    }
    diffs: list[str] = []
    for key in sorted(set(by_key) | set(sql_by_key)):
        py = by_key.get(key)
        sql = sql_by_key.get(key)
        if py is None or sql is None:
            diffs.append(f"{key}: row only on {'python' if py else 'sql'} side")
            continue
        for field in LEADERBOARD_FIELDS[3:]:
            left = py.get(field)
            right = sql.get(field)
            if field in _FLOAT_FIELDS:
                if left is None or right is None:
                    if left != right:
                        diffs.append(f"{key}.{field}: {left!r} vs {right!r}")
                elif not math.isclose(
                    float(left), float(right), rel_tol=1e-9, abs_tol=1e-9
                ):
                    diffs.append(f"{key}.{field}: {left!r} vs {right!r}")
            elif int(left) != int(right):
                diffs.append(f"{key}.{field}: {left!r} vs {right!r}")
    return diffs


def parity_check(
    py_rows: list[dict[str, Any]],
    con: duckdb.DuckDBPyConnection,
    paths: Iterable[str],
) -> list[str]:
    """SQL rows from canonical files, diffed against Python rows."""
    raw = leaderboard_rows(con, paths)
    columns = LEADERBOARD_FIELDS
    sql_rows = [dict(zip(columns, row)) for row in raw]
    return parity_compare(py_rows, sql_rows)


__all__ = [
    "LEADERBOARD_FIELDS",
    "LEADERBOARD_SQL",
    "leaderboard_rows",
    "parity_check",
    "parity_compare",
    "runs_sql",
]
