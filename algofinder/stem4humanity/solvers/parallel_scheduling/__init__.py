"""Parallel-scheduling solver package: importing this registers all solvers."""

from stem4humanity.solvers.parallel_scheduling import (  # noqa: F401
    cp_list,
    exact,
    hu,
    learned_priority,
    list_scheduling,
)

__all__ = ["cp_list", "exact", "hu", "learned_priority", "list_scheduling"]
