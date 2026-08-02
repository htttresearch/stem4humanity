"""Shared test setup: ensure the solver registry is populated.

The harness discovers solvers through registration side effects, so
``stem4humanity.solvers.registry`` must be imported in the parent process
before any benchmark call.
"""

import stem4humanity.solvers.registry  # noqa: F401
