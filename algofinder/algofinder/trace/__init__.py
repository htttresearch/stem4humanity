"""Universal dev-mode observability: event traces for every solve.

The harness owns lifecycle, identity, ordering, serialization, storage,
and completeness. Solvers and shared kernels emit semantic events
through a ``SolveContext``; production uses a null recorder with zero
I/O. See ``docs/dev-mode-observability-design.md``.
"""

from algofinder.trace.artifacts import ArtifactRef, ArtifactStore
from algofinder.trace.contract import SolveContext, TraceContract, trace_contract
from algofinder.trace.model import (
    Invocation,
    Status,
    TraceConfig,
    TraceProfile,
    TraceSummary,
)
from algofinder.trace.recorder import (
    CATEGORIES,
    ENVELOPE_SCHEMA,
    NullTraceRecorder,
    RecorderError,
    StateRef,
    TraceRecorder,
)
from algofinder.trace.session import DevSession
from algofinder.trace.serialize import sha256_hex, strict_dumps, to_json_value

__all__ = [
    "CATEGORIES",
    "ENVELOPE_SCHEMA",
    "ArtifactRef",
    "ArtifactStore",
    "DevSession",
    "Invocation",
    "NullTraceRecorder",
    "RecorderError",
    "SolveContext",
    "StateRef",
    "Status",
    "TraceConfig",
    "TraceContract",
    "TraceProfile",
    "TraceRecorder",
    "TraceSummary",
    "sha256_hex",
    "strict_dumps",
    "to_json_value",
    "trace_contract",
]
