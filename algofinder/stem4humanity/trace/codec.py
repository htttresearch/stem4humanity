"""Problem trace codecs: problem-owned instance/state/solution encoding.

The problem, not the solver, defines what a valid solution is and what
immutable solver-visible state means. Each implemented problem family
registers a codec so the harness can snapshot and fingerprint every
cell uniformly. ``fingerprint_state`` is the canonical hash of
``encode_state``; it identifies the immutable problem seen by the
solver.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from stem4humanity.trace.serialize import sha256_hex, to_json_value


@runtime_checkable
class ProblemTraceCodec(Protocol):
    schema_id: str
    schema_version: int

    def encode_instance(self, instance: Any) -> dict[str, Any]: ...
    def encode_state(self, state: Any) -> dict[str, Any]: ...
    def encode_solution(self, solution: Any) -> Any: ...

    def fingerprint_state(self, state: Any) -> str: ...


_CODECS: dict[str, ProblemTraceCodec] = {}


def register_codec(problem_id: str):
    """Class decorator registering a codec for one implemented problem."""

    def decorate(cls: type[ProblemTraceCodec]) -> type[ProblemTraceCodec]:
        instance = cls()
        if problem_id in _CODECS:
            raise ValueError(f"duplicate codec for problem: {problem_id}")
        _CODECS[problem_id] = instance
        return cls

    return decorate


def get_codec(problem_id: str) -> ProblemTraceCodec:
    if problem_id not in _CODECS:
        raise KeyError(
            f"no trace codec for problem {problem_id!r}; "
            "register one with @register_codec"
        )
    return _CODECS[problem_id]


def has_codec(problem_id: str) -> bool:
    return problem_id in _CODECS


class _BaseCodec:
    """Common fingerprint/solution helpers; subclasses define encode_state."""

    schema_version = 1

    def encode_instance(self, instance: Any) -> dict[str, Any]:
        return to_json_value(instance.to_mapping())

    def fingerprint_state(self, state: Any) -> str:
        return sha256_hex(self.encode_state(state))

    def encode_solution(self, solution: Any) -> Any:
        return to_json_value(solution)


def _array(rows: Any) -> list[Any]:
    return to_json_value(rows)


# Import the per-family codecs so their @register_codec decorators run.
from stem4humanity.problems import codecs  # noqa: E402, F401


__all__ = [
    "ProblemTraceCodec",
    "_BaseCodec",
    "get_codec",
    "has_codec",
    "register_codec",
]
