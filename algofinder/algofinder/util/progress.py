"""Lightweight progress logging for long-running pipelines."""

from __future__ import annotations

from time import perf_counter


def log(message: str) -> None:
    """Print a timestamped status line."""
    print(message, flush=True)


def log_phase(title: str) -> None:
    log(f"\n=== {title} ===")


class ProgressTracker:
    """Track and print incremental progress for a fixed-size workload."""

    def __init__(
        self,
        total: int,
        label: str,
        *,
        report_every: int = 1,
    ) -> None:
        if total < 1:
            raise ValueError("total must be positive")
        self.total = total
        self.label = label
        self.report_every = max(1, report_every)
        self.completed = 0
        self.started_at = perf_counter()
        log(f"[{self.label}] starting ({self.total} items)")

    def step(self, detail: str = "") -> None:
        self.completed += 1
        if (
            self.completed == 1
            or self.completed == self.total
            or self.completed % self.report_every == 0
        ):
            elapsed = perf_counter() - self.started_at
            rate = self.completed / elapsed if elapsed > 0 else 0.0
            remaining = (self.total - self.completed) / rate if rate > 0 else 0.0
            suffix = f" | {detail}" if detail else ""
            log(
                f"[{self.label}] {self.completed}/{self.total} "
                f"({100.0 * self.completed / self.total:.1f}%) "
                f"elapsed={elapsed:.1f}s eta={remaining:.1f}s{suffix}"
            )

    def finish(self, detail: str = "") -> None:
        elapsed = perf_counter() - self.started_at
        suffix = f" | {detail}" if detail else ""
        log(
            f"[{self.label}] done {self.completed}/{self.total} "
            f"in {elapsed:.1f}s{suffix}"
        )
