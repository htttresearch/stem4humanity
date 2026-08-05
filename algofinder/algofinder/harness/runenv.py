"""Immutable environment record and ``environment_id`` (benchmark spec 9.3).

Every run is stamped with the digest of a stable, redacted environment
snapshot: CPU/OS/platform, Python and key library versions, core counts,
and selected environment variables. Secrets and volatile state (PID, temp
paths, hosts) never enter the record.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import socket
import sys
from dataclasses import dataclass

RUNTIME_MODULES = ("numpy", "sklearn", "networkx", "joblib", "tomllib")

REDACTED_ENV = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AZURE_CLIENT_SECRET",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "KAGGLE_KEY",
        "PYTHONBREAKPOINT",
    }
)

KEEP_ENV = ("ALGOFINDER_LKH", "ALGOFINDER_CONCORDE", "ALGOFINDER_GAEAX", "MKL_NUM_THREADS", "OMP_NUM_THREADS")


def _module_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in RUNTIME_MODULES:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "?")
        except ImportError:
            continue
    return versions


def _cpu_identity() -> dict[str, object]:
    identity: dict[str, object] = {"logical_cpus": os.cpu_count() or 0}
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as handle:
            lines = [line for line in handle if line.startswith("model name")]
        if lines:
            identity["cpu_model"] = lines[0].split(":", 1)[1].strip()[:128]
    except OSError:
        pass
    try:
        with open(
            "/sys/devices/system/cpu/physical_package_id",
            encoding="utf-8",
            errors="ignore",
        ) as handle:
            identity["sockets"] = int(handle.read().strip())
    except (OSError, ValueError):
        pass
    return identity


def environment_record() -> dict[str, object]:
    """A stable, redacted snapshot of the running environment."""
    record: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "hostname": socket.gethostname(),
        "env": {name: os.environ.get(name) for name in sorted(KEEP_ENV)},
    }
    record.update(_cpu_identity())
    record["modules"] = _module_versions()
    return record


def environment_id() -> str:
    """Digest of the environment record (immutable per identical environment)."""
    payload = json.dumps(environment_record(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def peak_rss_bytes() -> int:
    """Peak resident set size in bytes for the current process."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def children_peak_rss_bytes() -> int:
    """High-water resident set across all child processes (RUSAGE_CHILDREN)."""
    return int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * 1024


def cpu_seconds() -> float:
    """Process-tree user+system CPU time for the current process."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def children_cpu_seconds() -> float:
    """Cumulative process-tree user+system CPU time of child processes."""
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime + usage.ru_stime)


@dataclass(frozen=True)
class ResourceProbe:
    """Point-in-time resource snapshot for one cell (spec 9.1/9.2)."""

    cpu_seconds: float
    peak_rss_bytes: int

    def delta(self, other: "ResourceProbe") -> "ResourceProbe":
        """Resources attributable to work between probes."""
        return ResourceProbe(
            cpu_seconds=max(0.0, self.cpu_seconds - other.cpu_seconds),
            peak_rss_bytes=max(0, self.peak_rss_bytes - other.peak_rss_bytes),
        )


def probe() -> ResourceProbe:
    """Snapshot current self+child CPU time and peak RSS (spec 9.1/9.2)."""
    return ResourceProbe(
        cpu_seconds=cpu_seconds() + children_cpu_seconds(),
        peak_rss_bytes=max(peak_rss_bytes(), children_peak_rss_bytes()),
    )


__all__ = [
    "KEEP_ENV",
    "REDACTED_ENV",
    "ResourceProbe",
    "cpu_seconds",
    "children_cpu_seconds",
    "children_peak_rss_bytes",
    "environment_id",
    "environment_record",
    "peak_rss_bytes",
    "probe",
]