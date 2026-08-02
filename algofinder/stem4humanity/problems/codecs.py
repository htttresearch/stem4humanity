"""Problem trace codecs for the seven implemented problem families.

Each codec captures the immutable solver-visible state (including
derived data that matters, e.g. the Euclidean TSP distance matrix) and
owns solution encoding. State snapshots are content-addressed by
``fingerprint_state`` so a session stores each distinct problem once.
"""

from __future__ import annotations

import numpy as np

from stem4humanity.trace.codec import _BaseCodec, register_codec
from stem4humanity.trace.serialize import to_json_value


def _matrices(values: object) -> list[object]:
    return to_json_value(values)


@register_codec("knapsack")
class KnapsackCodec(_BaseCodec):
    schema_id = "knapsack-state/v1"

    def encode_state(self, state: object) -> dict:
        return {
            "weights": _matrices(state.weights),
            "values": _matrices(state.values),
            "capacities": _matrices(state.capacities),
        }


@register_codec("bin-packing")
class BinPackingCodec(_BaseCodec):
    schema_id = "bin-packing-state/v1"

    def encode_state(self, state: object) -> dict:
        return {
            "items": _matrices(state.sizes),
            "capacities": _matrices(state.capacities),
        }


@register_codec("tsp")
class TSPCodec(_BaseCodec):
    schema_id = "tsp-state/v1"

    def encode_state(self, state: object) -> dict:
        if hasattr(state, "points"):
            return {
                "variant": "euclidean",
                "points": _matrices(state.points),
                "distance_matrix": _matrices(state.distances),
            }
        return {"variant": "general", "distances": _matrices(state.distances)}


@register_codec("scheduling")
class SchedulingCodec(_BaseCodec):
    schema_id = "scheduling-state/v1"

    def encode_state(self, state: object) -> dict:
        if hasattr(state, "n_jobs"):
            return {
                "variant": "job-shop",
                "n_jobs": int(state.n_jobs),
                "n_machines": int(state.n_machines),
                "operations": to_json_value(
                    [
                        [list(step) for step in job_ops]
                        for job_ops in state.operations
                    ]
                ),
            }
        return {"variant": "flow-shop-2", "times": _matrices(state.times)}


@register_codec("shortest-path")
class ShortestPathCodec(_BaseCodec):
    schema_id = "shortest-path-state/v1"

    def encode_state(self, state: object) -> dict:
        return {
            "node_count": int(state.node_count),
            "source": int(state.source),
            "arcs": _matrices(state.arcs),
        }


@register_codec("parallel-scheduling")
class ParallelSchedulingCodec(_BaseCodec):
    schema_id = "parallel-scheduling-state/v1"

    def encode_state(self, state: object) -> dict:
        return {
            "task_count": int(state.task_count),
            "machines": int(state.machines),
            "times": _matrices(state.times),
            "predecessors": [list(preds) for preds in state.predecessors],
            "successors": [list(succs) for succs in state.successors],
        }


@register_codec("unit-commitment")
class UnitCommitmentCodec(_BaseCodec):
    schema_id = "unit-commitment-state/v1"

    def encode_state(self, state: object) -> dict:
        if hasattr(state, "prices"):
            return {
                "variant": "storage",
                "periods": int(state.periods),
                "prices": _matrices(state.prices),
                "capacity": float(state.capacity),
                "rate": float(state.rate),
                "efficiency": float(state.efficiency),
                "initial_energy": float(state.initial_energy),
                "target_energy": (
                    None if state.target_energy is None else float(state.target_energy)
                ),
            }
        return {
            "variant": "classic",
            "generators": [
                {
                    "p_min": float(state.p_min[g]),
                    "p_max": float(state.p_max[g]),
                    "c_var": float(state.c_var[g]),
                    "c_start": float(state.c_start[g]),
                    "min_up": int(state.min_up[g]),
                    "min_down": int(state.min_down[g]),
                    "initial_age": int(state.initial_age[g]),
                }
                for g in range(state.generator_count)
            ],
            "demand": _matrices(state.demand),
            "reserve": _matrices(state.reserve),
        }


__all__ = [
    "BinPackingCodec",
    "KnapsackCodec",
    "ParallelSchedulingCodec",
    "SchedulingCodec",
    "ShortestPathCodec",
    "TSPCodec",
    "UnitCommitmentCodec",
]
