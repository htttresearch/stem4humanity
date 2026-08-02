"""TSP solver package: importing this module registers all solvers."""

from stem4humanity.solvers.tsp import (  # noqa: F401
    distance_ranked,
    euclidean_local_search,
    general_two_opt,
    held_karp,
    incremental_exact,
    learned_candidate,
)

__all__ = [
    "distance_ranked",
    "euclidean_local_search",
    "general_two_opt",
    "held_karp",
    "incremental_exact",
    "learned_candidate",
]
