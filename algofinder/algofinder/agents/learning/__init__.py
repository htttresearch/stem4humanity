"""Provenance-first learning layer for AlgoFinder research agents.

The learning package deliberately has no dependency on a model runtime.  It
records exactly what a research agent saw and did, builds datasets only from
eligible evidence, and keeps learned policies as immutable artifacts.
"""

from .contracts import (
    AgentEpisode,
    AgentEvaluation,
    AgentPolicy,
    AgentTransition,
    DistributionProfile,
    LearningRun,
    ResearchAttempt,
)

__all__ = [
    "AgentEpisode",
    "AgentEvaluation",
    "AgentPolicy",
    "AgentTransition",
    "DistributionProfile",
    "LearningRun",
    "ResearchAttempt",
]
