"""Reproducible campaign infrastructure and evaluator-driven program search."""

from .campaign import Campaign, CampaignError
from .contracts import AgentSpec, CampaignSpec
from .hir_evolution import EvolutionaryHIRModel
from .ledger import CampaignLedger
from .learning.contracts import AgentEpisode, AgentEvaluation, AgentPolicy, AgentTransition, DistributionProfile, LearningRun, ResearchAttempt
from .learning.recorder import EpisodeRecorder
from .ollama_model import OllamaModelConfig, OllamaResearchModel
from .quality_diversity import QualityDiversitySearch
from .research import AIDETreeResearchAgent, QualityDiversityResearchAgent
from .tsp_hir import HIREdit, TspGenome

__all__ = [
    "AIDETreeResearchAgent",
    "AgentEpisode",
    "AgentEvaluation",
    "AgentPolicy",
    "AgentSpec",
    "AgentTransition",
    "Campaign",
    "CampaignError",
    "CampaignLedger",
    "CampaignSpec",
    "EvolutionaryHIRModel",
    "DistributionProfile",
    "EpisodeRecorder",
    "HIREdit",
    "OllamaModelConfig",
    "OllamaResearchModel",
    "QualityDiversityResearchAgent",
    "QualityDiversitySearch",
    "LearningRun",
    "ResearchAttempt",
    "TspGenome",
]
