"""Trusted infrastructure for reproducible AlgoFinder research campaigns.

This package deliberately contains no solver-improvement agent or search
methodology. It supplies immutable records, campaign lifecycle management,
candidate workspaces, and evaluator-facing safety boundaries that such
methods can use.
"""

from .campaign import Campaign, CampaignError
from .contracts import AgentSpec, CampaignSpec
from .ledger import CampaignLedger

__all__ = ["AgentSpec", "Campaign", "CampaignError", "CampaignLedger", "CampaignSpec"]
