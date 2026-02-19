"""Backward-compat re-export — canonical location: ``agentic_trading.intelligence.analysis.smc_confluence``.

Will be removed in PR 16.
"""

from agentic_trading.intelligence.analysis.smc_confluence import *  # noqa: F401, F403
from agentic_trading.intelligence.analysis.smc_confluence import (  # noqa: F811
    SMCConfluenceResult,
    SMCConfluenceScorer,
    SMCTimeframeSummary,
)

__all__ = ["SMCConfluenceResult", "SMCConfluenceScorer", "SMCTimeframeSummary"]
