"""Backward-compat re-export — canonical location: ``agentic_trading.intelligence.historical``.

Will be removed in PR 16.
"""

from agentic_trading.intelligence.historical import *  # noqa: F401, F403
from agentic_trading.intelligence.historical import HistoricalDataLoader  # noqa: F811

__all__ = ["HistoricalDataLoader"]
