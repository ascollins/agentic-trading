"""Backward-compat re-export — canonical location: ``agentic_trading.intelligence.candle_builder``.

Will be removed in PR 16.
"""

from agentic_trading.intelligence.candle_builder import *  # noqa: F401, F403
from agentic_trading.intelligence.candle_builder import CandleBuilder  # noqa: F811

__all__ = ["CandleBuilder"]
