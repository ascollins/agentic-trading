"""Backward-compat re-export — canonical location: ``agentic_trading.intelligence.analysis.market_context``.

Will be removed in PR 16.
"""

from agentic_trading.intelligence.analysis.market_context import *  # noqa: F401, F403
from agentic_trading.intelligence.analysis.market_context import (  # noqa: F811
    MarketContext,
    assess_macro_regime,
)

__all__ = ["MarketContext", "assess_macro_regime"]
