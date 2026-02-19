"""Backward-compat re-export — canonical location: ``agentic_trading.intelligence.analysis.trade_plan``.

Will be removed in PR 16.
"""

from agentic_trading.intelligence.analysis.trade_plan import *  # noqa: F401, F403
from agentic_trading.intelligence.analysis.trade_plan import (  # noqa: F811
    EntryZone,
    TargetSpec,
    TradePlan,
)

__all__ = ["EntryZone", "TargetSpec", "TradePlan"]
