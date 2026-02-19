"""Backward-compatibility shim — see ``agentic_trading.execution.risk.post_trade``.

Will be removed in PR 16.
"""

from agentic_trading.execution.risk.post_trade import *  # noqa: F401, F403
from agentic_trading.execution.risk.post_trade import PostTradeChecker  # noqa: F811

__all__ = ["PostTradeChecker"]
