"""Backward-compat re-export — canonical location: ``agentic_trading.signal.portfolio.intent_converter``.

Will be removed in PR 16.
"""

from agentic_trading.signal.portfolio.intent_converter import *  # noqa: F401, F403
from agentic_trading.signal.portfolio.intent_converter import (  # noqa: F811
    build_order_intents,
)

__all__ = ["build_order_intents"]
