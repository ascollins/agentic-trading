"""Backward-compat re-export — canonical location: ``agentic_trading.intelligence.feed_manager``.

Will be removed in PR 16.
"""

from agentic_trading.intelligence.feed_manager import *  # noqa: F401, F403
from agentic_trading.intelligence.feed_manager import FeedManager  # noqa: F811

__all__ = ["FeedManager"]
