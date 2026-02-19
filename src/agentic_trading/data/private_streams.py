"""Backward-compat re-export — canonical location: ``agentic_trading.intelligence.private_streams``.

Will be removed in PR 16.
"""

from agentic_trading.intelligence.private_streams import *  # noqa: F401, F403
from agentic_trading.intelligence.private_streams import PrivateStreamManager  # noqa: F811

__all__ = ["PrivateStreamManager"]
