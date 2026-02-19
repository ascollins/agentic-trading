"""Backward-compat re-export — canonical location: ``agentic_trading.bus.bus``.

Will be removed in PR 16.
"""

from agentic_trading.bus.bus import *  # noqa: F401, F403
from agentic_trading.bus.bus import create_event_bus  # noqa: F811

__all__ = ["create_event_bus"]
