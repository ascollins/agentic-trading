"""Backward-compat re-export — canonical location: ``agentic_trading.bus.redis_streams``.

Will be removed in PR 16.
"""

from agentic_trading.bus.redis_streams import *  # noqa: F401, F403
from agentic_trading.bus.redis_streams import DeadLetter, RedisStreamsBus  # noqa: F811

__all__ = ["DeadLetter", "RedisStreamsBus"]
