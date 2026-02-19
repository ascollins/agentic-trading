"""Backward-compat re-export — canonical location: ``agentic_trading.bus.schemas``.

Will be removed in PR 16.
"""

from agentic_trading.bus.schemas import *  # noqa: F401, F403
from agentic_trading.bus.schemas import (  # noqa: F811
    EVENT_TYPE_MAP,
    TOPIC_SCHEMAS,
    get_event_class,
    get_topic_for_event,
)

__all__ = ["TOPIC_SCHEMAS", "EVENT_TYPE_MAP", "get_event_class", "get_topic_for_event"]
