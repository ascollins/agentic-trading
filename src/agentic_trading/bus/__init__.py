"""Legacy event bus — topic-string-routed MemoryEventBus & RedisStreamsBus.

This package is the new canonical location for the event bus modules that
were previously at ``agentic_trading.event_bus``.  The old path still works
via backward-compat re-exports (see ``event_bus/__init__.py``).
"""

from agentic_trading.bus.manager import BusManager

__all__ = ["BusManager"]
