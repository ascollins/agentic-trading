"""Event bus factory.

Creates the appropriate event bus implementation based on mode.
"""

from __future__ import annotations

from agentic_trading.core.enums import Mode
from .memory_bus import MemoryEventBus
from .redis_streams import RedisStreamsBus


def create_event_bus(
    mode: Mode,
    redis_url: str = "redis://localhost:6379/0",
) -> MemoryEventBus | RedisStreamsBus:
    """Create an event bus for the given mode.

    - BACKTEST: MemoryEventBus (no external deps, deterministic)
    - PAPER/LIVE: RedisStreamsBus (persistent, observable)
    """
    if mode == Mode.BACKTEST:
        return MemoryEventBus()
    return RedisStreamsBus(redis_url=redis_url)
