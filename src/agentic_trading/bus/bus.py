"""Event bus factory.

Creates the appropriate event bus implementation based on mode.
"""

from __future__ import annotations

from collections.abc import Callable

from agentic_trading.core.enums import Mode

from .memory_bus import MemoryEventBus
from .redis_streams import RedisStreamsBus


def create_event_bus(
    mode: Mode,
    redis_url: str = "redis://localhost:6379/0",
    on_handler_error: Callable[
        [str, str, str, Exception], None
    ] | None = None,
) -> MemoryEventBus | RedisStreamsBus:
    """Create an event bus for the given mode.

    - BACKTEST: MemoryEventBus (no external deps, deterministic)
    - PAPER/LIVE: RedisStreamsBus (persistent, observable)

    Args:
        mode: Trading mode (backtest/paper/live).
        redis_url: Redis connection URL (paper/live only).
        on_handler_error: Optional callback ``(topic, group, msg_id, exc)``
            invoked when a handler raises.  Useful for external metrics.
    """
    if mode == Mode.BACKTEST:
        return MemoryEventBus(on_handler_error=on_handler_error)
    return RedisStreamsBus(
        redis_url=redis_url,
        on_handler_error=on_handler_error,
    )
