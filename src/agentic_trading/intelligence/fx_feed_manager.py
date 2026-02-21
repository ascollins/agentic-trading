"""FX market data feed manager (Phase 0 stub).

In Phase 0 (paper trading), candles are injected manually via
``inject_candle()`` or fed from backtest data.  Phase 1 will add
OANDA streaming API support.

Follows the same lifecycle pattern as :class:`FeedManager`.
"""

from __future__ import annotations

import logging
from typing import Any

from agentic_trading.core.enums import Timeframe
from agentic_trading.core.events import CandleEvent
from agentic_trading.core.interfaces import IEventBus
from agentic_trading.core.models import Candle

logger = logging.getLogger(__name__)


class FXFeedManager:
    """FX market data feed manager.

    Phase 0 stub: candles must be injected externally.  A future
    implementation will stream candles from OANDA / LMAX.

    Parameters
    ----------
    event_bus:
        Event bus for publishing ``CandleEvent`` messages.
    symbols:
        List of FX symbols to manage (e.g. ``["EUR/USD", "GBP/USD"]``).
    base_timeframe:
        Timeframe for raw candle ingestion (default ``1m``).
    candle_topic:
        Event bus topic for candle events (default ``"market.candle"``).
    """

    def __init__(
        self,
        event_bus: IEventBus | None = None,
        symbols: list[str] | None = None,
        base_timeframe: Timeframe = Timeframe.M1,
        candle_topic: str = "market.candle",
    ) -> None:
        self._event_bus = event_bus
        self._symbols = set(symbols or [])
        self._base_timeframe = base_timeframe
        self._candle_topic = candle_topic
        self._running = False

    async def start(self) -> None:
        """Mark the feed manager as running.

        Phase 0: no-op beyond setting the flag.  Phase 1 will open
        WebSocket connections to the broker streaming API.
        """
        self._running = True
        logger.info(
            "FXFeedManager started (phase 0 stub): symbols=%s",
            sorted(self._symbols),
        )

    async def stop(self) -> None:
        """Stop the feed manager."""
        self._running = False
        logger.info("FXFeedManager stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    async def inject_candle(self, candle: Candle) -> None:
        """Manually inject a candle and publish to the event bus.

        This is the primary ingestion path during Phase 0 paper trading.
        Callers (e.g. a polling loop or backtest harness) produce candles
        and push them here.
        """
        if self._event_bus is None:
            return

        event = CandleEvent(
            symbol=candle.symbol,
            exchange=candle.exchange,
            timeframe=candle.timeframe,
            timestamp=candle.timestamp,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            quote_volume=candle.quote_volume,
            trades=candle.trades,
            is_closed=candle.is_closed,
        )
        await self._event_bus.publish(self._candle_topic, event)
