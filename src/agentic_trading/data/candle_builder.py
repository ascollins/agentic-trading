"""Multi-timeframe candle aggregation.

``CandleBuilder`` ingests 1-minute candles and aggregates them into higher
timeframes (5m, 15m, 1h, 4h, 1d).  Completed higher-timeframe candles are
emitted to the event bus as ``CandleEvent`` instances.

Design decisions
----------------
* **Alignment** -- Higher-timeframe candle boundaries are aligned to UTC
  midnight.  A 4h candle opens at 00:00, 04:00, 08:00, ... UTC; a 1d candle
  opens at 00:00 UTC.
* **Partial candles** -- An in-progress (partial) candle is always available
  via ``get_partial`` for strategies that need real-time data, but it is
  *not* emitted to the bus until it closes.
* **Gaps** -- If 1m candles arrive with a gap, the builder will close the
  current aggregation window at the gap boundary and start a new window.
  A warning is logged when this happens.
* **Thread safety** -- Not thread-safe.  Intended to run on a single asyncio
  event loop.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from agentic_trading.core.enums import Exchange, Timeframe
from agentic_trading.core.events import CandleEvent
from agentic_trading.core.interfaces import IEventBus
from agentic_trading.core.models import Candle

logger = logging.getLogger(__name__)

# How many 1m candles fit in each target timeframe.
_TF_MULTIPLIER: dict[Timeframe, int] = {
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.H1: 60,
    Timeframe.H4: 240,
    Timeframe.D1: 1440,
}


def _align_timestamp(ts: datetime, tf: Timeframe) -> datetime:
    """Return the candle-open time for the window that *ts* falls into.

    All alignments are relative to UTC midnight so that e.g. 4h candles
    always open at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC.
    """
    epoch_minutes = int(ts.timestamp()) // 60
    bucket_minutes = tf.minutes
    aligned_minutes = (epoch_minutes // bucket_minutes) * bucket_minutes
    return datetime.fromtimestamp(aligned_minutes * 60, tz=timezone.utc)


class _RunningCandle:
    """Mutable accumulator for an in-progress higher-timeframe candle."""

    __slots__ = (
        "symbol",
        "exchange",
        "timeframe",
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "count",
    )

    def __init__(
        self,
        symbol: str,
        exchange: Exchange,
        timeframe: Timeframe,
        open_time: datetime,
        first_candle: Candle,
    ) -> None:
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.open_time = open_time
        self.open = first_candle.open
        self.high = first_candle.high
        self.low = first_candle.low
        self.close = first_candle.close
        self.volume = first_candle.volume
        self.quote_volume = first_candle.quote_volume
        self.trades = first_candle.trades
        self.count = 1

    def update(self, candle: Candle) -> None:
        """Fold a new 1m candle into this running aggregate."""
        self.high = max(self.high, candle.high)
        self.low = min(self.low, candle.low)
        self.close = candle.close
        self.volume += candle.volume
        self.quote_volume += candle.quote_volume
        self.trades += candle.trades
        self.count += 1

    def to_candle(self, *, is_closed: bool = True) -> Candle:
        return Candle(
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe,
            timestamp=self.open_time,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            quote_volume=self.quote_volume,
            trades=self.trades,
            is_closed=is_closed,
        )

    def to_event(self, *, is_closed: bool = True) -> CandleEvent:
        return CandleEvent(
            source_module="data.candle_builder",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            quote_volume=self.quote_volume,
            trades=self.trades,
            is_closed=is_closed,
            timestamp=self.open_time,
        )


# Maps (symbol, exchange, timeframe) -> _RunningCandle
_StateKey = tuple[str, Exchange, Timeframe]


class CandleBuilder:
    """Aggregates 1-minute candles into higher timeframes.

    Parameters
    ----------
    event_bus:
        Event bus used to emit completed candle events.
    target_timeframes:
        Timeframes to build.  Defaults to all standard higher timeframes.
    emit_topic:
        Event bus topic for emitted ``CandleEvent`` instances.
    """

    def __init__(
        self,
        event_bus: IEventBus,
        target_timeframes: list[Timeframe] | None = None,
        emit_topic: str = "market.candle",
    ) -> None:
        self._event_bus = event_bus
        self._emit_topic = emit_topic

        if target_timeframes is None:
            self._target_timeframes = [
                Timeframe.M5,
                Timeframe.M15,
                Timeframe.H1,
                Timeframe.H4,
                Timeframe.D1,
            ]
        else:
            # Filter out M1 since that is our input granularity.
            self._target_timeframes = [
                tf for tf in target_timeframes if tf != Timeframe.M1
            ]

        self._state: dict[_StateKey, _RunningCandle] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(self, candle: Candle) -> list[CandleEvent]:
        """Ingest a 1-minute candle and emit any completed higher-TF candles.

        Parameters
        ----------
        candle:
            A closed (or partially closed) 1m candle.

        Returns
        -------
        list[CandleEvent]
            List of ``CandleEvent`` instances that were emitted.  This is
            returned for convenience (e.g. testing); events are also
            published to the event bus.
        """
        if candle.timeframe != Timeframe.M1:
            logger.warning(
                "CandleBuilder received non-1m candle (%s %s %s); ignoring.",
                candle.symbol,
                candle.timeframe.value,
                candle.timestamp.isoformat(),
            )
            return []

        emitted: list[CandleEvent] = []

        for tf in self._target_timeframes:
            event = await self._aggregate(candle, tf)
            if event is not None:
                emitted.append(event)

        return emitted

    def get_partial(
        self, symbol: str, exchange: Exchange, timeframe: Timeframe
    ) -> Candle | None:
        """Return the current in-progress candle (not yet closed).

        Returns ``None`` if no data has been accumulated for the key.
        """
        key: _StateKey = (symbol, exchange, timeframe)
        running = self._state.get(key)
        if running is None:
            return None
        return running.to_candle(is_closed=False)

    def reset(self) -> None:
        """Clear all running state.  Useful between backtest runs."""
        self._state.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _aggregate(
        self, candle: Candle, target_tf: Timeframe
    ) -> CandleEvent | None:
        """Try to fold *candle* into the running window for *target_tf*.

        Returns a ``CandleEvent`` if the window just closed, else ``None``.
        """
        key: _StateKey = (candle.symbol, candle.exchange, target_tf)
        aligned = _align_timestamp(candle.timestamp, target_tf)
        running = self._state.get(key)

        # -- No running window yet: start one. ---------------------------------
        if running is None:
            self._state[key] = _RunningCandle(
                symbol=candle.symbol,
                exchange=candle.exchange,
                timeframe=target_tf,
                open_time=aligned,
                first_candle=candle,
            )
            return None

        # -- Same window: fold in. ---------------------------------------------
        if running.open_time == aligned:
            running.update(candle)
            return None

        # -- New window: the old one is closed. --------------------------------
        # This also handles gaps: if aligned > running.open_time the old
        # window is closed regardless of how many 1m candles it received.
        if aligned > running.open_time:
            completed_event = running.to_event(is_closed=True)
            await self._event_bus.publish(self._emit_topic, completed_event)

            expected_count = _TF_MULTIPLIER.get(target_tf, 1)
            if running.count < expected_count:
                logger.warning(
                    "Candle gap detected: %s %s %s had %d/%d bars.",
                    candle.symbol,
                    target_tf.value,
                    running.open_time.isoformat(),
                    running.count,
                    expected_count,
                )

            # Start the new window with the incoming candle.
            self._state[key] = _RunningCandle(
                symbol=candle.symbol,
                exchange=candle.exchange,
                timeframe=target_tf,
                open_time=aligned,
                first_candle=candle,
            )
            return completed_event

        # -- Out-of-order candle (aligned < running.open_time). ----------------
        logger.warning(
            "Out-of-order 1m candle ignored: %s %s ts=%s, current window=%s.",
            candle.symbol,
            target_tf.value,
            candle.timestamp.isoformat(),
            running.open_time.isoformat(),
        )
        return None
