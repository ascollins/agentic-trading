"""Live market-data feed manager built on CCXT Pro.

``FeedManager`` manages WebSocket connections to one or more exchanges
(Binance, Bybit) and streams OHLCV candles into the platform's event bus.

Architecture
------------
* One asyncio task per (exchange, symbol, timeframe) subscription.
* CCXT Pro handles WebSocket reconnection, heartbeats, and rate-limiting
  internally; we simply ``await exchange.watch_ohlcv(...)`` in a loop.
* Incoming raw OHLCV data is normalized via :mod:`agentic_trading.data.normalizer`
  and optionally aggregated by :class:`agentic_trading.data.candle_builder.CandleBuilder`.
* Closed 1m candles are published to the event bus as ``CandleEvent``.
  The ``CandleBuilder`` publishes higher-timeframe candles.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import ccxt.pro as ccxtpro

from agentic_trading.core.config import ExchangeConfig
from agentic_trading.core.enums import Exchange, Timeframe
from agentic_trading.core.events import CandleEvent
from agentic_trading.core.interfaces import IEventBus
from agentic_trading.data.candle_builder import CandleBuilder
from agentic_trading.data.normalizer import (
    candle_to_event,
    normalize_ccxt_ohlcv,
)

logger = logging.getLogger(__name__)

# Map our Exchange enum to CCXT Pro exchange class constructors.
_EXCHANGE_CLASSES: dict[Exchange, type] = {
    Exchange.BINANCE: ccxtpro.binance,
    Exchange.BYBIT: ccxtpro.bybit,
}


def _build_ccxt_exchange(config: ExchangeConfig) -> Any:
    """Instantiate a CCXT Pro exchange from an ``ExchangeConfig``."""
    cls = _EXCHANGE_CLASSES.get(config.name)
    if cls is None:
        raise ValueError(f"Unsupported exchange: {config.name.value}")

    options: dict[str, Any] = {}

    # Binance-specific: use the combined-stream endpoint for efficiency.
    if config.name == Exchange.BINANCE:
        options["defaultType"] = "future"

    # Bybit-specific: prefer v5 unified endpoint.
    if config.name == Exchange.BYBIT:
        options["defaultType"] = "swap"

    exchange = cls(
        {
            "apiKey": config.api_key,
            "secret": config.secret,
            "enableRateLimit": True,
            "rateLimit": config.rate_limit,
            "timeout": config.timeout,
            "options": options,
        }
    )

    if config.testnet:
        exchange.set_sandbox_mode(True)

    return exchange


class FeedManager:
    """Manages CCXT Pro WebSocket feeds for multiple exchanges.

    Parameters
    ----------
    event_bus:
        Platform event bus for publishing ``CandleEvent`` instances.
    candle_builder:
        Multi-timeframe candle aggregator.  1m candle events are also
        forwarded here for aggregation into higher timeframes.
    exchange_configs:
        List of exchange configurations to connect to.
    symbols:
        List of unified symbols to subscribe to (e.g. ``["BTC/USDT"]``).
    base_timeframe:
        The lowest-resolution timeframe fetched from the exchange.
        Defaults to ``Timeframe.M1``.
    candle_topic:
        Event bus topic for candle events.
    """

    def __init__(
        self,
        event_bus: IEventBus,
        candle_builder: CandleBuilder,
        exchange_configs: list[ExchangeConfig],
        symbols: list[str],
        base_timeframe: Timeframe = Timeframe.M1,
        candle_topic: str = "market.candle",
    ) -> None:
        self._event_bus = event_bus
        self._candle_builder = candle_builder
        self._exchange_configs = exchange_configs
        self._symbols = symbols
        self._base_timeframe = base_timeframe
        self._candle_topic = candle_topic

        # Runtime state
        self._exchanges: dict[Exchange, Any] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False

        # Track last-seen candle timestamp per (exchange, symbol, tf) to
        # detect new (closed) candles.
        self._last_candle_ts: dict[tuple[Exchange, str, Timeframe], int] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to all configured exchanges and launch feed tasks."""
        if self._running:
            logger.warning("FeedManager.start() called while already running.")
            return

        self._running = True
        logger.info(
            "Starting FeedManager: exchanges=%s, symbols=%s, base_tf=%s",
            [c.name.value for c in self._exchange_configs],
            self._symbols,
            self._base_timeframe.value,
        )

        for config in self._exchange_configs:
            try:
                exchange = _build_ccxt_exchange(config)
                self._exchanges[config.name] = exchange
                logger.info(
                    "Initialized CCXT Pro exchange: %s (testnet=%s)",
                    config.name.value,
                    config.testnet,
                )
            except Exception:
                logger.exception(
                    "Failed to initialize exchange %s", config.name.value
                )
                continue

            # Spawn one task per (exchange, symbol).
            for symbol in self._symbols:
                task = asyncio.create_task(
                    self._watch_ohlcv_loop(config.name, exchange, symbol),
                    name=f"feed:{config.name.value}:{symbol}:{self._base_timeframe.value}",
                )
                self._tasks.append(task)

        logger.info("FeedManager launched %d feed tasks.", len(self._tasks))

    async def stop(self) -> None:
        """Cancel all feed tasks and close exchange connections."""
        self._running = False

        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        for name, exchange in self._exchanges.items():
            try:
                await exchange.close()
                logger.info("Closed exchange connection: %s", name.value)
            except Exception:
                logger.exception(
                    "Error closing exchange %s", name.value
                )
        self._exchanges.clear()

        logger.info("FeedManager stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Feed loop
    # ------------------------------------------------------------------

    async def _watch_ohlcv_loop(
        self,
        exchange_name: Exchange,
        exchange: Any,
        symbol: str,
    ) -> None:
        """Continuously watch OHLCV data for a single symbol on one exchange.

        CCXT Pro's ``watch_ohlcv`` returns the latest cached candles each time
        the exchange pushes an update.  We compare timestamps to detect when a
        candle has closed and emit it.
        """
        tf_value = self._base_timeframe.value
        state_key = (exchange_name, symbol, self._base_timeframe)

        logger.info(
            "Feed loop started: %s %s %s", exchange_name.value, symbol, tf_value
        )

        consecutive_errors = 0
        max_consecutive_errors = 20
        base_backoff = 1.0  # seconds
        max_backoff = 60.0

        while self._running:
            try:
                ohlcv_list = await exchange.watch_ohlcv(symbol, tf_value)

                if not ohlcv_list:
                    continue

                # Reset error counter on success.
                consecutive_errors = 0

                await self._process_ohlcv_update(
                    exchange_name, symbol, ohlcv_list, state_key
                )

            except asyncio.CancelledError:
                logger.info(
                    "Feed loop cancelled: %s %s", exchange_name.value, symbol
                )
                raise
            except Exception:
                consecutive_errors += 1
                backoff = min(
                    base_backoff * (2 ** (consecutive_errors - 1)), max_backoff
                )
                logger.exception(
                    "Error in feed loop %s %s (attempt %d/%d, backoff %.1fs)",
                    exchange_name.value,
                    symbol,
                    consecutive_errors,
                    max_consecutive_errors,
                    backoff,
                )
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(
                        "Feed loop %s %s exceeded max errors (%d); stopping.",
                        exchange_name.value,
                        symbol,
                        max_consecutive_errors,
                    )
                    break
                await asyncio.sleep(backoff)

        logger.info(
            "Feed loop exited: %s %s %s", exchange_name.value, symbol, tf_value
        )

    async def _process_ohlcv_update(
        self,
        exchange_name: Exchange,
        symbol: str,
        ohlcv_list: list[list],
        state_key: tuple[Exchange, str, Timeframe],
    ) -> None:
        """Process an OHLCV update from CCXT Pro and emit closed candles."""
        prev_ts = self._last_candle_ts.get(state_key)

        for raw in ohlcv_list:
            ts_ms = int(raw[0])

            # First time: just record the timestamp, don't emit.
            if prev_ts is None:
                self._last_candle_ts[state_key] = ts_ms
                prev_ts = ts_ms
                continue

            # Same candle still forming -- update but don't emit.
            if ts_ms == prev_ts:
                continue

            # New candle arrived -> the *previous* candle is now closed.
            if ts_ms > prev_ts:
                # We need the previous candle data, but CCXT Pro only gives
                # us the latest buffer.  When the timestamp advances, the
                # new raw entry with ts_ms is the currently forming candle.
                # The previous entry is the closed candle.  We look backwards
                # in the list for the entry with prev_ts.
                closed_raw = self._find_raw_by_ts(ohlcv_list, prev_ts)
                if closed_raw is not None:
                    await self._emit_closed_candle(
                        exchange_name, symbol, closed_raw
                    )

                self._last_candle_ts[state_key] = ts_ms
                prev_ts = ts_ms

        # Also treat the latest candle as the new "previous" for next update.
        if ohlcv_list:
            latest_ts = int(ohlcv_list[-1][0])
            self._last_candle_ts[state_key] = latest_ts

    @staticmethod
    def _find_raw_by_ts(
        ohlcv_list: list[list], target_ts: int
    ) -> list | None:
        """Find the OHLCV entry with the given timestamp in the buffer."""
        for raw in ohlcv_list:
            if int(raw[0]) == target_ts:
                return raw
        return None

    async def _emit_closed_candle(
        self,
        exchange_name: Exchange,
        symbol: str,
        raw_ohlcv: list,
    ) -> None:
        """Normalize a raw OHLCV, emit to bus, and forward to CandleBuilder."""
        candle = normalize_ccxt_ohlcv(
            exchange=exchange_name,
            symbol=symbol,
            timeframe=self._base_timeframe,
            raw_ohlcv=raw_ohlcv,
            is_closed=True,
        )

        # Publish the 1m candle event.
        event = candle_to_event(candle)
        await self._event_bus.publish(self._candle_topic, event)

        # Forward to CandleBuilder for higher-timeframe aggregation.
        await self._candle_builder.process(candle)

        logger.debug(
            "Emitted closed candle: %s %s %s @ %s  O=%.2f H=%.2f L=%.2f C=%.2f V=%.4f",
            exchange_name.value,
            symbol,
            self._base_timeframe.value,
            candle.timestamp.isoformat(),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_exchange(self, name: Exchange) -> Any | None:
        """Return the CCXT Pro exchange instance (for testing / diagnostics)."""
        return self._exchanges.get(name)

    @property
    def active_task_count(self) -> int:
        return sum(1 for t in self._tasks if not t.done())
