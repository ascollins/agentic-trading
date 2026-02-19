"""Data normalization from exchange-specific formats to canonical models.

Converts raw CCXT responses (OHLCV arrays, ticker dicts, order-book dicts)
into the platform's canonical Pydantic models and events.  All exchange
quirks are handled here so that downstream code never sees raw data.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from agentic_trading.core.enums import Exchange, Timeframe
from agentic_trading.core.events import CandleEvent, OrderBookSnapshot, TickEvent
from agentic_trading.core.models import Candle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CCXT OHLCV -> Candle / CandleEvent
# ---------------------------------------------------------------------------

# CCXT OHLCV format: [timestamp_ms, open, high, low, close, volume]
_IDX_TS = 0
_IDX_OPEN = 1
_IDX_HIGH = 2
_IDX_LOW = 3
_IDX_CLOSE = 4
_IDX_VOLUME = 5


def normalize_ccxt_ohlcv(
    exchange: Exchange,
    symbol: str,
    timeframe: Timeframe,
    raw_ohlcv: list,
    *,
    is_closed: bool = True,
) -> Candle:
    """Convert a single CCXT OHLCV array to a canonical ``Candle``.

    Parameters
    ----------
    exchange:
        Source exchange enum.
    symbol:
        Unified symbol, e.g. ``"BTC/USDT"``.
    timeframe:
        Candle timeframe.
    raw_ohlcv:
        CCXT OHLCV list ``[ts_ms, o, h, l, c, v]``.
    is_closed:
        Whether the candle period has completed.

    Returns
    -------
    Candle
        Canonical candle model.
    """
    ts_ms: int = int(raw_ohlcv[_IDX_TS])
    timestamp = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)

    return Candle(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        timestamp=timestamp,
        open=float(raw_ohlcv[_IDX_OPEN]),
        high=float(raw_ohlcv[_IDX_HIGH]),
        low=float(raw_ohlcv[_IDX_LOW]),
        close=float(raw_ohlcv[_IDX_CLOSE]),
        volume=float(raw_ohlcv[_IDX_VOLUME]),
        is_closed=is_closed,
    )


def normalize_ccxt_ohlcv_batch(
    exchange: Exchange,
    symbol: str,
    timeframe: Timeframe,
    raw_ohlcv_list: list[list],
    *,
    is_closed: bool = True,
) -> list[Candle]:
    """Convert a batch of CCXT OHLCV arrays to canonical ``Candle`` objects.

    The last candle in a live stream is typically still forming; pass
    ``is_closed=False`` if appropriate.
    """
    candles: list[Candle] = []
    for raw in raw_ohlcv_list:
        candles.append(
            normalize_ccxt_ohlcv(
                exchange, symbol, timeframe, raw, is_closed=is_closed
            )
        )
    return candles


def candle_to_event(candle: Candle) -> CandleEvent:
    """Promote a ``Candle`` model to a ``CandleEvent`` for the event bus."""
    return CandleEvent(
        source_module="data.normalizer",
        symbol=candle.symbol,
        exchange=candle.exchange,
        timeframe=candle.timeframe,
        open=candle.open,
        high=candle.high,
        low=candle.low,
        close=candle.close,
        volume=candle.volume,
        quote_volume=candle.quote_volume,
        trades=candle.trades,
        is_closed=candle.is_closed,
        timestamp=candle.timestamp,
    )


# ---------------------------------------------------------------------------
# CCXT Ticker -> TickEvent
# ---------------------------------------------------------------------------


def normalize_ccxt_ticker(exchange: Exchange, raw_ticker: dict) -> TickEvent:
    """Convert a CCXT ticker dict to a canonical ``TickEvent``.

    CCXT ticker fields used:
        - symbol, bid, ask, bidVolume, askVolume, last

    Exchange-specific quirks handled:
        - Bybit may return ``None`` for bid/ask volume on some symbols.
        - Binance may include ``info`` sub-dict with extra fields.
    """
    symbol: str = raw_ticker.get("symbol", "")
    bid: float = _safe_float(raw_ticker.get("bid"))
    ask: float = _safe_float(raw_ticker.get("ask"))
    last: float = _safe_float(raw_ticker.get("last"))
    bid_size: float = _safe_float(raw_ticker.get("bidVolume"))
    ask_size: float = _safe_float(raw_ticker.get("askVolume"))

    # Sanity: if bid/ask are zero but last is available, fall back
    if bid == 0.0 and last > 0.0:
        bid = last
    if ask == 0.0 and last > 0.0:
        ask = last

    return TickEvent(
        source_module="data.normalizer",
        symbol=symbol,
        exchange=exchange,
        bid=bid,
        ask=ask,
        bid_size=bid_size,
        ask_size=ask_size,
        last=last,
    )


# ---------------------------------------------------------------------------
# CCXT Order Book -> OrderBookSnapshot
# ---------------------------------------------------------------------------


def normalize_ccxt_orderbook(
    exchange: Exchange, raw_ob: dict
) -> OrderBookSnapshot:
    """Convert a CCXT order-book dict to a canonical ``OrderBookSnapshot``.

    CCXT order-book format::

        {
            "symbol": "BTC/USDT",
            "bids": [[price, qty], ...],
            "asks": [[price, qty], ...],
            "timestamp": 1234567890123,
            "datetime": "...",
            "nonce": ...,
        }

    Exchange-specific quirks:
        - Bybit may include ``"u"`` / ``"seq"`` nonce fields.
        - Binance returns ``lastUpdateId`` in the ``info`` sub-dict.
    """
    symbol: str = raw_ob.get("symbol", "")

    # Ensure each level is [float, float]
    bids: list[list[float]] = [
        [float(lvl[0]), float(lvl[1])]
        for lvl in (raw_ob.get("bids") or [])
    ]
    asks: list[list[float]] = [
        [float(lvl[0]), float(lvl[1])]
        for lvl in (raw_ob.get("asks") or [])
    ]

    return OrderBookSnapshot(
        source_module="data.normalizer",
        symbol=symbol,
        exchange=exchange,
        bids=bids,
        asks=asks,
        depth=max(len(bids), len(asks)),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: object, default: float = 0.0) -> float:
    """Coerce a value to float, returning *default* on ``None`` or error."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
