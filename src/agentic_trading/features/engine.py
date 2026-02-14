"""Feature computation engine.

Subscribes to ``CandleEvent`` on the event bus, computes a full set of
technical indicators, and publishes the resulting ``FeatureVector``.

The engine maintains per-symbol, per-timeframe candle buffers so that
indicators with long warmup periods always have enough history.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any

import numpy as np

from agentic_trading.core.enums import Timeframe
from agentic_trading.core.events import BaseEvent, CandleEvent, FeatureVector
from agentic_trading.core.interfaces import IEventBus
from agentic_trading.core.models import Candle

from .indicators import (
    compute_adx,
    compute_atr,
    compute_bollinger_bands,
    compute_donchian,
    compute_ema,
    compute_macd,
    compute_rsi,
    compute_sma,
    compute_vwap,
)

logger = logging.getLogger(__name__)

# Maximum candles retained per symbol/timeframe buffer.  This is generous
# enough for any indicator warmup (e.g. 200-period SMA on daily candles).
_DEFAULT_BUFFER_SIZE = 500


class FeatureEngine:
    """Stateful engine that converts raw candle streams into feature vectors.

    Typical lifecycle::

        engine = FeatureEngine(event_bus)
        await engine.start()   # subscribes to CandleEvent
        # ... candle events flow in, FeatureVectors are published ...
        await engine.stop()

    The engine can also be called directly (useful in backtesting)::

        fv = engine.compute_features(symbol, timeframe, candle_list)
    """

    def __init__(
        self,
        event_bus: IEventBus | None = None,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
        indicator_config: dict[str, Any] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._buffer_size = buffer_size

        # Per (symbol, timeframe) candle history.
        # Key: (symbol, Timeframe) -> deque[Candle]
        self._buffers: dict[tuple[str, Timeframe], deque[Candle]] = defaultdict(
            lambda: deque(maxlen=self._buffer_size)
        )

        # Indicator parameters -- allow callers to override defaults.
        cfg = indicator_config or {}
        self._ema_periods: list[int] = cfg.get("ema_periods", [9, 12, 21, 26, 50, 200])
        self._sma_periods: list[int] = cfg.get("sma_periods", [20, 50, 200])
        self._rsi_period: int = cfg.get("rsi_period", 14)
        self._bb_period: int = cfg.get("bb_period", 20)
        self._bb_std: float = cfg.get("bb_std", 2.0)
        self._adx_period: int = cfg.get("adx_period", 14)
        self._atr_period: int = cfg.get("atr_period", 14)
        self._macd_fast: int = cfg.get("macd_fast", 12)
        self._macd_slow: int = cfg.get("macd_slow", 26)
        self._macd_signal: int = cfg.get("macd_signal", 9)
        self._donchian_period: int = cfg.get("donchian_period", 20)

    # ------------------------------------------------------------------
    # Event bus lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to candle events on the event bus."""
        if self._event_bus is None:
            logger.warning("FeatureEngine started without event bus - direct mode only")
            return
        await self._event_bus.subscribe(
            topic="market.candle",
            group="feature_engine",
            handler=self._handle_candle_event,
        )
        logger.info("FeatureEngine subscribed to market.candle")

    async def stop(self) -> None:
        """Clean up resources (currently a no-op)."""
        logger.info("FeatureEngine stopped")

    async def _handle_candle_event(self, event: BaseEvent) -> None:
        """Internal handler for ``CandleEvent``."""
        if not isinstance(event, CandleEvent):
            return

        # Only process closed candles to avoid partial-bar noise.
        if not event.is_closed:
            return

        candle = Candle(
            symbol=event.symbol,
            exchange=event.exchange,
            timeframe=event.timeframe,
            timestamp=event.timestamp,
            open=event.open,
            high=event.high,
            low=event.low,
            close=event.close,
            volume=event.volume,
            quote_volume=event.quote_volume,
            trades=event.trades,
            is_closed=event.is_closed,
        )

        key = (event.symbol, event.timeframe)
        self._buffers[key].append(candle)

        candles = list(self._buffers[key])
        fv = self.compute_features(event.symbol, event.timeframe, candles)

        if self._event_bus is not None:
            await self._event_bus.publish("feature.vector", fv)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_candle(self, candle: Candle) -> None:
        """Manually add a candle to the internal buffer.

        Useful in backtesting when you want to feed candles without the
        event bus and then call :meth:`compute_features` directly.
        """
        key = (candle.symbol, candle.timeframe)
        self._buffers[key].append(candle)

    def compute_features(
        self,
        symbol: str,
        timeframe: Timeframe,
        candles: list[Candle],
    ) -> FeatureVector:
        """Compute the full indicator suite for a candle series.

        Args:
            symbol: Trading pair, e.g. ``"BTC/USDT"``.
            timeframe: Candle timeframe.
            candles: Ordered list of candles (oldest first).

        Returns:
            A :class:`FeatureVector` event carrying a ``features`` dict
            whose keys follow the pattern ``<indicator>_<param>``.
        """
        if not candles:
            return FeatureVector(symbol=symbol, timeframe=timeframe, features={})

        # Extract OHLCV arrays
        opens = np.array([c.open for c in candles], dtype=np.float64)
        highs = np.array([c.high for c in candles], dtype=np.float64)
        lows = np.array([c.low for c in candles], dtype=np.float64)
        closes = np.array([c.close for c in candles], dtype=np.float64)
        volumes = np.array([c.volume for c in candles], dtype=np.float64)

        features: dict[str, float] = {}

        # Latest raw OHLCV
        features["open"] = float(opens[-1])
        features["high"] = float(highs[-1])
        features["low"] = float(lows[-1])
        features["close"] = float(closes[-1])
        features["volume"] = float(volumes[-1])

        # ----- EMA -----
        for p in self._ema_periods:
            ema = compute_ema(closes, p)
            features[f"ema_{p}"] = _safe_last(ema)

        # ----- SMA -----
        for p in self._sma_periods:
            sma = compute_sma(closes, p)
            features[f"sma_{p}"] = _safe_last(sma)

        # ----- RSI -----
        rsi = compute_rsi(closes, self._rsi_period)
        features[f"rsi_{self._rsi_period}"] = _safe_last(rsi)

        # ----- Bollinger Bands -----
        bb_upper, bb_mid, bb_lower = compute_bollinger_bands(
            closes, self._bb_period, self._bb_std
        )
        features["bb_upper"] = _safe_last(bb_upper)
        features["bb_middle"] = _safe_last(bb_mid)
        features["bb_lower"] = _safe_last(bb_lower)

        # Percent-B: how close the price is to the upper band (0 = lower, 1 = upper)
        bw = _safe_last(bb_upper) - _safe_last(bb_lower)
        if bw > 0 and not np.isnan(bw):
            features["bb_pct_b"] = (
                float(closes[-1]) - _safe_last(bb_lower)
            ) / bw
        else:
            features["bb_pct_b"] = float("nan")

        # ----- ADX -----
        adx = compute_adx(highs, lows, closes, self._adx_period)
        features[f"adx_{self._adx_period}"] = _safe_last(adx)

        # ----- ATR -----
        atr = compute_atr(highs, lows, closes, self._atr_period)
        features[f"atr_{self._atr_period}"] = _safe_last(atr)
        # Normalised ATR (as % of close)
        if closes[-1] != 0:
            features[f"atr_{self._atr_period}_pct"] = (
                _safe_last(atr) / closes[-1]
            ) * 100.0
        else:
            features[f"atr_{self._atr_period}_pct"] = float("nan")

        # ----- MACD -----
        macd_line, signal_line, histogram = compute_macd(
            closes, self._macd_fast, self._macd_slow, self._macd_signal
        )
        features["macd"] = _safe_last(macd_line)
        features["macd_signal"] = _safe_last(signal_line)
        features["macd_histogram"] = _safe_last(histogram)

        # ----- Donchian Channel -----
        dc_upper, dc_lower = compute_donchian(highs, lows, self._donchian_period)
        features[f"donchian_upper_{self._donchian_period}"] = _safe_last(dc_upper)
        features[f"donchian_lower_{self._donchian_period}"] = _safe_last(dc_lower)

        # ----- VWAP -----
        vwap = compute_vwap(highs, lows, closes, volumes)
        features["vwap"] = _safe_last(vwap)

        # ----- Derived / composite features -----
        # Price vs. key MAs
        for p in self._ema_periods:
            ema_val = features.get(f"ema_{p}", float("nan"))
            if ema_val and not np.isnan(ema_val) and ema_val != 0:
                features[f"close_vs_ema_{p}"] = (closes[-1] / ema_val) - 1.0
            else:
                features[f"close_vs_ema_{p}"] = float("nan")

        # Simple return features
        if len(closes) >= 2:
            features["return_1"] = (closes[-1] / closes[-2]) - 1.0
        else:
            features["return_1"] = float("nan")

        if len(closes) >= 6:
            features["return_5"] = (closes[-1] / closes[-6]) - 1.0
        else:
            features["return_5"] = float("nan")

        if len(closes) >= 21:
            features["return_20"] = (closes[-1] / closes[-21]) - 1.0
        else:
            features["return_20"] = float("nan")

        # Realised volatility (20 bar)
        if len(closes) >= 21:
            log_returns = np.diff(np.log(closes[-21:]))
            features["realised_vol_20"] = float(np.std(log_returns, ddof=1))
        else:
            features["realised_vol_20"] = float("nan")

        return FeatureVector(
            symbol=symbol,
            timeframe=timeframe,
            features=features,
            source_module="features.engine",
        )

    def get_buffer(
        self, symbol: str, timeframe: Timeframe
    ) -> list[Candle]:
        """Return a copy of the candle buffer for a symbol/timeframe pair."""
        key = (symbol, timeframe)
        return list(self._buffers.get(key, []))

    def clear_buffers(self) -> None:
        """Drop all candle history.  Primarily for testing."""
        self._buffers.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_last(arr: np.ndarray) -> float:
    """Return the last element of *arr* as a Python float, or NaN."""
    if arr is None or len(arr) == 0:
        return float("nan")
    val = arr[-1]
    if np.isnan(val):
        return float("nan")
    return float(val)
