"""Open interest features for perpetual futures.

Computes features that quantify open-interest dynamics, essential for:

* **Fresh-money confirmation** -- rising OI with rising price = new longs.
* **Unwinding detection** -- falling OI = positions closing.
* **Crowding / squeeze risk** -- OI spikes signal potential liquidation
  cascades.

The engine periodically polls OI via CCXT and publishes features into the
``FeatureVector`` pipeline.  It can also be called directly in backtesting.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

from agentic_trading.core.events import BaseEvent, FeatureVector, OpenInterestEvent
from agentic_trading.core.enums import Timeframe
from agentic_trading.core.interfaces import IEventBus

logger = logging.getLogger(__name__)

_DEFAULT_HISTORY_SIZE = 500


class OpenInterestEngine:
    """Computes open-interest features from OI snapshots.

    Usage (event-bus driven)::

        engine = OpenInterestEngine(event_bus=bus)
        await engine.start()

    Or direct (backtesting)::

        engine = OpenInterestEngine()
        features = engine.compute_oi_features(
            symbol="BTC/USDT",
            open_interest=500_000_000.0,
        )
    """

    def __init__(
        self,
        event_bus: IEventBus | None = None,
        history_size: int = _DEFAULT_HISTORY_SIZE,
    ) -> None:
        self._event_bus = event_bus
        self._history_size = history_size

        # Rolling history of OI per symbol.
        # Key: symbol -> deque of (timestamp, open_interest)
        self._oi_history: dict[
            str, deque[tuple[datetime, float]]
        ] = defaultdict(lambda: deque(maxlen=self._history_size))

    # ------------------------------------------------------------------
    # Event bus lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to open interest events on the event bus."""
        if self._event_bus is None:
            logger.warning(
                "OpenInterestEngine started without event bus - "
                "call compute_oi_features() directly"
            )
            return

        await self._event_bus.subscribe(
            topic="state.open_interest",
            group="open_interest_engine",
            handler=self._handle_oi_event,
        )
        logger.info("OpenInterestEngine subscribed to state.open_interest")

    async def stop(self) -> None:
        """Clean up (currently a no-op)."""
        logger.info("OpenInterestEngine stopped")

    async def _handle_oi_event(self, event: BaseEvent) -> None:
        """Handle incoming open interest events."""
        if not isinstance(event, OpenInterestEvent):
            return

        features = self.compute_oi_features(
            symbol=event.symbol,
            open_interest=event.open_interest,
            open_interest_value=event.open_interest_value,
            timestamp=event.timestamp,
        )

        if self._event_bus is not None:
            fv = FeatureVector(
                symbol=event.symbol,
                timeframe=Timeframe.M1,
                features=features,
                source_module="features.open_interest",
            )
            await self._event_bus.publish("feature.vector", fv)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_oi_features(
        self,
        symbol: str,
        open_interest: float,
        open_interest_value: float = 0.0,
        timestamp: datetime | None = None,
    ) -> dict[str, float]:
        """Compute OI features for *symbol*.

        Args:
            symbol: Trading pair, e.g. ``"BTC/USDT"``.
            open_interest: Current open interest (contracts or base units).
            open_interest_value: Notional value in quote currency.
            timestamp: Observation time.  Defaults to now (UTC).

        Returns:
            Dict of feature name to float value.
        """
        ts = timestamp or datetime.now(timezone.utc)

        self._oi_history[symbol].append((ts, open_interest))

        features: dict[str, float] = {}
        features["oi_current"] = open_interest
        if open_interest_value > 0:
            features["oi_value"] = open_interest_value

        history = self._oi_history[symbol]
        oi_values = np.array([oi for _, oi in history], dtype=np.float64)

        # --- Change percentages ---
        # Approximate 1h and 24h changes based on observation count.
        # (In practice the polling interval determines how many samples
        #  map to a real-time window; we use position-based proxies.)
        if len(oi_values) >= 2:
            prev = oi_values[-2]
            if prev > 0:
                features["oi_change_pct_latest"] = (
                    (open_interest - prev) / prev
                ) * 100.0

        # ~1h proxy: last 12 samples (at 5-minute polling)
        if len(oi_values) >= 13:
            prev_1h = float(oi_values[-13])
            if prev_1h > 0:
                features["oi_change_pct_1h"] = (
                    (open_interest - prev_1h) / prev_1h
                ) * 100.0

        # ~24h proxy: last 288 samples (at 5-minute polling)
        lookback_24h = min(289, len(oi_values))
        if lookback_24h >= 50:
            prev_24h = float(oi_values[-lookback_24h])
            if prev_24h > 0:
                features["oi_change_pct_24h"] = (
                    (open_interest - prev_24h) / prev_24h
                ) * 100.0

        # --- Trend direction ---
        # Simple regression slope over last 20 observations.
        if len(oi_values) >= 20:
            window = oi_values[-20:]
            x = np.arange(len(window), dtype=np.float64)
            mean_x = np.mean(x)
            mean_y = np.mean(window)
            denom = np.sum((x - mean_x) ** 2)
            if denom > 0:
                slope = float(np.sum((x - mean_x) * (window - mean_y)) / denom)
                # Normalise slope as % of current value
                slope_pct = (slope / open_interest) * 100.0 if open_interest > 0 else 0.0
                if slope_pct > 0.05:
                    features["oi_trend"] = 1.0
                elif slope_pct < -0.05:
                    features["oi_trend"] = -1.0
                else:
                    features["oi_trend"] = 0.0
            else:
                features["oi_trend"] = 0.0
        else:
            features["oi_trend"] = 0.0

        # --- Z-score ---
        if len(oi_values) >= 20:
            mean_oi = float(np.mean(oi_values))
            std_oi = float(np.std(oi_values, ddof=1))
            if std_oi > 0:
                features["oi_zscore"] = (open_interest - mean_oi) / std_oi
            else:
                features["oi_zscore"] = 0.0
        else:
            features["oi_zscore"] = float("nan")

        return features

    def get_oi_history(
        self, symbol: str,
    ) -> list[tuple[datetime, float]]:
        """Return a copy of the OI history for *symbol*."""
        return list(self._oi_history.get(symbol, []))

    def clear(self, symbol: str | None = None) -> None:
        """Reset history.  If *symbol* is ``None``, clears all symbols."""
        if symbol is None:
            self._oi_history.clear()
        else:
            self._oi_history.pop(symbol, None)
