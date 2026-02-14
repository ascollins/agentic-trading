"""Funding rate and basis features for perpetual futures.

Computes features that quantify the cost-of-carry between spot and
perpetual contracts.  These are essential inputs for:

* **Carry/basis trading** -- capturing the funding rate premium.
* **Sentiment gauging** -- extreme funding rates signal crowded
  positioning.
* **Regime detection** -- sustained positive/negative basis can
  indicate trending vs. mean-reverting regimes.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

from agentic_trading.core.events import BaseEvent, FeatureVector
from agentic_trading.core.enums import Timeframe
from agentic_trading.core.interfaces import IEventBus

logger = logging.getLogger(__name__)

# Number of funding rate observations to keep for z-score.
_DEFAULT_HISTORY_SIZE = 500

# Standard funding interval for most perp exchanges (hours).
_FUNDING_INTERVAL_HOURS = 8

# Annualisation factor: 365 days * (24h / interval).
_FUNDING_PERIODS_PER_YEAR = 365 * (24 / _FUNDING_INTERVAL_HOURS)


class FundingBasisEngine:
    """Computes funding-rate and spot-perp basis features.

    Usage (event-bus driven)::

        engine = FundingBasisEngine(event_bus=bus)
        await engine.start()

    Or direct (backtesting)::

        engine = FundingBasisEngine()
        features = engine.compute_funding_features(
            symbol="BTC/USDT",
            funding_rate=0.0001,
            spot_price=67_000.0,
            perp_price=67_050.0,
        )
    """

    def __init__(
        self,
        event_bus: IEventBus | None = None,
        history_size: int = _DEFAULT_HISTORY_SIZE,
        funding_interval_hours: int = _FUNDING_INTERVAL_HOURS,
    ) -> None:
        self._event_bus = event_bus
        self._history_size = history_size
        self._funding_interval_hours = funding_interval_hours
        self._periods_per_year = 365 * (24 / funding_interval_hours)

        # Rolling history of funding rates per symbol.
        # Key: symbol -> deque of (timestamp, funding_rate)
        self._funding_history: dict[
            str, deque[tuple[datetime, float]]
        ] = defaultdict(lambda: deque(maxlen=self._history_size))

        # Rolling history of basis values per symbol (for z-score).
        self._basis_history: dict[
            str, deque[float]
        ] = defaultdict(lambda: deque(maxlen=self._history_size))

    # ------------------------------------------------------------------
    # Event bus lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to funding payment events on the event bus."""
        if self._event_bus is None:
            logger.warning(
                "FundingBasisEngine started without event bus - "
                "call compute_funding_features() directly"
            )
            return

        await self._event_bus.subscribe(
            topic="state.funding",
            group="funding_basis_engine",
            handler=self._handle_funding_event,
        )
        logger.info("FundingBasisEngine subscribed to state.funding")

    async def stop(self) -> None:
        """Clean up (currently a no-op)."""
        logger.info("FundingBasisEngine stopped")

    async def _handle_funding_event(self, event: BaseEvent) -> None:
        """Handle incoming funding payment events.

        Note: This handler expects ``FundingPaymentEvent`` which carries
        ``funding_rate`` but not spot/perp prices.  For full basis
        features the caller must also supply price data.  In event-bus
        mode we compute what we can (funding-rate-only features) and
        publish them.
        """
        from agentic_trading.core.events import FundingPaymentEvent

        if not isinstance(event, FundingPaymentEvent):
            return

        features = self.compute_funding_features(
            symbol=event.symbol,
            funding_rate=float(event.funding_rate),
            spot_price=None,
            perp_price=None,
            timestamp=event.timestamp,
        )

        if self._event_bus is not None:
            fv = FeatureVector(
                symbol=event.symbol,
                timeframe=Timeframe.M1,  # Funding features are point-in-time.
                features=features,
                source_module="features.funding_basis",
            )
            await self._event_bus.publish("feature.vector", fv)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_funding_features(
        self,
        symbol: str,
        funding_rate: float,
        spot_price: float | None = None,
        perp_price: float | None = None,
        timestamp: datetime | None = None,
    ) -> dict[str, float]:
        """Compute funding / basis features for *symbol*.

        Args:
            symbol: Trading pair, e.g. ``"BTC/USDT"``.
            funding_rate: The current (or most recent) funding rate as a
                decimal fraction (e.g. 0.0001 for 0.01%).
            spot_price: Current spot price.  ``None`` if unavailable.
            perp_price: Current perpetual price.  ``None`` if unavailable.
            timestamp: Observation time.  Defaults to now (UTC).

        Returns:
            Dict of feature name to float value.
        """
        ts = timestamp or datetime.now(timezone.utc)

        # Record funding rate history.
        self._funding_history[symbol].append((ts, funding_rate))

        features: dict[str, float] = {}

        # ---- Funding rate features ----
        features["funding_rate"] = funding_rate
        features["annualized_funding"] = funding_rate * self._periods_per_year

        # Funding rate z-score (how extreme is the current rate?).
        fr_values = np.array(
            [fr for _, fr in self._funding_history[symbol]], dtype=np.float64
        )
        if len(fr_values) >= 20:
            mean_fr = float(np.mean(fr_values))
            std_fr = float(np.std(fr_values, ddof=1))
            if std_fr > 0:
                features["funding_zscore"] = (funding_rate - mean_fr) / std_fr
            else:
                features["funding_zscore"] = 0.0
        else:
            features["funding_zscore"] = float("nan")

        # Rolling mean funding rate (last 24h worth of periods).
        periods_24h = int(24 / self._funding_interval_hours)
        if len(fr_values) >= periods_24h:
            features["funding_mean_24h"] = float(np.mean(fr_values[-periods_24h:]))
        else:
            features["funding_mean_24h"] = float("nan")

        # ---- Basis features (require both spot and perp prices) ----
        if spot_price is not None and perp_price is not None and spot_price > 0:
            basis = perp_price - spot_price
            basis_bps = (basis / spot_price) * 10_000.0

            features["basis"] = basis
            features["basis_bps"] = basis_bps
            features["basis_pct"] = (basis / spot_price) * 100.0
            features["annualized_basis"] = (
                basis_bps / 10_000.0
            ) * self._periods_per_year * 100.0

            # Record basis for z-score.
            self._basis_history[symbol].append(basis_bps)

            basis_arr = np.array(self._basis_history[symbol], dtype=np.float64)
            if len(basis_arr) >= 20:
                mean_b = float(np.mean(basis_arr))
                std_b = float(np.std(basis_arr, ddof=1))
                if std_b > 0:
                    features["basis_zscore"] = (basis_bps - mean_b) / std_b
                else:
                    features["basis_zscore"] = 0.0
            else:
                features["basis_zscore"] = float("nan")
        else:
            features["basis"] = float("nan")
            features["basis_bps"] = float("nan")
            features["basis_pct"] = float("nan")
            features["annualized_basis"] = float("nan")
            features["basis_zscore"] = float("nan")

        return features

    def get_funding_history(
        self, symbol: str
    ) -> list[tuple[datetime, float]]:
        """Return a copy of the funding rate history for *symbol*."""
        return list(self._funding_history.get(symbol, []))

    def clear(self, symbol: str | None = None) -> None:
        """Reset history.  If *symbol* is ``None``, clears all symbols."""
        if symbol is None:
            self._funding_history.clear()
            self._basis_history.clear()
        else:
            self._funding_history.pop(symbol, None)
            self._basis_history.pop(symbol, None)
