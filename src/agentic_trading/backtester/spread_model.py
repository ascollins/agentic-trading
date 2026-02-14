"""Spread widening and feed outage simulation for backtesting."""

from __future__ import annotations

import random


class SpreadModel:
    """Simulates bid-ask spread dynamics.

    Models:
    - Normal spread based on instrument tick size
    - Widening during high volatility / low liquidity
    - Feed outage gaps
    """

    def __init__(
        self,
        base_spread_bps: float = 2.0,
        volatility_widening_factor: float = 2.0,
        outage_probability: float = 0.001,  # 0.1% chance per candle
        outage_duration_candles: int = 5,
        seed: int = 42,
    ) -> None:
        self._base_spread = base_spread_bps / 10_000.0
        self._vol_factor = volatility_widening_factor
        self._outage_prob = outage_probability
        self._outage_duration = outage_duration_candles
        self._rng = random.Random(seed)
        self._outage_remaining = 0

    def get_spread(
        self,
        price: float,
        atr: float = 0.0,
        avg_atr: float = 0.0,
    ) -> float:
        """Get current spread in price units.

        Args:
            price: Current mid price.
            atr: Current ATR.
            avg_atr: Historical average ATR.

        Returns:
            Spread in price units (half-spread on each side).
        """
        base = price * self._base_spread

        # Widen in high vol
        if atr > 0 and avg_atr > 0:
            vol_ratio = atr / avg_atr
            if vol_ratio > 1.5:
                widening = self._vol_factor * (vol_ratio - 1.0)
                base *= (1.0 + widening)

        return base

    def check_outage(self) -> bool:
        """Check if a feed outage is occurring.

        Returns True if data should be considered stale/missing.
        """
        if self._outage_remaining > 0:
            self._outage_remaining -= 1
            return True

        if self._rng.random() < self._outage_prob:
            self._outage_remaining = self._outage_duration
            return True

        return False

    def get_bid_ask(
        self,
        mid_price: float,
        atr: float = 0.0,
        avg_atr: float = 0.0,
    ) -> tuple[float, float]:
        """Get bid and ask prices."""
        half_spread = self.get_spread(mid_price, atr, avg_atr) / 2.0
        bid = mid_price - half_spread
        ask = mid_price + half_spread
        return bid, ask
