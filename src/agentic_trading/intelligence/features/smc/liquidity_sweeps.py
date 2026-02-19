"""Liquidity sweep detection (BSL / SSL).

Detects when price sweeps above buy-side liquidity (BSL — swing highs) or
below sell-side liquidity (SSL — swing lows) and then reverses.  The
pattern is: wick extends beyond a swing level but the candle body closes
back inside, indicating a liquidity grab rather than a genuine breakout.

Usage::

    sweeps = detect_liquidity_sweeps(
        highs, lows, opens, closes, swings, atr=atr,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .swing_detection import SwingPoint, SwingType

logger = logging.getLogger(__name__)


class SweepType(str, Enum):
    """Type of liquidity sweep."""

    BSL = "BSL"  # Buy-Side Liquidity sweep (above swing high)
    SSL = "SSL"  # Sell-Side Liquidity sweep (below swing low)


@dataclass
class LiquiditySweep:
    """A confirmed liquidity sweep event."""

    index: int  # Bar index where the sweep occurred
    sweep_type: SweepType
    swing_level: float  # The swing level that was swept
    swing_index: int  # Index of the original swing point
    wick_price: float  # How far the wick extended beyond the level
    close_price: float  # Where the bar actually closed
    penetration_pct: float  # (|wick_price - swing_level|) / swing_level
    reversal_confirmed: bool  # True if next bar confirms reversal direction


def detect_liquidity_sweeps(
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
    swings: list[SwingPoint],
    max_age_bars: int = 100,
    min_penetration_atr: float = 0.1,
    atr: np.ndarray | None = None,
) -> list[LiquiditySweep]:
    """Detect BSL and SSL liquidity sweeps.

    A BSL sweep occurs when a bar's wick extends above a swing high but
    the body closes below that level.  An SSL sweep is the mirror: wick
    below a swing low, body closes above.

    Args:
        highs: Array of high prices.
        lows: Array of low prices.
        opens: Array of open prices.
        closes: Array of close prices.
        swings: List of swing points from swing detection.
        max_age_bars: Maximum bars after a swing to search for sweeps.
        min_penetration_atr: Minimum wick penetration as a multiple of
            ATR to filter noise.  Set to 0.0 to disable.
        atr: Optional ATR array for penetration filtering.

    Returns:
        List of :class:`LiquiditySweep` objects sorted by index.
    """
    n = len(closes)
    if n < 3 or not swings:
        return []

    sweeps: list[LiquiditySweep] = []
    swept_levels: set[tuple[int, str]] = set()  # (swing_index, type) dedup

    for swing in swings:
        if swing.index >= n - 1:
            continue

        key = (swing.index, swing.swing_type.value)
        if key in swept_levels:
            continue

        # Search window: from 1 bar after the swing to max_age_bars
        search_start = swing.index + 1
        search_end = min(n, swing.index + max_age_bars + 1)

        for i in range(search_start, search_end):
            if swing.swing_type == SwingType.HIGH:
                # BSL: wick above swing high, body closes below
                if highs[i] > swing.price and closes[i] <= swing.price:
                    penetration = abs(float(highs[i]) - swing.price)
                    penetration_pct = penetration / swing.price if swing.price > 0 else 0.0

                    # ATR filter
                    if min_penetration_atr > 0 and atr is not None:
                        atr_val = float(atr[i]) if not np.isnan(atr[i]) else 0.0
                        if atr_val > 0 and penetration < min_penetration_atr * atr_val:
                            continue

                    # Reversal confirmation: next bar closes lower
                    reversal = False
                    if i + 1 < n:
                        reversal = bool(closes[i + 1] < closes[i])

                    sweeps.append(
                        LiquiditySweep(
                            index=i,
                            sweep_type=SweepType.BSL,
                            swing_level=swing.price,
                            swing_index=swing.index,
                            wick_price=float(highs[i]),
                            close_price=float(closes[i]),
                            penetration_pct=round(penetration_pct, 6),
                            reversal_confirmed=reversal,
                        )
                    )
                    swept_levels.add(key)
                    break  # Only first sweep per swing level

            else:  # SwingType.LOW — SSL
                if lows[i] < swing.price and closes[i] >= swing.price:
                    penetration = abs(swing.price - float(lows[i]))
                    penetration_pct = penetration / swing.price if swing.price > 0 else 0.0

                    # ATR filter
                    if min_penetration_atr > 0 and atr is not None:
                        atr_val = float(atr[i]) if not np.isnan(atr[i]) else 0.0
                        if atr_val > 0 and penetration < min_penetration_atr * atr_val:
                            continue

                    # Reversal confirmation: next bar closes higher
                    reversal = False
                    if i + 1 < n:
                        reversal = bool(closes[i + 1] > closes[i])

                    sweeps.append(
                        LiquiditySweep(
                            index=i,
                            sweep_type=SweepType.SSL,
                            swing_level=swing.price,
                            swing_index=swing.index,
                            wick_price=float(lows[i]),
                            close_price=float(closes[i]),
                            penetration_pct=round(penetration_pct, 6),
                            reversal_confirmed=reversal,
                        )
                    )
                    swept_levels.add(key)
                    break  # Only first sweep per swing level

    sweeps.sort(key=lambda s: s.index)
    return sweeps
