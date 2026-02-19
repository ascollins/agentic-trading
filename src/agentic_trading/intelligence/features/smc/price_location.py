"""Premium / Discount zone classification.

Computes the dealing range from swing extremes, the equilibrium (50%
level), and classifies where the current price sits relative to the range.
Includes Optimal Trade Entry (OTE) zone detection at the 61.8–78.6%
Fibonacci retracement level.

Usage::

    dealing_range = compute_dealing_range(swings, atr_value=atr[-1])
    if dealing_range is not None:
        location = classify_price_location(
            current_price, dealing_range[0], dealing_range[1],
        )
        print(location.zone, location.equilibrium)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from .swing_detection import SwingPoint, SwingType

logger = logging.getLogger(__name__)


class PriceZone(str, Enum):
    """Where current price sits within the dealing range."""

    DEEP_PREMIUM = "deep_premium"      # Above 78.6% of range
    PREMIUM = "premium"                # 50% to 78.6%
    EQUILIBRIUM = "equilibrium"        # 45% to 55%
    DISCOUNT = "discount"              # 21.4% to 50%
    DEEP_DISCOUNT = "deep_discount"    # Below 21.4%


@dataclass
class PriceLocation:
    """Complete price location assessment within a dealing range."""

    equilibrium: float  # 50% of dealing range
    zone: PriceZone
    deviation_pct: float  # Signed deviation from equilibrium as fraction of range
    dealing_range_high: float
    dealing_range_low: float
    dealing_range_size: float  # Absolute size
    ote_high: float  # OTE upper bound (for longs: 38.2% from low)
    ote_low: float  # OTE lower bound (for longs: 21.4% from low)
    in_ote: bool  # True if price is within any OTE zone
    range_position_pct: float  # 0.0 = at low, 1.0 = at high


def compute_dealing_range(
    swings: list[SwingPoint],
    min_range_atr: float = 2.0,
    atr_value: float | None = None,
) -> tuple[float, float] | None:
    """Find the most recent significant dealing range from swing extremes.

    Searches for the most recent swing high and swing low that form a
    range large enough to be significant (at least ``min_range_atr * ATR``
    if an ATR value is provided, otherwise accepts any non-zero range).

    Args:
        swings: Sorted list of swing points (from swing detection).
        min_range_atr: Minimum range as a multiple of ATR.
        atr_value: Current ATR value for range filtering.  If *None*,
            any non-zero range is accepted.

    Returns:
        Tuple of ``(range_high, range_low)`` or *None* if no valid
        range can be formed.
    """
    if len(swings) < 2:
        return None

    # Find the most recent swing high and swing low
    last_high: SwingPoint | None = None
    last_low: SwingPoint | None = None

    for swing in reversed(swings):
        if swing.swing_type == SwingType.HIGH and last_high is None:
            last_high = swing
        elif swing.swing_type == SwingType.LOW and last_low is None:
            last_low = swing

        if last_high is not None and last_low is not None:
            break

    if last_high is None or last_low is None:
        return None

    range_high = last_high.price
    range_low = last_low.price

    # Ensure high > low
    if range_high <= range_low:
        return None

    range_size = range_high - range_low

    # ATR-based minimum range filter
    if atr_value is not None and atr_value > 0:
        if range_size < min_range_atr * atr_value:
            return None

    return (range_high, range_low)


def classify_price_location(
    current_price: float,
    dealing_range_high: float,
    dealing_range_low: float,
) -> PriceLocation:
    """Classify current price position within the dealing range.

    Zones are defined by the position of the price within the range,
    measured as a fraction from 0.0 (at the low) to 1.0 (at the high):

    - **DEEP_PREMIUM**: position > 0.786
    - **PREMIUM**: 0.55 < position ≤ 0.786
    - **EQUILIBRIUM**: 0.45 ≤ position ≤ 0.55
    - **DISCOUNT**: 0.214 ≤ position < 0.45
    - **DEEP_DISCOUNT**: position < 0.214

    OTE (Optimal Trade Entry) for longs is at the 61.8–78.6% retracement
    from the high (i.e., range_position between 0.214 and 0.382).
    OTE for shorts is the mirror: range_position between 0.618 and 0.786.

    Args:
        current_price: Current market price.
        dealing_range_high: Upper bound of the dealing range.
        dealing_range_low: Lower bound of the dealing range.

    Returns:
        :class:`PriceLocation` with zone classification and metrics.
    """
    range_size = dealing_range_high - dealing_range_low
    if range_size <= 0:
        # Degenerate range — return equilibrium defaults
        return PriceLocation(
            equilibrium=current_price,
            zone=PriceZone.EQUILIBRIUM,
            deviation_pct=0.0,
            dealing_range_high=dealing_range_high,
            dealing_range_low=dealing_range_low,
            dealing_range_size=0.0,
            ote_high=current_price,
            ote_low=current_price,
            in_ote=False,
            range_position_pct=0.5,
        )

    equilibrium = dealing_range_low + range_size * 0.5

    # Position within range: 0.0 = at low, 1.0 = at high
    # Clamp is NOT applied — price may be outside the range
    range_position = (current_price - dealing_range_low) / range_size

    # Deviation from equilibrium as fraction of range (signed)
    deviation = (current_price - equilibrium) / range_size

    # Classify zone
    if range_position > 0.786:
        zone = PriceZone.DEEP_PREMIUM
    elif range_position > 0.55:
        zone = PriceZone.PREMIUM
    elif range_position >= 0.45:
        zone = PriceZone.EQUILIBRIUM
    elif range_position >= 0.214:
        zone = PriceZone.DISCOUNT
    else:
        zone = PriceZone.DEEP_DISCOUNT

    # OTE zones (Fibonacci 61.8% to 78.6% retracement)
    # For longs: retracement from high = buying in the discount
    ote_long_high = dealing_range_low + range_size * 0.382
    ote_long_low = dealing_range_low + range_size * 0.214

    # For shorts: retracement from low = selling in the premium
    ote_short_high = dealing_range_low + range_size * 0.786
    ote_short_low = dealing_range_low + range_size * 0.618

    in_ote_long = ote_long_low <= current_price <= ote_long_high
    in_ote_short = ote_short_low <= current_price <= ote_short_high
    in_ote = in_ote_long or in_ote_short

    # Report the OTE zone relevant to the price location
    if range_position < 0.5:
        # Price is in discount → long OTE is relevant
        ote_high = ote_long_high
        ote_low = ote_long_low
    else:
        # Price is in premium → short OTE is relevant
        ote_high = ote_short_high
        ote_low = ote_short_low

    return PriceLocation(
        equilibrium=round(equilibrium, 8),
        zone=zone,
        deviation_pct=round(deviation, 6),
        dealing_range_high=dealing_range_high,
        dealing_range_low=dealing_range_low,
        dealing_range_size=round(range_size, 8),
        ote_high=round(ote_high, 8),
        ote_low=round(ote_low, 8),
        in_ote=in_ote,
        range_position_pct=round(range_position, 6),
    )
