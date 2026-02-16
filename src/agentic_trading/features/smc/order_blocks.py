"""Order Block (OB) and Fair Value Gap (FVG) detection.

Order blocks are consolidation zones before a displacement move. They represent
institutional accumulation/distribution zones and act as supply/demand areas.

Fair Value Gaps are imbalances in price delivery — gaps between candle ranges
that price tends to revisit (fill).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class BlockType(str, Enum):
    BULLISH = "bullish"  # Demand zone — last down candle before up move
    BEARISH = "bearish"  # Supply zone — last up candle before down move


class GapType(str, Enum):
    BULLISH = "bullish"  # Gap up — buying imbalance
    BEARISH = "bearish"  # Gap down — selling imbalance


@dataclass
class OrderBlock:
    """An identified order block zone."""

    index: int  # Index of the OB candle
    ob_high: float
    ob_low: float
    block_type: BlockType
    displacement_size: float  # Size of the move after the OB
    is_mitigated: bool = False  # True if price has returned to the zone


@dataclass
class FairValueGap:
    """An identified fair value gap (imbalance)."""

    index: int  # Index of the middle candle (candle 2 of the 3-candle pattern)
    gap_high: float  # Upper bound of the gap
    gap_low: float  # Lower bound of the gap
    gap_type: GapType
    gap_size: float
    is_filled: bool = False  # True if price has filled the gap


def detect_order_blocks(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    atr: np.ndarray,
    displacement_mult: float = 2.0,
    max_blocks: int = 20,
) -> list[OrderBlock]:
    """Detect order blocks using displacement analysis.

    An order block is the last opposing candle before a strong displacement
    move (range > displacement_mult * ATR).

    Bullish OB: Last bearish candle before a bullish displacement.
    Bearish OB: Last bullish candle before a bearish displacement.

    Args:
        opens: Open prices.
        highs: High prices.
        lows: Low prices.
        closes: Close prices.
        volumes: Volume data.
        atr: ATR values (same length as OHLCV).
        displacement_mult: Multiple of ATR to qualify as displacement.
        max_blocks: Maximum number of blocks to return (most recent).

    Returns:
        List of OrderBlock objects, most recent first.
    """
    n = len(closes)
    if n < 3:
        return []

    blocks: list[OrderBlock] = []
    candle_ranges = highs - lows
    is_bullish = closes > opens
    is_bearish = closes < opens

    for i in range(1, n - 1):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        displacement_threshold = atr[i] * displacement_mult

        # Check if candle i+1 (or i itself) is a displacement candle
        if i + 1 >= n:
            continue

        disp_range = candle_ranges[i + 1]

        # Bullish displacement: strong up candle
        if is_bullish[i + 1] and disp_range > displacement_threshold:
            # Look for the last bearish candle before displacement
            if is_bearish[i]:
                blocks.append(
                    OrderBlock(
                        index=i,
                        ob_high=float(highs[i]),
                        ob_low=float(lows[i]),
                        block_type=BlockType.BULLISH,
                        displacement_size=float(disp_range),
                    )
                )

        # Bearish displacement: strong down candle
        elif is_bearish[i + 1] and disp_range > displacement_threshold:
            # Look for the last bullish candle before displacement
            if is_bullish[i]:
                blocks.append(
                    OrderBlock(
                        index=i,
                        ob_high=float(highs[i]),
                        ob_low=float(lows[i]),
                        block_type=BlockType.BEARISH,
                        displacement_size=float(disp_range),
                    )
                )

    # Check mitigation: has price returned to the OB zone?
    if blocks and n > 0:
        current_price = float(closes[-1])
        for block in blocks:
            if block.block_type == BlockType.BULLISH:
                # Mitigated if price dropped back into or below the bullish OB
                mitigated = any(
                    float(lows[j]) <= block.ob_high
                    for j in range(block.index + 2, n)
                )
                block.is_mitigated = mitigated
            else:
                # Mitigated if price rose back into or above the bearish OB
                mitigated = any(
                    float(highs[j]) >= block.ob_low
                    for j in range(block.index + 2, n)
                )
                block.is_mitigated = mitigated

    # Return most recent blocks
    return blocks[-max_blocks:]


def detect_fvgs(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    min_gap_atr: float = 0.0,
    atr: np.ndarray | None = None,
    max_gaps: int = 20,
) -> list[FairValueGap]:
    """Detect Fair Value Gaps (3-candle imbalance patterns).

    Bullish FVG: candle_1.high < candle_3.low (gap between candle 1 and 3)
    Bearish FVG: candle_1.low > candle_3.high (gap between candle 1 and 3)

    Args:
        highs: High prices.
        lows: Low prices.
        closes: Close prices.
        min_gap_atr: Minimum gap size as multiple of ATR (0 = no filter).
        atr: ATR values for gap size filtering.
        max_gaps: Maximum number of gaps to return.

    Returns:
        List of FairValueGap objects.
    """
    n = len(highs)
    if n < 3:
        return []

    gaps: list[FairValueGap] = []

    for i in range(1, n - 1):
        candle1_high = highs[i - 1]
        candle3_low = lows[i + 1]
        candle1_low = lows[i - 1]
        candle3_high = highs[i + 1]

        # Bullish FVG: gap between candle 1 high and candle 3 low
        if candle3_low > candle1_high:
            gap_size = float(candle3_low - candle1_high)

            if min_gap_atr > 0 and atr is not None and not np.isnan(atr[i]):
                if gap_size < min_gap_atr * atr[i]:
                    continue

            gaps.append(
                FairValueGap(
                    index=i,
                    gap_high=float(candle3_low),
                    gap_low=float(candle1_high),
                    gap_type=GapType.BULLISH,
                    gap_size=gap_size,
                )
            )

        # Bearish FVG: gap between candle 3 high and candle 1 low
        elif candle1_low > candle3_high:
            gap_size = float(candle1_low - candle3_high)

            if min_gap_atr > 0 and atr is not None and not np.isnan(atr[i]):
                if gap_size < min_gap_atr * atr[i]:
                    continue

            gaps.append(
                FairValueGap(
                    index=i,
                    gap_high=float(candle1_low),
                    gap_low=float(candle3_high),
                    gap_type=GapType.BEARISH,
                    gap_size=gap_size,
                )
            )

    # Check fill status
    for gap in gaps:
        for j in range(gap.index + 2, n):
            if gap.gap_type == GapType.BULLISH:
                # Filled if price drops into the gap
                if float(lows[j]) <= gap.gap_high:
                    gap.is_filled = True
                    break
            else:
                # Filled if price rises into the gap
                if float(highs[j]) >= gap.gap_low:
                    gap.is_filled = True
                    break

    return gaps[-max_gaps:]
