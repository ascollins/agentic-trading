"""Break of Structure (BOS) and Change of Character (CHoCH) detection.

BOS occurs when price closes beyond a previous swing point in the
direction of the existing trend — confirming continuation.

CHoCH occurs when price closes beyond a previous swing point AGAINST
the existing trend — signalling a potential reversal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .swing_detection import StructureLabel, SwingPoint, SwingType


class BreakType(str, Enum):
    BOS = "BOS"    # Break of Structure (trend continuation)
    CHOCH = "CHoCH"  # Change of Character (trend reversal)


class BreakDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class StructureBreak:
    """A detected structure break event."""

    index: int  # Bar index where the break was confirmed
    break_type: BreakType
    direction: BreakDirection
    level: float  # The swing level that was broken
    confirmed: bool = True  # Body close beyond level (not just wick)


def detect_structure_breaks(
    swings: list[SwingPoint],
    closes: np.ndarray,
) -> list[StructureBreak]:
    """Detect BOS and CHoCH events from swing points and closes.

    Logic:
    - Track the current trend from the swing sequence (HH/HL = uptrend, LH/LL = downtrend)
    - When a close exceeds a previous swing high → bullish break
    - When a close drops below a previous swing low → bearish break
    - If the break is WITH the trend → BOS
    - If the break is AGAINST the trend → CHoCH

    Args:
        swings: Sorted list of confirmed swing points.
        closes: Array of close prices.

    Returns:
        List of StructureBreak events, sorted by index.
    """
    if len(swings) < 3 or len(closes) < 2:
        return []

    breaks: list[StructureBreak] = []

    # Determine current trend from swing sequence
    trend = _determine_trend(swings)

    # Track the most recent unbroken swing high and low
    last_swing_high: SwingPoint | None = None
    last_swing_low: SwingPoint | None = None

    for swing in swings:
        if swing.swing_type == SwingType.HIGH:
            last_swing_high = swing
        else:
            last_swing_low = swing

    # Check if any close after the last swing breaks the level
    if last_swing_high is not None:
        high_level = last_swing_high.price
        for i in range(last_swing_high.index + 1, len(closes)):
            if closes[i] > high_level:
                # Bullish break — is it BOS or CHoCH?
                if trend == "bullish":
                    break_type = BreakType.BOS
                else:
                    break_type = BreakType.CHOCH

                breaks.append(
                    StructureBreak(
                        index=i,
                        break_type=break_type,
                        direction=BreakDirection.BULLISH,
                        level=high_level,
                        confirmed=True,
                    )
                )
                break  # Only detect the first break per swing level

    if last_swing_low is not None:
        low_level = last_swing_low.price
        for i in range(last_swing_low.index + 1, len(closes)):
            if closes[i] < low_level:
                # Bearish break — is it BOS or CHoCH?
                if trend == "bearish":
                    break_type = BreakType.BOS
                else:
                    break_type = BreakType.CHOCH

                breaks.append(
                    StructureBreak(
                        index=i,
                        break_type=break_type,
                        direction=BreakDirection.BEARISH,
                        level=low_level,
                        confirmed=True,
                    )
                )
                break

    breaks.sort(key=lambda b: b.index)
    return breaks


def detect_all_structure_breaks(
    swings: list[SwingPoint],
    closes: np.ndarray,
) -> list[StructureBreak]:
    """Detect all historical BOS/CHoCH events across the full swing sequence.

    Walks through swings chronologically, tracking trend state and detecting
    breaks as they occur.

    Args:
        swings: Sorted list of confirmed swing points.
        closes: Array of close prices.

    Returns:
        All structure break events detected historically.
    """
    if len(swings) < 4 or len(closes) < 2:
        return []

    breaks: list[StructureBreak] = []
    trend = "unknown"

    # Separate into swing highs and lows
    swing_highs: list[SwingPoint] = []
    swing_lows: list[SwingPoint] = []

    for swing in swings:
        if swing.swing_type == SwingType.HIGH:
            # Check for break of previous swing high
            if swing_highs:
                prev_high = swing_highs[-1]
                # Check if any close between previous high and this one broke it
                for j in range(prev_high.index + 1, min(swing.index, len(closes))):
                    if closes[j] > prev_high.price:
                        direction = BreakDirection.BULLISH
                        bt = BreakType.BOS if trend == "bullish" else BreakType.CHOCH
                        if trend == "unknown":
                            bt = BreakType.BOS

                        breaks.append(
                            StructureBreak(
                                index=j,
                                break_type=bt,
                                direction=direction,
                                level=prev_high.price,
                            )
                        )
                        if bt == BreakType.CHOCH:
                            trend = "bullish"
                        break

            swing_highs.append(swing)

            # Update trend from structure
            if len(swing_highs) >= 2:
                if swing_highs[-1].price > swing_highs[-2].price:
                    if trend != "bullish":
                        trend = "bullish"

        else:  # LOW
            # Check for break of previous swing low
            if swing_lows:
                prev_low = swing_lows[-1]
                for j in range(prev_low.index + 1, min(swing.index, len(closes))):
                    if closes[j] < prev_low.price:
                        direction = BreakDirection.BEARISH
                        bt = BreakType.BOS if trend == "bearish" else BreakType.CHOCH
                        if trend == "unknown":
                            bt = BreakType.BOS

                        breaks.append(
                            StructureBreak(
                                index=j,
                                break_type=bt,
                                direction=direction,
                                level=prev_low.price,
                            )
                        )
                        if bt == BreakType.CHOCH:
                            trend = "bearish"
                        break

            swing_lows.append(swing)

            if len(swing_lows) >= 2:
                if swing_lows[-1].price < swing_lows[-2].price:
                    if trend != "bearish":
                        trend = "bearish"

    breaks.sort(key=lambda b: b.index)
    return breaks


def _determine_trend(swings: list[SwingPoint]) -> str:
    """Determine current trend from the last few swings.

    Returns "bullish", "bearish", or "unknown".
    """
    # Get the last 4 swings and check for HH/HL or LH/LL patterns
    recent = swings[-6:] if len(swings) >= 6 else swings
    highs = [s for s in recent if s.swing_type == SwingType.HIGH]
    lows = [s for s in recent if s.swing_type == SwingType.LOW]

    bullish_signals = 0
    bearish_signals = 0

    if len(highs) >= 2:
        if highs[-1].price > highs[-2].price:
            bullish_signals += 1  # Higher high
        else:
            bearish_signals += 1  # Lower high

    if len(lows) >= 2:
        if lows[-1].price > lows[-2].price:
            bullish_signals += 1  # Higher low
        else:
            bearish_signals += 1  # Lower low

    if bullish_signals > bearish_signals:
        return "bullish"
    elif bearish_signals > bullish_signals:
        return "bearish"
    return "unknown"
