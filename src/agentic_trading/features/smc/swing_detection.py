"""Swing point detection and structure classification.

Identifies swing highs/lows using a fractal-based approach, then classifies
the sequence into Higher Highs (HH), Higher Lows (HL), Lower Highs (LH),
and Lower Lows (LL) to determine market structure bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class SwingType(str, Enum):
    HIGH = "high"
    LOW = "low"


class StructureLabel(str, Enum):
    HH = "HH"  # Higher High
    HL = "HL"  # Higher Low
    LH = "LH"  # Lower High
    LL = "LL"  # Lower Low


@dataclass
class SwingPoint:
    """A confirmed swing high or low."""

    index: int
    price: float
    swing_type: SwingType


@dataclass
class StructureClassification:
    """Classification of a swing point within market structure."""

    index: int
    price: float
    swing_type: SwingType
    label: StructureLabel


def detect_swing_highs(
    highs: np.ndarray,
    lookback: int = 5,
) -> list[SwingPoint]:
    """Detect swing highs using a fractal approach.

    A swing high at index i requires that highs[i] is the highest
    value in the window [i - lookback, i + lookback].

    Args:
        highs: Array of high prices.
        lookback: Number of bars on each side to confirm the swing.

    Returns:
        List of SwingPoint objects for confirmed swing highs.
    """
    swings: list[SwingPoint] = []
    n = len(highs)

    for i in range(lookback, n - lookback):
        window = highs[i - lookback : i + lookback + 1]
        if highs[i] == np.max(window) and np.sum(window == highs[i]) == 1:
            swings.append(
                SwingPoint(index=i, price=float(highs[i]), swing_type=SwingType.HIGH)
            )

    return swings


def detect_swing_lows(
    lows: np.ndarray,
    lookback: int = 5,
) -> list[SwingPoint]:
    """Detect swing lows using a fractal approach.

    A swing low at index i requires that lows[i] is the lowest
    value in the window [i - lookback, i + lookback].

    Args:
        lows: Array of low prices.
        lookback: Number of bars on each side to confirm the swing.

    Returns:
        List of SwingPoint objects for confirmed swing lows.
    """
    swings: list[SwingPoint] = []
    n = len(lows)

    for i in range(lookback, n - lookback):
        window = lows[i - lookback : i + lookback + 1]
        if lows[i] == np.min(window) and np.sum(window == lows[i]) == 1:
            swings.append(
                SwingPoint(index=i, price=float(lows[i]), swing_type=SwingType.LOW)
            )

    return swings


def detect_all_swings(
    highs: np.ndarray,
    lows: np.ndarray,
    lookback: int = 5,
) -> list[SwingPoint]:
    """Detect all swing points (highs and lows), sorted by index."""
    swing_highs = detect_swing_highs(highs, lookback)
    swing_lows = detect_swing_lows(lows, lookback)
    all_swings = swing_highs + swing_lows
    all_swings.sort(key=lambda s: s.index)
    return all_swings


def classify_structure(
    swings: list[SwingPoint],
) -> list[StructureClassification]:
    """Classify swing points into HH/HL/LH/LL structure.

    Compares each swing to the most recent swing of the same type
    (high vs high, low vs low) to determine if it's higher or lower.

    Args:
        swings: Sorted list of swing points (mixed highs and lows).

    Returns:
        List of StructureClassification objects.
    """
    if len(swings) < 2:
        return []

    classifications: list[StructureClassification] = []
    last_high: SwingPoint | None = None
    last_low: SwingPoint | None = None

    for swing in swings:
        if swing.swing_type == SwingType.HIGH:
            if last_high is not None:
                label = (
                    StructureLabel.HH
                    if swing.price > last_high.price
                    else StructureLabel.LH
                )
                classifications.append(
                    StructureClassification(
                        index=swing.index,
                        price=swing.price,
                        swing_type=swing.swing_type,
                        label=label,
                    )
                )
            last_high = swing
        else:  # LOW
            if last_low is not None:
                label = (
                    StructureLabel.HL
                    if swing.price > last_low.price
                    else StructureLabel.LL
                )
                classifications.append(
                    StructureClassification(
                        index=swing.index,
                        price=swing.price,
                        swing_type=swing.swing_type,
                        label=label,
                    )
                )
            last_low = swing

    return classifications


def compute_swing_bias(classifications: list[StructureClassification]) -> float:
    """Compute a bias score from structure classifications.

    Returns:
        Float in [-1, +1]. Positive = bullish (HH/HL dominant),
        negative = bearish (LH/LL dominant).
    """
    if not classifications:
        return 0.0

    # Weight recent classifications more heavily
    score = 0.0
    total_weight = 0.0

    for i, c in enumerate(classifications):
        weight = 1.0 + i * 0.5  # Linearly increasing weight for recency
        if c.label in (StructureLabel.HH, StructureLabel.HL):
            score += weight
        else:  # LH or LL
            score -= weight
        total_weight += weight

    return score / total_weight if total_weight > 0 else 0.0
