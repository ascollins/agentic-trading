"""Smart Money Concepts (SMC) feature detection.

Provides institutional order flow analysis:
- Swing point detection (HH/HL/LH/LL)
- Order block identification
- Fair Value Gap (FVG) detection
- Break of Structure (BOS) / Change of Character (CHoCH)
- Liquidity sweep detection (BSL/SSL)
- Premium / Discount zone classification
"""

from .computer import SMCFeatureComputer
from .liquidity_sweeps import LiquiditySweep, SweepType, detect_liquidity_sweeps
from .price_location import (
    PriceLocation,
    PriceZone,
    classify_price_location,
    compute_dealing_range,
)

__all__ = [
    "SMCFeatureComputer",
    "LiquiditySweep",
    "SweepType",
    "detect_liquidity_sweeps",
    "PriceLocation",
    "PriceZone",
    "classify_price_location",
    "compute_dealing_range",
]
