"""Backward-compat re-export — canonical location: ``agentic_trading.intelligence.features.smc``.

Will be removed in PR 16.
"""

from agentic_trading.intelligence.features.smc import *  # noqa: F401, F403
from agentic_trading.intelligence.features.smc import (  # noqa: F811
    SMCFeatureComputer,
    LiquiditySweep,
    SweepType,
    detect_liquidity_sweeps,
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
