"""Intelligence layer — market data ingestion, feature computation.

This package is the new canonical location for modules that were previously
split across ``agentic_trading.data`` and ``agentic_trading.features``.

Sub-packages:
    features/   Technical indicators, feature engine, SMC detection.
    analysis/   HTF analysis, SMC confluence, trade plan generation.
"""

from agentic_trading.intelligence.manager import IntelligenceManager

__all__ = ["IntelligenceManager"]
