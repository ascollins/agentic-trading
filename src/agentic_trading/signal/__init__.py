"""Signal layer — strategy execution, signal generation, portfolio management.

This package groups the signal-generation side of the pipeline.

Sub-packages:
    strategies/   All trading strategies, registry, regime detection, research tooling.
    portfolio/    Position sizing, allocation, correlation risk, intent conversion.
"""

from agentic_trading.signal.manager import SignalManager

__all__ = ["SignalManager"]
