"""Simulated clock for deterministic backtesting.

Re-exports SimClock from core.clock for convenience.
Adds backtest-specific helpers.
"""

from agentic_trading.core.clock import SimClock

__all__ = ["SimClock"]
