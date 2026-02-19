"""No-lookahead / data leakage verification.

Tests that strategies cannot access future data.
Should be run as part of the test suite and fail the build if violated.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from agentic_trading.core.clock import SimClock
from agentic_trading.core.enums import Timeframe
from agentic_trading.core.events import FeatureVector
from agentic_trading.core.interfaces import TradingContext
from agentic_trading.core.models import Candle

logger = logging.getLogger(__name__)


class LeakageDetector:
    """Detects data leakage / lookahead bias in strategies.

    Technique: feed candles one-by-one, verify that no feature or signal
    references timestamps beyond the current candle.

    Also checks:
    - Features only use data up to current candle timestamp
    - Strategy does not modify past state based on future data
    - No global mutable state shared between runs
    """

    def __init__(self) -> None:
        self._violations: list[dict[str, Any]] = []

    def check_feature_vector(
        self,
        features: FeatureVector,
        current_time: datetime,
    ) -> list[str]:
        """Check that a FeatureVector doesn't reference future data."""
        violations = []

        if features.timestamp > current_time:
            violations.append(
                f"FeatureVector timestamp {features.timestamp} > current {current_time}"
            )

        return violations

    def check_strategy_determinism(
        self,
        strategy: Any,
        candles: list[Candle],
        feature_vectors: list[FeatureVector],
        ctx_factory: Any,
    ) -> list[str]:
        """Run strategy twice on same data, verify identical outputs.

        If outputs differ, there's hidden mutable state (potential leakage source).
        """
        violations = []

        # First run
        signals_1 = []
        for candle, fv in zip(candles, feature_vectors):
            ctx = ctx_factory(candle.timestamp)
            sig = strategy.on_candle(ctx, candle, fv)
            signals_1.append(sig)

        # Second run (reset strategy state)
        strategy_copy = type(strategy)(
            strategy_id=strategy.strategy_id,
            params=strategy.get_parameters(),
        )
        signals_2 = []
        for candle, fv in zip(candles, feature_vectors):
            ctx = ctx_factory(candle.timestamp)
            sig = strategy_copy.on_candle(ctx, candle, fv)
            signals_2.append(sig)

        # Compare
        for i, (s1, s2) in enumerate(zip(signals_1, signals_2)):
            if s1 is None and s2 is None:
                continue
            if s1 is None or s2 is None:
                violations.append(
                    f"Candle {i}: signal mismatch (one None, one not)"
                )
                continue
            if s1.direction != s2.direction or abs(s1.confidence - s2.confidence) > 1e-9:
                violations.append(
                    f"Candle {i}: signal differs between runs "
                    f"({s1.direction}/{s1.confidence} vs {s2.direction}/{s2.confidence})"
                )

        return violations

    def check_no_future_candles(
        self,
        candle_index: int,
        all_candles: list[Candle],
        accessed_indices: set[int],
    ) -> list[str]:
        """Verify that only candles up to candle_index were accessed."""
        violations = []
        for idx in accessed_indices:
            if idx > candle_index:
                violations.append(
                    f"Accessed candle at index {idx} (future) "
                    f"while processing candle {candle_index}"
                )
        return violations
