"""Shared fixtures for narration tests."""

from __future__ import annotations

import pytest

from agentic_trading.narration.schema import (
    ConsideredSetup,
    DecisionExplanation,
    PositionSnapshot,
    RiskSummary,
)


@pytest.fixture
def enter_explanation() -> DecisionExplanation:
    """Sample DecisionExplanation for an ENTER action."""
    return DecisionExplanation(
        symbol="BTC/USDT",
        timeframe="5m",
        market_summary="Price has been rising steadily over the last hour.",
        active_strategy="trend_following",
        active_regime="trend",
        regime_confidence=0.82,
        considered_setups=[
            ConsideredSetup(name="momentum breakout", direction="long", confidence=0.8, status="triggered"),
            ConsideredSetup(name="range fade", direction="short", confidence=0.3, status="invalidated"),
        ],
        action="ENTER",
        reasons=["Strong upward momentum", "Volume confirming the move"],
        reason_confidences=[0.85, 0.72],
        risk=RiskSummary(
            intended_size_pct=0.03,
            stop_invalidation="Below recent support",
            health_score=0.95,
            maturity_level="L3_constrained",
        ),
        position=PositionSnapshot(
            open_positions=1,
            gross_exposure_usd=5000.0,
            unrealized_pnl_usd=120.0,
        ),
        trace_id="trace-abc-123",
    )


@pytest.fixture
def no_trade_explanation() -> DecisionExplanation:
    """Sample DecisionExplanation for a NO_TRADE action."""
    return DecisionExplanation(
        symbol="ETH/USDT",
        timeframe="15m",
        market_summary="Market is choppy with no clear direction.",
        active_strategy="mean_reversion",
        active_regime="range",
        action="NO_TRADE",
        reasons=["No clear setup"],
        why_not=["Spread too wide", "Volatility above threshold"],
        what_would_change=["Spread narrows below 5 bps", "Volatility drops below 2%"],
        risk=RiskSummary(
            active_blocks=["circuit_breaker_spread"],
            health_score=0.65,
        ),
        trace_id="trace-def-456",
    )


@pytest.fixture
def hold_explanation() -> DecisionExplanation:
    """Sample DecisionExplanation for a HOLD action."""
    return DecisionExplanation(
        symbol="BTC/USDT",
        active_strategy="trend_following",
        active_regime="trend",
        action="HOLD",
        reasons=["Trend still intact"],
        position=PositionSnapshot(
            open_positions=1,
            gross_exposure_usd=10000.0,
            unrealized_pnl_usd=350.0,
        ),
        trace_id="trace-ghi-789",
    )
