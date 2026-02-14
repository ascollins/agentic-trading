"""Tests for narration schemas — DecisionExplanation and NarrationItem."""

from __future__ import annotations

import pytest

from agentic_trading.narration.schema import (
    ConsideredSetup,
    DecisionExplanation,
    NarrationItem,
    PositionSnapshot,
    RiskSummary,
)


class TestDecisionExplanation:
    def test_content_hash_stable(self):
        """Same content should produce the same hash."""
        exp1 = DecisionExplanation(symbol="BTC/USDT", action="ENTER")
        exp2 = DecisionExplanation(symbol="BTC/USDT", action="ENTER")
        assert exp1.content_hash() == exp2.content_hash()

    def test_content_hash_changes_with_action(self):
        exp1 = DecisionExplanation(symbol="BTC/USDT", action="ENTER")
        exp2 = DecisionExplanation(symbol="BTC/USDT", action="EXIT")
        assert exp1.content_hash() != exp2.content_hash()

    def test_content_hash_ignores_timestamp(self):
        """Hash should not change with different timestamps."""
        from datetime import datetime, timezone, timedelta
        t1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        t2 = t1 + timedelta(hours=1)
        exp1 = DecisionExplanation(symbol="BTC/USDT", action="HOLD", timestamp=t1)
        exp2 = DecisionExplanation(symbol="BTC/USDT", action="HOLD", timestamp=t2)
        assert exp1.content_hash() == exp2.content_hash()

    def test_default_values(self):
        exp = DecisionExplanation()
        assert exp.symbol == ""
        assert exp.action == ""
        assert exp.reasons == []
        assert exp.why_not == []
        assert exp.risk.health_score == 1.0

    def test_considered_setups(self):
        exp = DecisionExplanation(
            considered_setups=[
                ConsideredSetup(name="breakout", direction="long", confidence=0.8),
            ]
        )
        assert len(exp.considered_setups) == 1
        assert exp.considered_setups[0].name == "breakout"


class TestNarrationItem:
    def test_default_values(self):
        item = NarrationItem()
        assert item.script_text == ""
        assert item.published_text is False
        assert item.published_avatar is False

    def test_full_item(self):
        item = NarrationItem(
            script_id="abc123",
            script_text="Test narration text.",
            verbosity="normal",
            decision_ref="trace-123",
            sources=["action", "symbol"],
            playback_url="http://example.com/video",
            tavus_session_id="session-1",
            published_text=True,
            published_avatar=True,
        )
        assert item.script_id == "abc123"
        assert len(item.sources) == 2
        assert item.playback_url == "http://example.com/video"


class TestRiskSummary:
    def test_defaults(self):
        risk = RiskSummary()
        assert risk.health_score == 1.0
        assert risk.active_blocks == []

    def test_with_blocks(self):
        risk = RiskSummary(active_blocks=["circuit_breaker", "kill_switch"])
        assert len(risk.active_blocks) == 2


class TestPositionSnapshot:
    def test_defaults(self):
        pos = PositionSnapshot()
        assert pos.open_positions == 0
        assert pos.gross_exposure_usd == 0.0

    def test_with_values(self):
        pos = PositionSnapshot(
            open_positions=3,
            gross_exposure_usd=50000.0,
            unrealized_pnl_usd=-200.0,
        )
        assert pos.open_positions == 3
        assert pos.unrealized_pnl_usd == -200.0
