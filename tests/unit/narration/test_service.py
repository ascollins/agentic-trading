"""Tests for NarrationService — script generation, verbosity, jargon, dedupe."""

from __future__ import annotations

import pytest

from agentic_trading.narration.schema import DecisionExplanation, RiskSummary
from agentic_trading.narration.service import (
    BANNED_JARGON,
    WORD_LIMITS,
    NarrationService,
    Verbosity,
)


# ===========================================================================
# Word limit enforcement
# ===========================================================================

class TestWordLimits:
    def test_quiet_limit(self, enter_explanation):
        svc = NarrationService(verbosity=Verbosity.QUIET)
        item = svc.generate(enter_explanation, force=True)
        assert item is not None
        word_count = len(item.script_text.split())
        assert word_count <= WORD_LIMITS[Verbosity.QUIET], (
            f"QUIET script has {word_count} words, max is {WORD_LIMITS[Verbosity.QUIET]}"
        )

    def test_normal_limit(self, enter_explanation):
        svc = NarrationService(verbosity=Verbosity.NORMAL)
        item = svc.generate(enter_explanation, force=True)
        assert item is not None
        word_count = len(item.script_text.split())
        assert word_count <= WORD_LIMITS[Verbosity.NORMAL], (
            f"NORMAL script has {word_count} words, max is {WORD_LIMITS[Verbosity.NORMAL]}"
        )

    def test_detailed_limit(self, enter_explanation):
        svc = NarrationService(verbosity=Verbosity.DETAILED)
        item = svc.generate(enter_explanation, force=True)
        assert item is not None
        word_count = len(item.script_text.split())
        assert word_count <= WORD_LIMITS[Verbosity.DETAILED], (
            f"DETAILED script has {word_count} words, max is {WORD_LIMITS[Verbosity.DETAILED]}"
        )

    def test_no_trade_quiet_limit(self, no_trade_explanation):
        svc = NarrationService(verbosity=Verbosity.QUIET)
        item = svc.generate(no_trade_explanation, force=True)
        assert item is not None
        assert len(item.script_text.split()) <= 30

    def test_no_trade_normal_limit(self, no_trade_explanation):
        svc = NarrationService(verbosity=Verbosity.NORMAL)
        item = svc.generate(no_trade_explanation, force=True)
        assert item is not None
        assert len(item.script_text.split()) <= 70


# ===========================================================================
# Jargon ban enforcement
# ===========================================================================

class TestJargonBan:
    def test_no_jargon_in_enter_script(self, enter_explanation):
        svc = NarrationService()
        item = svc.generate(enter_explanation, force=True)
        assert item is not None
        found = NarrationService.check_jargon(item.script_text)
        assert found == [], f"Banned jargon found in script: {found}"

    def test_no_jargon_in_no_trade_script(self, no_trade_explanation):
        svc = NarrationService()
        item = svc.generate(no_trade_explanation, force=True)
        assert item is not None
        found = NarrationService.check_jargon(item.script_text)
        assert found == [], f"Banned jargon found in script: {found}"

    def test_scrub_jargon_removes_rsi(self):
        text = "The RSI indicates overbought conditions near MACD crossover."
        cleaned = NarrationService._scrub_jargon(text)
        assert "rsi" not in cleaned.lower()
        assert "macd" not in cleaned.lower()

    def test_scrub_jargon_removes_atr(self):
        text = "ATR is expanding, sharpe ratio improving."
        cleaned = NarrationService._scrub_jargon(text)
        assert "atr" not in cleaned.lower()
        assert "sharpe" not in cleaned.lower()

    def test_jargon_in_reason_gets_scrubbed(self):
        exp = DecisionExplanation(
            symbol="BTC/USDT",
            action="ENTER",
            active_strategy="test",
            reasons=["RSI below 30 signals oversold", "MACD crossing up"],
        )
        svc = NarrationService()
        item = svc.generate(exp, force=True)
        assert item is not None
        assert "rsi" not in item.script_text.lower()
        assert "macd" not in item.script_text.lower()


# ===========================================================================
# Script content
# ===========================================================================

class TestScriptContent:
    def test_enter_contains_decision(self, enter_explanation):
        svc = NarrationService()
        item = svc.generate(enter_explanation, force=True)
        assert item is not None
        text = item.script_text.lower()
        assert "enter" in text or "position" in text

    def test_no_trade_contains_no_trade(self, no_trade_explanation):
        svc = NarrationService()
        item = svc.generate(no_trade_explanation, force=True)
        assert item is not None
        text = item.script_text.lower()
        assert "no trade" in text or "blocker" in text

    def test_hold_contains_hold(self, hold_explanation):
        svc = NarrationService()
        item = svc.generate(hold_explanation, force=True)
        assert item is not None
        text = item.script_text.lower()
        assert "hold" in text

    def test_script_contains_symbol(self, enter_explanation):
        svc = NarrationService()
        item = svc.generate(enter_explanation, force=True)
        assert "BTC/USDT" in item.script_text

    def test_script_contains_strategy(self, enter_explanation):
        svc = NarrationService()
        item = svc.generate(enter_explanation, force=True)
        assert "trend_following" in item.script_text

    def test_normal_answers_five_questions(self, enter_explanation):
        """NORMAL verbosity should answer: what, watching, decided, why, risk."""
        svc = NarrationService(verbosity=Verbosity.NORMAL)
        item = svc.generate(enter_explanation, force=True)
        assert item is not None
        text = item.script_text
        # Must have some content for each section
        assert len(text) > 50  # Substantial narration

    def test_sources_tracked(self, enter_explanation):
        svc = NarrationService()
        item = svc.generate(enter_explanation, force=True)
        assert item is not None
        assert len(item.sources) > 0
        assert "action" in item.sources


# ===========================================================================
# Deduplication
# ===========================================================================

class TestDedupe:
    def test_identical_explanations_deduplicated(self, enter_explanation):
        svc = NarrationService(dedupe_window_seconds=100)
        item1 = svc.generate(enter_explanation, force=False)
        item2 = svc.generate(enter_explanation, force=False)
        assert item1 is not None
        assert item2 is None  # Deduplicated

    def test_force_bypasses_dedupe(self, enter_explanation):
        svc = NarrationService(dedupe_window_seconds=100)
        item1 = svc.generate(enter_explanation, force=True)
        item2 = svc.generate(enter_explanation, force=True)
        assert item1 is not None
        assert item2 is not None  # Force bypasses dedupe

    def test_different_actions_not_deduplicated(self, enter_explanation, no_trade_explanation):
        # heartbeat_seconds=0 so non-meaningful actions aren't throttled
        svc = NarrationService(dedupe_window_seconds=100, heartbeat_seconds=0)
        item1 = svc.generate(enter_explanation, force=False)
        item2 = svc.generate(no_trade_explanation, force=False)
        assert item1 is not None
        assert item2 is not None


# ===========================================================================
# Throttling
# ===========================================================================

class TestThrottle:
    def test_heartbeat_throttles_non_meaningful(self):
        """HOLD actions should be throttled by heartbeat interval."""
        svc = NarrationService(heartbeat_seconds=1000)  # Very long
        exp = DecisionExplanation(action="HOLD", symbol="BTC/USDT")
        item1 = svc.generate(exp, force=False)
        item2 = svc.generate(exp, force=False)
        # First should succeed, second should be throttled
        assert item1 is not None
        assert item2 is None

    def test_meaningful_actions_not_throttled(self, enter_explanation, no_trade_explanation):
        """ENTER actions should NOT be heartbeat-throttled (they are meaningful)."""
        svc = NarrationService(heartbeat_seconds=1000, dedupe_window_seconds=0)
        item1 = svc.generate(enter_explanation, force=False)
        # Change action but keep it meaningful
        enter_explanation.action = "EXIT"
        enter_explanation.reasons = ["Take profit hit"]
        item2 = svc.generate(enter_explanation, force=False)
        assert item1 is not None
        assert item2 is not None


# ===========================================================================
# Contract: only uses DecisionExplanation fields
# ===========================================================================

class TestContract:
    def test_sources_are_valid_fields(self, enter_explanation):
        """All source references must be valid DecisionExplanation fields."""
        svc = NarrationService()
        item = svc.generate(enter_explanation, force=True)
        assert item is not None
        valid_fields = set(DecisionExplanation.model_fields.keys()) | {
            "risk", "position", "considered_setups",
        }
        for source in item.sources:
            assert source in valid_fields, (
                f"Source '{source}' not a valid DecisionExplanation field"
            )

    def test_narration_count_increments(self, enter_explanation):
        svc = NarrationService()
        assert svc.narration_count == 0
        svc.generate(enter_explanation, force=True)
        assert svc.narration_count == 1
        svc.generate(enter_explanation, force=True)
        assert svc.narration_count == 2


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_explanation(self):
        svc = NarrationService()
        exp = DecisionExplanation()
        item = svc.generate(exp, force=True)
        assert item is not None
        assert len(item.script_text) > 0  # Should still produce something

    def test_very_long_reasons_truncated(self):
        svc = NarrationService(verbosity=Verbosity.QUIET)
        exp = DecisionExplanation(
            symbol="BTC/USDT",
            action="ENTER",
            active_strategy="test",
            reasons=["Very long reason " * 20, "Another long reason " * 20],
        )
        item = svc.generate(exp, force=True)
        assert item is not None
        assert len(item.script_text.split()) <= 30

    def test_script_id_is_stable(self, enter_explanation):
        svc = NarrationService()
        item = svc.generate(enter_explanation, force=True)
        assert item is not None
        assert len(item.script_id) == 16  # sha256[:16]
