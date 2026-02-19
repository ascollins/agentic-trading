"""Tests for the CMT Analysis Engine."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from agentic_trading.intelligence.analysis.cmt_engine import CMTAnalysisEngine
from agentic_trading.intelligence.analysis.cmt_models import (
    CMTAssessmentRequest,
    CMTAssessmentResponse,
)


def _make_engine(**overrides) -> CMTAnalysisEngine:
    """Factory for a CMTAnalysisEngine with test defaults."""
    defaults = {
        "skill_path": "/nonexistent",  # _load_skill returns ""
        "api_key_env": "TEST_API_KEY",
        "model": "claude-sonnet-4-5-20250929",
        "max_daily_calls": 10,
        "min_confluence": 5,
    }
    defaults.update(overrides)
    return CMTAnalysisEngine(**defaults)


def _make_request(**overrides) -> CMTAssessmentRequest:
    """Factory for a CMTAssessmentRequest with test defaults."""
    defaults = {
        "symbol": "BTC/USDT",
        "timeframes": ["H1", "H4", "D1"],
        "ohlcv_summary": {
            "H1": [{"t": "2025-01-01T00:00:00", "o": 100, "h": 105, "l": 99, "c": 103, "v": 1000}],
        },
        "indicator_values": {"H1_rsi": 55.0, "H4_macd_hist": 0.5},
    }
    defaults.update(overrides)
    return CMTAssessmentRequest(**defaults)


def _make_valid_response_json(**overrides) -> str:
    """Generate a valid CMT response JSON string."""
    data = {
        "symbol": "BTC/USDT",
        "timeframes_analyzed": ["H1", "H4", "D1"],
        "layers": [
            {
                "layer": 1,
                "name": "Trend Identification",
                "direction": "bullish",
                "confidence": "high",
                "score": 1.5,
                "key_findings": ["Uptrend intact"],
                "warnings": [],
            },
        ],
        "confluence": {
            "trend_alignment": 2.0,
            "key_level_proximity": 1.0,
            "pattern_signal": 1.0,
            "indicator_consensus": 1.5,
            "sentiment_alignment": 0.5,
            "volatility_regime": 0.0,
            "macro_alignment": 0.0,
        },
        "trade_plan": {
            "direction": "LONG",
            "entry_price": 100000.0,
            "entry_trigger": "Break above resistance",
            "stop_loss": 97000.0,
            "stop_reasoning": "Below swing low",
            "targets": [{"price": 105000.0, "pct": 50.0, "source": "measured_move"}],
            "rr_ratio": 1.67,
            "blended_rr": 1.5,
            "position_size_pct": 2.0,
            "invalidation": "Close below 96k",
            "thesis": "Bullish breakout",
        },
        "thesis": "Strong bullish setup with multi-timeframe confirmation",
        "system_health": "green",
        "watchlist_action": "enter",
        "no_trade_reason": "",
    }
    data.update(overrides)
    return json.dumps(data)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_includes_symbol(self):
        engine = _make_engine()
        prompt = engine.build_prompt(_make_request())
        assert "BTC/USDT" in prompt

    def test_includes_timeframes(self):
        engine = _make_engine()
        prompt = engine.build_prompt(_make_request())
        assert "H1" in prompt
        assert "H4" in prompt
        assert "D1" in prompt

    def test_includes_ohlcv_data(self):
        engine = _make_engine()
        prompt = engine.build_prompt(_make_request())
        assert "OHLCV Summary" in prompt

    def test_includes_indicator_values(self):
        engine = _make_engine()
        prompt = engine.build_prompt(_make_request())
        assert "Pre-Computed Indicator Values" in prompt
        assert "H1_rsi" in prompt

    def test_includes_response_schema(self):
        engine = _make_engine()
        prompt = engine.build_prompt(_make_request())
        assert "Respond with ONLY valid JSON" in prompt

    def test_includes_min_confluence(self):
        engine = _make_engine(min_confluence=7)
        prompt = engine.build_prompt(_make_request())
        assert "7" in prompt

    def test_omits_empty_sections(self):
        engine = _make_engine()
        req = _make_request(
            htf_assessment={},
            smc_confluence={},
            regime_state={},
            portfolio_state={},
            performance_metrics={},
        )
        prompt = engine.build_prompt(req)
        assert "Higher-Timeframe Assessment" not in prompt
        assert "SMC Confluence" not in prompt
        assert "Market Regime" not in prompt

    def test_includes_htf_when_present(self):
        engine = _make_engine()
        req = _make_request(htf_assessment={"bias": "bullish", "strength": 0.8})
        prompt = engine.build_prompt(req)
        assert "Higher-Timeframe Assessment" in prompt
        assert "bullish" in prompt

    def test_includes_smc_when_present(self):
        engine = _make_engine()
        req = _make_request(smc_confluence={"score": 8, "signals": ["order_block"]})
        prompt = engine.build_prompt(req)
        assert "SMC Confluence" in prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_valid_json(self):
        engine = _make_engine()
        resp = engine.parse_response(_make_valid_response_json())
        assert resp.symbol == "BTC/USDT"
        assert len(resp.layers) == 1
        assert resp.layers[0].layer == 1
        assert resp.layers[0].direction == "bullish"
        assert resp.trade_plan is not None
        assert resp.trade_plan.direction == "LONG"
        assert resp.thesis == "Strong bullish setup with multi-timeframe confirmation"

    def test_confluence_computed(self):
        engine = _make_engine()
        resp = engine.parse_response(_make_valid_response_json())
        assert resp.confluence.total == pytest.approx(6.0)
        assert resp.confluence.threshold_met is True
        assert resp.confluence.veto is False

    def test_strips_markdown_fences(self):
        engine = _make_engine()
        raw = "```json\n" + _make_valid_response_json() + "\n```"
        resp = engine.parse_response(raw)
        assert resp.symbol == "BTC/USDT"
        assert len(resp.layers) == 1

    def test_invalid_json_returns_error_response(self):
        engine = _make_engine()
        resp = engine.parse_response("this is not valid json {{{")
        assert resp.symbol == "unknown"
        assert "Parse error" in resp.thesis
        assert resp.system_health == "amber"

    def test_null_trade_plan(self):
        engine = _make_engine()
        raw = _make_valid_response_json(trade_plan=None)
        resp = engine.parse_response(raw)
        assert resp.trade_plan is None

    def test_no_trade_reason(self):
        engine = _make_engine()
        raw = _make_valid_response_json(
            trade_plan=None,
            no_trade_reason="Insufficient confluence",
        )
        resp = engine.parse_response(raw)
        assert resp.no_trade_reason == "Insufficient confluence"

    def test_threshold_not_met_below_min(self):
        """Confluence below min_confluence should not meet threshold."""
        engine = _make_engine(min_confluence=8)
        resp = engine.parse_response(_make_valid_response_json())
        # Default valid response has total ~6.0
        assert resp.confluence.threshold_met is False

    def test_veto_blocks_threshold(self):
        engine = _make_engine(min_confluence=3)
        raw_data = json.loads(_make_valid_response_json())
        raw_data["confluence"]["trend_alignment"] = -2.0
        resp = engine.parse_response(json.dumps(raw_data))
        assert resp.confluence.veto is True
        assert resp.confluence.threshold_met is False

    def test_missing_layers(self):
        engine = _make_engine()
        raw = json.dumps({"symbol": "BTC/USDT", "thesis": "test"})
        resp = engine.parse_response(raw)
        assert resp.layers == []
        assert resp.confluence.total == 0.0


# ---------------------------------------------------------------------------
# Budget management
# ---------------------------------------------------------------------------


class TestBudgetManagement:
    def test_initial_budget_available(self):
        engine = _make_engine(max_daily_calls=5)
        assert engine.calls_remaining_today == 5

    def test_budget_decrements(self):
        engine = _make_engine(max_daily_calls=5)
        # Sync the budget day counter first
        engine._check_budget()
        engine._record_call()
        assert engine.calls_remaining_today == 4
        engine._record_call()
        assert engine.calls_remaining_today == 3

    def test_budget_exhaustion(self):
        engine = _make_engine(max_daily_calls=2)
        assert engine._check_budget() is True
        engine._record_call()
        assert engine._check_budget() is True
        engine._record_call()
        assert engine._check_budget() is False

    def test_calls_remaining_never_negative(self):
        engine = _make_engine(max_daily_calls=1)
        engine._check_budget()  # Sync the budget day counter
        engine._record_call()
        engine._record_call()
        engine._record_call()
        assert engine.calls_remaining_today == 0


# ---------------------------------------------------------------------------
# Full assess pipeline
# ---------------------------------------------------------------------------


class TestAssess:
    @pytest.mark.asyncio
    async def test_assess_budget_exhausted_returns_none(self):
        engine = _make_engine(max_daily_calls=0)
        req = _make_request()
        result = await engine.assess(req)
        assert result is None

    @pytest.mark.asyncio
    async def test_assess_api_failure_returns_none(self):
        engine = _make_engine(max_daily_calls=10)
        engine.call_api = AsyncMock(side_effect=RuntimeError("API down"))
        req = _make_request()
        result = await engine.assess(req)
        assert result is None

    @pytest.mark.asyncio
    async def test_assess_success(self):
        engine = _make_engine(max_daily_calls=10)
        engine.call_api = AsyncMock(return_value=_make_valid_response_json())
        req = _make_request()
        result = await engine.assess(req)
        assert result is not None
        assert result.symbol == "BTC/USDT"
        assert engine.calls_remaining_today == 9

    @pytest.mark.asyncio
    async def test_assess_records_call_and_timestamp(self):
        engine = _make_engine(max_daily_calls=10)
        engine.call_api = AsyncMock(return_value=_make_valid_response_json())
        req = _make_request()
        await engine.assess(req)
        assert "BTC/USDT" in engine._last_call_per_symbol
        assert engine._calls_today == 1

    @pytest.mark.asyncio
    async def test_assess_overwrites_symbol_from_request(self):
        """The response symbol should match the request, not the JSON."""
        engine = _make_engine(max_daily_calls=10)
        engine.call_api = AsyncMock(
            return_value=_make_valid_response_json(symbol="WRONG/PAIR")
        )
        req = _make_request(symbol="ETH/USDT")
        result = await engine.assess(req)
        assert result is not None
        assert result.symbol == "ETH/USDT"
