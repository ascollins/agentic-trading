"""Tests for the CMTAnalystAgent."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_trading.agents.cmt_analyst import CMTAnalystAgent
from agentic_trading.core.config import CMTConfig
from agentic_trading.core.enums import AgentType, SignalDirection, Timeframe
from agentic_trading.core.events import CMTAssessment, Signal
from agentic_trading.core.interfaces import IAgent
from agentic_trading.intelligence.analysis.cmt_models import (
    CMTAssessmentResponse,
    CMTConfluenceScore,
    CMTLayerResult,
    CMTTarget,
    CMTTradePlan,
)


def _make_config(**overrides) -> CMTConfig:
    defaults = {
        "enabled": True,
        "analysis_interval_seconds": 60,
        "min_confluence_score": 5,
        "max_daily_api_calls": 10,
        "timeframes": ["1h", "4h"],
    }
    defaults.update(overrides)
    return CMTConfig(**defaults)


def _make_im_with_data() -> MagicMock:
    """Factory for an IntelligenceManager mock with candle data."""
    candle = MagicMock(timestamp="t", open=1, high=2, low=0, close=1.5, volume=10)
    im = MagicMock()
    im.get_buffer.return_value = [candle] * 5

    fe = MagicMock()
    fe.get_buffer.return_value = [candle] * 5
    fv = MagicMock()
    fv.features = {"rsi": 50.0}
    fe.compute_features.return_value = fv
    im.feature_engine = fe

    # HTF and SMC return proper dicts via model_dump
    htf_result = MagicMock()
    htf_result.model_dump.return_value = {"bias": "bullish"}
    im.analyze_htf.return_value = htf_result

    smc_result = MagicMock()
    smc_result.model_dump.return_value = {"score": 8}
    im.score_smc_confluence.return_value = smc_result

    return im


def _make_agent(
    *,
    symbols: list[str] | None = None,
    config: CMTConfig | None = None,
    engine: MagicMock | None = None,
    bus: AsyncMock | None = None,
    im: MagicMock | None = None,
) -> CMTAnalystAgent:
    return CMTAnalystAgent(
        intelligence_manager=im or MagicMock(),
        event_bus=bus or AsyncMock(),
        symbols=symbols or ["BTC/USDT"],
        engine=engine or MagicMock(),
        config=config or _make_config(),
        agent_id="cmt-test",
    )


def _make_response(
    *,
    threshold_met: bool = True,
    with_trade_plan: bool = True,
    total: float = 7.0,
) -> CMTAssessmentResponse:
    confluence = CMTConfluenceScore(
        trend_alignment=2.0,
        key_level_proximity=1.0,
        pattern_signal=1.0,
        indicator_consensus=1.5,
        sentiment_alignment=0.5,
        volatility_regime=0.5,
        macro_alignment=0.5,
    )
    confluence.total = total
    confluence.threshold_met = threshold_met
    confluence.veto = False

    trade_plan = None
    if with_trade_plan:
        trade_plan = CMTTradePlan(
            direction="LONG",
            entry_price=100_000.0,
            entry_trigger="Break above 100k",
            stop_loss=97_000.0,
            stop_reasoning="Below swing low",
            targets=[CMTTarget(price=105_000.0, pct=100.0, source="sr_level")],
            rr_ratio=1.67,
            blended_rr=1.5,
            position_size_pct=2.0,
        )

    return CMTAssessmentResponse(
        symbol="BTC/USDT",
        timeframes_analyzed=["H1", "H4"],
        layers=[
            CMTLayerResult(
                layer=1,
                name="Trend Identification",
                direction="bullish",
                confidence="high",
                score=1.5,
            ),
        ],
        confluence=confluence,
        trade_plan=trade_plan,
        thesis="Strong bullish setup with multi-timeframe alignment.",
        system_health="green",
    )


# ---------------------------------------------------------------------------
# Identity & protocol
# ---------------------------------------------------------------------------


class TestCMTAnalystAgentIdentity:
    def test_satisfies_iagent(self):
        agent = _make_agent()
        assert isinstance(agent, IAgent)

    def test_agent_type(self):
        agent = _make_agent()
        assert agent.agent_type == AgentType.CMT_ANALYST

    def test_capabilities(self):
        agent = _make_agent()
        caps = agent.capabilities()
        assert caps.subscribes_to == []
        assert "intelligence.cmt" in caps.publishes_to
        assert "strategy.signal" in caps.publishes_to

    def test_agent_id(self):
        agent = _make_agent()
        assert agent.agent_id == "cmt-test"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestCMTAnalystAgentLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        agent = _make_agent()
        await agent.start()
        assert agent.is_running
        await agent.stop()
        assert not agent.is_running

    @pytest.mark.asyncio
    async def test_health_check_when_running(self):
        agent = _make_agent()
        await agent.start()
        health = agent.health_check()
        assert health.healthy is True
        assert health.error_count == 0
        await agent.stop()

    @pytest.mark.asyncio
    async def test_health_check_when_stopped(self):
        agent = _make_agent()
        health = agent.health_check()
        assert health.healthy is False


# ---------------------------------------------------------------------------
# _build_request
# ---------------------------------------------------------------------------


class TestBuildRequest:
    def test_returns_none_when_no_data(self):
        im = MagicMock()
        im.get_buffer.return_value = None
        agent = _make_agent(im=im)
        result = agent._build_request("BTC/USDT")
        assert result is None

    def test_returns_request_with_data(self):
        candle = MagicMock()
        candle.timestamp = "2025-01-01T00:00:00"
        candle.open = 100
        candle.high = 105
        candle.low = 99
        candle.close = 103
        candle.volume = 1000

        im = MagicMock()
        im.get_buffer.return_value = [candle] * 5

        # Feature engine mock
        fe = MagicMock()
        fe.get_buffer.return_value = [candle] * 5
        fv = MagicMock()
        fv.features = {"rsi": 55.0}
        fe.compute_features.return_value = fv
        im.feature_engine = fe

        # HTF and SMC return proper dicts via model_dump
        htf_result = MagicMock()
        htf_result.model_dump.return_value = {"bias": "bullish"}
        im.analyze_htf.return_value = htf_result

        smc_result = MagicMock()
        smc_result.model_dump.return_value = {"score": 8}
        im.score_smc_confluence.return_value = smc_result

        agent = _make_agent(im=im)
        result = agent._build_request("BTC/USDT")
        assert result is not None
        assert result.symbol == "BTC/USDT"
        assert len(result.ohlcv_summary) > 0

    def test_handles_invalid_timeframe_gracefully(self):
        im = MagicMock()
        im.get_buffer.return_value = None
        config = _make_config(timeframes=["INVALID_TF"])
        agent = _make_agent(im=im, config=config)
        result = agent._build_request("BTC/USDT")
        assert result is None


# ---------------------------------------------------------------------------
# _work (main loop)
# ---------------------------------------------------------------------------


class TestWork:
    @pytest.mark.asyncio
    async def test_work_calls_analyze_for_each_symbol(self):
        engine = MagicMock()
        engine.assess = AsyncMock(return_value=_make_response())

        agent = _make_agent(
            symbols=["BTC/USDT", "ETH/USDT"],
            engine=engine,
        )
        # Mock _analyze_symbol to track calls
        agent._analyze_symbol = AsyncMock()
        await agent._work()
        assert agent._analyze_symbol.call_count == 2
        agent._analyze_symbol.assert_any_call("BTC/USDT")
        agent._analyze_symbol.assert_any_call("ETH/USDT")

    @pytest.mark.asyncio
    async def test_work_continues_on_symbol_failure(self):
        agent = _make_agent(symbols=["BTC/USDT", "ETH/USDT"])
        call_count = 0

        async def _analyze(symbol):
            nonlocal call_count
            call_count += 1
            if symbol == "BTC/USDT":
                raise RuntimeError("API error")

        agent._analyze_symbol = _analyze
        await agent._work()
        assert call_count == 2  # Both symbols attempted


# ---------------------------------------------------------------------------
# _analyze_symbol (full pipeline)
# ---------------------------------------------------------------------------


class TestAnalyzeSymbol:
    @pytest.mark.asyncio
    async def test_publishes_cmt_assessment_event(self):
        bus = AsyncMock()
        engine = MagicMock()
        resp = _make_response(threshold_met=False, with_trade_plan=False)
        engine.assess_with_thinking = AsyncMock(return_value=(resp, ""))
        engine.assess = AsyncMock(return_value=resp)
        im = _make_im_with_data()

        agent = _make_agent(bus=bus, engine=engine, im=im)
        await agent._analyze_symbol("BTC/USDT")

        # Should publish CMTAssessment on intelligence.cmt
        bus.publish.assert_called_once()
        call_args = bus.publish.call_args
        assert call_args[0][0] == "intelligence.cmt"
        event = call_args[0][1]
        assert isinstance(event, CMTAssessment)
        assert event.symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_emits_signal_when_threshold_met(self):
        bus = AsyncMock()
        engine = MagicMock()
        resp = _make_response(threshold_met=True)
        engine.assess_with_thinking = AsyncMock(return_value=(resp, ""))
        engine.assess = AsyncMock(return_value=resp)
        im = _make_im_with_data()

        agent = _make_agent(bus=bus, engine=engine, im=im)
        await agent._analyze_symbol("BTC/USDT")

        # Should have 2 publish calls: CMTAssessment + Signal
        assert bus.publish.call_count == 2

        signal_call = bus.publish.call_args_list[1]
        assert signal_call[0][0] == "strategy.signal"
        signal = signal_call[0][1]
        assert isinstance(signal, Signal)
        assert signal.strategy_id == "cmt_analyst"
        assert signal.symbol == "BTC/USDT"
        assert signal.direction == SignalDirection.LONG

    @pytest.mark.asyncio
    async def test_no_signal_when_threshold_not_met(self):
        bus = AsyncMock()
        engine = MagicMock()
        resp = _make_response(threshold_met=False, with_trade_plan=False)
        engine.assess_with_thinking = AsyncMock(return_value=(resp, ""))
        engine.assess = AsyncMock(return_value=resp)
        im = _make_im_with_data()

        agent = _make_agent(bus=bus, engine=engine, im=im)
        await agent._analyze_symbol("BTC/USDT")

        # Only 1 publish: CMTAssessment (no Signal)
        assert bus.publish.call_count == 1
        assert bus.publish.call_args[0][0] == "intelligence.cmt"

    @pytest.mark.asyncio
    async def test_skips_when_insufficient_data(self):
        bus = AsyncMock()
        engine = MagicMock()
        engine.assess_with_thinking = AsyncMock()
        engine.assess = AsyncMock()

        im = MagicMock()
        im.get_buffer.return_value = None  # No data

        agent = _make_agent(bus=bus, engine=engine, im=im)
        await agent._analyze_symbol("BTC/USDT")

        # Engine should not be called
        engine.assess_with_thinking.assert_not_called()
        bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_engine_returning_none(self):
        bus = AsyncMock()
        engine = MagicMock()
        engine.assess_with_thinking = AsyncMock(return_value=(None, ""))
        engine.assess = AsyncMock(return_value=None)
        im = _make_im_with_data()

        agent = _make_agent(bus=bus, engine=engine, im=im)
        await agent._analyze_symbol("BTC/USDT")

        # Nothing published when engine returns None
        bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_captures_reasoning_trace(self):
        bus = AsyncMock()
        engine = MagicMock()
        resp = _make_response(threshold_met=True)
        engine.assess_with_thinking = AsyncMock(
            return_value=(resp, "Extended thinking about the market...")
        )
        im = _make_im_with_data()

        agent = _make_agent(bus=bus, engine=engine, im=im)
        trace = await agent._analyze_symbol("BTC/USDT")

        assert trace is not None
        assert trace.outcome == "signal_emitted"
        assert trace.symbol == "BTC/USDT"
        assert len(trace.steps) > 0
        assert trace.raw_thinking == "Extended thinking about the market..."

    @pytest.mark.asyncio
    async def test_reasoning_trace_no_signal(self):
        bus = AsyncMock()
        engine = MagicMock()
        resp = _make_response(threshold_met=False, with_trade_plan=False)
        engine.assess_with_thinking = AsyncMock(return_value=(resp, ""))
        im = _make_im_with_data()

        agent = _make_agent(bus=bus, engine=engine, im=im)
        trace = await agent._analyze_symbol("BTC/USDT")

        assert trace is not None
        assert trace.outcome == "no_signal"


# ---------------------------------------------------------------------------
# _emit_signal
# ---------------------------------------------------------------------------


class TestEmitSignal:
    @pytest.mark.asyncio
    async def test_long_signal_properties(self):
        bus = AsyncMock()
        agent = _make_agent(bus=bus)
        response = _make_response(total=7.0)
        await agent._emit_signal("BTC/USDT", response)

        bus.publish.assert_called_once()
        signal = bus.publish.call_args[0][1]
        assert signal.strategy_id == "cmt_analyst"
        assert signal.direction == SignalDirection.LONG
        assert signal.stop_loss == Decimal("97000.0")
        assert signal.take_profit == Decimal("105000.0")
        assert signal.timeframe == Timeframe.H4
        assert 0.0 <= signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_short_signal(self):
        bus = AsyncMock()
        agent = _make_agent(bus=bus)
        response = _make_response()
        # Build a valid SHORT plan (stop above entry, target below entry)
        response.trade_plan = CMTTradePlan(
            direction="SHORT",
            entry_price=100_000.0,
            entry_trigger="Break below 100k support",
            stop_loss=103_000.0,
            stop_reasoning="Above swing high",
            targets=[CMTTarget(price=95_000.0, pct=100.0, source="sr_level")],
            rr_ratio=1.67,
            blended_rr=1.5,
            position_size_pct=2.0,
        )
        await agent._emit_signal("ETH/USDT", response)

        signal = bus.publish.call_args[0][1]
        assert signal.direction == SignalDirection.SHORT
        assert signal.symbol == "ETH/USDT"

    @pytest.mark.asyncio
    async def test_no_signal_for_null_trade_plan(self):
        bus = AsyncMock()
        agent = _make_agent(bus=bus)
        response = _make_response(with_trade_plan=False)
        await agent._emit_signal("BTC/USDT", response)
        bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_signal_for_unknown_direction(self):
        bus = AsyncMock()
        agent = _make_agent(bus=bus)
        response = _make_response()
        response.trade_plan.direction = "SIDEWAYS"
        await agent._emit_signal("BTC/USDT", response)
        bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_confidence_mapping(self):
        """Confluence of 7.0 should map to (7-3)/8 = 0.5."""
        bus = AsyncMock()
        agent = _make_agent(bus=bus)
        response = _make_response(total=7.0)
        await agent._emit_signal("BTC/USDT", response)

        signal = bus.publish.call_args[0][1]
        assert signal.confidence == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_confidence_clamped_to_zero(self):
        """Low confluence should clamp confidence to 0."""
        bus = AsyncMock()
        agent = _make_agent(bus=bus)
        response = _make_response(total=-5.0)
        await agent._emit_signal("BTC/USDT", response)

        signal = bus.publish.call_args[0][1]
        assert signal.confidence == 0.0

    @pytest.mark.asyncio
    async def test_confidence_clamped_to_one(self):
        """Very high confluence should clamp confidence to 1."""
        bus = AsyncMock()
        agent = _make_agent(bus=bus)
        response = _make_response(total=15.0)
        await agent._emit_signal("BTC/USDT", response)

        signal = bus.publish.call_args[0][1]
        assert signal.confidence == 1.0

    @pytest.mark.asyncio
    async def test_risk_constraints_included(self):
        bus = AsyncMock()
        agent = _make_agent(bus=bus)
        response = _make_response(total=7.0)
        await agent._emit_signal("BTC/USDT", response)

        signal = bus.publish.call_args[0][1]
        assert "cmt_confluence" in signal.risk_constraints
        assert "cmt_rr_ratio" in signal.risk_constraints
        assert "cmt_system_health" in signal.risk_constraints
        assert signal.risk_constraints["cmt_rr_ratio"] == 1.67
