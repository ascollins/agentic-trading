"""Tests for ReasoningChain."""

from __future__ import annotations

from agentic_trading.core.enums import ReasoningPhase
from agentic_trading.reasoning.chain import ReasoningChain
from agentic_trading.reasoning.models import ReasoningTrace


class TestReasoningChain:
    def test_creation(self):
        chain = ReasoningChain("pipe-123")
        assert chain.pipeline_id == "pipe-123"
        assert chain.traces == []

    def test_create_trace(self):
        chain = ReasoningChain("pipe-123")
        trace = chain.create_trace(
            "agent-1", "cmt_analyst", symbol="BTC/USDT"
        )
        assert isinstance(trace, ReasoningTrace)
        assert trace.pipeline_id == "pipe-123"
        assert trace.agent_id == "agent-1"
        assert trace.agent_type == "cmt_analyst"
        assert trace.symbol == "BTC/USDT"
        assert len(chain.traces) == 1

    def test_add_trace(self):
        chain = ReasoningChain("pipe-123")
        trace = ReasoningTrace(
            agent_id="agent-2",
            agent_type="risk_gate",
            pipeline_id="pipe-123",
        )
        chain.add_trace(trace)
        assert len(chain.traces) == 1
        assert chain.traces[0].agent_id == "agent-2"

    def test_multiple_traces(self):
        chain = ReasoningChain("pipe-123")
        chain.create_trace("agent-1", "intelligence")
        chain.create_trace("agent-2", "signal")
        chain.create_trace("agent-3", "execution")
        assert len(chain.traces) == 3

    def test_format_chain_of_thought(self):
        chain = ReasoningChain("pipe-123")

        t1 = chain.create_trace("agent-1", "intelligence", "BTC/USDT")
        t1.add_step(
            ReasoningPhase.PERCEPTION,
            "Price at 65000",
            confidence=1.0,
        )
        t1.add_step(
            ReasoningPhase.HYPOTHESIS,
            "Trend is bullish",
            confidence=0.8,
        )
        t1.complete("signal_emitted")

        t2 = chain.create_trace("agent-2", "risk_gate", "BTC/USDT")
        t2.add_step(
            ReasoningPhase.EVALUATION,
            "Risk check passed",
            confidence=1.0,
        )
        t2.complete("no_signal")

        output = chain.format_chain_of_thought()
        assert "Pipeline pipe-123" in output
        assert "intelligence" in output
        assert "risk_gate" in output
        assert "Price at 65000" in output
        assert "[perception]" in output
        assert "[80%]" in output

    def test_format_includes_raw_thinking(self):
        chain = ReasoningChain("pipe-123")
        trace = chain.create_trace("agent-1", "cmt_analyst")
        trace.raw_thinking = "Extended thinking content here"
        trace.complete("done")

        output = chain.format_chain_of_thought()
        assert "[extended_thinking]" in output
        assert "Extended thinking content" in output

    def test_to_dict(self):
        chain = ReasoningChain("pipe-123")
        trace = chain.create_trace("agent-1", "cmt_analyst")
        trace.add_step(
            ReasoningPhase.PERCEPTION, "Data gathered"
        )
        trace.complete("done")

        data = chain.to_dict()
        assert data["pipeline_id"] == "pipe-123"
        assert len(data["traces"]) == 1
        assert data["traces"][0]["agent_id"] == "agent-1"

    def test_traces_returns_copy(self):
        chain = ReasoningChain("pipe-123")
        chain.create_trace("agent-1", "test")
        traces = chain.traces
        traces.clear()  # Modifying the returned list
        assert len(chain.traces) == 1  # Original not affected
