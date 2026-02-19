"""Tests for ReasoningTrace and ReasoningStep models."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from agentic_trading.core.enums import ReasoningPhase
from agentic_trading.reasoning.models import ReasoningStep, ReasoningTrace


class TestReasoningStep:
    def test_creation(self):
        step = ReasoningStep(
            phase=ReasoningPhase.PERCEPTION,
            content="Observed price at 65000",
            confidence=0.9,
        )
        assert step.phase == ReasoningPhase.PERCEPTION
        assert step.content == "Observed price at 65000"
        assert step.confidence == 0.9
        assert step.step_id  # Auto-generated

    def test_default_confidence(self):
        step = ReasoningStep(
            phase=ReasoningPhase.DECISION,
            content="No trade",
        )
        assert step.confidence == 0.0

    def test_metadata(self):
        step = ReasoningStep(
            phase=ReasoningPhase.EVALUATION,
            content="Checking confluence",
            metadata={"score": 7.5},
        )
        assert step.metadata["score"] == 7.5

    def test_serialization_round_trip(self):
        step = ReasoningStep(
            phase=ReasoningPhase.HYPOTHESIS,
            content="Trend is up",
            confidence=0.85,
            evidence={"ema_21": 65000, "ema_50": 63000},
        )
        data = step.model_dump(mode="json")
        restored = ReasoningStep.model_validate(data)
        assert restored.phase == step.phase
        assert restored.content == step.content
        assert restored.confidence == step.confidence
        assert restored.evidence == step.evidence


class TestReasoningTrace:
    def test_creation(self):
        trace = ReasoningTrace(
            agent_id="agent-1",
            agent_type="cmt_analyst",
            symbol="BTC/USDT",
        )
        assert trace.agent_id == "agent-1"
        assert trace.agent_type == "cmt_analyst"
        assert trace.symbol == "BTC/USDT"
        assert trace.trace_id  # Auto-generated
        assert trace.steps == []
        assert trace.outcome == ""

    def test_add_step(self):
        trace = ReasoningTrace(agent_id="a", agent_type="b")
        step = trace.add_step(
            ReasoningPhase.PERCEPTION,
            "Price is 65000",
            confidence=1.0,
        )
        assert len(trace.steps) == 1
        assert trace.steps[0] is step
        assert step.phase == ReasoningPhase.PERCEPTION

    def test_add_step_with_evidence(self):
        trace = ReasoningTrace(agent_id="a", agent_type="b")
        step = trace.add_step(
            ReasoningPhase.HYPOTHESIS,
            "Bullish trend",
            confidence=0.8,
            evidence={"adx": 28.0},
        )
        assert step.evidence == {"adx": 28.0}

    def test_add_step_with_metadata(self):
        trace = ReasoningTrace(agent_id="a", agent_type="b")
        step = trace.add_step(
            ReasoningPhase.DECISION,
            "Go long",
            confidence=0.9,
            strategy="trend_following",
            timeframe="4h",
        )
        assert step.metadata["strategy"] == "trend_following"

    def test_complete(self):
        trace = ReasoningTrace(agent_id="a", agent_type="b")
        assert trace.completed_at is None
        trace.complete("signal_emitted")
        assert trace.outcome == "signal_emitted"
        assert trace.completed_at is not None

    def test_duration_ms(self):
        trace = ReasoningTrace(agent_id="a", agent_type="b")
        assert trace.duration_ms == 0.0
        trace.complete("done")
        # Duration should be >= 0 (very small but non-negative)
        assert trace.duration_ms >= 0.0

    def test_duration_ms_not_completed(self):
        trace = ReasoningTrace(agent_id="a", agent_type="b")
        assert trace.duration_ms == 0.0

    def test_raw_thinking(self):
        trace = ReasoningTrace(
            agent_id="a",
            agent_type="cmt_analyst",
            raw_thinking="I think the market is trending up because...",
        )
        assert "trending up" in trace.raw_thinking

    def test_serialization_round_trip(self):
        trace = ReasoningTrace(
            agent_id="agent-1",
            agent_type="cmt_analyst",
            symbol="BTC/USDT",
            pipeline_id="pipe-123",
        )
        trace.add_step(
            ReasoningPhase.PERCEPTION, "Price at 65000", confidence=1.0
        )
        trace.add_step(
            ReasoningPhase.DECISION, "Go long", confidence=0.8
        )
        trace.complete("signal_emitted")

        data = trace.model_dump(mode="json")
        json_str = json.dumps(data, default=str)
        restored_data = json.loads(json_str)
        restored = ReasoningTrace.model_validate(restored_data)

        assert restored.agent_id == "agent-1"
        assert restored.symbol == "BTC/USDT"
        assert len(restored.steps) == 2
        assert restored.outcome == "signal_emitted"

    def test_multiple_phases(self):
        trace = ReasoningTrace(agent_id="a", agent_type="b")
        trace.add_step(ReasoningPhase.PERCEPTION, "Data gathered")
        trace.add_step(ReasoningPhase.HYPOTHESIS, "Bullish signal")
        trace.add_step(ReasoningPhase.EVALUATION, "Confluence 8/14")
        trace.add_step(ReasoningPhase.DECISION, "Enter long")
        trace.add_step(ReasoningPhase.ACTION, "Signal published")
        trace.complete("signal_emitted")

        assert len(trace.steps) == 5
        phases = [s.phase for s in trace.steps]
        assert phases == [
            ReasoningPhase.PERCEPTION,
            ReasoningPhase.HYPOTHESIS,
            ReasoningPhase.EVALUATION,
            ReasoningPhase.DECISION,
            ReasoningPhase.ACTION,
        ]
