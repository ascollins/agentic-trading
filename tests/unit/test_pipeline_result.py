"""Tests for PipelineResult and PipelineLog."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agentic_trading.core.enums import PipelineOutcome, ReasoningPhase
from agentic_trading.reasoning.pipeline_log import (
    InMemoryPipelineLog,
    PipelineLog,
)
from agentic_trading.reasoning.pipeline_result import PipelineResult


def _make_pipeline_result(
    symbol: str = "BTC/USDT",
    outcome: PipelineOutcome = PipelineOutcome.SIGNAL_EMITTED,
) -> PipelineResult:
    """Create a test pipeline result."""
    return PipelineResult(
        trigger_event_type="feature_vector",
        trigger_symbol=symbol,
        trigger_timeframe="4h",
        outcome=outcome,
        reasoning_traces=[
            {
                "agent_id": "agent-1",
                "agent_type": "cmt_analyst",
                "symbol": symbol,
                "outcome": "signal_emitted",
                "steps": [
                    {
                        "phase": ReasoningPhase.PERCEPTION.value,
                        "content": "Price at 65000",
                        "confidence": 1.0,
                    },
                    {
                        "phase": ReasoningPhase.DECISION.value,
                        "content": "Go long",
                        "confidence": 0.8,
                    },
                ],
            }
        ],
        signals=[{"strategy_id": "cmt_analyst", "direction": "long"}],
    )


class TestPipelineResult:
    def test_creation(self):
        result = PipelineResult()
        assert result.pipeline_id  # Auto-generated
        assert result.outcome == PipelineOutcome.NO_SIGNAL
        assert result.reasoning_traces == []

    def test_duration_ms_not_completed(self):
        result = PipelineResult()
        assert result.duration_ms == 0.0

    def test_duration_ms_completed(self):
        result = PipelineResult()
        result.finalize()
        assert result.duration_ms >= 0.0
        assert result.completed_at is not None

    def test_print_chain_of_thought(self):
        result = _make_pipeline_result()
        result.finalize()

        output = result.print_chain_of_thought()
        assert "Pipeline" in output
        assert "feature_vector" in output
        assert "BTC/USDT" in output
        assert "cmt_analyst" in output
        assert "Price at 65000" in output
        assert "[perception]" in output
        assert "[decision]" in output
        assert "signal_emitted" in output

    def test_print_chain_includes_extended_thinking(self):
        result = PipelineResult(
            reasoning_traces=[
                {
                    "agent_id": "a",
                    "agent_type": "cmt",
                    "steps": [],
                    "raw_thinking": "Claude was thinking deeply about..."
                }
            ]
        )
        output = result.print_chain_of_thought()
        assert "[extended_thinking]" in output
        assert "Claude was thinking deeply" in output

    def test_to_json(self):
        result = _make_pipeline_result()
        result.finalize()
        json_str = result.to_json()
        data = json.loads(json_str)
        assert data["pipeline_id"] == result.pipeline_id
        assert data["outcome"] == "signal_emitted"

    def test_serialization_round_trip(self):
        result = _make_pipeline_result()
        result.finalize()
        data = result.model_dump(mode="json")
        restored = PipelineResult.model_validate(data)
        assert restored.pipeline_id == result.pipeline_id
        assert restored.outcome == result.outcome
        assert len(restored.reasoning_traces) == 1


class TestInMemoryPipelineLog:
    def test_save_and_load(self):
        log = InMemoryPipelineLog()
        result = _make_pipeline_result()
        log.save(result)

        loaded = log.load(result.pipeline_id)
        assert loaded is not None
        assert loaded.pipeline_id == result.pipeline_id

    def test_load_unknown_returns_none(self):
        log = InMemoryPipelineLog()
        assert log.load("nonexistent") is None

    def test_count(self):
        log = InMemoryPipelineLog()
        assert log.count == 0
        log.save(_make_pipeline_result())
        assert log.count == 1

    def test_query_by_symbol(self):
        log = InMemoryPipelineLog()
        log.save(_make_pipeline_result(symbol="BTC/USDT"))
        log.save(_make_pipeline_result(symbol="ETH/USDT"))

        results = log.query(symbol="BTC/USDT")
        assert len(results) == 1
        assert results[0].trigger_symbol == "BTC/USDT"

    def test_query_limit(self):
        log = InMemoryPipelineLog()
        for _ in range(10):
            log.save(_make_pipeline_result())

        results = log.query(limit=3)
        assert len(results) == 3

    def test_query_since(self):
        log = InMemoryPipelineLog()

        old = _make_pipeline_result()
        old.started_at = datetime.now(timezone.utc) - timedelta(hours=2)
        log.save(old)

        recent = _make_pipeline_result()
        log.save(recent)

        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        results = log.query(since=cutoff)
        assert len(results) == 1

    def test_clear(self):
        log = InMemoryPipelineLog()
        log.save(_make_pipeline_result())
        log.clear()
        assert log.count == 0


class TestPipelineLog:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipelines.jsonl"
            log = PipelineLog(path)

            result = _make_pipeline_result()
            log.save(result)

            loaded = log.load(result.pipeline_id)
            assert loaded is not None
            assert loaded.pipeline_id == result.pipeline_id

    def test_persistence_across_instances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipelines.jsonl"

            result = _make_pipeline_result()

            log1 = PipelineLog(path)
            log1.save(result)
            del log1

            log2 = PipelineLog(path)
            loaded = log2.load(result.pipeline_id)
            assert loaded is not None
            assert loaded.trigger_symbol == "BTC/USDT"

    def test_query_by_symbol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipelines.jsonl"
            log = PipelineLog(path)
            log.save(_make_pipeline_result(symbol="BTC/USDT"))
            log.save(_make_pipeline_result(symbol="ETH/USDT"))

            results = log.query(symbol="ETH/USDT")
            assert len(results) == 1
