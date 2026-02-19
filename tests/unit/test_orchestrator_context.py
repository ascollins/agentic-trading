"""Tests for Orchestrator context, pipeline log, and explain() features."""

from __future__ import annotations

from agentic_trading.context.fact_table import FactTable
from agentic_trading.context.manager import ContextManager
from agentic_trading.context.memory_store import InMemoryMemoryStore
from agentic_trading.core.enums import PipelineOutcome
from agentic_trading.reasoning.pipeline_log import InMemoryPipelineLog
from agentic_trading.reasoning.pipeline_result import PipelineResult


def _make_orchestrator_parts():
    """Create minimal orchestrator parts for testing."""
    context_mgr = ContextManager(
        fact_table=FactTable(),
        memory_store=InMemoryMemoryStore(),
    )
    pipeline_log = InMemoryPipelineLog()
    return context_mgr, pipeline_log


class TestOrchestratorExplain:
    def test_explain_unknown_returns_not_found(self):
        _, pipeline_log = _make_orchestrator_parts()

        # Simulate orchestrator.explain() logic
        result = pipeline_log.load("nonexistent-id")
        assert result is None

    def test_explain_known_returns_chain_of_thought(self):
        _, pipeline_log = _make_orchestrator_parts()

        pr = PipelineResult(
            trigger_event_type="feature_vector",
            trigger_symbol="BTC/USDT",
            trigger_timeframe="4h",
            outcome=PipelineOutcome.SIGNAL_EMITTED,
            reasoning_traces=[
                {
                    "agent_id": "agent-1",
                    "agent_type": "cmt_analyst",
                    "symbol": "BTC/USDT",
                    "steps": [
                        {
                            "phase": "perception",
                            "content": "Observed uptrend",
                            "confidence": 0.9,
                        }
                    ],
                }
            ],
        )
        pr.finalize()
        pipeline_log.save(pr)

        loaded = pipeline_log.load(pr.pipeline_id)
        assert loaded is not None
        output = loaded.print_chain_of_thought()
        assert "cmt_analyst" in output
        assert "Observed uptrend" in output


class TestOrchestratorPipelineHistory:
    def test_empty_history(self):
        _, pipeline_log = _make_orchestrator_parts()
        results = pipeline_log.query()
        assert results == []

    def test_query_by_symbol(self):
        _, pipeline_log = _make_orchestrator_parts()

        for sym in ["BTC/USDT", "ETH/USDT", "BTC/USDT"]:
            pr = PipelineResult(
                trigger_symbol=sym,
                outcome=PipelineOutcome.NO_SIGNAL,
            )
            pipeline_log.save(pr)

        results = pipeline_log.query(symbol="BTC/USDT")
        assert len(results) == 2

    def test_query_with_limit(self):
        _, pipeline_log = _make_orchestrator_parts()

        for i in range(10):
            pr = PipelineResult(
                trigger_symbol="BTC/USDT",
                outcome=PipelineOutcome.NO_SIGNAL,
            )
            pipeline_log.save(pr)

        results = pipeline_log.query(limit=5)
        assert len(results) == 5


class TestContextManagerInOrchestrator:
    def test_context_manager_initialized(self):
        context_mgr, _ = _make_orchestrator_parts()
        assert context_mgr.facts is not None
        assert context_mgr.memory is not None

    def test_read_context_before_any_data(self):
        context_mgr, _ = _make_orchestrator_parts()
        ctx = context_mgr.read_context()
        assert ctx.fact_snapshot is not None
        assert ctx.relevant_memories == []

    def test_write_and_read_memory(self):
        from agentic_trading.core.enums import MemoryEntryType

        context_mgr, _ = _make_orchestrator_parts()

        context_mgr.write_analysis(
            MemoryEntryType.CMT_ASSESSMENT,
            {"thesis": "Bullish momentum building"},
            symbol="BTC/USDT",
            summary="Bullish BTC",
        )

        ctx = context_mgr.read_context(symbol="BTC/USDT")
        assert len(ctx.relevant_memories) == 1
        assert ctx.relevant_memories[0].summary == "Bullish BTC"
