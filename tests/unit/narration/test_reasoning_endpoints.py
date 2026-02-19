"""Tests for reasoning & chain-of-thought HTTP endpoints.

Covers all /reasoning/* endpoints used by the Grafana dashboard.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agentic_trading.context.fact_table import FactTable, PortfolioSnapshot
from agentic_trading.context.manager import ContextManager
from agentic_trading.context.memory_store import InMemoryMemoryStore, MemoryEntry
from agentic_trading.core.enums import (
    MemoryEntryType,
    PipelineOutcome,
)
from agentic_trading.narration.server import create_narration_app
from agentic_trading.narration.store import NarrationStore
from agentic_trading.narration.tavus import MockTavusAdapter
from agentic_trading.reasoning.pipeline_log import InMemoryPipelineLog
from agentic_trading.reasoning.pipeline_result import PipelineResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_trace(
    agent_type: str = "CMT_ANALYST",
    symbol: str = "BTC/USDT",
    outcome: str = "signal_emitted",
) -> dict:
    """Build a sample reasoning trace dict."""
    now = datetime.now(timezone.utc)
    return {
        "trace_id": "trace-001",
        "pipeline_id": "",
        "agent_id": "agent-abc12345",
        "agent_type": agent_type,
        "symbol": symbol,
        "started_at": now.isoformat(),
        "completed_at": now.isoformat(),
        "steps": [
            {
                "step_id": "step-1",
                "phase": "perception",
                "content": "BTC at 65000, uptrend on 4H",
                "confidence": 0.0,
                "timestamp": now.isoformat(),
                "metadata": {},
                "evidence": {},
            },
            {
                "step_id": "step-2",
                "phase": "hypothesis",
                "content": "Bullish continuation expected",
                "confidence": 0.75,
                "timestamp": now.isoformat(),
                "metadata": {},
                "evidence": {},
            },
            {
                "step_id": "step-3",
                "phase": "evaluation",
                "content": "Confluence score 8.2/10 above threshold",
                "confidence": 0.82,
                "timestamp": now.isoformat(),
                "metadata": {},
                "evidence": {"confluence_score": 8.2},
            },
            {
                "step_id": "step-4",
                "phase": "decision",
                "content": "Emit LONG signal with 0.82 confidence",
                "confidence": 0.82,
                "timestamp": now.isoformat(),
                "metadata": {},
                "evidence": {},
            },
            {
                "step_id": "step-5",
                "phase": "action",
                "content": "Published signal to strategy.signal topic",
                "confidence": 0.0,
                "timestamp": now.isoformat(),
                "metadata": {},
                "evidence": {},
            },
        ],
        "outcome": outcome,
        "raw_thinking": "I see price at 65000 trending up...",
    }


def _make_pipeline_result(
    pipeline_id: str = "pipe-001",
    symbol: str = "BTC/USDT",
    outcome: PipelineOutcome = PipelineOutcome.SIGNAL_EMITTED,
    trace_count: int = 1,
) -> PipelineResult:
    """Build a sample PipelineResult."""
    now = datetime.now(timezone.utc)
    traces = []
    for i in range(trace_count):
        t = _make_trace(symbol=symbol)
        t["pipeline_id"] = pipeline_id
        traces.append(t)

    return PipelineResult(
        pipeline_id=pipeline_id,
        started_at=now,
        completed_at=now,
        trigger_event_type="market.candle",
        trigger_symbol=symbol,
        trigger_timeframe="4h",
        reasoning_traces=traces,
        outcome=outcome,
        signals=[{"strategy_id": "smc_trend", "direction": "long"}],
    )


@pytest.fixture
def pipeline_log() -> InMemoryPipelineLog:
    log = InMemoryPipelineLog()
    log.save(_make_pipeline_result("pipe-001", "BTC/USDT", PipelineOutcome.SIGNAL_EMITTED))
    log.save(_make_pipeline_result("pipe-002", "ETH/USDT", PipelineOutcome.NO_SIGNAL))
    log.save(_make_pipeline_result("pipe-003", "BTC/USDT", PipelineOutcome.SIGNAL_EMITTED, trace_count=2))
    return log


@pytest.fixture
def context_manager() -> ContextManager:
    facts = FactTable()
    facts.update_portfolio(PortfolioSnapshot(
        total_equity=50000.0,
        gross_exposure=12000.0,
        daily_pnl=350.0,
        open_position_count=1,
        positions={"BTC/USDT": {"qty": 0.15, "entry_price": 64000.0}},
    ))
    facts.update_risk(current_drawdown_pct=0.02)
    facts.update_regime("BTC/USDT", {"regime": "trending"})
    facts.update_price("BTC/USDT", last=65000.0, bid=64999.0, ask=65001.0)

    memory = InMemoryMemoryStore()
    memory.store(MemoryEntry(
        entry_type=MemoryEntryType.CMT_ASSESSMENT,
        content={"thesis": "bullish", "confluence": 8.2},
        symbol="BTC/USDT",
        summary="CMT bullish on BTC 4H — confluence 8.2/10",
        tags=["bullish", "4h"],
    ))
    memory.store(MemoryEntry(
        entry_type=MemoryEntryType.RISK_EVENT,
        content={"event": "circuit_breaker_cleared"},
        symbol="",
        summary="Daily loss circuit breaker cleared",
        tags=["risk"],
    ))

    return ContextManager(fact_table=facts, memory_store=memory)


@pytest.fixture
def app(pipeline_log, context_manager):
    store = NarrationStore()
    tavus = MockTavusAdapter(base_url="http://localhost:8099")
    return create_narration_app(
        store=store,
        tavus=tavus,
        pipeline_log=pipeline_log,
        context_manager=context_manager,
    )


@pytest.fixture
def app_no_reasoning():
    """App without pipeline_log or context_manager."""
    store = NarrationStore()
    tavus = MockTavusAdapter(base_url="http://localhost:8099")
    return create_narration_app(store=store, tavus=tavus)


# ===========================================================================
# GET /reasoning/stats
# ===========================================================================

class TestReasoningStats:
    @pytest.mark.asyncio
    async def test_returns_stats(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/stats")
        assert resp.status == 200
        data = await resp.json()
        assert data["total_pipelines"] == 3
        assert data["signals_emitted"] == 2
        assert data["no_signals"] == 1
        assert data["errors"] == 0
        assert data["agents_active"] >= 1
        assert data["total_memories"] == 2
        assert data["avg_confidence"] > 0

    @pytest.mark.asyncio
    async def test_stats_without_pipeline_log(self, app_no_reasoning, aiohttp_client):
        client = await aiohttp_client(app_no_reasoning)
        resp = await client.get("/reasoning/stats")
        assert resp.status == 200
        data = await resp.json()
        assert data["total_pipelines"] == 0
        assert data["total_memories"] == 0


# ===========================================================================
# GET /reasoning/pipelines
# ===========================================================================

class TestReasoningPipelines:
    @pytest.mark.asyncio
    async def test_returns_all_pipelines(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/pipelines")
        assert resp.status == 200
        data = await resp.json()
        assert isinstance(data, list)
        assert len(data) == 3

    @pytest.mark.asyncio
    async def test_pipeline_fields(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/pipelines?limit=1")
        data = await resp.json()
        row = data[0]
        assert "pipeline_id" in row
        assert "timestamp" in row
        assert "symbol" in row
        assert "outcome" in row
        assert "duration_ms" in row
        assert "agents" in row
        assert "trace_count" in row
        assert "signal_count" in row

    @pytest.mark.asyncio
    async def test_filter_by_symbol(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/pipelines?symbol=ETH/USDT")
        data = await resp.json()
        assert len(data) == 1
        assert data[0]["symbol"] == "ETH/USDT"

    @pytest.mark.asyncio
    async def test_limit_param(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/pipelines?limit=2")
        data = await resp.json()
        assert len(data) == 2

    @pytest.mark.asyncio
    async def test_empty_when_no_log(self, app_no_reasoning, aiohttp_client):
        client = await aiohttp_client(app_no_reasoning)
        resp = await client.get("/reasoning/pipelines")
        data = await resp.json()
        assert data == []


# ===========================================================================
# GET /reasoning/pipeline/{pipeline_id}
# ===========================================================================

class TestReasoningPipelineDetail:
    @pytest.mark.asyncio
    async def test_returns_detail(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/pipeline/pipe-001")
        assert resp.status == 200
        data = await resp.json()
        assert data["pipeline_id"] == "pipe-001"
        assert "chain_of_thought" in data
        assert "Pipeline pipe-001" in data["chain_of_thought"]
        assert data["outcome"] == "signal_emitted"
        assert isinstance(data["traces"], list)
        assert len(data["traces"]) == 1

    @pytest.mark.asyncio
    async def test_not_found(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/pipeline/nonexistent")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_no_log_returns_404(self, app_no_reasoning, aiohttp_client):
        client = await aiohttp_client(app_no_reasoning)
        resp = await client.get("/reasoning/pipeline/pipe-001")
        assert resp.status == 404


# ===========================================================================
# GET /reasoning/traces
# ===========================================================================

class TestReasoningTraces:
    @pytest.mark.asyncio
    async def test_returns_traces(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/traces")
        assert resp.status == 200
        data = await resp.json()
        assert isinstance(data, list)
        # 3 pipelines: pipe-001 has 1 trace, pipe-002 has 1, pipe-003 has 2 = 4
        assert len(data) == 4

    @pytest.mark.asyncio
    async def test_trace_fields(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/traces?limit=1")
        data = await resp.json()
        row = data[0]
        assert "agent_type" in row
        assert "symbol" in row
        assert "decision" in row
        assert "confidence" in row
        assert "phase_summary" in row
        assert "step_count" in row
        assert row["step_count"] == 5

    @pytest.mark.asyncio
    async def test_filter_by_symbol(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/traces?symbol=ETH/USDT")
        data = await resp.json()
        assert len(data) == 1
        assert data[0]["symbol"] == "ETH/USDT"

    @pytest.mark.asyncio
    async def test_confidence_extracted(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/traces?limit=1")
        data = await resp.json()
        # Max confidence across steps is 0.82
        assert data[0]["confidence"] == 0.82

    @pytest.mark.asyncio
    async def test_empty_when_no_log(self, app_no_reasoning, aiohttp_client):
        client = await aiohttp_client(app_no_reasoning)
        resp = await client.get("/reasoning/traces")
        data = await resp.json()
        assert data == []


# ===========================================================================
# GET /reasoning/steps
# ===========================================================================

class TestReasoningSteps:
    @pytest.mark.asyncio
    async def test_returns_steps(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/steps")
        assert resp.status == 200
        data = await resp.json()
        assert isinstance(data, list)
        # 3 pipelines with 1+1+2 traces * 5 steps each = 20
        assert len(data) == 20

    @pytest.mark.asyncio
    async def test_step_fields(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/steps?limit=1")
        data = await resp.json()
        row = data[0]
        assert "phase" in row
        assert "content" in row
        assert "confidence" in row
        assert "agent_type" in row
        assert "symbol" in row

    @pytest.mark.asyncio
    async def test_filter_by_phase(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/steps?phase=decision")
        data = await resp.json()
        # Each trace has 1 decision step, 4 traces total = 4
        assert len(data) == 4
        assert all(r["phase"] == "decision" for r in data)

    @pytest.mark.asyncio
    async def test_perception_phase(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/steps?phase=perception")
        data = await resp.json()
        assert len(data) == 4
        assert all(r["phase"] == "perception" for r in data)

    @pytest.mark.asyncio
    async def test_filter_by_symbol(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/steps?symbol=ETH/USDT")
        data = await resp.json()
        # 1 pipeline with 1 trace * 5 steps = 5
        assert len(data) == 5

    @pytest.mark.asyncio
    async def test_empty_when_no_log(self, app_no_reasoning, aiohttp_client):
        client = await aiohttp_client(app_no_reasoning)
        resp = await client.get("/reasoning/steps")
        data = await resp.json()
        assert data == []


# ===========================================================================
# GET /reasoning/context
# ===========================================================================

class TestReasoningContext:
    @pytest.mark.asyncio
    async def test_returns_context(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/context")
        assert resp.status == 200
        data = await resp.json()
        assert "trending" in data["regime"]
        assert data["equity"] == 50000.0
        assert data["gross_exposure"] == 12000.0
        assert data["daily_pnl"] == 350.0
        assert data["drawdown_pct"] == 0.02
        assert data["kill_switch"] is False

    @pytest.mark.asyncio
    async def test_includes_prices(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/context")
        data = await resp.json()
        assert isinstance(data["prices"], list)
        assert len(data["prices"]) == 1
        btc = data["prices"][0]
        assert btc["symbol"] == "BTC/USDT"
        assert btc["last"] == 65000.0

    @pytest.mark.asyncio
    async def test_no_context_manager(self, app_no_reasoning, aiohttp_client):
        client = await aiohttp_client(app_no_reasoning)
        resp = await client.get("/reasoning/context")
        data = await resp.json()
        assert "error" in data


# ===========================================================================
# GET /reasoning/memories
# ===========================================================================

class TestReasoningMemories:
    @pytest.mark.asyncio
    async def test_returns_memories(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/memories")
        assert resp.status == 200
        data = await resp.json()
        assert isinstance(data, list)
        assert len(data) == 2

    @pytest.mark.asyncio
    async def test_memory_fields(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/memories")
        data = await resp.json()
        row = data[0]
        assert "entry_type" in row
        assert "symbol" in row
        assert "summary" in row
        assert "relevance" in row
        assert "tags" in row

    @pytest.mark.asyncio
    async def test_filter_by_symbol(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/memories?symbol=BTC/USDT")
        data = await resp.json()
        assert len(data) == 1
        assert "BTC" in data[0]["summary"]

    @pytest.mark.asyncio
    async def test_filter_by_type(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/memories?type=risk_event")
        data = await resp.json()
        assert len(data) == 1
        assert data[0]["entry_type"] == "risk_event"

    @pytest.mark.asyncio
    async def test_limit_param(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/reasoning/memories?limit=1")
        data = await resp.json()
        assert len(data) == 1

    @pytest.mark.asyncio
    async def test_empty_when_no_context(self, app_no_reasoning, aiohttp_client):
        client = await aiohttp_client(app_no_reasoning)
        resp = await client.get("/reasoning/memories")
        data = await resp.json()
        assert data == []


# ===========================================================================
# CORS
# ===========================================================================

class TestReasoningCORS:
    @pytest.mark.asyncio
    async def test_cors_on_reasoning_endpoints(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        for path in [
            "/reasoning/stats",
            "/reasoning/pipelines",
            "/reasoning/traces",
            "/reasoning/steps",
            "/reasoning/context",
            "/reasoning/memories",
        ]:
            resp = await client.get(path)
            assert resp.headers.get("Access-Control-Allow-Origin") == "*", (
                f"Missing CORS header on {path}"
            )
