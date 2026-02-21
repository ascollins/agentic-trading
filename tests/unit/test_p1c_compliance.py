"""Tests for P1c sprint: compliance and modelling infrastructure.

Covers:
- SurveillanceAgent detection rules (Task #13)
- Compliance case management (Task #14)
- ModelRegistry lifecycle (Task #15)
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agentic_trading.agents.surveillance import SurveillanceAgent
from agentic_trading.compliance.case_manager import (
    CaseManager,
    ComplianceCase,
    _VALID_TRANSITIONS,
)
from agentic_trading.core.enums import (
    AgentType,
    Exchange,
    OrderStatus,
    Side,
)
from agentic_trading.core.events import (
    FillEvent,
    OrderAck,
    OrderIntent,
    OrderUpdate,
    SurveillanceCaseEvent,
)
from agentic_trading.intelligence.model_registry import (
    ModelRecord,
    ModelRegistry,
    ModelStage,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_bus() -> AsyncMock:
    bus = AsyncMock()
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    return bus


def _make_fill(
    symbol: str = "BTCUSDT",
    side: Side = Side.BUY,
    price: float = 50000.0,
    qty: float = 0.01,
    strategy_id: str = "strat-1",
) -> FillEvent:
    return FillEvent(
        fill_id=f"fill-{id(side)}",
        order_id="ord-1",
        client_order_id="dk-1",
        symbol=symbol,
        exchange=Exchange.BYBIT,
        side=side,
        price=Decimal(str(price)),
        qty=Decimal(str(qty)),
        fee=Decimal("0.01"),
        fee_currency="USDT",
        strategy_id=strategy_id,
    )


def _make_intent(
    symbol: str = "BTCUSDT",
    side: Side = Side.BUY,
    qty: float = 1.0,
    dedupe_key: str = "dk-1",
    strategy_id: str = "strat-1",
) -> OrderIntent:
    return OrderIntent(
        dedupe_key=dedupe_key,
        strategy_id=strategy_id,
        symbol=symbol,
        exchange=Exchange.BYBIT,
        side=side,
        qty=Decimal(str(qty)),
        price=Decimal("50000"),
    )


def _make_ack(
    order_id: str = "ord-1",
    client_order_id: str = "dk-1",
    status: OrderStatus = OrderStatus.SUBMITTED,
) -> OrderAck:
    return OrderAck(
        order_id=order_id,
        client_order_id=client_order_id,
        symbol="BTCUSDT",
        exchange=Exchange.BYBIT,
        status=status,
    )


def _make_update(
    order_id: str = "ord-1",
    client_order_id: str = "dk-1",
    status: OrderStatus = OrderStatus.CANCELLED,
) -> OrderUpdate:
    return OrderUpdate(
        order_id=order_id,
        client_order_id=client_order_id,
        symbol="BTCUSDT",
        exchange=Exchange.BYBIT,
        status=status,
    )


# ===============================================================
# Task #13: SurveillanceAgent
# ===============================================================


class TestSurveillanceAgentIdentity:
    """Test agent metadata."""

    def test_agent_type(self):
        agent = SurveillanceAgent(event_bus=_make_bus())
        assert agent.agent_type == AgentType.SURVEILLANCE

    def test_capabilities(self):
        agent = SurveillanceAgent(event_bus=_make_bus())
        caps = agent.capabilities()
        assert "execution.fill" in caps.subscribes_to
        assert "surveillance" in caps.publishes_to


class TestWashTradeDetection:
    """Test wash-trade detection logic."""

    def test_opposite_fills_within_window_flagged(self):
        """Buy + sell on same symbol/strategy within window → case."""
        bus = _make_bus()
        agent = SurveillanceAgent(
            event_bus=bus, wash_trade_window_sec=60.0,
        )

        buy_fill = _make_fill(side=Side.BUY, strategy_id="strat-1")
        sell_fill = _make_fill(side=Side.SELL, strategy_id="strat-1")

        loop = asyncio.get_event_loop()
        loop.run_until_complete(agent._on_fill(buy_fill))
        loop.run_until_complete(agent._on_fill(sell_fill))

        assert agent.cases_created == 1
        # Verify published event
        bus.publish.assert_called()
        call_args = bus.publish.call_args
        assert call_args[0][0] == "surveillance"
        event = call_args[0][1]
        assert isinstance(event, SurveillanceCaseEvent)
        assert event.case_type == "wash_trade"

    def test_same_side_fills_not_flagged(self):
        """Two buys from same strategy → no case."""
        bus = _make_bus()
        agent = SurveillanceAgent(event_bus=bus, wash_trade_window_sec=60.0)

        fill1 = _make_fill(side=Side.BUY, strategy_id="strat-1")
        fill2 = _make_fill(side=Side.BUY, strategy_id="strat-1")

        loop = asyncio.get_event_loop()
        loop.run_until_complete(agent._on_fill(fill1))
        loop.run_until_complete(agent._on_fill(fill2))

        assert agent.cases_created == 0

    def test_different_strategies_not_flagged(self):
        """Buy + sell from different strategies → no case."""
        bus = _make_bus()
        agent = SurveillanceAgent(event_bus=bus, wash_trade_window_sec=60.0)

        buy_fill = _make_fill(side=Side.BUY, strategy_id="strat-1")
        sell_fill = _make_fill(side=Side.SELL, strategy_id="strat-2")

        loop = asyncio.get_event_loop()
        loop.run_until_complete(agent._on_fill(buy_fill))
        loop.run_until_complete(agent._on_fill(sell_fill))

        assert agent.cases_created == 0

    def test_different_symbols_not_flagged(self):
        """Buy BTC + sell ETH → no case."""
        bus = _make_bus()
        agent = SurveillanceAgent(event_bus=bus, wash_trade_window_sec=60.0)

        buy_fill = _make_fill(symbol="BTCUSDT", side=Side.BUY)
        sell_fill = _make_fill(symbol="ETHUSDT", side=Side.SELL)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(agent._on_fill(buy_fill))
        loop.run_until_complete(agent._on_fill(sell_fill))

        assert agent.cases_created == 0


class TestSpoofingDetection:
    """Test spoofing/layering detection logic."""

    def test_quick_cancel_flagged(self):
        """Order submitted then cancelled quickly → case."""
        bus = _make_bus()
        agent = SurveillanceAgent(
            event_bus=bus,
            spoof_cancel_window_sec=10.0,
            spoof_min_qty=0.0,
        )

        intent = _make_intent(dedupe_key="dk-1")
        ack = _make_ack(order_id="ord-1", client_order_id="dk-1")
        cancel = _make_update(order_id="ord-1", status=OrderStatus.CANCELLED)

        loop = asyncio.get_event_loop()
        # Process intent (tracked), ack (gets order_id), then cancel
        loop.run_until_complete(agent._on_execution_event(intent))
        loop.run_until_complete(agent._on_execution_event(ack))
        loop.run_until_complete(agent._on_execution_event(cancel))

        assert agent.cases_created == 1
        call_args = bus.publish.call_args
        event = call_args[0][1]
        assert event.case_type == "spoofing"

    def test_fill_not_flagged_as_spoof(self):
        """Order that gets filled → no spoofing case."""
        bus = _make_bus()
        agent = SurveillanceAgent(event_bus=bus, spoof_cancel_window_sec=10.0)

        intent = _make_intent(dedupe_key="dk-1")
        ack = _make_ack(
            order_id="ord-1", client_order_id="dk-1",
            status=OrderStatus.FILLED,
        )

        loop = asyncio.get_event_loop()
        loop.run_until_complete(agent._on_execution_event(intent))
        loop.run_until_complete(agent._on_execution_event(ack))

        assert agent.cases_created == 0
        assert agent.pending_orders_count == 0  # Cleaned up

    def test_small_order_below_threshold_not_flagged(self):
        """Small order cancelled quickly but below min_qty → no case."""
        bus = _make_bus()
        agent = SurveillanceAgent(
            event_bus=bus,
            spoof_cancel_window_sec=10.0,
            spoof_min_qty=5.0,  # Only flag orders >= 5.0
        )

        intent = _make_intent(dedupe_key="dk-1", qty=1.0)  # Too small
        ack = _make_ack(order_id="ord-1", client_order_id="dk-1")
        cancel = _make_update(order_id="ord-1", status=OrderStatus.CANCELLED)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(agent._on_execution_event(intent))
        loop.run_until_complete(agent._on_execution_event(ack))
        loop.run_until_complete(agent._on_execution_event(cancel))

        assert agent.cases_created == 0


class TestSurveillanceWithCaseManager:
    """Test SurveillanceAgent integration with CaseManager."""

    def test_case_persisted_to_manager(self):
        bus = _make_bus()
        cm = CaseManager()
        agent = SurveillanceAgent(
            event_bus=bus,
            case_manager=cm,
            wash_trade_window_sec=60.0,
        )

        buy_fill = _make_fill(side=Side.BUY)
        sell_fill = _make_fill(side=Side.SELL)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(agent._on_fill(buy_fill))
        loop.run_until_complete(agent._on_fill(sell_fill))

        assert cm.total_cases == 1
        open_cases = cm.list_open()
        assert len(open_cases) == 1
        assert open_cases[0].case_type == "wash_trade"


# ===============================================================
# Task #14: Compliance Case Management
# ===============================================================


class TestComplianceCaseModel:
    """Test ComplianceCase Pydantic model."""

    def test_default_fields(self):
        case = ComplianceCase(case_type="wash_trade", severity="high")
        assert case.case_id
        assert case.status == "open"
        assert case.disposition == ""
        assert case.created_at is not None

    def test_timeline_on_creation(self):
        cm = CaseManager()
        case = cm.open_case(
            case_type="spoofing", severity="medium",
            description="Test case",
        )
        assert len(case.timeline) == 1
        assert case.timeline[0].action == "opened"


class TestCaseManagerLifecycle:
    """Test case lifecycle transitions."""

    def test_open_to_investigating(self):
        cm = CaseManager()
        case = cm.open_case(case_type="wash_trade", severity="high")
        updated = cm.transition(
            case.case_id, "investigating",
            actor="analyst-1", detail="Reviewing fills",
        )
        assert updated is not None
        assert updated.status == "investigating"
        assert len(updated.timeline) == 2

    def test_investigating_to_closed(self):
        cm = CaseManager()
        case = cm.open_case(case_type="spoofing", severity="medium")
        cm.transition(case.case_id, "investigating")
        updated = cm.transition(
            case.case_id, "closed",
            actor="analyst-1",
            disposition="false_positive",
        )
        assert updated is not None
        assert updated.status == "closed"
        assert updated.disposition == "false_positive"
        assert updated.closed_at is not None

    def test_invalid_transition_returns_none(self):
        cm = CaseManager()
        case = cm.open_case(case_type="wash_trade", severity="high")
        cm.transition(case.case_id, "closed")
        # Cannot transition from closed
        result = cm.transition(case.case_id, "investigating")
        assert result is None

    def test_open_to_escalated(self):
        cm = CaseManager()
        case = cm.open_case(case_type="wash_trade", severity="critical")
        updated = cm.transition(case.case_id, "escalated", actor="system")
        assert updated is not None
        assert updated.status == "escalated"

    def test_assign_case(self):
        cm = CaseManager()
        case = cm.open_case(case_type="spoofing", severity="medium")
        updated = cm.assign(case.case_id, "analyst-2", actor="admin")
        assert updated is not None
        assert updated.assigned_to == "analyst-2"

    def test_add_evidence(self):
        cm = CaseManager()
        case = cm.open_case(case_type="wash_trade", severity="high")
        evidence = {"type": "fill", "fill_id": "fill-123"}
        updated = cm.add_evidence(case.case_id, evidence, actor="system")
        assert updated is not None
        assert len(updated.evidence) == 1

    def test_add_evidence_to_closed_case_fails(self):
        cm = CaseManager()
        case = cm.open_case(case_type="wash_trade", severity="high")
        cm.transition(case.case_id, "closed")
        result = cm.add_evidence(
            case.case_id, {"type": "fill"}, actor="system",
        )
        assert result is None


class TestCaseManagerQueries:
    """Test case query methods."""

    def test_list_open(self):
        cm = CaseManager()
        c1 = cm.open_case(case_type="wash_trade", severity="high")
        c2 = cm.open_case(case_type="spoofing", severity="medium")
        cm.transition(c1.case_id, "closed")

        open_cases = cm.list_open()
        assert len(open_cases) == 1
        assert open_cases[0].case_id == c2.case_id

    def test_list_by_status(self):
        cm = CaseManager()
        cm.open_case(case_type="wash_trade", severity="high")
        c2 = cm.open_case(case_type="spoofing", severity="medium")
        cm.transition(c2.case_id, "investigating")

        investigating = cm.list_by_status("investigating")
        assert len(investigating) == 1

    def test_list_by_symbol(self):
        cm = CaseManager()
        cm.open_case(case_type="wash_trade", severity="high", symbol="BTCUSDT")
        cm.open_case(case_type="spoofing", severity="medium", symbol="ETHUSDT")

        btc = cm.list_by_symbol("BTCUSDT")
        assert len(btc) == 1

    def test_total_and_open_count(self):
        cm = CaseManager()
        c1 = cm.open_case(case_type="wash_trade", severity="high")
        cm.open_case(case_type="spoofing", severity="medium")
        cm.transition(c1.case_id, "closed")

        assert cm.total_cases == 2
        assert cm.open_count == 1


class TestCaseManagerPersistence:
    """Test JSONL persistence."""

    def test_persist_and_reload(self, tmp_path: Path):
        path = tmp_path / "cases.jsonl"
        cm1 = CaseManager(persistence_path=path)
        cm1.open_case(
            case_type="wash_trade", severity="high", symbol="BTCUSDT",
        )
        cm1.open_case(
            case_type="spoofing", severity="medium", symbol="ETHUSDT",
        )
        assert path.exists()

        # Reload from disk
        cm2 = CaseManager(persistence_path=path)
        assert cm2.total_cases == 2

    def test_nonexistent_file_no_error(self, tmp_path: Path):
        path = tmp_path / "nonexistent.jsonl"
        cm = CaseManager(persistence_path=path)
        assert cm.total_cases == 0


# ===============================================================
# Task #15: ModelRegistry
# ===============================================================


class TestModelRecord:
    """Test ModelRecord model."""

    def test_default_fields(self):
        record = ModelRecord(name="btc_lstm")
        assert record.model_id
        assert record.version == 1
        assert record.stage == ModelStage.RESEARCH
        assert record.created_at is not None

    def test_custom_fields(self):
        record = ModelRecord(
            name="btc_lstm",
            version=3,
            stage=ModelStage.PAPER,
            training_data_hash="abc123",
            metrics={"sharpe": 1.5, "mse": 0.01},
        )
        assert record.version == 3
        assert record.metrics["sharpe"] == 1.5


class TestModelRegistryRegistration:
    """Test model registration."""

    def test_register_auto_version(self):
        reg = ModelRegistry()
        r1 = reg.register(name="btc_lstm")
        r2 = reg.register(name="btc_lstm")
        assert r1.version == 1
        assert r2.version == 2
        assert r1.model_id != r2.model_id

    def test_register_different_names(self):
        reg = ModelRegistry()
        r1 = reg.register(name="btc_lstm")
        r2 = reg.register(name="eth_dense")
        assert r1.version == 1
        assert r2.version == 1

    def test_register_with_metadata(self):
        reg = ModelRegistry()
        r = reg.register(
            name="btc_lstm",
            training_data_hash="abc123",
            hyperparameters={"layers": 3, "hidden_size": 128},
            metrics={"sharpe": 1.5},
            tags=["crypto", "momentum"],
        )
        assert r.training_data_hash == "abc123"
        assert r.hyperparameters["layers"] == 3
        assert "crypto" in r.tags


class TestModelRegistryPromotion:
    """Test stage transitions."""

    def test_research_to_paper(self):
        reg = ModelRegistry()
        r = reg.register(name="test_model")
        updated = reg.promote(
            r.model_id, ModelStage.PAPER,
            approved_by="admin", reason="Paper testing",
        )
        assert updated is not None
        assert updated.stage == ModelStage.PAPER
        assert updated.approved_by == "admin"
        assert len(updated.transitions) == 1

    def test_full_promotion_path(self):
        reg = ModelRegistry()
        r = reg.register(name="test_model")
        reg.promote(r.model_id, ModelStage.PAPER, approved_by="admin")
        reg.promote(r.model_id, ModelStage.LIMITED, approved_by="risk")
        updated = reg.promote(
            r.model_id, ModelStage.PRODUCTION, approved_by="cto",
        )
        assert updated is not None
        assert updated.stage == ModelStage.PRODUCTION
        assert len(updated.transitions) == 3

    def test_invalid_promotion_returns_none(self):
        reg = ModelRegistry()
        r = reg.register(name="test_model")
        # Cannot go directly from RESEARCH to PRODUCTION
        result = reg.promote(r.model_id, ModelStage.PRODUCTION)
        assert result is None
        # Stage unchanged
        assert reg.get(r.model_id).stage == ModelStage.RESEARCH

    def test_demotion_allowed(self):
        reg = ModelRegistry()
        r = reg.register(name="test_model")
        reg.promote(r.model_id, ModelStage.PAPER)
        # Demote back to research
        result = reg.promote(r.model_id, ModelStage.RESEARCH, reason="Needs more work")
        assert result is not None
        assert result.stage == ModelStage.RESEARCH

    def test_retire(self):
        reg = ModelRegistry()
        r = reg.register(name="test_model")
        result = reg.retire(r.model_id, reason="Obsolete", actor="admin")
        assert result is not None
        assert result.stage == ModelStage.RETIRED

    def test_retired_is_terminal(self):
        reg = ModelRegistry()
        r = reg.register(name="test_model")
        reg.retire(r.model_id)
        result = reg.promote(r.model_id, ModelStage.PAPER)
        assert result is None


class TestModelRegistryQueries:
    """Test query methods."""

    def test_get_latest(self):
        reg = ModelRegistry()
        reg.register(name="btc_lstm")
        r2 = reg.register(name="btc_lstm")
        latest = reg.get_latest("btc_lstm")
        assert latest.model_id == r2.model_id
        assert latest.version == 2

    def test_get_production(self):
        reg = ModelRegistry()
        r1 = reg.register(name="btc_lstm")
        r2 = reg.register(name="btc_lstm")
        reg.promote(r1.model_id, ModelStage.PAPER)
        reg.promote(r1.model_id, ModelStage.LIMITED)
        reg.promote(r1.model_id, ModelStage.PRODUCTION)

        prod = reg.get_production("btc_lstm")
        assert prod is not None
        assert prod.model_id == r1.model_id

    def test_get_production_none_when_no_prod(self):
        reg = ModelRegistry()
        reg.register(name="btc_lstm")
        assert reg.get_production("btc_lstm") is None

    def test_list_by_name(self):
        reg = ModelRegistry()
        reg.register(name="btc_lstm")
        reg.register(name="btc_lstm")
        reg.register(name="eth_dense")
        records = reg.list_by_name("btc_lstm")
        assert len(records) == 2
        assert records[0].version < records[1].version

    def test_list_by_stage(self):
        reg = ModelRegistry()
        r1 = reg.register(name="model_a")
        r2 = reg.register(name="model_b")
        reg.promote(r1.model_id, ModelStage.PAPER)

        research = reg.list_by_stage(ModelStage.RESEARCH)
        paper = reg.list_by_stage(ModelStage.PAPER)
        assert len(research) == 1
        assert len(paper) == 1

    def test_update_metrics(self):
        reg = ModelRegistry()
        r = reg.register(name="test", metrics={"sharpe": 1.0})
        reg.update_metrics(r.model_id, {"sharpe": 1.5, "mse": 0.005})
        updated = reg.get(r.model_id)
        assert updated.metrics["sharpe"] == 1.5
        assert updated.metrics["mse"] == 0.005

    def test_total_models(self):
        reg = ModelRegistry()
        reg.register(name="a")
        reg.register(name="b")
        assert reg.total_models == 2


class TestModelRegistryPersistence:
    """Test JSONL persistence."""

    def test_persist_and_reload(self, tmp_path: Path):
        path = tmp_path / "models.jsonl"
        reg1 = ModelRegistry(persistence_path=path)
        r = reg1.register(name="btc_lstm", metrics={"sharpe": 1.5})
        reg1.promote(r.model_id, ModelStage.PAPER, approved_by="admin")

        # Reload
        reg2 = ModelRegistry(persistence_path=path)
        assert reg2.total_models >= 1
        loaded = reg2.get(r.model_id)
        assert loaded is not None
        # Should have latest state (PAPER stage from the last persist)
        assert loaded.stage == ModelStage.PAPER

    def test_nonexistent_file_no_error(self, tmp_path: Path):
        path = tmp_path / "nonexistent.jsonl"
        reg = ModelRegistry(persistence_path=path)
        assert reg.total_models == 0


# ===============================================================
# Schema registration
# ===============================================================


class TestSurveillanceTopic:
    """Test surveillance topic is registered."""

    def test_topic_in_schema_registry(self):
        from agentic_trading.bus.schemas import TOPIC_SCHEMAS
        assert "surveillance" in TOPIC_SCHEMAS
        assert SurveillanceCaseEvent in TOPIC_SCHEMAS["surveillance"]

    def test_event_type_map(self):
        from agentic_trading.bus.schemas import EVENT_TYPE_MAP
        assert "SurveillanceCaseEvent" in EVENT_TYPE_MAP
