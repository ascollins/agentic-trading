"""Tests for P1a sprint: institutional features.

Covers:
- ExecutionPlannerAgent (Task #5)
- ExecutionQualityTracker (Task #6)
- DailyEffectivenessScorecard formula updates (Task #7)
- Degraded mode expansion (Task #8)
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_trading.agents.execution_planner import ExecutionPlannerAgent
from agentic_trading.agents.incident_response import IncidentResponseAgent
from agentic_trading.control_plane.action_types import (
    ActionScope,
    ApprovalTier,
    DegradedMode,
    ProposedAction,
    ToolName,
)
from agentic_trading.control_plane.policy_evaluator import CPPolicyEvaluator
from agentic_trading.core.enums import AgentType, Exchange, OrderType, Side
from agentic_trading.core.events import (
    ExecutionPlanCreated,
    IncidentCreated,
    OrderIntent,
)
from agentic_trading.execution.plan import ExecutionPlan, OrderSlice
from agentic_trading.governance.policy_engine import PolicyEngine
from agentic_trading.observability.daily_scorecard import DailyEffectivenessScorecard
from agentic_trading.observability.execution_quality import (
    ExecutionQualityTracker,
    OrderMetrics,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_intent(**overrides) -> OrderIntent:
    defaults = {
        "symbol": "BTCUSDT",
        "exchange": Exchange.BYBIT,
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "qty": 0.01,
        "price": 50000.0,
        "strategy_id": "test-strat",
        "dedupe_key": "dk-001",
        "trace_id": "tr-001",
    }
    defaults.update(overrides)
    return OrderIntent(**defaults)


def _make_proposed(tool: ToolName = ToolName.SUBMIT_ORDER, **overrides) -> ProposedAction:
    defaults = {
        "tool_name": tool,
        "scope": ActionScope(strategy_id="test-strat", symbol="BTCUSDT"),
    }
    defaults.update(overrides)
    return ProposedAction(**defaults)


def _make_event_bus() -> AsyncMock:
    bus = AsyncMock()
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    return bus


# ===============================================================
# Task #5: ExecutionPlannerAgent
# ===============================================================


class TestExecutionPlan:
    """Test ExecutionPlan data model."""

    def test_plan_fields(self):
        plan = ExecutionPlan(
            intent_dedupe_key="dk-001",
            trace_id="tr-001",
            strategy_id="strat-1",
            symbol="BTCUSDT",
            slices=[
                OrderSlice(
                    sequence=0,
                    symbol="BTCUSDT",
                    exchange=Exchange.BYBIT,
                    side=Side.BUY,
                    order_type=OrderType.LIMIT,
                    qty=0.01,
                    price=50000.0,
                ),
            ],
        )
        assert plan.plan_id  # auto-generated UUID
        assert plan.slice_count == 1
        assert plan.symbol == "BTCUSDT"

    def test_plan_total_qty(self):
        slices = [
            OrderSlice(sequence=0, symbol="BTCUSDT", exchange=Exchange.BYBIT,
                       side=Side.BUY, order_type=OrderType.MARKET, qty=0.5),
            OrderSlice(sequence=1, symbol="BTCUSDT", exchange=Exchange.BYBIT,
                       side=Side.BUY, order_type=OrderType.MARKET, qty=0.3),
        ]
        plan = ExecutionPlan(
            intent_dedupe_key="dk-002",
            symbol="BTCUSDT",
            slices=slices,
        )
        assert float(plan.total_qty) == pytest.approx(0.8)
        assert plan.slice_count == 2


class TestExecutionPlannerAgent:
    """Test ExecutionPlannerAgent plan creation."""

    def test_create_plan_single_slice(self):
        bus = _make_event_bus()
        agent = ExecutionPlannerAgent(event_bus=bus)
        intent = _make_intent()

        plan = agent.create_plan(intent)

        assert isinstance(plan, ExecutionPlan)
        assert plan.intent_dedupe_key == intent.dedupe_key
        assert plan.trace_id == intent.trace_id
        assert plan.strategy_id == intent.strategy_id
        assert plan.slice_count == 1
        assert plan.slices[0].qty == intent.qty
        assert plan.slices[0].symbol == intent.symbol
        assert agent.plans_created == 1

    def test_agent_type(self):
        bus = _make_event_bus()
        agent = ExecutionPlannerAgent(event_bus=bus)
        assert agent.agent_type == AgentType.EXECUTION_PLANNER

    def test_capabilities(self):
        bus = _make_event_bus()
        agent = ExecutionPlannerAgent(event_bus=bus)
        caps = agent.capabilities()
        assert "execution" in caps.subscribes_to
        assert "execution.plan" in caps.publishes_to

    def test_create_and_publish(self):
        bus = _make_event_bus()
        agent = ExecutionPlannerAgent(event_bus=bus)
        intent = _make_intent()

        plan = asyncio.get_event_loop().run_until_complete(
            agent.create_and_publish_plan(intent)
        )

        assert isinstance(plan, ExecutionPlan)
        bus.publish.assert_called_once()
        call_args = bus.publish.call_args
        assert call_args[0][0] == "execution.plan"
        event = call_args[0][1]
        assert isinstance(event, ExecutionPlanCreated)
        assert event.plan_id == plan.plan_id

    def test_contingencies_default(self):
        bus = _make_event_bus()
        agent = ExecutionPlannerAgent(event_bus=bus)
        intent = _make_intent()

        plan = agent.create_plan(intent)

        triggers = [c.trigger for c in plan.contingencies]
        assert "rejection" in triggers
        assert "timeout" in triggers


# ===============================================================
# Task #6: ExecutionQualityTracker
# ===============================================================


class TestExecutionQualityTracker:
    """Test execution quality metrics tracking."""

    def test_record_lifecycle(self):
        tracker = ExecutionQualityTracker()
        tracker.record_intent("dk-1", "BTCUSDT", reference_price=50000.0)
        tracker.record_submission("dk-1")
        tracker.record_ack("dk-1")
        result = tracker.record_fill("dk-1", fill_price=50005.0, fill_qty=0.01)

        assert result is not None
        assert result.filled
        assert result.slippage_bps == pytest.approx(1.0, rel=0.01)
        assert result.total_latency_ms > 0

    def test_avg_slippage(self):
        tracker = ExecutionQualityTracker()
        # Two fills with known slippage
        tracker.record_intent("dk-1", "BTCUSDT", reference_price=50000.0)
        tracker.record_fill("dk-1", fill_price=50010.0, fill_qty=0.01)  # 2 bps

        tracker.record_intent("dk-2", "BTCUSDT", reference_price=50000.0)
        tracker.record_fill("dk-2", fill_price=50005.0, fill_qty=0.01)  # 1 bps

        assert tracker.avg_slippage_bps == pytest.approx(1.5, rel=0.01)

    def test_fill_rate(self):
        tracker = ExecutionQualityTracker()
        tracker.record_intent("dk-1", "BTCUSDT")
        tracker.record_intent("dk-2", "BTCUSDT")
        tracker.record_fill("dk-1", fill_price=50000.0, fill_qty=0.01)
        tracker.record_unfilled("dk-2")

        assert tracker.fill_rate == pytest.approx(0.5)

    def test_slippage_score_perfect(self):
        tracker = ExecutionQualityTracker(target_slippage_bps=10.0)
        # No orders = 0 avg slippage = perfect score
        assert tracker.slippage_score == pytest.approx(10.0)

    def test_slippage_score_at_target(self):
        tracker = ExecutionQualityTracker(target_slippage_bps=10.0)
        tracker.record_intent("dk-1", "BTCUSDT", reference_price=50000.0)
        # 10 bps slippage = at target = score 0
        tracker.record_fill("dk-1", fill_price=50050.0, fill_qty=0.01)
        assert tracker.slippage_score == pytest.approx(0.0, abs=0.1)

    def test_latency_score(self):
        tracker = ExecutionQualityTracker(target_latency_ms=5000.0)
        tracker.record_intent("dk-1", "BTCUSDT")
        # Fill immediately (latency << target)
        tracker.record_fill("dk-1", fill_price=50000.0, fill_qty=0.01)
        assert tracker.latency_score == pytest.approx(10.0)

    def test_participation_score_no_data(self):
        tracker = ExecutionQualityTracker()
        # No data → neutral 10.0
        assert tracker.participation_score == pytest.approx(10.0)

    def test_composite_score(self):
        tracker = ExecutionQualityTracker()
        # No fills = perfect scores
        score = tracker.composite_execution_score
        assert score == pytest.approx(10.0)

    def test_window_size_limit(self):
        tracker = ExecutionQualityTracker(window_size=5)
        for i in range(10):
            tracker.record_intent(f"dk-{i}", "BTCUSDT", reference_price=50000.0)
            tracker.record_fill(f"dk-{i}", fill_price=50000.0 + i, fill_qty=0.01)

        assert tracker.window_size == 5
        assert tracker.total_fills == 10

    def test_get_recent_metrics(self):
        tracker = ExecutionQualityTracker()
        for i in range(5):
            tracker.record_intent(f"dk-{i}", "BTCUSDT")
            tracker.record_fill(f"dk-{i}", fill_price=50000.0, fill_qty=0.01)

        recent = tracker.get_recent_metrics(n=3)
        assert len(recent) == 3


# ===============================================================
# Task #7: DailyEffectivenessScorecard
# ===============================================================


class TestDailyEffectivenessScorecard:
    """Test scorecard computation."""

    def test_neutral_defaults(self):
        """All None providers → neutral 5.0 for each score."""
        scorecard = DailyEffectivenessScorecard()
        scores = scorecard.compute()

        assert scores["edge_quality"] == 5.0
        assert scores["execution_quality"] == 5.0
        assert scores["risk_discipline"] == 5.0
        assert scores["operational_integrity"] == 10.0  # No issues = perfect
        assert "total" in scores

    def test_edge_quality_with_journal(self):
        """Edge quality from journal stats."""
        journal = MagicMock()
        journal.get_aggregate_stats.return_value = {
            "total_trades": 100,
            "information_ratio": 2.0,  # IR/2 = 1.0 → clamped to 1.0
            "win_rate": 0.6,           # 0.6 * 10 = 6.0
            "sharpe_ratio": 1.5,       # clamped to 1.5
        }
        scorecard = DailyEffectivenessScorecard(journal=journal)
        scores = scorecard.compute()
        edge = scores["edge_quality"]
        # 0.5 * clamp(2/2, 0, 10) + 0.3 * clamp(0.6*10, 0, 10) + 0.2 * clamp(1.5, 0, 10)
        # = 0.5*1.0 + 0.3*6.0 + 0.2*1.5 = 0.5 + 1.8 + 0.3 = 2.6
        assert edge == pytest.approx(2.6, abs=0.1)

    def test_edge_quality_insufficient_trades(self):
        """Less than 5 trades → neutral 5.0."""
        journal = MagicMock()
        journal.get_aggregate_stats.return_value = {"total_trades": 3}
        scorecard = DailyEffectivenessScorecard(journal=journal)
        scores = scorecard.compute()
        assert scores["edge_quality"] == 5.0

    def test_execution_quality_with_tracker(self):
        """Execution quality from quality tracker scores."""
        tracker = MagicMock()
        tracker.slippage_score = 8.0
        tracker.participation_score = 7.0
        tracker.latency_score = 9.0
        scorecard = DailyEffectivenessScorecard(quality_tracker=tracker)
        scores = scorecard.compute()
        # (8 + 7 + 9) / 3 = 8.0
        assert scores["execution_quality"] == 8.0

    def test_risk_discipline_with_manager(self):
        """Risk discipline from risk manager attributes."""
        rm = MagicMock()
        rm.current_exposure = 5000.0
        rm.max_exposure = 10000.0
        rm.circuit_breaker_trips_today = 1
        rm.var_limit = 1000.0
        rm.realised_loss_today = 500.0

        scorecard = DailyEffectivenessScorecard(risk_manager=rm)
        scores = scorecard.compute()

        # utilisation_score = 10 * (1 - 5000/10000) = 5.0
        # breach_penalty = max(0, 10 - 1*2) = 8.0
        # var_score = 10 * (1 - max(0, 500-1000)/1000) = 10.0 (no excess)
        # risk = (5 + 8 + 10) / 3 ≈ 7.67
        assert scores["risk_discipline"] == pytest.approx(7.7, abs=0.1)

    def test_operational_integrity_all_healthy(self):
        """No issues → 10.0 operational integrity."""
        scorecard = DailyEffectivenessScorecard()
        scores = scorecard.compute()
        assert scores["operational_integrity"] == 10.0

    def test_operational_integrity_with_breaks(self):
        """Recon breaks reduce operational score."""
        recon = lambda: {"break_count": 3}
        scorecard = DailyEffectivenessScorecard(recon_provider=recon)
        scores = scorecard.compute()
        # break_score = 10 - 3 = 7
        # data_quality = 10 (no bus/registry)
        # incident = 10 (no provider)
        # canary = 10 (healthy default)
        # (7 + 10 + 10 + 10) / 4 = 9.25
        assert scores["operational_integrity"] == pytest.approx(9.2, abs=0.1)

    def test_operational_integrity_with_canary_down(self):
        """Canary failure drops score."""
        canary = lambda: {"healthy": False}
        scorecard = DailyEffectivenessScorecard(canary_provider=canary)
        scores = scorecard.compute()
        # break=10, data=10, incident=10, canary=0
        # (10+10+10+0)/4 = 7.5
        assert scores["operational_integrity"] == 7.5

    def test_total_is_weighted_sum(self):
        """Total = weighted sum of 4 scores."""
        scorecard = DailyEffectivenessScorecard()
        scores = scorecard.compute()
        expected_total = (
            scores["edge_quality"] * 0.30
            + scores["execution_quality"] * 0.25
            + scores["risk_discipline"] * 0.25
            + scores["operational_integrity"] * 0.20
        )
        assert scores["total"] == pytest.approx(expected_total, abs=0.1)

    def test_last_scores_cached(self):
        scorecard = DailyEffectivenessScorecard()
        assert scorecard.last_scores is None
        scorecard.compute()
        assert scorecard.last_scores is not None


# ===============================================================
# Task #8: Degraded Mode Expansion
# ===============================================================


class TestDegradedModeEnum:
    """Test DegradedMode enum has all expected members."""

    def test_all_modes_present(self):
        modes = {m.value for m in DegradedMode}
        assert modes == {
            "normal", "cautious", "stop_new_orders",
            "risk_off_only", "read_only", "full_stop",
        }

    def test_mode_values(self):
        assert DegradedMode.CAUTIOUS.value == "cautious"
        assert DegradedMode.STOP_NEW_ORDERS.value == "stop_new_orders"


class TestDegradedModePolicyEvaluator:
    """Test CPPolicyEvaluator degraded mode enforcement."""

    def _make_evaluator(self, mode: str) -> CPPolicyEvaluator:
        engine = PolicyEngine()
        evaluator = CPPolicyEvaluator(policy_engine=engine)
        evaluator.set_system_state("degraded_mode", mode)
        return evaluator

    def test_normal_allows_all(self):
        ev = self._make_evaluator("normal")
        decision = ev.evaluate(_make_proposed(ToolName.SUBMIT_ORDER))
        assert decision.allowed

    def test_cautious_allows_with_half_sizing(self):
        ev = self._make_evaluator("cautious")
        decision = ev.evaluate(_make_proposed(ToolName.SUBMIT_ORDER))
        assert decision.allowed
        assert decision.sizing_multiplier == pytest.approx(0.5)
        assert "cautious" in decision.reasons[0]

    def test_cautious_signals_block_new_symbols(self):
        ev = self._make_evaluator("cautious")
        decision = ev.evaluate(_make_proposed(ToolName.SUBMIT_ORDER))
        assert decision.context_snapshot.get("block_new_symbols") is True

    def test_cautious_reads_unaffected(self):
        ev = self._make_evaluator("cautious")
        decision = ev.evaluate(_make_proposed(ToolName.GET_POSITIONS))
        assert decision.allowed
        assert decision.sizing_multiplier == 1.0

    def test_stop_new_orders_blocks_submit(self):
        ev = self._make_evaluator("stop_new_orders")
        decision = ev.evaluate(_make_proposed(ToolName.SUBMIT_ORDER))
        assert not decision.allowed
        assert "stop_new_orders" in decision.reasons[0]

    def test_stop_new_orders_blocks_batch_submit(self):
        ev = self._make_evaluator("stop_new_orders")
        decision = ev.evaluate(_make_proposed(ToolName.BATCH_SUBMIT_ORDERS))
        assert not decision.allowed

    def test_stop_new_orders_allows_cancel(self):
        ev = self._make_evaluator("stop_new_orders")
        decision = ev.evaluate(_make_proposed(ToolName.CANCEL_ORDER))
        assert decision.allowed

    def test_stop_new_orders_allows_amend(self):
        ev = self._make_evaluator("stop_new_orders")
        decision = ev.evaluate(_make_proposed(ToolName.AMEND_ORDER))
        assert decision.allowed

    def test_stop_new_orders_allows_reads(self):
        ev = self._make_evaluator("stop_new_orders")
        decision = ev.evaluate(_make_proposed(ToolName.GET_POSITIONS))
        assert decision.allowed

    def test_stop_new_orders_allows_tp_sl(self):
        ev = self._make_evaluator("stop_new_orders")
        decision = ev.evaluate(_make_proposed(ToolName.SET_TRADING_STOP))
        assert decision.allowed

    def test_risk_off_blocks_submit(self):
        ev = self._make_evaluator("risk_off_only")
        decision = ev.evaluate(_make_proposed(ToolName.SUBMIT_ORDER))
        assert not decision.allowed

    def test_risk_off_allows_cancel(self):
        ev = self._make_evaluator("risk_off_only")
        decision = ev.evaluate(_make_proposed(ToolName.CANCEL_ORDER))
        assert decision.allowed

    def test_read_only_blocks_cancel(self):
        ev = self._make_evaluator("read_only")
        decision = ev.evaluate(_make_proposed(ToolName.CANCEL_ORDER))
        assert not decision.allowed

    def test_read_only_allows_reads(self):
        ev = self._make_evaluator("read_only")
        decision = ev.evaluate(_make_proposed(ToolName.GET_BALANCES))
        assert decision.allowed

    def test_full_stop_blocks_everything(self):
        ev = self._make_evaluator("full_stop")
        decision = ev.evaluate(_make_proposed(ToolName.GET_POSITIONS))
        assert not decision.allowed


class TestIncidentResponseModes:
    """Test IncidentResponseAgent severity → mode mapping."""

    def test_warning_maps_to_cautious(self):
        assert IncidentResponseAgent._severity_to_mode("warning") == "cautious"

    def test_error_maps_to_stop_new_orders(self):
        assert IncidentResponseAgent._severity_to_mode("error") == "stop_new_orders"

    def test_critical_maps_to_risk_off(self):
        assert IncidentResponseAgent._severity_to_mode("critical") == "risk_off_only"

    def test_emergency_maps_to_full_stop(self):
        assert IncidentResponseAgent._severity_to_mode("emergency") == "full_stop"

    def test_info_no_mode_change(self):
        assert IncidentResponseAgent._severity_to_mode("info") is None

    def test_mode_rank_ordering(self):
        rank = IncidentResponseAgent._mode_rank
        assert rank("normal") < rank("cautious")
        assert rank("cautious") < rank("stop_new_orders")
        assert rank("stop_new_orders") < rank("risk_off_only")
        assert rank("risk_off_only") < rank("read_only")
        assert rank("read_only") < rank("full_stop")

    def test_escalation_warning_to_cautious(self):
        """Warning incident escalates from normal → cautious."""
        bus = _make_event_bus()
        agent = IncidentResponseAgent(event_bus=bus)

        incident = IncidentCreated(
            incident_id="inc-1",
            severity="warning",
            component="feed",
            description="Feed latency spike",
        )
        asyncio.get_event_loop().run_until_complete(
            agent._handle_incident(incident)
        )
        assert agent.current_mode == "cautious"

    def test_escalation_error_to_stop_new_orders(self):
        """Error incident escalates from normal → stop_new_orders."""
        bus = _make_event_bus()
        agent = IncidentResponseAgent(event_bus=bus)

        incident = IncidentCreated(
            incident_id="inc-2",
            severity="error",
            component="execution",
            description="Exchange API errors",
        )
        asyncio.get_event_loop().run_until_complete(
            agent._handle_incident(incident)
        )
        assert agent.current_mode == "stop_new_orders"

    def test_no_downgrade(self):
        """Less severe incident doesn't downgrade mode."""
        bus = _make_event_bus()
        agent = IncidentResponseAgent(event_bus=bus)

        # First escalate to risk_off via critical
        critical = IncidentCreated(
            incident_id="inc-1",
            severity="critical",
            component="risk",
            description="Risk breach",
        )
        asyncio.get_event_loop().run_until_complete(
            agent._handle_incident(critical)
        )
        assert agent.current_mode == "risk_off_only"

        # Warning should not downgrade
        warning = IncidentCreated(
            incident_id="inc-2",
            severity="warning",
            component="feed",
            description="Feed delay",
        )
        asyncio.get_event_loop().run_until_complete(
            agent._handle_incident(warning)
        )
        assert agent.current_mode == "risk_off_only"  # Unchanged

    def test_policy_evaluator_updated_on_escalation(self):
        """Policy evaluator's system state is updated on mode change."""
        bus = _make_event_bus()
        evaluator = MagicMock()
        agent = IncidentResponseAgent(event_bus=bus, policy_evaluator=evaluator)

        incident = IncidentCreated(
            incident_id="inc-1",
            severity="error",
            component="execution",
            description="Exchange errors",
        )
        asyncio.get_event_loop().run_until_complete(
            agent._handle_incident(incident)
        )
        evaluator.set_system_state.assert_called_once_with(
            "degraded_mode", "stop_new_orders",
        )
