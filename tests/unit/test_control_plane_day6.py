"""Day 6 tests: Event schemas, degraded mode, agent stubs, recon integration.

Acceptance tests:
    A5: Degraded mode enforcement in PolicyEvaluator
    A6: Event schema completeness and wiring

Unit tests:
    - DegradedMode levels block/allow correct tool sets
    - IncidentResponseAgent processes incidents and escalates
    - DataQualityAgent tracks staleness and emits incidents
    - New event types (ToolCallRecorded, IncidentCreated, DegradedModeEnabled)
    - Schema registry includes new topics/events
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_trading.control_plane.action_types import (
    ActionScope,
    ApprovalTier,
    CPPolicyDecision,
    DegradedMode,
    ProposedAction,
    ToolName,
)
from agentic_trading.control_plane.policy_evaluator import (
    CPPolicyEvaluator,
    _READ_ONLY_TOOLS,
    _RISK_OFF_TOOLS,
)
from agentic_trading.core.enums import AgentType
from agentic_trading.core.events import (
    BaseEvent,
    DegradedModeEnabled,
    IncidentCreated,
    ToolCallRecorded,
)
from agentic_trading.governance.policy_engine import PolicyEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_proposed(tool: ToolName = ToolName.SUBMIT_ORDER) -> ProposedAction:
    return ProposedAction(
        tool_name=tool,
        scope=ActionScope(strategy_id="test", symbol="BTC/USDT"),
        request_params={"qty": 1.0},
    )


# ===========================================================================
# A5: Degraded mode enforcement
# ===========================================================================


class TestDegradedModeNormal:
    """A5a: NORMAL mode allows all tools."""

    def test_normal_allows_submit(self):
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "normal")
        result = evaluator.evaluate(_make_proposed(ToolName.SUBMIT_ORDER))
        assert result.allowed

    def test_normal_allows_reads(self):
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "normal")
        result = evaluator.evaluate(_make_proposed(ToolName.GET_POSITIONS))
        assert result.allowed


class TestDegradedModeRiskOff:
    """A5b: RISK_OFF_ONLY allows cancels/reads, blocks new orders."""

    def test_blocks_submit_order(self):
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "risk_off_only")
        result = evaluator.evaluate(_make_proposed(ToolName.SUBMIT_ORDER))
        assert not result.allowed
        assert any("risk_off" in r for r in result.reasons)

    def test_blocks_batch_submit(self):
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "risk_off_only")
        result = evaluator.evaluate(_make_proposed(ToolName.BATCH_SUBMIT_ORDERS))
        assert not result.allowed

    def test_allows_cancel(self):
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "risk_off_only")
        result = evaluator.evaluate(_make_proposed(ToolName.CANCEL_ORDER))
        assert result.allowed

    def test_allows_cancel_all(self):
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "risk_off_only")
        result = evaluator.evaluate(_make_proposed(ToolName.CANCEL_ALL_ORDERS))
        assert result.allowed

    def test_allows_set_trading_stop(self):
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "risk_off_only")
        result = evaluator.evaluate(_make_proposed(ToolName.SET_TRADING_STOP))
        assert result.allowed

    def test_allows_read_positions(self):
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "risk_off_only")
        result = evaluator.evaluate(_make_proposed(ToolName.GET_POSITIONS))
        assert result.allowed

    def test_allows_all_risk_off_tools(self):
        """Every tool in _RISK_OFF_TOOLS is allowed."""
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "risk_off_only")
        for tool in _RISK_OFF_TOOLS:
            result = evaluator.evaluate(_make_proposed(tool))
            assert result.allowed, f"{tool} should be allowed in risk_off_only"


class TestDegradedModeReadOnly:
    """A5c: READ_ONLY allows only reads, blocks all mutations."""

    def test_blocks_submit_order(self):
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "read_only")
        result = evaluator.evaluate(_make_proposed(ToolName.SUBMIT_ORDER))
        assert not result.allowed
        assert any("read_only" in r for r in result.reasons)

    def test_blocks_cancel(self):
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "read_only")
        result = evaluator.evaluate(_make_proposed(ToolName.CANCEL_ORDER))
        assert not result.allowed

    def test_blocks_set_trading_stop(self):
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "read_only")
        result = evaluator.evaluate(_make_proposed(ToolName.SET_TRADING_STOP))
        assert not result.allowed

    def test_allows_read_positions(self):
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "read_only")
        result = evaluator.evaluate(_make_proposed(ToolName.GET_POSITIONS))
        assert result.allowed

    def test_allows_all_read_tools(self):
        """Every tool in _READ_ONLY_TOOLS is allowed."""
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "read_only")
        for tool in _READ_ONLY_TOOLS:
            result = evaluator.evaluate(_make_proposed(tool))
            assert result.allowed, f"{tool} should be allowed in read_only"

    def test_blocks_all_mutating_tools(self):
        """All mutating tools (submit, amend, set_leverage, etc.) blocked."""
        from agentic_trading.control_plane.action_types import MUTATING_TOOLS

        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "read_only")
        for tool in MUTATING_TOOLS:
            result = evaluator.evaluate(_make_proposed(tool))
            assert not result.allowed, f"{tool} should be blocked in read_only"


class TestDegradedModeFullStop:
    """A5d: FULL_STOP blocks everything."""

    def test_blocks_submit_order(self):
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "full_stop")
        result = evaluator.evaluate(_make_proposed(ToolName.SUBMIT_ORDER))
        assert not result.allowed
        assert any("full_stop" in r for r in result.reasons)

    def test_blocks_reads(self):
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "full_stop")
        result = evaluator.evaluate(_make_proposed(ToolName.GET_POSITIONS))
        assert not result.allowed

    def test_blocks_cancels(self):
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "full_stop")
        result = evaluator.evaluate(_make_proposed(ToolName.CANCEL_ORDER))
        assert not result.allowed

    def test_blocks_all_tools(self):
        """Every single ToolName is blocked in FULL_STOP."""
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "full_stop")
        for tool in ToolName:
            result = evaluator.evaluate(_make_proposed(tool))
            assert not result.allowed, f"{tool} should be blocked in full_stop"


class TestDegradedModeStateManagement:
    """System state get/set and degraded_mode property."""

    def test_default_mode_is_normal(self):
        evaluator = CPPolicyEvaluator(PolicyEngine())
        assert evaluator.degraded_mode == "normal"

    def test_set_and_get_state(self):
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "full_stop")
        assert evaluator.get_system_state("degraded_mode") == "full_stop"
        assert evaluator.degraded_mode == "full_stop"

    def test_invalid_mode_falls_back_to_normal(self):
        evaluator = CPPolicyEvaluator(PolicyEngine())
        evaluator.set_system_state("degraded_mode", "invalid_mode_xyz")
        result = evaluator.evaluate(_make_proposed(ToolName.SUBMIT_ORDER))
        assert result.allowed  # Falls back to NORMAL


# ===========================================================================
# A6: Event schema completeness
# ===========================================================================


class TestEventSchemas:
    """A6: New event types exist and have correct fields."""

    def test_incident_created_fields(self):
        ev = IncidentCreated(
            severity="critical",
            component="data_quality",
            description="Feed stale",
            affected_symbols=["BTC/USDT"],
        )
        assert ev.severity == "critical"
        assert ev.component == "data_quality"
        assert ev.incident_id  # auto-generated
        assert ev.source_module == "control_plane"
        assert "BTC/USDT" in ev.affected_symbols

    def test_incident_created_auto_action(self):
        ev = IncidentCreated(
            severity="warning",
            component="test",
            description="test",
            auto_action="degraded_mode",
        )
        assert ev.auto_action == "degraded_mode"

    def test_tool_call_recorded_fields(self):
        ev = ToolCallRecorded(
            action_id="act-123",
            tool_name="submit_order",
            success=True,
            latency_ms=42.5,
            request_hash="abc123",
            response_hash="def456",
        )
        assert ev.action_id == "act-123"
        assert ev.tool_name == "submit_order"
        assert ev.success is True
        assert ev.latency_ms == 42.5
        assert ev.source_module == "control_plane.tool_gateway"

    def test_tool_call_recorded_failure(self):
        ev = ToolCallRecorded(
            action_id="act-456",
            tool_name="cancel_order",
            success=False,
            error="timeout",
        )
        assert not ev.success
        assert ev.error == "timeout"

    def test_degraded_mode_enabled_enhanced(self):
        ev = DegradedModeEnabled(
            mode="risk_off_only",
            previous_mode="normal",
            reason="feed stale",
            triggered_by="incident:inc-123",
            blocked_tools=["submit_order"],
            allowed_tools=["cancel_order", "get_positions"],
        )
        assert ev.mode == "risk_off_only"
        assert ev.triggered_by == "incident:inc-123"
        assert "submit_order" in ev.blocked_tools
        assert "cancel_order" in ev.allowed_tools
        assert ev.source_module == "control_plane"

    def test_all_events_inherit_base_event(self):
        assert issubclass(IncidentCreated, BaseEvent)
        assert issubclass(ToolCallRecorded, BaseEvent)
        assert issubclass(DegradedModeEnabled, BaseEvent)


class TestSchemaRegistry:
    """A6b: Schema registry includes new topics and events."""

    def test_system_incident_topic(self):
        from agentic_trading.event_bus.schemas import TOPIC_SCHEMAS

        assert "system.incident" in TOPIC_SCHEMAS
        names = [c.__name__ for c in TOPIC_SCHEMAS["system.incident"]]
        assert "IncidentCreated" in names

    def test_system_degraded_mode_topic(self):
        from agentic_trading.event_bus.schemas import TOPIC_SCHEMAS

        assert "system.degraded_mode" in TOPIC_SCHEMAS
        names = [c.__name__ for c in TOPIC_SCHEMAS["system.degraded_mode"]]
        assert "DegradedModeEnabled" in names

    def test_control_plane_tool_call_topic(self):
        from agentic_trading.event_bus.schemas import TOPIC_SCHEMAS

        assert "control_plane.tool_call" in TOPIC_SCHEMAS
        names = [c.__name__ for c in TOPIC_SCHEMAS["control_plane.tool_call"]]
        assert "ToolCallRecorded" in names

    def test_event_type_map_lookup(self):
        from agentic_trading.event_bus.schemas import get_event_class

        assert get_event_class("IncidentCreated") is IncidentCreated
        assert get_event_class("ToolCallRecorded") is ToolCallRecorded
        assert get_event_class("DegradedModeEnabled") is DegradedModeEnabled

    def test_topic_for_event(self):
        from agentic_trading.event_bus.schemas import get_topic_for_event

        ev = IncidentCreated(
            severity="warning", component="test", description="test",
        )
        assert get_topic_for_event(ev) == "system.incident"

        ev2 = ToolCallRecorded(
            action_id="a", tool_name="submit_order", success=True,
        )
        assert get_topic_for_event(ev2) == "control_plane.tool_call"


# ===========================================================================
# DataQualityAgent unit tests
# ===========================================================================


class TestDataQualityAgent:
    """DataQualityAgent stub tests."""

    def test_satisfies_iagent(self):
        from agentic_trading.agents.data_quality import DataQualityAgent
        from agentic_trading.core.interfaces import IAgent

        agent = DataQualityAgent(
            event_bus=AsyncMock(),
            agent_id="dq-test",
        )
        assert isinstance(agent, IAgent)
        assert agent.agent_type == AgentType.DATA_QUALITY

    def test_capabilities(self):
        from agentic_trading.agents.data_quality import DataQualityAgent

        agent = DataQualityAgent(event_bus=AsyncMock())
        caps = agent.capabilities()
        assert "feature.vector" in caps.subscribes_to
        assert "system" in caps.publishes_to

    @pytest.mark.asyncio
    async def test_start_subscribes_to_feature_vector(self):
        from agentic_trading.agents.data_quality import DataQualityAgent

        event_bus = AsyncMock()
        agent = DataQualityAgent(event_bus=event_bus, agent_id="dq-test")

        await agent.start()
        assert agent.is_running

        # Verify subscription was created
        event_bus.subscribe.assert_called_once()
        call_kwargs = event_bus.subscribe.call_args
        assert call_kwargs.kwargs.get("topic") == "feature.vector" or \
               (call_kwargs.args and call_kwargs.args[0] == "feature.vector")

        await agent.stop()

    @pytest.mark.asyncio
    async def test_feature_vector_updates_last_seen(self):
        from agentic_trading.agents.data_quality import DataQualityAgent

        agent = DataQualityAgent(event_bus=AsyncMock(), agent_id="dq-test")

        # Simulate receiving a feature vector event
        mock_event = MagicMock()
        mock_event.symbol = "BTC/USDT"
        await agent._on_feature_vector(mock_event)

        assert "BTC/USDT" in agent.last_seen
        assert agent.last_seen["BTC/USDT"] > 0

    @pytest.mark.asyncio
    async def test_staleness_emits_incident(self):
        from agentic_trading.agents.data_quality import DataQualityAgent

        event_bus = AsyncMock()
        agent = DataQualityAgent(
            event_bus=event_bus,
            staleness_threshold=0.01,  # Very short for testing
            agent_id="dq-test",
        )

        # Simulate a feature vector from the past
        mock_event = MagicMock()
        mock_event.symbol = "ETH/USDT"
        await agent._on_feature_vector(mock_event)

        # Force last_seen to be stale
        agent._last_seen["ETH/USDT"] = time.monotonic() - 100.0

        # Run the work cycle
        await agent._work()

        # Should have published an IncidentCreated event
        event_bus.publish.assert_called_once()
        call_args = event_bus.publish.call_args
        assert call_args.args[0] == "system"
        published_event = call_args.args[1]
        assert isinstance(published_event, IncidentCreated)
        assert published_event.severity == "warning"
        assert "ETH/USDT" in published_event.affected_symbols

    @pytest.mark.asyncio
    async def test_staleness_alert_not_repeated(self):
        """Once alerted, same symbol doesn't trigger again until refreshed."""
        from agentic_trading.agents.data_quality import DataQualityAgent

        event_bus = AsyncMock()
        agent = DataQualityAgent(
            event_bus=event_bus,
            staleness_threshold=0.01,
            agent_id="dq-test",
        )

        # Set stale data
        agent._last_seen["BTC/USDT"] = time.monotonic() - 100.0

        await agent._work()
        assert event_bus.publish.call_count == 1

        # Second work cycle — no new alert for same symbol
        await agent._work()
        assert event_bus.publish.call_count == 1

    @pytest.mark.asyncio
    async def test_fresh_data_clears_alert(self):
        """New feature vector clears the alert status."""
        from agentic_trading.agents.data_quality import DataQualityAgent

        agent = DataQualityAgent(
            event_bus=AsyncMock(),
            staleness_threshold=0.01,
            agent_id="dq-test",
        )

        agent._last_seen["BTC/USDT"] = time.monotonic() - 100.0
        agent._alerted.add("BTC/USDT")

        # Fresh data arrives
        mock_event = MagicMock()
        mock_event.symbol = "BTC/USDT"
        await agent._on_feature_vector(mock_event)

        assert "BTC/USDT" not in agent._alerted


# ===========================================================================
# IncidentResponseAgent unit tests
# ===========================================================================


class TestIncidentResponseAgent:
    """IncidentResponseAgent stub tests."""

    def test_satisfies_iagent(self):
        from agentic_trading.agents.incident_response import (
            IncidentResponseAgent,
        )
        from agentic_trading.core.interfaces import IAgent

        agent = IncidentResponseAgent(
            event_bus=AsyncMock(),
            agent_id="ir-test",
        )
        assert isinstance(agent, IAgent)
        assert agent.agent_type == AgentType.INCIDENT_RESPONSE

    def test_capabilities(self):
        from agentic_trading.agents.incident_response import (
            IncidentResponseAgent,
        )

        agent = IncidentResponseAgent(event_bus=AsyncMock())
        caps = agent.capabilities()
        assert "system" in caps.subscribes_to
        assert "system" in caps.publishes_to

    @pytest.mark.asyncio
    async def test_start_subscribes_to_system(self):
        from agentic_trading.agents.incident_response import (
            IncidentResponseAgent,
        )

        event_bus = AsyncMock()
        agent = IncidentResponseAgent(event_bus=event_bus, agent_id="ir-test")

        await agent.start()
        assert agent.is_running
        event_bus.subscribe.assert_called_once()

        await agent.stop()

    @pytest.mark.asyncio
    async def test_warning_escalates_to_cautious(self):
        from agentic_trading.agents.incident_response import (
            IncidentResponseAgent,
        )

        event_bus = AsyncMock()
        agent = IncidentResponseAgent(event_bus=event_bus, agent_id="ir-test")

        incident = IncidentCreated(
            severity="warning",
            component="data_quality",
            description="Feed stale for BTC/USDT",
        )
        await agent._handle_incident(incident)

        assert agent.current_mode == "cautious"
        assert incident.incident_id in agent.active_incidents
        # DegradedModeEnabled published for warning → cautious
        event_bus.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_critical_escalates_to_risk_off(self):
        from agentic_trading.agents.incident_response import (
            IncidentResponseAgent,
        )

        event_bus = AsyncMock()
        agent = IncidentResponseAgent(event_bus=event_bus, agent_id="ir-test")

        incident = IncidentCreated(
            severity="critical",
            component="exchange",
            description="Exchange connectivity lost",
        )
        await agent._handle_incident(incident)

        assert agent.current_mode == "risk_off_only"
        event_bus.publish.assert_called_once()
        call_args = event_bus.publish.call_args
        assert call_args.args[0] == "system"
        mode_event = call_args.args[1]
        assert isinstance(mode_event, DegradedModeEnabled)
        assert mode_event.mode == "risk_off_only"
        assert mode_event.previous_mode == "normal"

    @pytest.mark.asyncio
    async def test_emergency_escalates_to_full_stop(self):
        from agentic_trading.agents.incident_response import (
            IncidentResponseAgent,
        )

        event_bus = AsyncMock()
        agent = IncidentResponseAgent(event_bus=event_bus, agent_id="ir-test")

        incident = IncidentCreated(
            severity="emergency",
            component="risk",
            description="Critical risk breach",
        )
        await agent._handle_incident(incident)

        assert agent.current_mode == "full_stop"
        mode_event = event_bus.publish.call_args.args[1]
        assert mode_event.mode == "full_stop"

    @pytest.mark.asyncio
    async def test_mode_only_escalates_not_downgrades(self):
        """Once at full_stop, a critical incident doesn't downgrade."""
        from agentic_trading.agents.incident_response import (
            IncidentResponseAgent,
        )

        event_bus = AsyncMock()
        agent = IncidentResponseAgent(event_bus=event_bus, agent_id="ir-test")

        # First: emergency → full_stop
        emergency = IncidentCreated(
            severity="emergency",
            component="risk",
            description="Critical risk breach",
        )
        await agent._handle_incident(emergency)
        assert agent.current_mode == "full_stop"

        # Second: critical → should NOT downgrade to risk_off_only
        critical = IncidentCreated(
            severity="critical",
            component="data",
            description="Data issue",
        )
        await agent._handle_incident(critical)
        assert agent.current_mode == "full_stop"  # Still full_stop

        # Only one publish call (for the emergency)
        assert event_bus.publish.call_count == 1

    @pytest.mark.asyncio
    async def test_updates_policy_evaluator(self):
        """When policy_evaluator is provided, updates its system state."""
        from agentic_trading.agents.incident_response import (
            IncidentResponseAgent,
        )

        event_bus = AsyncMock()
        mock_evaluator = MagicMock()
        agent = IncidentResponseAgent(
            event_bus=event_bus,
            policy_evaluator=mock_evaluator,
            agent_id="ir-test",
        )

        incident = IncidentCreated(
            severity="critical",
            component="exchange",
            description="Exchange down",
        )
        await agent._handle_incident(incident)

        mock_evaluator.set_system_state.assert_called_once_with(
            "degraded_mode", "risk_off_only",
        )

    @pytest.mark.asyncio
    async def test_system_event_routing(self):
        """Non-incident events on system topic are ignored."""
        from agentic_trading.agents.incident_response import (
            IncidentResponseAgent,
        )

        event_bus = AsyncMock()
        agent = IncidentResponseAgent(event_bus=event_bus, agent_id="ir-test")

        # Non-incident event
        other_event = BaseEvent(source_module="test")
        await agent._on_system_event(other_event)

        assert len(agent.active_incidents) == 0
        assert agent.current_mode == "normal"


# ===========================================================================
# Tool allowlist consistency checks
# ===========================================================================


class TestToolAllowlistConsistency:
    """Verify tool allowlists are correct subsets."""

    def test_read_only_tools_are_not_mutating(self):
        from agentic_trading.control_plane.action_types import MUTATING_TOOLS

        for tool in _READ_ONLY_TOOLS:
            assert tool not in MUTATING_TOOLS, (
                f"{tool} is in both READ_ONLY and MUTATING"
            )

    def test_risk_off_contains_all_reads(self):
        for tool in _READ_ONLY_TOOLS:
            assert tool in _RISK_OFF_TOOLS, (
                f"{tool} in READ_ONLY but not in RISK_OFF"
            )

    def test_risk_off_has_protective_tools(self):
        assert ToolName.CANCEL_ORDER in _RISK_OFF_TOOLS
        assert ToolName.CANCEL_ALL_ORDERS in _RISK_OFF_TOOLS
        assert ToolName.SET_TRADING_STOP in _RISK_OFF_TOOLS

    def test_read_only_count(self):
        """Read-only has exactly the 6 read tools."""
        assert len(_READ_ONLY_TOOLS) == 6

    def test_risk_off_count(self):
        """Risk-off = 6 reads + 3 protective = 9 tools."""
        assert len(_RISK_OFF_TOOLS) == 9


# ===========================================================================
# AgentType enum completeness
# ===========================================================================


class TestAgentTypeEnum:
    """Verify new agent types exist in the enum."""

    def test_data_quality_type_exists(self):
        assert AgentType.DATA_QUALITY == "data_quality"

    def test_incident_response_type_exists(self):
        assert AgentType.INCIDENT_RESPONSE == "incident_response"
