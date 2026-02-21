"""Tests for P2 sprint: compliance polish & institutional features.

Covers:
- Regulatory mapping on PolicyRule/PolicySet (Task #16)
- Information barriers (Task #17)
- Audit bundle generation (Task #18)
- ReportingAgent (Task #19)
- FeatureComputationAgent (Task #20)
- Strategy lifecycle scorecard automation (Task #21)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_trading.core.enums import (
    AgentType,
    Exchange,
    GovernanceAction,
    MaturityLevel,
    OrderStatus,
    Side,
    StrategyStage,
)


# ---------------------------------------------------------------
# Task #16: Regulatory mapping
# ---------------------------------------------------------------


class TestRegulatoryMapping:
    def test_policy_rule_has_regulatory_refs(self):
        from agentic_trading.policy.models import (
            Operator,
            PolicyRule,
            PolicyType,
        )

        rule = PolicyRule(
            rule_id="max_notional",
            name="Max Notional",
            field="order_notional_usd",
            operator=Operator.LE,
            threshold=500_000.0,
            action=GovernanceAction.BLOCK,
            regulatory_refs=["SEC-15c3-5", "FINRA-5210"],
        )
        assert rule.regulatory_refs == ["SEC-15c3-5", "FINRA-5210"]

    def test_policy_rule_default_empty_refs(self):
        from agentic_trading.policy.models import Operator, PolicyRule

        rule = PolicyRule(
            rule_id="test",
            name="Test",
            field="x",
            operator=Operator.LT,
            threshold=1.0,
        )
        assert rule.regulatory_refs == []

    def test_policy_set_has_regulatory_refs(self):
        from agentic_trading.policy.models import PolicySet

        ps = PolicySet(
            set_id="risk_v1",
            name="Risk Limits",
            regulatory_refs=["MiFID-II-RTS-6"],
        )
        assert ps.regulatory_refs == ["MiFID-II-RTS-6"]

    def test_policy_set_rules_for_regulation(self):
        from agentic_trading.policy.models import (
            Operator,
            PolicyRule,
            PolicySet,
        )

        rules = [
            PolicyRule(
                rule_id="r1",
                name="Rule 1",
                field="x",
                operator=Operator.LT,
                threshold=1.0,
                regulatory_refs=["SEC-15c3-5"],
            ),
            PolicyRule(
                rule_id="r2",
                name="Rule 2",
                field="y",
                operator=Operator.GT,
                threshold=0.0,
                regulatory_refs=["FINRA-5210"],
            ),
            PolicyRule(
                rule_id="r3",
                name="Rule 3",
                field="z",
                operator=Operator.EQ,
                threshold="ok",
                regulatory_refs=["SEC-15c3-5", "FINRA-5210"],
            ),
        ]
        ps = PolicySet(set_id="test", name="Test", rules=rules)
        sec_rules = ps.rules_for_regulation("SEC-15c3-5")
        assert len(sec_rules) == 2
        assert {r.rule_id for r in sec_rules} == {"r1", "r3"}

    def test_eval_result_carries_regulatory_refs(self):
        from agentic_trading.policy.models import PolicyEvalResult

        result = PolicyEvalResult(
            rule_id="r1",
            rule_name="Rule 1",
            passed=False,
            field="x",
            regulatory_refs=["SEC-15c3-5"],
        )
        assert result.regulatory_refs == ["SEC-15c3-5"]


# ---------------------------------------------------------------
# Task #17: Information barriers
# ---------------------------------------------------------------


class TestInformationBarriers:
    def test_proposed_action_has_required_role(self):
        from agentic_trading.control_plane.action_types import (
            ProposedAction,
            ToolName,
        )

        action = ProposedAction(
            tool_name=ToolName.SUBMIT_ORDER,
            required_role="trader",
        )
        assert action.required_role == "trader"

    def test_proposed_action_default_no_role(self):
        from agentic_trading.control_plane.action_types import (
            ProposedAction,
            ToolName,
        )

        action = ProposedAction(tool_name=ToolName.GET_POSITIONS)
        assert action.required_role is None

    def test_action_scope_has_actor_role(self):
        from agentic_trading.control_plane.action_types import ActionScope

        scope = ActionScope(actor="agent-1", actor_role="trader")
        assert scope.actor_role == "trader"

    @pytest.mark.asyncio
    async def test_tool_gateway_blocks_wrong_role(self):
        from agentic_trading.control_plane.action_types import (
            ActionScope,
            ProposedAction,
            ToolName,
        )
        from agentic_trading.control_plane.audit_log import AuditLog
        from agentic_trading.control_plane.tool_gateway import ToolGateway

        adapter = AsyncMock()
        audit_log = AuditLog()
        event_bus = AsyncMock()

        gw = ToolGateway(
            adapter=adapter,
            audit_log=audit_log,
            event_bus=event_bus,
        )

        action = ProposedAction(
            tool_name=ToolName.SUBMIT_ORDER,
            scope=ActionScope(actor="agent-1", actor_role="viewer"),
            required_role="trader",
            request_params={"intent": {}},
        )

        result = await gw.call(action)
        assert not result.success
        assert "information_barrier" in result.error

    @pytest.mark.asyncio
    async def test_tool_gateway_allows_matching_role(self):
        """When actor_role matches required_role, the barrier doesn't block."""
        from agentic_trading.control_plane.action_types import (
            ActionScope,
            ProposedAction,
            ToolName,
        )
        from agentic_trading.control_plane.audit_log import AuditLog
        from agentic_trading.control_plane.tool_gateway import ToolGateway

        adapter = AsyncMock()
        # submit_order returns an OrderAck-like dict
        adapter.submit_order = AsyncMock(return_value=MagicMock(
            model_dump=lambda: {"order_id": "o1", "status": "submitted"},
        ))
        audit_log = AuditLog()
        event_bus = AsyncMock()

        gw = ToolGateway(
            adapter=adapter,
            audit_log=audit_log,
            event_bus=event_bus,
        )

        action = ProposedAction(
            tool_name=ToolName.SUBMIT_ORDER,
            scope=ActionScope(actor="agent-1", actor_role="trader"),
            required_role="trader",
            request_params={"intent": {
                "dedupe_key": "dk-1",
                "strategy_id": "strat-1",
                "symbol": "BTCUSDT",
                "exchange": "bybit",
                "side": "buy",
                "qty": "0.01",
                "price": "50000",
            }},
        )

        result = await gw.call(action)
        # Should pass the barrier (may fail for other reasons like policy,
        # but NOT for information_barrier)
        if not result.success:
            assert "information_barrier" not in (result.error or "")

    @pytest.mark.asyncio
    async def test_tool_gateway_no_role_required_passes(self):
        """When no required_role is set, the barrier doesn't apply."""
        from agentic_trading.control_plane.action_types import (
            ActionScope,
            ProposedAction,
            ToolName,
        )
        from agentic_trading.control_plane.audit_log import AuditLog
        from agentic_trading.control_plane.tool_gateway import ToolGateway

        adapter = AsyncMock()
        adapter.get_positions = AsyncMock(return_value=[])
        audit_log = AuditLog()
        event_bus = AsyncMock()

        gw = ToolGateway(
            adapter=adapter,
            audit_log=audit_log,
            event_bus=event_bus,
        )

        action = ProposedAction(
            tool_name=ToolName.GET_POSITIONS,
            scope=ActionScope(actor="agent-1", actor_role="viewer"),
            # No required_role
            request_params={},
        )

        result = await gw.call(action)
        assert result.success


# ---------------------------------------------------------------
# Task #18: Audit bundle generation
# ---------------------------------------------------------------


class TestAuditBundleGenerator:
    def test_empty_bundle_for_unknown_trace(self):
        from agentic_trading.control_plane.audit_log import AuditLog
        from agentic_trading.observability.audit_bundle import (
            AuditBundleGenerator,
        )

        audit_log = AuditLog()
        gen = AuditBundleGenerator(audit_log=audit_log)
        bundle = gen.generate("nonexistent-trace")
        assert bundle.trace_id == "nonexistent-trace"
        assert len(bundle.entries) == 0
        assert bundle.summary["total_entries"] == 0
        assert bundle.summary["outcome"] == "no_data"

    @pytest.mark.asyncio
    async def test_bundle_with_audit_entries(self):
        from agentic_trading.control_plane.action_types import AuditEntry
        from agentic_trading.control_plane.audit_log import AuditLog
        from agentic_trading.observability.audit_bundle import (
            AuditBundleGenerator,
        )

        audit_log = AuditLog()
        await audit_log.append(AuditEntry(
            correlation_id="trace-1",
            actor="agent-1",
            event_type="tool_call_pre_execution",
            payload={"action_id": "a1", "tool_name": "submit_order"},
        ))
        await audit_log.append(AuditEntry(
            correlation_id="trace-1",
            actor="agent-1",
            event_type="tool_call_recorded",
            payload={"action_id": "a1", "success": True},
        ))

        gen = AuditBundleGenerator(audit_log=audit_log)
        bundle = gen.generate("trace-1")

        assert len(bundle.entries) == 2
        assert bundle.summary["total_entries"] == 2
        assert "agent-1" in bundle.summary["actors"]
        assert bundle.summary["outcome"] == "executed_success"

    def test_bundle_with_case_manager(self):
        from agentic_trading.compliance.case_manager import CaseManager
        from agentic_trading.observability.audit_bundle import (
            AuditBundleGenerator,
        )

        cm = CaseManager()
        cm.open_case(
            case_type="wash_trade",
            severity="high",
            symbol="BTCUSDT",
            description="trace-abc suspicious activity",
        )

        gen = AuditBundleGenerator(case_manager=cm)
        bundle = gen.generate("trace-abc")
        assert len(bundle.surveillance_cases) == 1
        assert bundle.summary["total_cases"] == 1

    def test_bundle_no_sources(self):
        from agentic_trading.observability.audit_bundle import (
            AuditBundleGenerator,
        )

        gen = AuditBundleGenerator()
        bundle = gen.generate("trace-1")
        assert bundle.trace_id == "trace-1"
        assert len(bundle.entries) == 0

    @pytest.mark.asyncio
    async def test_outcome_policy_blocked(self):
        from agentic_trading.control_plane.action_types import AuditEntry
        from agentic_trading.control_plane.audit_log import AuditLog
        from agentic_trading.observability.audit_bundle import (
            AuditBundleGenerator,
        )

        audit_log = AuditLog()
        await audit_log.append(AuditEntry(
            correlation_id="trace-2",
            event_type="policy_blocked",
            payload={"reasons": ["max_notional exceeded"]},
        ))

        gen = AuditBundleGenerator(audit_log=audit_log)
        bundle = gen.generate("trace-2")
        assert bundle.summary["outcome"] == "blocked_policy"


# ---------------------------------------------------------------
# Task #19: ReportingAgent
# ---------------------------------------------------------------


class TestReportingAgent:
    def test_agent_type_and_capabilities(self):
        from agentic_trading.agents.reporting import ReportingAgent

        bus = AsyncMock()
        agent = ReportingAgent(event_bus=bus)
        assert agent.agent_type == AgentType.REPORTING
        caps = agent.capabilities()
        assert "reporting" in caps.publishes_to

    @pytest.mark.asyncio
    async def test_work_publishes_report(self):
        from agentic_trading.agents.reporting import (
            DailyReportEvent,
            ReportingAgent,
        )

        bus = AsyncMock()
        bus.publish = AsyncMock()

        agent = ReportingAgent(event_bus=bus)
        await agent._work()

        bus.publish.assert_called_once()
        call_args = bus.publish.call_args
        assert call_args[0][0] == "reporting"
        report = call_args[0][1]
        assert isinstance(report, DailyReportEvent)
        assert report.report_id != ""
        assert report.report_date != ""
        assert agent.reports_generated == 1

    def test_report_with_case_manager(self):
        from agentic_trading.agents.reporting import ReportingAgent
        from agentic_trading.compliance.case_manager import CaseManager

        cm = CaseManager()
        cm.open_case(case_type="wash_trade", severity="high")
        cm.open_case(case_type="spoofing", severity="medium")

        bus = AsyncMock()
        agent = ReportingAgent(event_bus=bus, case_manager=cm)
        report = agent._compile_report()
        assert report.surveillance_summary["available"] is True
        assert report.surveillance_summary["total_cases"] == 2
        assert report.surveillance_summary["open_cases"] == 2

    def test_report_without_sources(self):
        from agentic_trading.agents.reporting import ReportingAgent

        bus = AsyncMock()
        agent = ReportingAgent(event_bus=bus)
        report = agent._compile_report()
        assert report.pnl_summary["available"] is False
        assert report.risk_summary["available"] is False
        assert report.surveillance_summary["available"] is False


class TestDailyReportEvent:
    def test_daily_report_in_schema_registry(self):
        from agentic_trading.bus.schemas import TOPIC_SCHEMAS

        assert "reporting" in TOPIC_SCHEMAS

    def test_daily_report_event_type_map(self):
        from agentic_trading.bus.schemas import EVENT_TYPE_MAP

        assert "DailyReportEvent" in EVENT_TYPE_MAP


# ---------------------------------------------------------------
# Task #20: FeatureComputationAgent
# ---------------------------------------------------------------


class TestFeatureComputationAgent:
    def test_agent_type(self):
        from agentic_trading.agents.feature_computation import (
            FeatureComputationAgent,
        )

        bus = AsyncMock()
        agent = FeatureComputationAgent(event_bus=bus)
        assert agent.agent_type == AgentType.FEATURE_COMPUTATION

    def test_capabilities(self):
        from agentic_trading.agents.feature_computation import (
            FeatureComputationAgent,
        )

        bus = AsyncMock()
        agent = FeatureComputationAgent(event_bus=bus)
        caps = agent.capabilities()
        assert "market.candle" in caps.subscribes_to
        assert "feature.vector" in caps.publishes_to

    def test_agent_type_enum_exists(self):
        assert hasattr(AgentType, "FEATURE_COMPUTATION")
        assert AgentType.FEATURE_COMPUTATION.value == "feature_computation"


class TestMarketIntelligenceAgentUpdated:
    def test_capabilities_updated(self):
        from agentic_trading.agents.market_intelligence import (
            MarketIntelligenceAgent,
        )

        bus = AsyncMock()
        agent = MarketIntelligenceAgent(event_bus=bus)
        caps = agent.capabilities()
        assert "market.candle" in caps.publishes_to


# ---------------------------------------------------------------
# Task #21: Strategy lifecycle scorecard automation
# ---------------------------------------------------------------


class TestScorecardAutomation:
    def _make_lifecycle(self, scorecard=None, **kwargs):
        from agentic_trading.policy.strategy_lifecycle import (
            StrategyLifecycleManager,
        )

        bus = AsyncMock()
        bus.publish = AsyncMock()
        journal = MagicMock()
        journal.get_strategy_stats = MagicMock(return_value={})
        gate = MagicMock()
        gate.drift.get_status = MagicMock(return_value={"metrics": {}})

        return StrategyLifecycleManager(
            event_bus=bus,
            journal=journal,
            governance_gate=gate,
            scorecard=scorecard,
            **kwargs,
        )

    def test_scorecard_triggers_exist(self):
        from agentic_trading.policy.strategy_lifecycle import (
            DEFAULT_SCORECARD_PROMOTION,
            DEFAULT_SCORECARD_TRIGGERS,
        )

        assert "edge_quality" in DEFAULT_SCORECARD_TRIGGERS
        assert "edge_quality" in DEFAULT_SCORECARD_PROMOTION

    def test_scorecard_demotion_after_sustained_poor_metrics(self):
        scorecard = MagicMock()
        scorecard.compute = MagicMock(return_value={
            "edge_quality": 2.0,  # Below 3.0 threshold
            "hit_rate": 0.30,      # Below 0.35 threshold
            "sharpe_ratio": -0.5,  # Below 0.0 threshold
            "profit_factor": 0.5,  # Below 0.8 threshold
        })

        manager = self._make_lifecycle(
            scorecard=scorecard,
            scorecard_triggers={
                "edge_quality": (3.0, "lt", 3),  # 3 cycles for test speed
            },
        )
        manager.register_strategy("strat-1", StrategyStage.LIMITED)

        # Run 3 cycles — should trigger after 3 consecutive
        loop = asyncio.get_event_loop()
        for _ in range(3):
            loop.run_until_complete(manager._work())

        assert manager.get_stage("strat-1") == StrategyStage.DEMOTED

    def test_scorecard_no_demotion_if_metric_recovers(self):
        call_count = 0

        def compute_varying(strategy_id):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return {"edge_quality": 2.0}  # Bad
            return {"edge_quality": 5.0}  # Recovered

        scorecard = MagicMock()
        scorecard.compute = compute_varying

        manager = self._make_lifecycle(
            scorecard=scorecard,
            scorecard_triggers={
                "edge_quality": (3.0, "lt", 3),
            },
        )
        manager.register_strategy("strat-1", StrategyStage.LIMITED)

        loop = asyncio.get_event_loop()
        for _ in range(5):
            loop.run_until_complete(manager._work())

        # Should NOT be demoted because metric recovered
        assert manager.get_stage("strat-1") != StrategyStage.DEMOTED

    def test_scorecard_promotion_after_sustained_good_metrics(self):
        scorecard = MagicMock()
        scorecard.compute = MagicMock(return_value={
            "edge_quality": 8.0,
            "hit_rate": 0.60,
            "sharpe_ratio": 2.0,
            "profit_factor": 2.0,
        })

        manager = self._make_lifecycle(
            scorecard=scorecard,
            # All must be sustained for 3 cycles
            scorecard_promotion={
                "edge_quality": (7.0, "gt", 3),
                "hit_rate": (0.55, "gt", 3),
                "sharpe_ratio": (1.5, "gt", 3),
                "profit_factor": (1.5, "gt", 3),
            },
        )
        manager.register_strategy("strat-1", StrategyStage.LIMITED)

        loop = asyncio.get_event_loop()
        for _ in range(3):
            loop.run_until_complete(manager._work())

        assert manager.get_stage("strat-1") == StrategyStage.SCALE

    def test_no_scorecard_no_automation(self):
        """Without a scorecard, no scorecard-based transitions happen."""
        manager = self._make_lifecycle(scorecard=None)
        manager.register_strategy("strat-1", StrategyStage.LIMITED)

        loop = asyncio.get_event_loop()
        for _ in range(5):
            loop.run_until_complete(manager._work())

        assert manager.get_stage("strat-1") == StrategyStage.LIMITED

    def test_demotion_triggers_still_work(self):
        """Existing demotion triggers still function alongside scorecard."""
        manager = self._make_lifecycle()
        manager._journal.get_strategy_stats.return_value = {
            "max_drawdown_pct": 15.0,  # Exceeds default 10.0
        }
        manager.register_strategy("strat-1", StrategyStage.PAPER)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(manager._work())

        assert manager.get_stage("strat-1") == StrategyStage.DEMOTED
