"""Tests for governance event schemas."""

from agentic_trading.core.enums import GovernanceAction, ImpactTier, MaturityLevel
from agentic_trading.core.events import (
    CanaryAlert,
    DriftAlert,
    GovernanceCanaryCheck,
    GovernanceDecision,
    HealthScoreUpdate,
    MaturityTransition,
    TokenEvent,
)


class TestGovernanceEvents:
    """Governance event construction and serialization."""

    def test_governance_decision_defaults(self):
        event = GovernanceDecision(
            strategy_id="s1",
            symbol="BTC/USDT",
            action=GovernanceAction.ALLOW,
        )
        assert event.strategy_id == "s1"
        assert event.action == GovernanceAction.ALLOW
        assert event.sizing_multiplier == 1.0
        assert event.source_module == "governance"

    def test_maturity_transition(self):
        event = MaturityTransition(
            strategy_id="s1",
            from_level=MaturityLevel.L1_PAPER,
            to_level=MaturityLevel.L2_GATED,
            reason="performance_criteria_met",
        )
        assert event.from_level == MaturityLevel.L1_PAPER
        assert event.to_level == MaturityLevel.L2_GATED
        assert event.source_module == "governance.maturity"

    def test_health_score_update(self):
        event = HealthScoreUpdate(
            strategy_id="s1",
            score=0.85,
            debt=1.5,
            credit=0.2,
            sizing_multiplier=0.85,
            window_trades=20,
        )
        assert event.score == 0.85
        assert event.window_trades == 20

    def test_canary_alert(self):
        event = CanaryAlert(
            component="redis",
            healthy=False,
            message="Connection refused",
            action_taken=GovernanceAction.KILL,
        )
        assert event.healthy is False
        assert event.action_taken == GovernanceAction.KILL

    def test_drift_alert(self):
        event = DriftAlert(
            strategy_id="s1",
            metric_name="win_rate",
            baseline_value=0.55,
            live_value=0.30,
            deviation_pct=45.5,
            action_taken=GovernanceAction.REDUCE_SIZE,
        )
        assert event.deviation_pct == 45.5

    def test_token_event(self):
        event = TokenEvent(
            token_id="tok-123",
            strategy_id="s1",
            action="issued",
            scope="order:BTC/USDT",
            ttl_seconds=300,
        )
        assert event.action == "issued"
        assert event.ttl_seconds == 300

    def test_canary_check(self):
        event = GovernanceCanaryCheck(
            all_healthy=False,
            components_checked=3,
            failed_components=["redis"],
        )
        assert event.all_healthy is False
        assert len(event.failed_components) == 1
