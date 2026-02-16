"""Tests for the approval workflow system.

Covers:
- ApprovalRule matching logic
- ApprovalManager lifecycle (request → approve/reject/expire/escalate)
- Auto-approval for L1 requests
- GovernanceGate + ApprovalManager integration
- Event publishing
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agentic_trading.governance.approval_manager import ApprovalManager
from agentic_trading.governance.approval_models import (
    ApprovalRequest,
    ApprovalRule,
    ApprovalStatus,
    ApprovalTrigger,
    EscalationLevel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rule(**overrides) -> ApprovalRule:
    defaults = dict(
        rule_id="test_rule",
        name="Test Rule",
        trigger=ApprovalTrigger.HIGH_IMPACT,
        escalation_level=EscalationLevel.L2_OPERATOR,
    )
    defaults.update(overrides)
    return ApprovalRule(**defaults)


def _make_context(**overrides) -> dict:
    defaults = dict(
        strategy_id="trend_following",
        symbol="BTC/USDT",
        notional_usd=50_000.0,
        impact_tier="high",
        maturity_level="L3_constrained",
        leverage=3,
    )
    defaults.update(overrides)
    return defaults


class FakeEventBus:
    """Minimal event bus for testing event publishing."""

    def __init__(self):
        self.published: list[tuple[str, object]] = []

    async def publish(self, topic: str, event) -> None:
        self.published.append((topic, event))


# ===========================================================================
# ApprovalRule matching
# ===========================================================================


class TestApprovalRule:
    """Test ApprovalRule.matches() with various conditions."""

    def test_no_conditions_always_matches(self):
        """A rule with no thresholds matches any context."""
        rule = _make_rule()
        assert rule.matches(_make_context())

    def test_disabled_rule_never_matches(self):
        rule = _make_rule(enabled=False)
        assert not rule.matches(_make_context())

    def test_notional_threshold_pass(self):
        rule = _make_rule(min_notional_usd=25_000)
        assert rule.matches(_make_context(notional_usd=50_000))

    def test_notional_threshold_fail(self):
        rule = _make_rule(min_notional_usd=100_000)
        assert not rule.matches(_make_context(notional_usd=50_000))

    def test_impact_tier_match(self):
        rule = _make_rule(impact_tiers=["high", "critical"])
        assert rule.matches(_make_context(impact_tier="high"))
        assert rule.matches(_make_context(impact_tier="critical"))

    def test_impact_tier_mismatch(self):
        rule = _make_rule(impact_tiers=["critical"])
        assert not rule.matches(_make_context(impact_tier="high"))

    def test_strategy_scope(self):
        rule = _make_rule(strategy_ids=["trend_following"])
        assert rule.matches(_make_context(strategy_id="trend_following"))
        assert not rule.matches(_make_context(strategy_id="mean_reversion"))

    def test_symbol_scope(self):
        rule = _make_rule(symbols=["BTC/USDT", "ETH/USDT"])
        assert rule.matches(_make_context(symbol="BTC/USDT"))
        assert not rule.matches(_make_context(symbol="SOL/USDT"))

    def test_maturity_level_scope(self):
        rule = _make_rule(maturity_levels=["L2_gated"])
        assert rule.matches(_make_context(maturity_level="L2_gated"))
        assert not rule.matches(_make_context(maturity_level="L4_autonomous"))

    def test_combined_conditions_all_must_match(self):
        """Multiple conditions use AND logic."""
        rule = _make_rule(
            min_notional_usd=10_000,
            impact_tiers=["high", "critical"],
            strategy_ids=["trend_following"],
        )
        # All conditions met
        ctx = _make_context(
            notional_usd=50_000,
            impact_tier="high",
            strategy_id="trend_following",
        )
        assert rule.matches(ctx)

        # One condition fails (wrong strategy)
        ctx2 = _make_context(
            notional_usd=50_000,
            impact_tier="high",
            strategy_id="mean_reversion",
        )
        assert not rule.matches(ctx2)


# ===========================================================================
# ApprovalRequest model
# ===========================================================================


class TestApprovalRequest:
    """Test ApprovalRequest model properties."""

    def test_is_expired(self):
        req = ApprovalRequest(
            strategy_id="s1",
            symbol="BTC/USDT",
            ttl_seconds=0,  # Expired immediately
            created_at=datetime.now(UTC) - timedelta(seconds=5),
        )
        assert req.is_expired

    def test_not_expired(self):
        req = ApprovalRequest(
            strategy_id="s1",
            symbol="BTC/USDT",
            ttl_seconds=300,
        )
        assert not req.is_expired

    def test_is_terminal_pending(self):
        req = ApprovalRequest(strategy_id="s1", symbol="BTC/USDT")
        assert not req.is_terminal

    def test_is_terminal_approved(self):
        req = ApprovalRequest(
            strategy_id="s1",
            symbol="BTC/USDT",
            status=ApprovalStatus.APPROVED,
        )
        assert req.is_terminal

    def test_is_terminal_rejected(self):
        req = ApprovalRequest(
            strategy_id="s1",
            symbol="BTC/USDT",
            status=ApprovalStatus.REJECTED,
        )
        assert req.is_terminal

    def test_is_terminal_escalated_is_not_terminal(self):
        """Escalated requests are NOT terminal — they can still be approved."""
        req = ApprovalRequest(
            strategy_id="s1",
            symbol="BTC/USDT",
            status=ApprovalStatus.ESCALATED,
        )
        assert not req.is_terminal


# ===========================================================================
# ApprovalManager: request lifecycle
# ===========================================================================


class TestApprovalManagerRequestApproval:
    """Test creating approval requests."""

    @pytest.mark.asyncio
    async def test_request_creates_pending(self):
        mgr = ApprovalManager()
        req = await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
        )
        assert req.status == ApprovalStatus.PENDING
        assert req.strategy_id == "s1"
        assert req.symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_request_stored_by_id(self):
        mgr = ApprovalManager()
        req = await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
        )
        assert mgr.get_request(req.request_id) is req

    @pytest.mark.asyncio
    async def test_pending_index_updated(self):
        mgr = ApprovalManager()
        await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
        )
        assert mgr.has_pending("s1")
        assert not mgr.has_pending("s2")

    @pytest.mark.asyncio
    async def test_get_pending_by_strategy(self):
        mgr = ApprovalManager()
        await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
        )
        await mgr.request_approval(
            strategy_id="s2",
            symbol="ETH/USDT",
            trigger=ApprovalTrigger.SIZE_THRESHOLD,
        )
        pending_s1 = mgr.get_pending("s1")
        assert len(pending_s1) == 1
        assert pending_s1[0].strategy_id == "s1"


class TestApprovalManagerApprove:
    """Test approving requests."""

    @pytest.mark.asyncio
    async def test_approve_success(self):
        mgr = ApprovalManager()
        req = await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
        )
        decision = await mgr.approve(req.request_id, "operator1", "looks good")
        assert decision is not None
        assert decision.status == ApprovalStatus.APPROVED
        assert decision.decided_by == "operator1"

        # Request updated
        updated = mgr.get_request(req.request_id)
        assert updated.status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_approve_removes_from_pending(self):
        mgr = ApprovalManager()
        req = await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
        )
        await mgr.approve(req.request_id, "op1")
        assert not mgr.has_pending("s1")

    @pytest.mark.asyncio
    async def test_approve_nonexistent_returns_none(self):
        mgr = ApprovalManager()
        result = await mgr.approve("nonexistent", "op1")
        assert result is None

    @pytest.mark.asyncio
    async def test_approve_already_approved_returns_none(self):
        mgr = ApprovalManager()
        req = await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
        )
        await mgr.approve(req.request_id, "op1")
        # Second approval should fail
        result = await mgr.approve(req.request_id, "op2")
        assert result is None


class TestApprovalManagerReject:
    """Test rejecting requests."""

    @pytest.mark.asyncio
    async def test_reject_success(self):
        mgr = ApprovalManager()
        req = await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.POLICY_VIOLATION,
        )
        decision = await mgr.reject(req.request_id, "risk_team", "too risky")
        assert decision is not None
        assert decision.status == ApprovalStatus.REJECTED
        assert decision.reason == "too risky"

    @pytest.mark.asyncio
    async def test_reject_removes_from_pending(self):
        mgr = ApprovalManager()
        req = await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
        )
        await mgr.reject(req.request_id, "op1")
        assert not mgr.has_pending("s1")

    @pytest.mark.asyncio
    async def test_reject_nonexistent_returns_none(self):
        mgr = ApprovalManager()
        result = await mgr.reject("nonexistent", "op1")
        assert result is None


class TestApprovalManagerEscalate:
    """Test escalating requests."""

    @pytest.mark.asyncio
    async def test_escalate_changes_level(self):
        mgr = ApprovalManager()
        req = await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
            escalation_level=EscalationLevel.L2_OPERATOR,
        )
        result = await mgr.escalate(
            req.request_id,
            EscalationLevel.L3_RISK,
            reason="operator unsure",
        )
        assert result is True
        updated = mgr.get_request(req.request_id)
        assert updated.escalation_level == EscalationLevel.L3_RISK

    @pytest.mark.asyncio
    async def test_escalate_terminal_returns_false(self):
        mgr = ApprovalManager()
        req = await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
        )
        await mgr.approve(req.request_id, "op1")
        result = await mgr.escalate(req.request_id, EscalationLevel.L3_RISK)
        assert result is False


class TestApprovalManagerCancel:
    """Test cancelling requests."""

    @pytest.mark.asyncio
    async def test_cancel_success(self):
        mgr = ApprovalManager()
        req = await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
        )
        result = await mgr.cancel(req.request_id, "no longer needed")
        assert result is True
        updated = mgr.get_request(req.request_id)
        assert updated.status == ApprovalStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_already_approved_returns_false(self):
        mgr = ApprovalManager()
        req = await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
        )
        await mgr.approve(req.request_id, "op1")
        result = await mgr.cancel(req.request_id)
        assert result is False


class TestApprovalManagerExpiry:
    """Test TTL-based expiry."""

    @pytest.mark.asyncio
    async def test_expire_stale_requests(self):
        mgr = ApprovalManager()
        req = await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
            ttl_seconds=0,  # Expire immediately
        )
        # Force creation time into the past
        req.created_at = datetime.now(UTC) - timedelta(seconds=10)

        expired = await mgr.expire_stale()
        assert req.request_id in expired
        assert mgr.get_request(req.request_id).status == ApprovalStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_non_expired_not_touched(self):
        mgr = ApprovalManager()
        await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
            ttl_seconds=600,
        )
        expired = await mgr.expire_stale()
        assert len(expired) == 0

    @pytest.mark.asyncio
    async def test_approve_expired_request_returns_none(self):
        """Attempting to approve an expired request should detect expiry."""
        mgr = ApprovalManager()
        req = await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
            ttl_seconds=0,
        )
        req.created_at = datetime.now(UTC) - timedelta(seconds=10)

        result = await mgr.approve(req.request_id, "op1")
        assert result is None
        assert mgr.get_request(req.request_id).status == ApprovalStatus.EXPIRED


# ===========================================================================
# Auto-approval (L1)
# ===========================================================================


class TestAutoApproval:
    """Test L1_AUTO escalation auto-approval."""

    @pytest.mark.asyncio
    async def test_l1_auto_approved(self):
        mgr = ApprovalManager(auto_approve_l1=True)
        req = await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
            escalation_level=EscalationLevel.L1_AUTO,
        )
        assert req.status == ApprovalStatus.APPROVED
        assert req.decided_by == "system_auto"

    @pytest.mark.asyncio
    async def test_l1_not_auto_approved_when_disabled(self):
        mgr = ApprovalManager(auto_approve_l1=False)
        req = await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
            escalation_level=EscalationLevel.L1_AUTO,
        )
        assert req.status == ApprovalStatus.PENDING

    @pytest.mark.asyncio
    async def test_l2_not_auto_approved(self):
        mgr = ApprovalManager(auto_approve_l1=True)
        req = await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
            escalation_level=EscalationLevel.L2_OPERATOR,
        )
        assert req.status == ApprovalStatus.PENDING


# ===========================================================================
# Rule checking
# ===========================================================================


class TestCheckApprovalRequired:
    """Test ApprovalManager.check_approval_required()."""

    def test_no_rules_returns_none(self):
        mgr = ApprovalManager()
        result = mgr.check_approval_required(_make_context())
        assert result is None

    def test_matching_rule_returned(self):
        rule = _make_rule(
            rule_id="big_trade",
            min_notional_usd=25_000,
            impact_tiers=["high", "critical"],
        )
        mgr = ApprovalManager(rules=[rule])
        result = mgr.check_approval_required(_make_context(
            notional_usd=50_000,
            impact_tier="high",
        ))
        assert result is not None
        assert result.rule_id == "big_trade"

    def test_no_match_returns_none(self):
        rule = _make_rule(min_notional_usd=1_000_000)
        mgr = ApprovalManager(rules=[rule])
        result = mgr.check_approval_required(_make_context(notional_usd=50_000))
        assert result is None

    def test_first_matching_rule_wins(self):
        rule1 = _make_rule(rule_id="r1", min_notional_usd=10_000)
        rule2 = _make_rule(rule_id="r2", min_notional_usd=20_000)
        mgr = ApprovalManager(rules=[rule1, rule2])
        result = mgr.check_approval_required(_make_context(notional_usd=50_000))
        assert result.rule_id == "r1"

    def test_add_and_remove_rule(self):
        mgr = ApprovalManager()
        rule = _make_rule(rule_id="r1")
        mgr.add_rule(rule)
        assert len(mgr.rules) == 1

        removed = mgr.remove_rule("r1")
        assert removed is True
        assert len(mgr.rules) == 0

        # Removing non-existent returns False
        assert mgr.remove_rule("r1") is False


# ===========================================================================
# Summary
# ===========================================================================


class TestApprovalSummary:
    """Test summary statistics."""

    @pytest.mark.asyncio
    async def test_summary_counts(self):
        mgr = ApprovalManager()

        # Create some requests with different outcomes
        r1 = await mgr.request_approval(
            strategy_id="s1", symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
        )
        await mgr.approve(r1.request_id, "op1")

        r2 = await mgr.request_approval(
            strategy_id="s2", symbol="ETH/USDT",
            trigger=ApprovalTrigger.POLICY_VIOLATION,
        )
        await mgr.reject(r2.request_id, "op2", "bad")

        await mgr.request_approval(
            strategy_id="s3", symbol="SOL/USDT",
            trigger=ApprovalTrigger.SIZE_THRESHOLD,
        )  # Still pending

        summary = mgr.get_summary()
        assert summary.total_requests == 3
        assert summary.approved == 1
        assert summary.rejected == 1
        assert summary.pending == 1


# ===========================================================================
# Event publishing
# ===========================================================================


class TestApprovalEventPublishing:
    """Test that events are published to the event bus."""

    @pytest.mark.asyncio
    async def test_request_publishes_approval_requested(self):
        bus = FakeEventBus()
        mgr = ApprovalManager(event_bus=bus)

        await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
        )

        # Should have published ApprovalRequested
        assert len(bus.published) >= 1
        topic, event = bus.published[0]
        assert topic == "governance.approval"
        assert event.strategy_id == "s1"

    @pytest.mark.asyncio
    async def test_approve_publishes_approval_resolved(self):
        bus = FakeEventBus()
        mgr = ApprovalManager(event_bus=bus)

        req = await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
        )
        await mgr.approve(req.request_id, "op1")

        # Second publish should be ApprovalResolved
        resolved_events = [
            (t, e) for t, e in bus.published
            if hasattr(e, "status") and e.status == "approved"
        ]
        assert len(resolved_events) >= 1

    @pytest.mark.asyncio
    async def test_no_event_bus_no_crash(self):
        """Manager works without an event bus (backward compat)."""
        mgr = ApprovalManager(event_bus=None)
        req = await mgr.request_approval(
            strategy_id="s1",
            symbol="BTC/USDT",
            trigger=ApprovalTrigger.HIGH_IMPACT,
        )
        await mgr.approve(req.request_id, "op1")
        # No exception raised


# ===========================================================================
# GovernanceGate + ApprovalManager integration
# ===========================================================================


class TestGovernanceGateWithApproval:
    """Test approval workflow integration with GovernanceGate."""

    @pytest.mark.asyncio
    async def test_high_impact_blocked_pending_approval(self):
        """High-impact trades matching an approval rule get BLOCKED pending approval."""
        from agentic_trading.core.config import GovernanceConfig
        from agentic_trading.core.enums import (
            GovernanceAction,
            MaturityLevel,
        )
        from agentic_trading.governance.drift_detector import DriftDetector
        from agentic_trading.governance.gate import GovernanceGate
        from agentic_trading.governance.health_score import HealthTracker
        from agentic_trading.governance.impact_classifier import ImpactClassifier
        from agentic_trading.governance.maturity import MaturityManager

        rule = _make_rule(
            rule_id="big_trade",
            min_notional_usd=10_000,
            impact_tiers=["high", "critical"],
        )
        approval_mgr = ApprovalManager(rules=[rule])

        config = GovernanceConfig(enabled=True)
        gate = GovernanceGate(
            config=config,
            maturity=MaturityManager(config.maturity),
            health=HealthTracker(config.health_score),
            impact=ImpactClassifier(config.impact_classifier),
            drift=DriftDetector(config.drift_detector),
            approval_manager=approval_mgr,
        )
        gate.maturity._levels["s1"] = MaturityLevel.L4_AUTONOMOUS

        decision = await gate.evaluate(
            strategy_id="s1",
            symbol="BTC/USDT",
            notional_usd=50_000,
            portfolio_pct=0.2,
            leverage=5,
        )

        assert decision.action == GovernanceAction.BLOCK
        assert "pending_approval" in decision.reason
        assert "approval_request_id" in decision.details

    @pytest.mark.asyncio
    async def test_low_impact_not_blocked(self):
        """Low-impact trades that don't match rules pass through normally."""
        from agentic_trading.core.config import GovernanceConfig
        from agentic_trading.core.enums import (
            GovernanceAction,
            MaturityLevel,
        )
        from agentic_trading.governance.drift_detector import DriftDetector
        from agentic_trading.governance.gate import GovernanceGate
        from agentic_trading.governance.health_score import HealthTracker
        from agentic_trading.governance.impact_classifier import ImpactClassifier
        from agentic_trading.governance.maturity import MaturityManager

        # Rule requires notional > 100k — our trade is only 1k
        rule = _make_rule(min_notional_usd=100_000)
        approval_mgr = ApprovalManager(rules=[rule])

        config = GovernanceConfig(enabled=True)
        gate = GovernanceGate(
            config=config,
            maturity=MaturityManager(config.maturity),
            health=HealthTracker(config.health_score),
            impact=ImpactClassifier(config.impact_classifier),
            drift=DriftDetector(config.drift_detector),
            approval_manager=approval_mgr,
        )
        gate.maturity._levels["s1"] = MaturityLevel.L4_AUTONOMOUS

        decision = await gate.evaluate(
            strategy_id="s1",
            symbol="BTC/USDT",
            notional_usd=1_000,
        )

        assert decision.action == GovernanceAction.ALLOW

    @pytest.mark.asyncio
    async def test_l1_auto_approval_continues(self):
        """L1_AUTO rules auto-approve and don't block the gate."""
        from agentic_trading.core.config import GovernanceConfig
        from agentic_trading.core.enums import (
            GovernanceAction,
            MaturityLevel,
        )
        from agentic_trading.governance.drift_detector import DriftDetector
        from agentic_trading.governance.gate import GovernanceGate
        from agentic_trading.governance.health_score import HealthTracker
        from agentic_trading.governance.impact_classifier import ImpactClassifier
        from agentic_trading.governance.maturity import MaturityManager

        rule = _make_rule(
            rule_id="auto_rule",
            min_notional_usd=1_000,
            escalation_level=EscalationLevel.L1_AUTO,
            auto_approve=True,
        )
        approval_mgr = ApprovalManager(rules=[rule], auto_approve_l1=True)

        config = GovernanceConfig(enabled=True)
        gate = GovernanceGate(
            config=config,
            maturity=MaturityManager(config.maturity),
            health=HealthTracker(config.health_score),
            impact=ImpactClassifier(config.impact_classifier),
            drift=DriftDetector(config.drift_detector),
            approval_manager=approval_mgr,
        )
        gate.maturity._levels["s1"] = MaturityLevel.L4_AUTONOMOUS

        decision = await gate.evaluate(
            strategy_id="s1",
            symbol="BTC/USDT",
            notional_usd=5_000,
        )

        # Auto-approved at L1, so gate should ALLOW
        assert decision.action == GovernanceAction.ALLOW
        assert decision.details.get("approval_auto_approved") is True

    @pytest.mark.asyncio
    async def test_gate_works_without_approval_manager(self):
        """GovernanceGate backward compat — no approval manager."""
        from agentic_trading.core.config import GovernanceConfig
        from agentic_trading.core.enums import (
            GovernanceAction,
            MaturityLevel,
        )
        from agentic_trading.governance.drift_detector import DriftDetector
        from agentic_trading.governance.gate import GovernanceGate
        from agentic_trading.governance.health_score import HealthTracker
        from agentic_trading.governance.impact_classifier import ImpactClassifier
        from agentic_trading.governance.maturity import MaturityManager

        config = GovernanceConfig(enabled=True)
        gate = GovernanceGate(
            config=config,
            maturity=MaturityManager(config.maturity),
            health=HealthTracker(config.health_score),
            impact=ImpactClassifier(config.impact_classifier),
            drift=DriftDetector(config.drift_detector),
            # No approval_manager
        )
        gate.maturity._levels["s1"] = MaturityLevel.L4_AUTONOMOUS

        decision = await gate.evaluate(
            strategy_id="s1",
            symbol="BTC/USDT",
            notional_usd=50_000,
        )

        assert decision.action == GovernanceAction.ALLOW
