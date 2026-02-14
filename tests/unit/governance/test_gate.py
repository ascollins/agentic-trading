"""Tests for governance.gate — GovernanceGate orchestrator."""

import pytest

from agentic_trading.core.config import (
    GovernanceConfig,
    MaturityConfig,
    HealthScoreConfig,
    ImpactClassifierConfig,
    DriftDetectorConfig,
    ExecutionTokenConfig,
)
from agentic_trading.core.enums import GovernanceAction, ImpactTier, MaturityLevel
from agentic_trading.governance.drift_detector import DriftDetector
from agentic_trading.governance.gate import GovernanceGate
from agentic_trading.governance.health_score import HealthTracker
from agentic_trading.governance.impact_classifier import ImpactClassifier
from agentic_trading.governance.maturity import MaturityManager
from agentic_trading.governance.tokens import TokenManager


class TestGovernanceDisabled:
    """When governance is disabled, everything should pass through."""

    @pytest.mark.asyncio
    async def test_disabled_returns_allow(self):
        cfg = GovernanceConfig(enabled=False)
        gate = GovernanceGate(
            config=cfg,
            maturity=MaturityManager(MaturityConfig()),
            health=HealthTracker(HealthScoreConfig()),
            impact=ImpactClassifier(ImpactClassifierConfig()),
            drift=DriftDetector(DriftDetectorConfig()),
        )
        decision = await gate.evaluate(
            strategy_id="s1", symbol="BTC/USDT"
        )
        assert decision.action == GovernanceAction.ALLOW
        assert decision.sizing_multiplier == 1.0

    @pytest.mark.asyncio
    async def test_disabled_reason(self):
        cfg = GovernanceConfig(enabled=False)
        gate = GovernanceGate(
            config=cfg,
            maturity=MaturityManager(MaturityConfig()),
            health=HealthTracker(HealthScoreConfig()),
            impact=ImpactClassifier(ImpactClassifierConfig()),
            drift=DriftDetector(DriftDetectorConfig()),
        )
        decision = await gate.evaluate(
            strategy_id="s1", symbol="BTC/USDT"
        )
        assert decision.reason == "governance_disabled"


class TestMaturityGating:
    """Maturity level blocks."""

    @pytest.mark.asyncio
    async def test_l0_blocked(self, governance_gate):
        governance_gate.maturity.set_level("s1", MaturityLevel.L0_SHADOW)
        decision = await governance_gate.evaluate(
            strategy_id="s1", symbol="BTC/USDT"
        )
        assert decision.action == GovernanceAction.BLOCK
        assert "maturity_level" in decision.reason

    @pytest.mark.asyncio
    async def test_l1_blocked(self, governance_gate):
        # L1 is the default
        decision = await governance_gate.evaluate(
            strategy_id="s1", symbol="BTC/USDT"
        )
        assert decision.action == GovernanceAction.BLOCK

    @pytest.mark.asyncio
    async def test_l2_allowed(self, governance_gate):
        governance_gate.maturity.set_level("s1", MaturityLevel.L2_GATED)
        decision = await governance_gate.evaluate(
            strategy_id="s1", symbol="BTC/USDT"
        )
        assert decision.action in (
            GovernanceAction.ALLOW,
            GovernanceAction.REDUCE_SIZE,
        )

    @pytest.mark.asyncio
    async def test_l4_full_sizing(self, governance_gate):
        governance_gate.maturity.set_level("s1", MaturityLevel.L4_AUTONOMOUS)
        decision = await governance_gate.evaluate(
            strategy_id="s1",
            symbol="BTC/USDT",
            notional_usd=1_000,
            portfolio_pct=0.01,
        )
        assert decision.action == GovernanceAction.ALLOW
        assert decision.sizing_multiplier == 1.0


class TestHealthScoreIntegration:
    """Health score affects sizing."""

    @pytest.mark.asyncio
    async def test_degraded_health_reduces_sizing(self, governance_gate):
        governance_gate.maturity.set_level("s1", MaturityLevel.L4_AUTONOMOUS)

        # Accumulate losses
        for _ in range(8):
            governance_gate.record_trade_outcome("s1", won=False, r_multiple=-1.0)

        decision = await governance_gate.evaluate(
            strategy_id="s1",
            symbol="BTC/USDT",
            notional_usd=1_000,
            portfolio_pct=0.01,
        )
        assert decision.sizing_multiplier < 1.0

    @pytest.mark.asyncio
    async def test_pristine_health_full_sizing(self, governance_gate):
        governance_gate.maturity.set_level("s1", MaturityLevel.L4_AUTONOMOUS)
        decision = await governance_gate.evaluate(
            strategy_id="s1",
            symbol="BTC/USDT",
            notional_usd=1_000,
            portfolio_pct=0.01,
        )
        assert decision.sizing_multiplier == 1.0


class TestDriftIntegration:
    """Drift detection affects decisions."""

    @pytest.mark.asyncio
    async def test_severe_drift_pauses(self, governance_gate):
        governance_gate.maturity.set_level("s1", MaturityLevel.L4_AUTONOMOUS)
        governance_gate.drift.set_baseline("s1", {"win_rate": 0.55})
        governance_gate.drift.update_live_metric("s1", "win_rate", 0.20)

        decision = await governance_gate.evaluate(
            strategy_id="s1",
            symbol="BTC/USDT",
            notional_usd=1_000,
        )
        assert decision.action == GovernanceAction.PAUSE

    @pytest.mark.asyncio
    async def test_moderate_drift_reduces_sizing(self, governance_gate):
        governance_gate.maturity.set_level("s1", MaturityLevel.L4_AUTONOMOUS)
        governance_gate.drift.set_baseline("s1", {"win_rate": 0.55})
        governance_gate.drift.update_live_metric("s1", "win_rate", 0.35)

        decision = await governance_gate.evaluate(
            strategy_id="s1",
            symbol="BTC/USDT",
            notional_usd=1_000,
        )
        assert decision.action == GovernanceAction.REDUCE_SIZE
        assert decision.sizing_multiplier < 1.0


class TestImpactIntegration:
    """Impact classification."""

    @pytest.mark.asyncio
    async def test_critical_impact_at_l2_blocked(self, governance_gate):
        governance_gate.maturity.set_level("s1", MaturityLevel.L2_GATED)
        decision = await governance_gate.evaluate(
            strategy_id="s1",
            symbol="BTC/USDT",
            notional_usd=500_000,
            portfolio_pct=0.50,
            leverage=10,
            existing_positions=10,
        )
        assert decision.action == GovernanceAction.BLOCK
        assert "critical_impact" in decision.reason

    @pytest.mark.asyncio
    async def test_impact_tier_in_decision(self, governance_gate):
        governance_gate.maturity.set_level("s1", MaturityLevel.L4_AUTONOMOUS)
        decision = await governance_gate.evaluate(
            strategy_id="s1",
            symbol="BTC/USDT",
            notional_usd=1_000,
        )
        assert decision.impact_tier is not None


class TestTokenIntegration:
    """Token requirement integration."""

    @pytest.mark.asyncio
    async def test_blocked_without_token_when_required(self):
        cfg = GovernanceConfig(
            enabled=True,
            execution_tokens=ExecutionTokenConfig(require_tokens=True),
        )
        token_mgr = TokenManager(cfg.execution_tokens)
        gate = GovernanceGate(
            config=cfg,
            maturity=MaturityManager(MaturityConfig(default_level="L4_autonomous")),
            health=HealthTracker(HealthScoreConfig()),
            impact=ImpactClassifier(ImpactClassifierConfig()),
            drift=DriftDetector(DriftDetectorConfig()),
            tokens=token_mgr,
        )
        decision = await gate.evaluate(
            strategy_id="s1", symbol="BTC/USDT"
        )
        assert decision.action == GovernanceAction.BLOCK
        assert "token" in decision.reason

    @pytest.mark.asyncio
    async def test_allowed_with_valid_token(self):
        cfg = GovernanceConfig(
            enabled=True,
            execution_tokens=ExecutionTokenConfig(require_tokens=True),
        )
        token_mgr = TokenManager(cfg.execution_tokens)
        gate = GovernanceGate(
            config=cfg,
            maturity=MaturityManager(MaturityConfig(default_level="L4_autonomous")),
            health=HealthTracker(HealthScoreConfig()),
            impact=ImpactClassifier(ImpactClassifierConfig()),
            drift=DriftDetector(DriftDetectorConfig()),
            tokens=token_mgr,
        )
        token = token_mgr.issue("s1", scope="order:BTC/USDT")
        decision = await gate.evaluate(
            strategy_id="s1",
            symbol="BTC/USDT",
            token_id=token.token_id,
        )
        assert decision.action == GovernanceAction.ALLOW
        assert token.used is True


class TestSizingComposition:
    """Multiplicative sizing composition."""

    @pytest.mark.asyncio
    async def test_maturity_caps_sizing(self, governance_gate):
        """L2 caps at 10%, L3 at 25%."""
        governance_gate.maturity.set_level("s1", MaturityLevel.L2_GATED)
        decision = await governance_gate.evaluate(
            strategy_id="s1",
            symbol="BTC/USDT",
            notional_usd=1_000,
            portfolio_pct=0.01,
        )
        assert decision.sizing_multiplier <= 0.10

    @pytest.mark.asyncio
    async def test_l3_sizing_cap(self, governance_gate):
        governance_gate.maturity.set_level("s1", MaturityLevel.L3_CONSTRAINED)
        decision = await governance_gate.evaluate(
            strategy_id="s1",
            symbol="BTC/USDT",
            notional_usd=1_000,
            portfolio_pct=0.01,
        )
        assert decision.sizing_multiplier <= 0.25

    @pytest.mark.asyncio
    async def test_decision_has_trace_id(self, governance_gate):
        governance_gate.maturity.set_level("s1", MaturityLevel.L4_AUTONOMOUS)
        decision = await governance_gate.evaluate(
            strategy_id="s1",
            symbol="BTC/USDT",
            trace_id="trace-abc",
        )
        assert decision.trace_id == "trace-abc"

    @pytest.mark.asyncio
    async def test_component_accessors(self, governance_gate):
        assert governance_gate.maturity is not None
        assert governance_gate.health is not None
        assert governance_gate.impact is not None
        assert governance_gate.drift is not None
