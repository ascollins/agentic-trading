"""Tests for FX-specific policy rules."""

from __future__ import annotations

import pytest

from agentic_trading.core.config import FXRiskConfig
from agentic_trading.core.enums import GovernanceAction
from agentic_trading.policy.default_policies import build_fx_policies
from agentic_trading.policy.engine import PolicyEngine
from agentic_trading.policy.models import PolicyMode


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_fx_engine(
    config: FXRiskConfig | None = None,
    mode: PolicyMode = PolicyMode.ENFORCED,
) -> PolicyEngine:
    """Build a PolicyEngine with FX rules registered."""
    engine = PolicyEngine()
    ps = build_fx_policies(config, mode=mode)
    engine.register(ps)
    return engine


def _base_context(**overrides) -> dict:
    """Build a passing FX context dict."""
    ctx = {
        "asset_class": "fx",
        "symbol": "EUR/USD",
        "projected_leverage": 10.0,
        "order_notional_usd": 100_000.0,
        "current_spread_pips": 1.5,
        "session_open": True,
        "fx_market_open": True,
        "expected_slippage_pips": 0.5,
        "daily_rollover_cost_usd": 50.0,
    }
    ctx.update(overrides)
    return ctx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFXPoliciesAllPass:
    def test_all_rules_pass(self):
        engine = _make_fx_engine()
        result = engine.evaluate("fx_risk", _base_context())
        assert result.all_passed is True
        assert result.action == GovernanceAction.ALLOW

    def test_sizing_multiplier_is_one(self):
        engine = _make_fx_engine()
        result = engine.evaluate("fx_risk", _base_context())
        assert result.sizing_multiplier == 1.0


class TestFXMaxLeverage:
    def test_leverage_exceeded_blocks(self):
        engine = _make_fx_engine()
        result = engine.evaluate(
            "fx_risk", _base_context(projected_leverage=60.0)
        )
        assert result.all_passed is False
        assert any(r.rule_id == "fx_max_leverage" for r in result.failed_rules)

    def test_leverage_at_limit_passes(self):
        engine = _make_fx_engine()
        result = engine.evaluate(
            "fx_risk", _base_context(projected_leverage=50.0)
        )
        failed_ids = {r.rule_id for r in result.failed_rules}
        assert "fx_max_leverage" not in failed_ids


class TestFXMaxNotional:
    def test_notional_exceeded_blocks(self):
        engine = _make_fx_engine()
        result = engine.evaluate(
            "fx_risk", _base_context(order_notional_usd=2_000_000.0)
        )
        assert result.all_passed is False
        assert any(r.rule_id == "fx_max_notional" for r in result.failed_rules)


class TestFXMaxSpread:
    def test_spread_exceeded_blocks(self):
        engine = _make_fx_engine()
        result = engine.evaluate(
            "fx_risk", _base_context(current_spread_pips=7.2)
        )
        assert result.all_passed is False
        assert any(r.rule_id == "fx_max_spread" for r in result.failed_rules)

    def test_spread_at_limit_passes(self):
        engine = _make_fx_engine()
        result = engine.evaluate(
            "fx_risk", _base_context(current_spread_pips=5.0)
        )
        failed_ids = {r.rule_id for r in result.failed_rules}
        assert "fx_max_spread" not in failed_ids


class TestFXSessionGuard:
    def test_session_closed_blocks(self):
        engine = _make_fx_engine()
        result = engine.evaluate(
            "fx_risk", _base_context(session_open=False)
        )
        assert result.all_passed is False
        assert any(
            r.rule_id == "fx_session_guard" for r in result.failed_rules
        )


class TestFXWeekendGuard:
    def test_weekend_closed_blocks(self):
        engine = _make_fx_engine()
        result = engine.evaluate(
            "fx_risk", _base_context(fx_market_open=False)
        )
        assert result.all_passed is False
        assert any(
            r.rule_id == "fx_weekend_guard" for r in result.failed_rules
        )

    def test_weekend_guard_disabled(self):
        config = FXRiskConfig(block_weekend_orders=False)
        engine = _make_fx_engine(config)
        result = engine.evaluate(
            "fx_risk", _base_context(fx_market_open=False)
        )
        # Weekend guard is disabled, so it shouldn't fail
        failed_ids = {r.rule_id for r in result.failed_rules}
        assert "fx_weekend_guard" not in failed_ids


class TestFXSlippageLimit:
    def test_slippage_exceeded_blocks(self):
        engine = _make_fx_engine()
        result = engine.evaluate(
            "fx_risk", _base_context(expected_slippage_pips=5.0)
        )
        assert result.all_passed is False
        assert any(
            r.rule_id == "fx_slippage_limit" for r in result.failed_rules
        )


class TestFXPairWhitelist:
    def test_non_whitelisted_pair_blocks(self):
        engine = _make_fx_engine()
        result = engine.evaluate(
            "fx_risk", _base_context(symbol="TRY/ZAR")
        )
        assert result.all_passed is False
        assert any(
            r.rule_id == "fx_pair_whitelist" for r in result.failed_rules
        )

    def test_whitelisted_pair_passes(self):
        engine = _make_fx_engine()
        result = engine.evaluate(
            "fx_risk", _base_context(symbol="GBP/USD")
        )
        failed_ids = {r.rule_id for r in result.failed_rules}
        assert "fx_pair_whitelist" not in failed_ids

    def test_whitelist_disabled(self):
        config = FXRiskConfig(major_pairs_only=False)
        engine = _make_fx_engine(config)
        result = engine.evaluate(
            "fx_risk", _base_context(symbol="TRY/ZAR")
        )
        failed_ids = {r.rule_id for r in result.failed_rules}
        assert "fx_pair_whitelist" not in failed_ids


class TestFXRolloverLimit:
    def test_rollover_exceeded_reduces_size(self):
        engine = _make_fx_engine()
        result = engine.evaluate(
            "fx_risk", _base_context(daily_rollover_cost_usd=600.0)
        )
        # rollover_limit action is REDUCE_SIZE, not BLOCK
        assert any(
            r.rule_id == "fx_rollover_limit" for r in result.failed_rules
        )
        # The aggregate action should be REDUCE_SIZE (not BLOCK)
        # since all other rules pass
        assert result.sizing_multiplier < 1.0


class TestFXShadowMode:
    def test_shadow_mode_logs_but_allows(self):
        engine = _make_fx_engine(mode=PolicyMode.SHADOW)
        result = engine.evaluate(
            "fx_risk",
            _base_context(
                projected_leverage=100.0,
                current_spread_pips=20.0,
            ),
        )
        # Shadow mode: violations logged but action is ALLOW
        assert result.all_passed is True
        assert len(result.shadow_violations) > 0


class TestFXCustomConfig:
    def test_custom_leverage_limit(self):
        config = FXRiskConfig(max_leverage=20)
        engine = _make_fx_engine(config)
        result = engine.evaluate(
            "fx_risk", _base_context(projected_leverage=25.0)
        )
        assert result.all_passed is False

    def test_custom_spread_limit(self):
        config = FXRiskConfig(max_spread_pips=2.0)
        engine = _make_fx_engine(config)
        result = engine.evaluate(
            "fx_risk", _base_context(current_spread_pips=3.0)
        )
        assert result.all_passed is False


class TestBuildFxPolicies:
    def test_policy_set_id(self):
        ps = build_fx_policies()
        assert ps.set_id == "fx_risk"

    def test_eight_rules(self):
        ps = build_fx_policies()
        assert len(ps.rules) == 8

    def test_rule_ids(self):
        ps = build_fx_policies()
        ids = {r.rule_id for r in ps.rules}
        expected = {
            "fx_max_leverage",
            "fx_max_notional",
            "fx_max_spread",
            "fx_session_guard",
            "fx_weekend_guard",
            "fx_slippage_limit",
            "fx_pair_whitelist",
            "fx_rollover_limit",
        }
        assert ids == expected
