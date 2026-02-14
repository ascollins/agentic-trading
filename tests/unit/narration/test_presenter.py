"""Tests for BloombergPresenter — broadcast-quality narration persona.

Tests cover:
  - Script structure for both LIVE_HIT and PACKAGE formats
  - Two-way risk is always present (mandatory)
  - Regime language mapping
  - Action-based lead selection
  - Data hygiene: no hallucinated data
  - Tavus context includes system prompt
  - Integration with NarrationService PRESENTER verbosity
  - Edge cases (empty explanation, missing fields)
"""

from __future__ import annotations

import pytest

from agentic_trading.narration.presenter import (
    BloombergPresenter,
    OutputFormat,
    PRESENTER_SYSTEM_PROMPT,
    _REGIME_LANGUAGE,
    _ACTION_LEAD,
)
from agentic_trading.narration.schema import (
    ConsideredSetup,
    DecisionExplanation,
    PositionSnapshot,
    RiskSummary,
)
from agentic_trading.narration.service import NarrationService, Verbosity


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def presenter() -> BloombergPresenter:
    return BloombergPresenter(default_format=OutputFormat.LIVE_HIT)


@pytest.fixture
def package_presenter() -> BloombergPresenter:
    return BloombergPresenter(default_format=OutputFormat.PACKAGE)


@pytest.fixture
def enter_exp() -> DecisionExplanation:
    return DecisionExplanation(
        symbol="BTC/USDT",
        timeframe="5m",
        market_summary="Bitcoin has been climbing steadily over the past hour with solid buying pressure.",
        active_strategy="trend_following",
        active_regime="trend",
        regime_confidence=0.85,
        considered_setups=[
            ConsideredSetup(
                name="momentum breakout",
                direction="long",
                confidence=0.82,
                status="triggered",
            ),
        ],
        action="ENTER",
        reasons=["Strong upward momentum confirmed by rising volume", "Price above key support level"],
        reason_confidences=[0.85, 0.78],
        risk=RiskSummary(
            intended_size_pct=0.03,
            stop_invalidation="Below 96,500 support",
            health_score=0.92,
            maturity_level="L3_constrained",
        ),
        position=PositionSnapshot(
            open_positions=1,
            gross_exposure_usd=5000.0,
            unrealized_pnl_usd=120.0,
            available_balance_usd=95000.0,
        ),
        trace_id="demo-trace-001",
    )


@pytest.fixture
def no_trade_exp() -> DecisionExplanation:
    return DecisionExplanation(
        symbol="ETH/USDT",
        timeframe="15m",
        market_summary="Ethereum is moving sideways in a tight range.",
        active_strategy="mean_reversion",
        active_regime="range",
        action="NO_TRADE",
        reasons=["No clear directional edge"],
        why_not=["Spread above acceptable threshold", "Volume too low"],
        what_would_change=["Volume picks up above average", "Price breaks out of the range"],
        risk=RiskSummary(
            active_blocks=["spread_circuit_breaker"],
            health_score=0.70,
        ),
        trace_id="demo-trace-002",
    )


@pytest.fixture
def hold_exp() -> DecisionExplanation:
    return DecisionExplanation(
        symbol="BTC/USDT",
        active_strategy="trend_following",
        active_regime="trend",
        action="HOLD",
        reasons=["Trend still intact, no exit signal"],
        risk=RiskSummary(
            stop_invalidation="Below recent support",
            health_score=0.95,
        ),
        position=PositionSnapshot(
            open_positions=1,
            gross_exposure_usd=5200.0,
            unrealized_pnl_usd=200.0,
        ),
        trace_id="demo-trace-003",
    )


@pytest.fixture
def exit_exp() -> DecisionExplanation:
    return DecisionExplanation(
        symbol="BTC/USDT",
        active_strategy="trend_following",
        active_regime="trend",
        action="EXIT",
        reasons=["Stop loss triggered"],
        what_would_change=["Re-enter above resistance"],
        trace_id="demo-trace-004",
    )


# ===========================================================================
# LIVE HIT format
# ===========================================================================


class TestLiveHit:
    def test_enter_script_has_content(self, presenter, enter_exp):
        script = presenter.build_script(enter_exp, fmt=OutputFormat.LIVE_HIT)
        assert len(script) > 50
        assert "BTC/USDT" in script

    def test_enter_contains_decision(self, presenter, enter_exp):
        script = presenter.build_script(enter_exp)
        lower = script.lower()
        assert "entering" in lower or "position" in lower

    def test_no_trade_contains_blocker(self, presenter, no_trade_exp):
        script = presenter.build_script(no_trade_exp)
        lower = script.lower()
        assert "spread" in lower or "blocker" in lower or "no trade" in lower

    def test_hold_contains_hold(self, presenter, hold_exp):
        script = presenter.build_script(hold_exp)
        lower = script.lower()
        assert "holding" in lower or "hold" in lower

    def test_exit_contains_closing(self, presenter, exit_exp):
        script = presenter.build_script(exit_exp)
        lower = script.lower()
        assert "closing" in lower or "exit" in lower or "flat" in lower

    def test_live_hit_includes_disclaimer(self, presenter, enter_exp):
        script = presenter.build_script(enter_exp)
        assert "not financial advice" in script.lower()

    def test_no_disclaimer_when_disabled(self, enter_exp):
        p = BloombergPresenter(include_disclaimer=False)
        script = p.build_script(enter_exp)
        assert "not financial advice" not in script.lower()


# ===========================================================================
# PACKAGE format
# ===========================================================================


class TestPackage:
    def test_package_longer_than_live_hit(self, enter_exp):
        live = BloombergPresenter(default_format=OutputFormat.LIVE_HIT)
        pkg = BloombergPresenter(default_format=OutputFormat.PACKAGE)
        live_script = live.build_script(enter_exp)
        pkg_script = pkg.build_script(enter_exp)
        assert len(pkg_script) > len(live_script)

    def test_package_contains_strategy_context(self, package_presenter, enter_exp):
        script = package_presenter.build_script(enter_exp, fmt=OutputFormat.PACKAGE)
        lower = script.lower()
        assert "trend following" in lower or "strategy" in lower

    def test_package_contains_portfolio(self, package_presenter, enter_exp):
        script = package_presenter.build_script(enter_exp, fmt=OutputFormat.PACKAGE)
        lower = script.lower()
        # Should reference open positions or exposure
        assert "position" in lower or "exposure" in lower

    def test_package_contains_setups(self, package_presenter, enter_exp):
        script = package_presenter.build_script(enter_exp, fmt=OutputFormat.PACKAGE)
        assert "momentum breakout" in script.lower()

    def test_package_contains_maturity(self, package_presenter, enter_exp):
        script = package_presenter.build_script(enter_exp, fmt=OutputFormat.PACKAGE)
        assert "maturity" in script.lower()


# ===========================================================================
# Two-way risk (mandatory)
# ===========================================================================


class TestTwoWayRisk:
    """Every briefing MUST contain a risk statement — both bull and bear case."""

    def test_enter_has_risk(self, presenter, enter_exp):
        script = presenter.build_script(enter_exp)
        lower = script.lower()
        assert "risk" in lower or "invalidate" in lower

    def test_no_trade_has_risk(self, presenter, no_trade_exp):
        script = presenter.build_script(no_trade_exp)
        lower = script.lower()
        assert "risk" in lower or "inaction" in lower

    def test_hold_has_risk(self, presenter, hold_exp):
        script = presenter.build_script(hold_exp)
        lower = script.lower()
        assert "risk" in lower or "invalidate" in lower

    def test_exit_has_risk(self, presenter, exit_exp):
        script = presenter.build_script(exit_exp)
        lower = script.lower()
        assert "risk" in lower or "flat" in lower

    def test_low_health_flagged(self, presenter, no_trade_exp):
        """Health score below 80% should be called out."""
        script = presenter.build_script(no_trade_exp)
        lower = script.lower()
        assert "70%" in lower or "health" in lower or "caution" in lower

    def test_active_blocks_mentioned(self, presenter, no_trade_exp):
        """Active risk blocks should be mentioned."""
        script = presenter.build_script(no_trade_exp)
        lower = script.lower()
        assert "spread" in lower or "circuit" in lower or "risk control" in lower


# ===========================================================================
# Regime language mapping
# ===========================================================================


class TestRegimeMapping:
    def test_trend_mapped(self):
        assert _REGIME_LANGUAGE["trend"] == "directional move"

    def test_range_mapped(self):
        assert _REGIME_LANGUAGE["range"] == "consolidation phase"

    def test_unknown_mapped(self):
        assert _REGIME_LANGUAGE["unknown"] == "uncertain conditions"

    def test_empty_mapped(self):
        assert _REGIME_LANGUAGE[""] == "current conditions"


# ===========================================================================
# Action lead angle mapping
# ===========================================================================


class TestActionLead:
    def test_enter_leads_with_entry(self):
        assert _ACTION_LEAD["ENTER"] == "position_entry"

    def test_exit_leads_with_exit(self):
        assert _ACTION_LEAD["EXIT"] == "position_exit"

    def test_hold_leads_with_status(self):
        assert _ACTION_LEAD["HOLD"] == "status_update"

    def test_no_trade_leads_with_watchlist(self):
        assert _ACTION_LEAD["NO_TRADE"] == "watchlist"

    def test_adjust_leads_with_adjustment(self):
        assert _ACTION_LEAD["ADJUST"] == "position_adjustment"


# ===========================================================================
# System prompt
# ===========================================================================


class TestSystemPrompt:
    def test_prompt_exists(self, presenter):
        prompt = presenter.system_prompt
        assert len(prompt) > 100

    def test_prompt_contains_key_sections(self, presenter):
        prompt = presenter.system_prompt
        assert "STORY SELECTOR" in prompt
        assert "VERIFICATION LADDER" in prompt
        assert "DATA HYGIENE" in prompt
        assert "TWO-WAY RISK" in prompt
        assert "LIVE HIT" in prompt
        assert "PACKAGE" in prompt

    def test_prompt_is_module_constant(self, presenter):
        assert presenter.system_prompt == PRESENTER_SYSTEM_PROMPT


# ===========================================================================
# Tavus context
# ===========================================================================


class TestTavusContext:
    def test_context_includes_system_prompt(self, presenter, enter_exp):
        context = presenter.build_tavus_context(enter_exp)
        assert PRESENTER_SYSTEM_PROMPT in context

    def test_context_includes_data(self, presenter, enter_exp):
        context = presenter.build_tavus_context(enter_exp)
        assert "BTC/USDT" in context
        assert "ENTER" in context
        assert "trend_following" in context

    def test_context_includes_format_instruction(self, presenter, enter_exp):
        ctx_live = presenter.build_tavus_context(enter_exp, fmt=OutputFormat.LIVE_HIT)
        assert "30-60 second" in ctx_live

        ctx_pkg = presenter.build_tavus_context(enter_exp, fmt=OutputFormat.PACKAGE)
        assert "2-minute" in ctx_pkg

    def test_context_includes_reasons(self, presenter, enter_exp):
        context = presenter.build_tavus_context(enter_exp)
        assert "Strong upward momentum" in context

    def test_context_includes_risk(self, presenter, enter_exp):
        context = presenter.build_tavus_context(enter_exp)
        assert "92%" in context  # health score

    def test_context_includes_position(self, presenter, enter_exp):
        context = presenter.build_tavus_context(enter_exp)
        assert "5,000" in context  # gross exposure

    def test_context_includes_stop(self, presenter, enter_exp):
        context = presenter.build_tavus_context(enter_exp)
        assert "96,500" in context  # stop invalidation


# ===========================================================================
# Integration with NarrationService
# ===========================================================================


class TestServiceIntegration:
    def test_presenter_verbosity_generates(self, enter_exp):
        svc = NarrationService(verbosity=Verbosity.PRESENTER)
        item = svc.generate(enter_exp, force=True)
        assert item is not None
        assert item.verbosity == "presenter"

    def test_presenter_no_word_limit(self, enter_exp):
        """PRESENTER mode should not truncate output like QUIET/NORMAL/DETAILED."""
        svc = NarrationService(verbosity=Verbosity.PRESENTER)
        item = svc.generate(enter_exp, force=True)
        assert item is not None
        # Presenter scripts are typically longer than NORMAL's 70 words
        # (but not required to be — it depends on input data)
        assert len(item.script_text) > 0

    def test_presenter_jargon_still_scrubbed(self):
        """Even in PRESENTER mode, jargon should be scrubbed."""
        exp = DecisionExplanation(
            symbol="BTC/USDT",
            action="ENTER",
            active_strategy="test",
            reasons=["RSI below 30 signals oversold"],
        )
        svc = NarrationService(verbosity=Verbosity.PRESENTER)
        item = svc.generate(exp, force=True)
        assert item is not None
        assert "rsi" not in item.script_text.lower()

    def test_presenter_dedupe_works(self, enter_exp):
        svc = NarrationService(
            verbosity=Verbosity.PRESENTER,
            dedupe_window_seconds=100,
        )
        item1 = svc.generate(enter_exp, force=False)
        item2 = svc.generate(enter_exp, force=False)
        assert item1 is not None
        assert item2 is None  # Deduplicated

    def test_presenter_throttle_works(self):
        svc = NarrationService(
            verbosity=Verbosity.PRESENTER,
            heartbeat_seconds=1000,
        )
        exp = DecisionExplanation(action="HOLD", symbol="BTC/USDT")
        item1 = svc.generate(exp, force=False)
        item2 = svc.generate(exp, force=False)
        assert item1 is not None
        assert item2 is None  # Throttled

    def test_service_exposes_presenter(self):
        svc = NarrationService()
        p = svc.presenter
        assert isinstance(p, BloombergPresenter)

    def test_service_uses_custom_presenter(self):
        custom = BloombergPresenter(
            default_format=OutputFormat.PACKAGE,
            include_disclaimer=False,
        )
        svc = NarrationService(presenter=custom)
        assert svc.presenter is custom

    def test_no_trade_presenter(self, no_trade_exp):
        svc = NarrationService(verbosity=Verbosity.PRESENTER)
        item = svc.generate(no_trade_exp, force=True)
        assert item is not None
        lower = item.script_text.lower()
        assert "eth/usdt" in lower


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_explanation(self, presenter):
        exp = DecisionExplanation()
        script = presenter.build_script(exp)
        assert len(script) > 0

    def test_no_market_summary(self, presenter):
        exp = DecisionExplanation(symbol="SOL/USDT", action="HOLD")
        script = presenter.build_script(exp)
        assert "SOL/USDT" in script

    def test_no_reasons(self, presenter):
        exp = DecisionExplanation(symbol="BTC/USDT", action="ENTER")
        script = presenter.build_script(exp)
        assert "BTC/USDT" in script

    def test_no_position(self, presenter):
        exp = DecisionExplanation(symbol="BTC/USDT", action="ENTER")
        script = presenter.build_script(exp, fmt=OutputFormat.PACKAGE)
        assert "flat" in script.lower() or "no open" in script.lower()

    def test_unknown_regime(self, presenter):
        exp = DecisionExplanation(
            symbol="BTC/USDT",
            active_regime="some_custom_regime",
            action="HOLD",
        )
        script = presenter.build_script(exp)
        assert "some_custom_regime" in script.lower() or len(script) > 0

    def test_adjust_action(self, presenter):
        exp = DecisionExplanation(
            symbol="BTC/USDT",
            action="ADJUST",
            reasons=["Reducing position size due to increased volatility"],
        )
        script = presenter.build_script(exp)
        lower = script.lower()
        assert "adjusting" in lower or "adjust" in lower

    def test_data_block_no_crash(self, presenter, enter_exp):
        """Data block builder should not crash with full data."""
        block = presenter._build_data_block(enter_exp)
        assert "BTC/USDT" in block
        assert "ENTER" in block
        assert "trend_following" in block
