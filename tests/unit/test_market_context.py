"""Tests for market context / macro regime assessment."""

import pytest

from agentic_trading.analysis.market_context import (
    MACRO_REGIMES,
    MarketContext,
    assess_macro_regime,
)


class TestAssessMacroRegime:
    def test_risk_on_easing(self):
        ctx = assess_macro_regime({
            "dxy_trend": "down",
            "yields_10y_trend": "down",
            "sp500_vs_200sma": 1.05,
            "stablecoin_supply_trend": "up",
            "funding_rates_avg": 0.005,
        })
        assert ctx.risk_regime == "risk_on"
        assert ctx.liquidity_conditions == "easing"
        assert ctx.regime_key == "risk_on_easing"
        assert "bullish" in ctx.impact_on_crypto.lower()

    def test_risk_off_tightening(self):
        ctx = assess_macro_regime({
            "dxy_trend": "up",
            "yields_10y_trend": "up",
            "sp500_vs_200sma": 0.95,
            "stablecoin_supply_trend": "down",
        })
        assert ctx.risk_regime == "risk_off"
        assert ctx.dollar_trend == "strengthening"
        assert ctx.regime_key == "risk_off_tightening"

    def test_neutral_ranging(self):
        ctx = assess_macro_regime({
            "dxy_trend": "flat",
            "yields_10y_trend": "flat",
            "sp500_vs_200sma": 1.0,
        })
        assert ctx.risk_regime == "neutral"
        assert ctx.regime_key == "neutral_ranging"

    def test_empty_inputs_defaults_neutral(self):
        ctx = assess_macro_regime({})
        assert ctx.risk_regime == "neutral"
        assert ctx.dollar_trend == "neutral"
        assert ctx.yield_environment == "neutral"
        assert ctx.regime_key == "neutral_ranging"

    def test_funding_overheated(self):
        ctx = assess_macro_regime({
            "sp500_vs_200sma": 1.05,
            "stablecoin_supply_trend": "up",
            "funding_rates_avg": 0.02,
        })
        assert ctx.crypto_specific["funding_sentiment"] == "overheated"

    def test_funding_fearful(self):
        ctx = assess_macro_regime({
            "funding_rates_avg": -0.01,
        })
        assert ctx.crypto_specific["funding_sentiment"] == "fearful"

    def test_btc_dominance_tracked(self):
        ctx = assess_macro_regime({
            "btc_dominance": 60.0,
        })
        assert ctx.crypto_specific["btc_dominance"] == 60.0
        assert ctx.crypto_specific["rotation"] == "btc_dominant"

    def test_alt_season_detection(self):
        ctx = assess_macro_regime({
            "btc_dominance": 40.0,
        })
        assert ctx.crypto_specific["rotation"] == "alt_season"

    def test_risk_off_crisis(self):
        ctx = assess_macro_regime({
            "dxy_trend": "flat",  # Not strengthening
            "sp500_vs_200sma": 0.90,
        })
        assert ctx.risk_regime == "risk_off"
        assert ctx.regime_key == "risk_off_crisis"

    def test_macro_regimes_dict_complete(self):
        """Verify all expected regime keys exist."""
        expected = [
            "risk_on_easing", "risk_on_neutral",
            "risk_off_tightening", "risk_off_crisis",
            "neutral_ranging",
        ]
        for key in expected:
            assert key in MACRO_REGIMES
            assert "description" in MACRO_REGIMES[key]
            assert "crypto_impact" in MACRO_REGIMES[key]
            assert "positioning" in MACRO_REGIMES[key]
