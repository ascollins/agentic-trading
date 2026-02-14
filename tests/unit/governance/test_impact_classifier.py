"""Tests for governance.impact_classifier — Trade Impact Scorer."""

import pytest

from agentic_trading.core.config import ImpactClassifierConfig
from agentic_trading.core.enums import ImpactTier
from agentic_trading.governance.impact_classifier import ImpactClassifier


class TestImpactClassification:
    """Impact tier assignment based on order properties."""

    def test_small_order_low_impact(self, impact_classifier):
        tier = impact_classifier.classify(
            symbol="BTC/USDT",
            notional_usd=1_000,
            portfolio_pct=0.01,
            is_reduce_only=True,
            leverage=1,
            existing_positions=0,
        )
        assert tier == ImpactTier.LOW

    def test_large_notional_high_impact(self, impact_classifier):
        tier = impact_classifier.classify(
            symbol="BTC/USDT",
            notional_usd=100_000,
            portfolio_pct=0.05,
            is_reduce_only=False,
            leverage=1,
            existing_positions=1,
        )
        assert tier in (ImpactTier.HIGH, ImpactTier.MEDIUM)

    def test_critical_notional(self, impact_classifier):
        tier = impact_classifier.classify(
            symbol="BTC/USDT",
            notional_usd=300_000,
            portfolio_pct=0.30,
            is_reduce_only=False,
            leverage=5,
            existing_positions=10,
        )
        assert tier == ImpactTier.CRITICAL

    def test_reduce_only_lower_impact(self, impact_classifier):
        """Reduce-only orders should score lower than new positions."""
        tier_new = impact_classifier.classify(
            symbol="ETH/USDT",
            notional_usd=30_000,
            portfolio_pct=0.05,
            is_reduce_only=False,
            leverage=1,
            existing_positions=2,
        )
        tier_reduce = impact_classifier.classify(
            symbol="ETH/USDT",
            notional_usd=30_000,
            portfolio_pct=0.05,
            is_reduce_only=True,
            leverage=1,
            existing_positions=2,
        )
        # reduce-only should be same or lower tier
        tier_order = [ImpactTier.LOW, ImpactTier.MEDIUM, ImpactTier.HIGH, ImpactTier.CRITICAL]
        assert tier_order.index(tier_reduce) <= tier_order.index(tier_new)

    def test_high_leverage_increases_impact(self, impact_classifier):
        tier_low_lev = impact_classifier.classify(
            symbol="BTC/USDT",
            notional_usd=25_000,
            portfolio_pct=0.05,
            is_reduce_only=False,
            leverage=1,
            existing_positions=0,
        )
        tier_high_lev = impact_classifier.classify(
            symbol="BTC/USDT",
            notional_usd=25_000,
            portfolio_pct=0.05,
            is_reduce_only=False,
            leverage=10,
            existing_positions=0,
        )
        tier_order = [ImpactTier.LOW, ImpactTier.MEDIUM, ImpactTier.HIGH, ImpactTier.CRITICAL]
        assert tier_order.index(tier_high_lev) >= tier_order.index(tier_low_lev)

    def test_high_concentration_increases_impact(self, impact_classifier):
        tier_low = impact_classifier.classify(
            symbol="BTC/USDT",
            notional_usd=10_000,
            portfolio_pct=0.01,
            is_reduce_only=False,
            leverage=1,
            existing_positions=0,
        )
        tier_high = impact_classifier.classify(
            symbol="BTC/USDT",
            notional_usd=10_000,
            portfolio_pct=0.20,
            is_reduce_only=False,
            leverage=1,
            existing_positions=0,
        )
        tier_order = [ImpactTier.LOW, ImpactTier.MEDIUM, ImpactTier.HIGH, ImpactTier.CRITICAL]
        assert tier_order.index(tier_high) >= tier_order.index(tier_low)


class TestCustomConfig:
    """Custom config thresholds."""

    def test_custom_notional_thresholds(self):
        cfg = ImpactClassifierConfig(
            high_notional_usd=10_000,
            critical_notional_usd=50_000,
        )
        clf = ImpactClassifier(cfg)
        tier = clf.classify(
            symbol="BTC/USDT",
            notional_usd=60_000,
            portfolio_pct=0.20,
            is_reduce_only=False,
            leverage=5,
            existing_positions=5,
        )
        assert tier == ImpactTier.CRITICAL

    def test_custom_concentration_threshold(self):
        cfg = ImpactClassifierConfig(concentration_threshold_pct=0.05)
        clf = ImpactClassifier(cfg)
        tier = clf.classify(
            symbol="BTC/USDT",
            notional_usd=5_000,
            portfolio_pct=0.06,
            is_reduce_only=False,
            leverage=1,
            existing_positions=0,
        )
        # Should be at least MEDIUM due to concentration
        assert tier in (ImpactTier.MEDIUM, ImpactTier.HIGH, ImpactTier.CRITICAL)

    def test_zero_values_low_impact(self, impact_classifier):
        tier = impact_classifier.classify(
            symbol="BTC/USDT",
            notional_usd=0,
            portfolio_pct=0.0,
            is_reduce_only=True,
            leverage=1,
            existing_positions=0,
        )
        assert tier == ImpactTier.LOW

    def test_many_positions_increases_blast(self, impact_classifier):
        """More existing positions increases blast radius."""
        tier = impact_classifier.classify(
            symbol="BTC/USDT",
            notional_usd=40_000,
            portfolio_pct=0.05,
            is_reduce_only=False,
            leverage=3,
            existing_positions=8,
        )
        assert tier in (ImpactTier.MEDIUM, ImpactTier.HIGH, ImpactTier.CRITICAL)

    def test_medium_order_medium_impact(self, impact_classifier):
        tier = impact_classifier.classify(
            symbol="BTC/USDT",
            notional_usd=30_000,
            portfolio_pct=0.07,
            is_reduce_only=False,
            leverage=2,
            existing_positions=2,
        )
        assert tier in (ImpactTier.MEDIUM, ImpactTier.HIGH)
