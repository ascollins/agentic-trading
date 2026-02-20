"""Tests for the EfficacyAnalyzer.

Verifies each diagnostic phase:
  Phase 0: Data integrity checks
  Phase 1: Segment analysis + loss driver diagnosis
  Phase 2: Recommendation generation
"""

from __future__ import annotations

import pytest

from agentic_trading.backtester.results import TradeDetail
from agentic_trading.optimizer.efficacy import EfficacyAnalyzer
from agentic_trading.optimizer.efficacy_models import (
    DRIVER_PRIORITY,
    EfficacyReport,
    LossDriverCategory,
    SegmentAnalysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade(
    strategy_id: str = "bb_squeeze",
    symbol: str = "BTC/USDT",
    direction: str = "long",
    entry_price: float = 50000.0,
    exit_price: float = 50500.0,
    return_pct: float = 0.01,
    fee_paid: float = 10.0,
    mae_pct: float = -0.005,
    mfe_pct: float = 0.015,
    exit_reason: str = "signal",
    hold_seconds: float = 3600.0,
    qty: float = 1.0,
) -> TradeDetail:
    """Create a test trade."""
    return TradeDetail(
        strategy_id=strategy_id,
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_time="2026-01-01T00:00:00+00:00",
        exit_time="2026-01-01T01:00:00+00:00",
        qty=qty,
        return_pct=return_pct,
        gross_return_pct=return_pct,
        fee_paid=fee_paid,
        slippage_cost=0.0,
        stop_price=49000.0,
        exit_reason=exit_reason,
        hold_seconds=hold_seconds,
        mae_pct=mae_pct,
        mfe_pct=mfe_pct,
    )


def _make_mixed_trades(n_winners: int = 30, n_losers: int = 20) -> list[TradeDetail]:
    """Create a mix of winning and losing trades."""
    trades: list[TradeDetail] = []
    for _ in range(n_winners):
        trades.append(_make_trade(return_pct=0.01, mfe_pct=0.02, mae_pct=-0.003))
    for _ in range(n_losers):
        trades.append(_make_trade(return_pct=-0.008, mfe_pct=0.002, mae_pct=-0.01))
    return trades


# ---------------------------------------------------------------------------
# Phase 0: Data Integrity
# ---------------------------------------------------------------------------


class TestDataIntegrity:
    """Test Phase 0 data integrity checks."""

    def test_empty_trades_fails(self):
        analyzer = EfficacyAnalyzer(min_trades=10)
        report = analyzer.analyze([], strategy_id="test")
        assert not report.data_integrity.passed
        assert report.total_trades == 0
        assert "No trades" in report.data_integrity.issues[0]

    def test_valid_trades_pass(self):
        trades = _make_mixed_trades(30, 20)
        analyzer = EfficacyAnalyzer(min_trades=10)
        report = analyzer.analyze(trades, strategy_id="test")
        assert report.data_integrity.passed
        assert report.total_trades == 50

    def test_zero_price_flagged(self):
        trades = [_make_trade(entry_price=0.0)]
        analyzer = EfficacyAnalyzer(min_trades=1)
        report = analyzer.analyze(trades, strategy_id="test")
        assert not report.data_integrity.passed
        assert not report.data_integrity.has_prices

    def test_no_fees_flagged(self):
        trades = [_make_trade(fee_paid=0.0) for _ in range(10)]
        analyzer = EfficacyAnalyzer(min_trades=5)
        report = analyzer.analyze(trades, strategy_id="test")
        assert report.data_integrity.passed  # Still passes, just warns
        assert not report.data_integrity.has_fees

    def test_no_mae_mfe_flagged(self):
        trades = [_make_trade(mae_pct=0.0, mfe_pct=0.0) for _ in range(10)]
        analyzer = EfficacyAnalyzer(min_trades=5)
        report = analyzer.analyze(trades, strategy_id="test")
        assert report.data_integrity.passed
        assert not report.data_integrity.has_mae_mfe


# ---------------------------------------------------------------------------
# Phase 1: Segmentation
# ---------------------------------------------------------------------------


class TestSegmentation:
    """Test segment analysis computation."""

    def test_segments_by_symbol(self):
        trades = [
            _make_trade(symbol="BTC/USDT", return_pct=0.01),
            _make_trade(symbol="BTC/USDT", return_pct=-0.005),
            _make_trade(symbol="ETH/USDT", return_pct=0.02),
        ]
        analyzer = EfficacyAnalyzer(min_trades=1)
        report = analyzer.analyze(trades, strategy_id="test")

        assert "symbol:BTC/USDT" in report.segments
        assert "symbol:ETH/USDT" in report.segments

        btc_seg = report.segments["symbol:BTC/USDT"]
        assert btc_seg.trade_count == 2
        assert btc_seg.win_rate == 0.5

        eth_seg = report.segments["symbol:ETH/USDT"]
        assert eth_seg.trade_count == 1
        assert eth_seg.win_rate == 1.0

    def test_segments_by_direction(self):
        trades = [
            _make_trade(direction="long", return_pct=0.01),
            _make_trade(direction="short", return_pct=-0.005),
        ]
        analyzer = EfficacyAnalyzer(min_trades=1)
        report = analyzer.analyze(trades, strategy_id="test")

        assert "direction:long" in report.segments
        assert "direction:short" in report.segments

    def test_segments_by_exit_reason(self):
        trades = [
            _make_trade(exit_reason="signal", return_pct=0.01),
            _make_trade(exit_reason="stop_loss", return_pct=-0.02),
            _make_trade(exit_reason="stop_loss", return_pct=-0.015),
        ]
        analyzer = EfficacyAnalyzer(min_trades=1)
        report = analyzer.analyze(trades, strategy_id="test")

        assert "exit:signal" in report.segments
        assert "exit:stop_loss" in report.segments

        stop_seg = report.segments["exit:stop_loss"]
        assert stop_seg.trade_count == 2
        assert stop_seg.win_rate == 0.0

    def test_segment_management_efficiency(self):
        """Winners with high MFE but low capture should have low efficiency."""
        trades = [
            _make_trade(return_pct=0.005, mfe_pct=0.05),  # Captured only 10%
            _make_trade(return_pct=0.04, mfe_pct=0.05),   # Captured 80%
        ]
        analyzer = EfficacyAnalyzer(min_trades=1)
        report = analyzer.analyze(trades, strategy_id="test")

        overall = report.segments.get("direction:long")
        assert overall is not None
        # avg efficiency = (0.005/0.05 + 0.04/0.05) / 2 = (0.1 + 0.8) / 2 = 0.45
        assert 0.4 < overall.management_efficiency < 0.5


# ---------------------------------------------------------------------------
# Phase 1: Loss Drivers
# ---------------------------------------------------------------------------


class TestLossDriverDiagnosis:
    """Test loss driver diagnosis for each category."""

    def test_all_driver_categories_present(self):
        """Every loss driver category should appear in the analysis."""
        trades = _make_mixed_trades(25, 25)
        analyzer = EfficacyAnalyzer(min_trades=10)
        report = analyzer.analyze(trades, strategy_id="test")

        categories = {d.category for d in report.loss_drivers}
        for cat in LossDriverCategory:
            assert cat in categories, f"Missing driver: {cat.value}"

    def test_high_fee_drag_is_critical(self):
        """Trades where fees dominate losses should flag cost as critical."""
        # Trades with tiny losses but relatively large fees
        trades = []
        for _ in range(50):
            trades.append(
                _make_trade(
                    return_pct=-0.001,   # -0.1% loss
                    fee_paid=100.0,       # Large fee relative to notional
                    entry_price=1000.0,   # notional = 1000
                    qty=1.0,
                )
            )
        analyzer = EfficacyAnalyzer(min_trades=10)
        report = analyzer.analyze(trades, strategy_id="test")

        cost_driver = next(
            d for d in report.loss_drivers
            if d.category == LossDriverCategory.EXECUTION_COST
        )
        assert cost_driver.severity == "critical"

    def test_poor_exit_geometry_detected(self):
        """Trades with high MFE but low capture should flag exit geometry."""
        trades = []
        for _ in range(30):
            # Winners that only captured 10% of MFE
            trades.append(
                _make_trade(return_pct=0.002, mfe_pct=0.02, mae_pct=-0.001)
            )
        for _ in range(20):
            # Losers that had positive MFE (could have exited profitably)
            trades.append(
                _make_trade(return_pct=-0.01, mfe_pct=0.005, mae_pct=-0.02)
            )
        analyzer = EfficacyAnalyzer(min_trades=10)
        report = analyzer.analyze(trades, strategy_id="test")

        exit_driver = next(
            d for d in report.loss_drivers
            if d.category == LossDriverCategory.EXIT_GEOMETRY
        )
        # Management efficiency is low, and many losers had positive MFE
        assert exit_driver.severity in ("critical", "warning")

    def test_no_edge_signal_detected(self):
        """Random-like win rate should flag signal edge as weak."""
        import random
        rng = random.Random(42)
        trades = []
        for _ in range(100):
            # Random returns around zero → no edge
            ret = rng.uniform(-0.01, 0.01)
            trades.append(_make_trade(return_pct=ret))

        analyzer = EfficacyAnalyzer(min_trades=10)
        report = analyzer.analyze(trades, strategy_id="test")

        signal_driver = next(
            d for d in report.loss_drivers
            if d.category == LossDriverCategory.SIGNAL_EDGE
        )
        # With random returns, actual WR should be ~50%, same as shuffle
        assert signal_driver.severity in ("info", "warning")

    def test_drivers_sorted_by_impact(self):
        """Loss drivers should be sorted by absolute impact."""
        trades = _make_mixed_trades(25, 25)
        analyzer = EfficacyAnalyzer(min_trades=10)
        report = analyzer.analyze(trades, strategy_id="test")

        impacts = [abs(d.total_pnl_impact) for d in report.loss_drivers]
        assert impacts == sorted(impacts, reverse=True)


# ---------------------------------------------------------------------------
# Phase 2: Recommendations
# ---------------------------------------------------------------------------


class TestRecommendations:
    """Test recommendation generation."""

    def test_has_recommendations(self):
        trades = _make_mixed_trades(20, 30)
        analyzer = EfficacyAnalyzer(min_trades=10)
        report = analyzer.analyze(trades, strategy_id="test")
        assert len(report.recommendations) > 0

    def test_critical_drivers_in_recommendations(self):
        """Critical drivers should appear in recommendations."""
        # All losers with high fees → critical cost driver
        trades = [
            _make_trade(
                return_pct=-0.001,
                fee_paid=100.0,
                entry_price=1000.0,
                qty=1.0,
            )
            for _ in range(50)
        ]
        analyzer = EfficacyAnalyzer(min_trades=10)
        report = analyzer.analyze(trades, strategy_id="test")

        rec_text = " ".join(report.recommendations)
        assert "CRITICAL" in rec_text

    def test_no_issues_gives_acceptable_message(self):
        """All winners → no critical issues → acceptable message."""
        trades = [
            _make_trade(
                return_pct=0.01,
                fee_paid=0.01,
                entry_price=50000.0,
                qty=1.0,
                mfe_pct=0.01,
                mae_pct=-0.001,
            )
            for _ in range(50)
        ]
        analyzer = EfficacyAnalyzer(min_trades=10)
        report = analyzer.analyze(trades, strategy_id="test")

        # Should not have critical recommendations
        critical_recs = [r for r in report.recommendations if "CRITICAL" in r]
        assert len(critical_recs) == 0


# ---------------------------------------------------------------------------
# Full Report
# ---------------------------------------------------------------------------


class TestEfficacyReport:
    """Test the full EfficacyReport."""

    def test_report_to_dict(self):
        trades = _make_mixed_trades(30, 20)
        analyzer = EfficacyAnalyzer(min_trades=10)
        report = analyzer.analyze(trades, strategy_id="test")

        d = report.to_dict()
        assert d["total_trades"] == 50
        assert "win_rate" in d
        assert "profit_factor" in d
        assert "loss_drivers" in d
        assert "segments" in d
        assert "recommendations" in d
        assert isinstance(d["loss_drivers"], list)
        assert len(d["loss_drivers"]) == 5  # All 5 categories

    def test_min_trades_flag(self):
        """min_trades_met should reflect the configured minimum."""
        trades = _make_mixed_trades(3, 2)
        analyzer = EfficacyAnalyzer(min_trades=50)
        report = analyzer.analyze(trades, strategy_id="test")
        assert not report.min_trades_met

        analyzer2 = EfficacyAnalyzer(min_trades=3)
        report2 = analyzer2.analyze(trades, strategy_id="test")
        assert report2.min_trades_met

    def test_aggregate_stats_correct(self):
        trades = [
            _make_trade(return_pct=0.01),
            _make_trade(return_pct=-0.005),
            _make_trade(return_pct=0.02),
        ]
        analyzer = EfficacyAnalyzer(min_trades=1)
        report = analyzer.analyze(trades, strategy_id="test")

        assert report.total_trades == 3
        assert abs(report.win_rate - 2 / 3) < 0.01
        assert abs(report.avg_return - 0.025 / 3) < 0.001
        assert report.profit_factor > 1.0
