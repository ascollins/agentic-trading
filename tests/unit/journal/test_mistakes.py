"""Tests for MistakeDetector — automated mistake detection and classification."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from agentic_trading.journal.mistakes import MistakeDetector, Mistake
from agentic_trading.journal.record import (
    FillLeg,
    MarkSample,
    TradePhase,
    TradeOutcome,
    TradeRecord,
)

from .conftest import make_fill, make_winning_trade, make_losing_trade


@pytest.fixture
def detector():
    return MistakeDetector(
        early_exit_threshold=0.4,
        moved_stop_mae_r_threshold=-1.5,
        chased_entry_pct=0.02,
        low_confidence_threshold=0.3,
        reversal_mfe_r=2.0,
        reversal_close_r=0.5,
    )


class TestMistakeDetectorBasics:
    """Test basic mistake detection."""

    def test_no_mistakes_on_good_trade(self, detector):
        """A clean winning trade should produce no mistakes."""
        trade = make_winning_trade(entry_price=100.0, exit_price=110.0)
        mistakes = detector.analyse(trade)
        assert len(mistakes) == 0

    def test_no_analysis_on_open_trade(self, detector):
        """Open trades should not be analysed."""
        trade = TradeRecord(
            trace_id="t1", strategy_id="trend",
            symbol="BTC/USDT", direction="long",
        )
        trade.add_entry_fill(make_fill(price=100.0))
        assert trade.phase == TradePhase.OPEN
        mistakes = detector.analyse(trade)
        assert len(mistakes) == 0

    def test_returns_mistake_objects(self, detector):
        """Detected mistakes should be proper Mistake instances."""
        # Create a trade with early exit pattern
        trade = _make_early_exit_trade()
        mistakes = detector.analyse(trade)
        for m in mistakes:
            assert isinstance(m, Mistake)
            assert m.mistake_type
            assert m.severity in ("low", "medium", "high")
            assert isinstance(m.pnl_impact, float)


class TestEarlyExit:
    """Test early exit detection."""

    def test_detects_early_exit(self, detector):
        """Trade that captured only small portion of MFE should be flagged."""
        trade = _make_early_exit_trade()
        mistakes = detector.analyse(trade)
        types = [m.mistake_type for m in mistakes]
        assert "early_exit" in types

    def test_no_early_exit_on_full_capture(self, detector):
        """Trade that captured most of MFE should not be flagged."""
        trade = make_winning_trade(entry_price=100.0, exit_price=112.0)
        mistakes = detector.analyse(trade)
        types = [m.mistake_type for m in mistakes]
        assert "early_exit" not in types


class TestMovedStop:
    """Test moved stop detection."""

    def test_detects_moved_stop(self, detector):
        """Trade where MAE exceeded initial risk should be flagged."""
        trade = _make_moved_stop_trade()
        mistakes = detector.analyse(trade)
        types = [m.mistake_type for m in mistakes]
        assert "moved_stop" in types

    def test_no_moved_stop_within_risk(self, detector):
        """Trade that stayed within risk should not be flagged."""
        trade = make_losing_trade(entry_price=100.0, exit_price=96.0)
        mistakes = detector.analyse(trade)
        types = [m.mistake_type for m in mistakes]
        assert "moved_stop" not in types


class TestHeldThroughReversal:
    """Test held through reversal detection."""

    def test_detects_reversal(self, detector):
        """Trade with big MFE that closed near breakeven should be flagged."""
        trade = _make_reversal_trade()
        mistakes = detector.analyse(trade)
        types = [m.mistake_type for m in mistakes]
        assert "held_through_reversal" in types


class TestLowConfidence:
    """Test low confidence entry detection."""

    def test_detects_low_confidence(self, detector):
        """Trade with very low confidence should be flagged."""
        trade = _make_low_confidence_trade()
        mistakes = detector.analyse(trade)
        types = [m.mistake_type for m in mistakes]
        assert "low_confidence_entry" in types

    def test_no_flag_on_decent_confidence(self, detector):
        """Trade with reasonable confidence should not be flagged."""
        trade = make_winning_trade()  # Default confidence is 0.8
        mistakes = detector.analyse(trade)
        types = [m.mistake_type for m in mistakes]
        assert "low_confidence_entry" not in types


class TestOversizedPosition:
    """Test oversized position detection."""

    def test_detects_oversized(self, detector):
        """Trade with governance multiplier < 1 should be flagged."""
        trade = make_winning_trade()
        trade.governance_sizing_multiplier = 0.5
        mistakes = detector.analyse(trade)
        types = [m.mistake_type for m in mistakes]
        assert "oversized_position" in types


class TestPoorHealthEntry:
    """Test poor health entry detection."""

    def test_detects_poor_health(self, detector):
        """Trade entered during low strategy health should be flagged."""
        trade = make_losing_trade()
        trade.health_score_at_entry = 0.3
        mistakes = detector.analyse(trade)
        types = [m.mistake_type for m in mistakes]
        assert "poor_health_entry" in types

    def test_no_flag_on_healthy_entry(self, detector):
        """Trade entered during good health should not be flagged."""
        trade = make_winning_trade()
        assert trade.health_score_at_entry == 1.0
        mistakes = detector.analyse(trade)
        types = [m.mistake_type for m in mistakes]
        assert "poor_health_entry" not in types


class TestMistakeReport:
    """Test aggregate mistake reporting."""

    def test_report_empty(self, detector):
        report = detector.report("nonexistent")
        assert report["total_mistakes"] == 0
        assert report["by_type"] == {}
        assert report["costliest_type"] is None

    def test_report_aggregation(self, detector):
        """Multiple trades should aggregate properly."""
        for _ in range(5):
            trade = _make_early_exit_trade()
            trade.trade_id = f"t_{_}"
            trade.trace_id = f"tr_{_}"
            detector.analyse(trade)

        report = detector.report("trend")
        assert report["total_mistakes"] >= 5
        assert "early_exit" in report["by_type"]
        assert report["by_type"]["early_exit"]["count"] >= 5

    def test_costliest_type_identified(self, detector):
        """The most expensive mistake type should be identified."""
        # Add early exit mistakes (medium cost)
        for i in range(3):
            trade = _make_early_exit_trade()
            trade.trade_id = f"ee_{i}"
            trade.trace_id = f"ee_tr_{i}"
            detector.analyse(trade)

        # Add moved stop mistakes (high cost)
        for i in range(3):
            trade = _make_moved_stop_trade()
            trade.trade_id = f"ms_{i}"
            trade.trace_id = f"ms_tr_{i}"
            detector.analyse(trade)

        report = detector.report("trend")
        assert report["costliest_type"] is not None
        assert report["total_pnl_impact"] < 0

    def test_mistakes_tagged_on_trade(self, detector):
        """Detected mistakes should be added to the trade's mistakes list."""
        trade = _make_early_exit_trade()
        detector.analyse(trade)
        assert "early_exit" in trade.mistakes

    def test_get_all_strategy_ids(self, detector):
        trade = _make_early_exit_trade()
        detector.analyse(trade)
        assert "trend" in detector.get_all_strategy_ids()


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_early_exit_trade() -> TradeRecord:
    """Create a trade that captured very little of its MFE."""
    bt = datetime(2024, 1, 1, 12, 0, 0)
    trade = TradeRecord(
        trace_id="trace_ee", strategy_id="trend",
        symbol="BTC/USDT", direction="long",
        signal_confidence=0.8,
        initial_risk_price=Decimal("95"),
    )
    trade.add_entry_fill(make_fill(price=100.0, qty=1.0, timestamp=bt))
    trade.compute_initial_risk()
    # MFE was 120 (big move)
    trade.add_mark_sample(MarkSample(
        timestamp=bt + timedelta(minutes=30),
        mark_price=Decimal("120"),
        unrealized_pnl=Decimal("20"),
    ))
    # But exited at only 102
    trade.add_exit_fill(make_fill(
        fill_id="exit_1", order_id="exit_order", side="sell",
        price=102.0, qty=1.0, timestamp=bt + timedelta(hours=1),
    ))
    return trade


def _make_moved_stop_trade() -> TradeRecord:
    """Create a trade where MAE exceeded planned risk significantly."""
    bt = datetime(2024, 1, 1, 12, 0, 0)
    trade = TradeRecord(
        trace_id="trace_ms", strategy_id="trend",
        symbol="BTC/USDT", direction="long",
        signal_confidence=0.8,
        initial_risk_price=Decimal("95"),  # 5% risk
    )
    trade.add_entry_fill(make_fill(price=100.0, qty=1.0, timestamp=bt))
    trade.compute_initial_risk()
    # MAE dropped to 88 (well below 95 stop)
    trade.add_mark_sample(MarkSample(
        timestamp=bt + timedelta(minutes=30),
        mark_price=Decimal("88"),
        unrealized_pnl=Decimal("-12"),
    ))
    trade.add_exit_fill(make_fill(
        fill_id="exit_1", order_id="exit_order", side="sell",
        price=92.0, qty=1.0, timestamp=bt + timedelta(hours=1),
    ))
    return trade


def _make_reversal_trade() -> TradeRecord:
    """Create a trade with big MFE that reversed to near breakeven."""
    bt = datetime(2024, 1, 1, 12, 0, 0)
    trade = TradeRecord(
        trace_id="trace_rev", strategy_id="trend",
        symbol="BTC/USDT", direction="long",
        signal_confidence=0.8,
        initial_risk_price=Decimal("95"),  # 5R risk
    )
    trade.add_entry_fill(make_fill(price=100.0, qty=1.0, timestamp=bt))
    trade.compute_initial_risk()
    # MFE reached 115 (3R)
    trade.add_mark_sample(MarkSample(
        timestamp=bt + timedelta(minutes=30),
        mark_price=Decimal("115"),
        unrealized_pnl=Decimal("15"),
    ))
    # Then reversed, closed at 101 (0.2R)
    trade.add_exit_fill(make_fill(
        fill_id="exit_1", order_id="exit_order", side="sell",
        price=101.0, qty=1.0, timestamp=bt + timedelta(hours=2),
    ))
    return trade


def _make_low_confidence_trade() -> TradeRecord:
    """Create a trade entered with very low confidence."""
    bt = datetime(2024, 1, 1, 12, 0, 0)
    trade = TradeRecord(
        trace_id="trace_lc", strategy_id="trend",
        symbol="BTC/USDT", direction="long",
        signal_confidence=0.15,
        initial_risk_price=Decimal("95"),
    )
    trade.add_entry_fill(make_fill(price=100.0, qty=1.0, timestamp=bt))
    trade.compute_initial_risk()
    trade.add_exit_fill(make_fill(
        fill_id="exit_1", order_id="exit_order", side="sell",
        price=98.0, qty=1.0, timestamp=bt + timedelta(hours=1),
    ))
    return trade
