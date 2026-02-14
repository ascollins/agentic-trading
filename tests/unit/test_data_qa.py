"""Test DataQualityChecker: gap detection, price sanity, staleness."""

from datetime import datetime, timedelta, timezone

import pytest

from agentic_trading.core.enums import Exchange, Timeframe
from agentic_trading.core.models import Candle
from agentic_trading.data.data_qa import DataQualityChecker, Severity


def _candle_at(minutes_offset: int, close: float = 67000.0) -> Candle:
    base = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    return Candle(
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        timeframe=Timeframe.M1,
        timestamp=base + timedelta(minutes=minutes_offset),
        open=close - 50,
        high=close + 50,
        low=close - 100,
        close=close,
        volume=10.0,
    )


class TestCheckGaps:
    def test_no_gaps_in_contiguous_candles(self):
        candles = [_candle_at(i) for i in range(10)]
        issues = DataQualityChecker.check_gaps(candles, Timeframe.M1)
        assert len(issues) == 0

    def test_detects_single_gap(self):
        candles = [_candle_at(0), _candle_at(1), _candle_at(5)]  # Gap at 2-4
        issues = DataQualityChecker.check_gaps(candles, Timeframe.M1)
        assert len(issues) == 1
        assert "gap" in issues[0].check.lower()
        assert issues[0].details["missing_count"] == 3

    def test_detects_multiple_gaps(self):
        candles = [_candle_at(0), _candle_at(5), _candle_at(10)]
        issues = DataQualityChecker.check_gaps(candles, Timeframe.M1)
        assert len(issues) == 2

    def test_single_candle_no_issues(self):
        candles = [_candle_at(0)]
        issues = DataQualityChecker.check_gaps(candles, Timeframe.M1)
        assert len(issues) == 0

    def test_large_gap_is_critical(self):
        candles = [_candle_at(0), _candle_at(10)]  # 9 missing candles
        issues = DataQualityChecker.check_gaps(candles, Timeframe.M1)
        assert len(issues) == 1
        assert issues[0].severity == Severity.CRITICAL


class TestCheckPriceSanity:
    def test_normal_candle_passes(self):
        candle = Candle(
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            timeframe=Timeframe.M1,
            timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
            open=67000.0,
            high=67100.0,
            low=66900.0,
            close=67050.0,
            volume=10.0,
        )
        issues = DataQualityChecker.check_price_sanity(candle, None)
        assert len(issues) == 0

    def test_detects_negative_price(self):
        candle = Candle(
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            timeframe=Timeframe.M1,
            timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
            open=-100.0,
            high=67100.0,
            low=66900.0,
            close=67050.0,
            volume=10.0,
        )
        issues = DataQualityChecker.check_price_sanity(candle, None)
        assert any(i.severity == Severity.CRITICAL for i in issues)

    def test_detects_zero_price(self):
        candle = Candle(
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            timeframe=Timeframe.M1,
            timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
            open=0.0,
            high=100.0,
            low=0.0,
            close=50.0,
            volume=10.0,
        )
        issues = DataQualityChecker.check_price_sanity(candle, None)
        assert len(issues) > 0

    def test_detects_extreme_close_to_close_change(self):
        prev = Candle(
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            timeframe=Timeframe.M1,
            timestamp=datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc),
            open=67000.0, high=67100.0, low=66900.0, close=67000.0,
            volume=10.0,
        )
        # Close jumps 20% (well above the 15% default threshold)
        curr = Candle(
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            timeframe=Timeframe.M1,
            timestamp=datetime(2024, 6, 1, 0, 1, 0, tzinfo=timezone.utc),
            open=80400.0, high=80400.0, low=80400.0, close=80400.0,
            volume=10.0,
        )
        issues = DataQualityChecker.check_price_sanity(curr, prev, max_change_pct=15.0)
        close_issues = [i for i in issues if "close-to-close" in i.message.lower()]
        assert len(close_issues) == 1
        assert close_issues[0].severity == Severity.CRITICAL


class TestCheckStaleness:
    def test_fresh_data_passes(self):
        now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        last = datetime(2024, 6, 1, 11, 59, 0, tzinfo=timezone.utc)
        issue = DataQualityChecker.check_staleness(last, max_age_seconds=300, now=now)
        assert issue is None

    def test_stale_data_detected(self):
        now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        last = datetime(2024, 6, 1, 11, 50, 0, tzinfo=timezone.utc)  # 10 min ago
        issue = DataQualityChecker.check_staleness(
            last, max_age_seconds=300, symbol="BTC/USDT", now=now
        )
        assert issue is not None
        assert issue.check == "staleness"
        assert issue.symbol == "BTC/USDT"

    def test_very_stale_is_critical(self):
        now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        last = datetime(2024, 6, 1, 11, 0, 0, tzinfo=timezone.utc)  # 1 hour ago
        issue = DataQualityChecker.check_staleness(
            last, max_age_seconds=300, now=now
        )
        assert issue is not None
        assert issue.severity == Severity.CRITICAL
