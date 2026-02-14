"""Test WallClock and SimClock."""

from datetime import datetime, timedelta, timezone

import pytest

from agentic_trading.core.clock import SimClock, WallClock


class TestWallClock:
    def test_now_returns_utc(self):
        clock = WallClock()
        now = clock.now()
        assert now.tzinfo is not None
        assert now.tzinfo == timezone.utc

    def test_now_is_recent(self):
        clock = WallClock()
        now = clock.now()
        diff = abs((datetime.now(timezone.utc) - now).total_seconds())
        assert diff < 1.0

    def test_now_ms_returns_int(self):
        clock = WallClock()
        ms = clock.now_ms()
        assert isinstance(ms, int)
        assert ms > 0


class TestSimClock:
    def test_default_start(self):
        clock = SimClock()
        assert clock.now() == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_custom_start(self, sim_clock):
        assert sim_clock.now() == datetime(2024, 6, 1, tzinfo=timezone.utc)

    def test_now_returns_datetime(self, sim_clock):
        result = sim_clock.now()
        assert isinstance(result, datetime)
        assert result.tzinfo == timezone.utc

    def test_now_ms_returns_int(self, sim_clock):
        ms = sim_clock.now_ms()
        assert isinstance(ms, int)
        assert ms > 0

    def test_set_time_advances(self, sim_clock):
        new_time = datetime(2024, 6, 2, tzinfo=timezone.utc)
        sim_clock.set_time(new_time)
        assert sim_clock.now() == new_time

    def test_set_time_cannot_go_backwards(self, sim_clock):
        earlier = datetime(2024, 5, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="cannot go backwards"):
            sim_clock.set_time(earlier)

    def test_set_time_same_time_ok(self, sim_clock):
        same = sim_clock.now()
        sim_clock.set_time(same)  # Should not raise
        assert sim_clock.now() == same

    def test_advance_ms(self, sim_clock):
        before = sim_clock.now()
        sim_clock.advance_ms(5000)
        after = sim_clock.now()
        diff = (after - before).total_seconds()
        assert diff == pytest.approx(5.0)

    def test_advance_ms_multiple_times(self, sim_clock):
        start = sim_clock.now()
        sim_clock.advance_ms(1000)
        sim_clock.advance_ms(2000)
        sim_clock.advance_ms(500)
        elapsed = (sim_clock.now() - start).total_seconds()
        assert elapsed == pytest.approx(3.5)

    def test_advance_ms_updates_now_ms(self, sim_clock):
        ms_before = sim_clock.now_ms()
        sim_clock.advance_ms(60_000)
        ms_after = sim_clock.now_ms()
        assert ms_after - ms_before == 60_000
