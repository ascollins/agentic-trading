"""Test DrawdownMonitor tracks peak equity and detects drawdown breaches."""

from agentic_trading.risk.drawdown import DrawdownMonitor


class TestDrawdownMonitor:
    def test_initial_peak_equity(self):
        monitor = DrawdownMonitor(initial_equity=100_000.0)
        assert monitor.peak_equity == 100_000.0

    def test_update_equity_tracks_peak(self):
        monitor = DrawdownMonitor(initial_equity=100_000.0)
        monitor.update_equity(110_000.0)
        assert monitor.peak_equity == 110_000.0

    def test_update_equity_does_not_reduce_peak(self):
        monitor = DrawdownMonitor(initial_equity=100_000.0)
        monitor.update_equity(110_000.0)
        monitor.update_equity(105_000.0)
        assert monitor.peak_equity == 110_000.0

    def test_drawdown_pct_computed(self):
        monitor = DrawdownMonitor(initial_equity=100_000.0)
        monitor.update_equity(110_000.0)  # New peak
        monitor.update_equity(99_000.0)  # Drawdown
        # DD = (110000 - 99000) / 110000 = 0.1
        assert abs(monitor.current_drawdown_pct - 0.1) < 0.001

    def test_check_drawdown_returns_false_when_ok(self):
        monitor = DrawdownMonitor(initial_equity=100_000.0)
        result = monitor.check_drawdown(95_000.0, max_drawdown_pct=0.15)
        assert result is False

    def test_check_drawdown_returns_true_on_breach(self):
        monitor = DrawdownMonitor(initial_equity=100_000.0)
        monitor.update_equity(100_000.0)  # Set peak
        result = monitor.check_drawdown(80_000.0, max_drawdown_pct=0.15)
        # DD = 20% > 15% limit
        assert result is True

    def test_check_drawdown_at_exact_threshold(self):
        monitor = DrawdownMonitor(initial_equity=100_000.0)
        # Exactly 15% drawdown from peak
        result = monitor.check_drawdown(85_000.0, max_drawdown_pct=0.15)
        assert result is True  # >= threshold

    def test_check_daily_loss_ok(self):
        monitor = DrawdownMonitor(initial_equity=100_000.0)
        result = monitor.check_daily_loss(
            daily_pnl=-2_000.0, max_daily_loss_pct=0.05, capital=100_000.0
        )
        assert result is False

    def test_check_daily_loss_breach(self):
        monitor = DrawdownMonitor(initial_equity=100_000.0)
        result = monitor.check_daily_loss(
            daily_pnl=-6_000.0, max_daily_loss_pct=0.05, capital=100_000.0
        )
        assert result is True

    def test_check_daily_loss_positive_pnl(self):
        monitor = DrawdownMonitor(initial_equity=100_000.0)
        result = monitor.check_daily_loss(
            daily_pnl=5_000.0, max_daily_loss_pct=0.05, capital=100_000.0
        )
        assert result is False

    def test_max_drawdown_from_peak_property(self):
        monitor = DrawdownMonitor(initial_equity=100_000.0)
        monitor.update_equity(110_000.0)
        monitor.update_equity(99_000.0)
        assert monitor.max_drawdown_from_peak == monitor.current_drawdown_pct

    def test_reset_daily(self):
        monitor = DrawdownMonitor(initial_equity=100_000.0)
        monitor.update_equity(95_000.0)
        monitor.reset_daily(current_equity=95_000.0)
        assert monitor.daily_pnl == 0.0
