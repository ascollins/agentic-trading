"""Test Settings loading and safety gates."""

import pytest

from agentic_trading.core.config import Settings
from agentic_trading.core.enums import Mode
from agentic_trading.core.errors import SafetyGateError


class TestSettingsLoadFromDict:
    def test_default_settings(self):
        settings = Settings()
        assert settings.mode == Mode.BACKTEST
        assert settings.read_only is False

    def test_load_with_mode_override(self):
        settings = Settings(mode=Mode.PAPER)
        assert settings.mode == Mode.PAPER

    def test_load_risk_config(self):
        settings = Settings()
        assert settings.risk.max_portfolio_leverage == 3.0
        assert settings.risk.max_daily_loss_pct == 0.05
        assert settings.risk.max_drawdown_pct == 0.15

    def test_load_backtest_config(self):
        settings = Settings()
        assert settings.backtest.initial_capital == 100_000.0
        assert settings.backtest.random_seed == 42


class TestValidateLiveMode:
    def test_backtest_mode_passes(self):
        settings = Settings(mode=Mode.BACKTEST)
        settings.validate_live_mode()  # Should not raise

    def test_paper_mode_passes(self):
        settings = Settings(mode=Mode.PAPER)
        settings.validate_live_mode()  # Should not raise

    def test_live_mode_raises_without_flag(self):
        settings = Settings(
            mode=Mode.LIVE,
            i_understand_live_trading=False,
            live_flag=True,
        )
        with pytest.raises(SafetyGateError, match="I_UNDERSTAND_LIVE_TRADING"):
            settings.validate_live_mode()

    def test_live_mode_raises_without_cli_flag(self):
        settings = Settings(
            mode=Mode.LIVE,
            i_understand_live_trading=True,
            live_flag=False,
        )
        with pytest.raises(SafetyGateError, match="--live CLI flag"):
            settings.validate_live_mode()

    def test_live_mode_passes_with_both_flags(self):
        settings = Settings(
            mode=Mode.LIVE,
            i_understand_live_trading=True,
            live_flag=True,
        )
        settings.validate_live_mode()  # Should not raise
