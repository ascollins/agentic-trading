"""Tests for the EfficacyAgent.

Tests config, lifecycle, and basic analysis workflow
without requiring actual historical data or backtest runs.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_trading.core.config import EfficacyAgentConfig
from agentic_trading.optimizer.efficacy_agent import EfficacyAgent
from agentic_trading.optimizer.efficacy_models import EfficacyReport


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestEfficacyAgentConfig:
    """Test EfficacyAgentConfig defaults and validation."""

    def test_default_values(self):
        cfg = EfficacyAgentConfig()
        assert cfg.enabled is False
        assert cfg.interval_hours == 24.0
        assert len(cfg.strategies) == 8
        assert "bb_squeeze" in cfg.strategies
        assert "multi_tf_ma" in cfg.strategies
        assert cfg.data_window_days == 90
        assert cfg.min_trades_per_segment == 50
        assert cfg.results_dir == "data/efficacy_results"

    def test_custom_values(self):
        cfg = EfficacyAgentConfig(
            enabled=True,
            interval_hours=12.0,
            strategies=["bb_squeeze"],
            min_trades_per_segment=30,
        )
        assert cfg.enabled is True
        assert cfg.interval_hours == 12.0
        assert cfg.strategies == ["bb_squeeze"]
        assert cfg.min_trades_per_segment == 30


# ---------------------------------------------------------------------------
# Settings integration
# ---------------------------------------------------------------------------


class TestSettingsIntegration:
    """Test EfficacyAgentConfig integration in Settings."""

    def test_settings_has_efficacy_agent(self):
        from agentic_trading.core.config import Settings

        settings = Settings()
        assert hasattr(settings, "efficacy_agent")
        assert isinstance(settings.efficacy_agent, EfficacyAgentConfig)
        assert settings.efficacy_agent.enabled is False

    def test_settings_load_with_efficacy_config(self):
        from agentic_trading.core.config import Settings

        settings = Settings(
            efficacy_agent=EfficacyAgentConfig(
                enabled=True,
                interval_hours=6.0,
                min_trades_per_segment=100,
            )
        )
        assert settings.efficacy_agent.enabled is True
        assert settings.efficacy_agent.interval_hours == 6.0
        assert settings.efficacy_agent.min_trades_per_segment == 100


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestEfficacyAgentLifecycle:
    """Test agent start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        cfg = EfficacyAgentConfig(
            enabled=True,
            interval_hours=999.0,  # Won't run during test
        )
        agent = EfficacyAgent(config=cfg, agent_id="test-efficacy")

        assert not agent.is_running
        await agent.start()
        assert agent.is_running
        assert agent._task is not None

        await agent.stop()
        assert not agent.is_running
        assert agent._task is None

    @pytest.mark.asyncio
    async def test_stop_without_start(self):
        cfg = EfficacyAgentConfig()
        agent = EfficacyAgent(config=cfg)
        # Should not raise
        await agent.stop()
        assert agent.run_count == 0

    @pytest.mark.asyncio
    async def test_properties(self):
        cfg = EfficacyAgentConfig()
        agent = EfficacyAgent(config=cfg)

        assert agent.is_running is False
        assert agent.last_run is None
        assert agent.run_count == 0
        assert agent.last_report is None
        assert agent.agent_type.value == "optimizer"

    def test_capabilities(self):
        cfg = EfficacyAgentConfig()
        agent = EfficacyAgent(config=cfg)
        caps = agent.capabilities()
        assert "optimizer.result" in caps.publishes_to


# ---------------------------------------------------------------------------
# Report persistence
# ---------------------------------------------------------------------------


class TestReportPersistence:
    """Test report saving and loading."""

    def test_save_and_load_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EfficacyAgentConfig(results_dir=tmpdir)
            agent = EfficacyAgent(config=cfg)

            report = EfficacyReport(
                timestamp="2026-01-01T00:00:00+00:00",
                strategy_id="test",
                total_trades=100,
                win_rate=0.45,
                profit_factor=0.95,
                recommendations=["Fix exit geometry"],
            )

            filepath = agent._save_report(report)
            assert filepath.exists()

            # Load back
            loaded = EfficacyAgent.load_latest_report(tmpdir)
            assert loaded is not None
            assert loaded.total_trades == 100
            assert loaded.win_rate == 0.45
            assert loaded.recommendations == ["Fix exit geometry"]

    def test_load_from_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert EfficacyAgent.load_latest_report(tmpdir) is None

    def test_load_from_nonexistent_dir(self):
        assert EfficacyAgent.load_latest_report("/nonexistent/path") is None

    def test_rotation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EfficacyAgentConfig(results_dir=tmpdir)
            agent = EfficacyAgent(config=cfg)

            import time
            from pathlib import Path

            # Create 5 files manually with distinct names
            for i in range(5):
                filepath = Path(tmpdir) / f"efficacy_2026010{i + 1}T000000.json"
                filepath.write_text(json.dumps({"run": i + 1}))
                time.sleep(0.01)  # Ensure different mtime

            # Verify all 5 exist
            all_files = list(Path(tmpdir).glob("efficacy_*.json"))
            assert len(all_files) == 5

            # Rotate keeping only 3
            agent._rotate_results(max_keep=3)
            remaining = list(Path(tmpdir).glob("efficacy_*.json"))
            assert len(remaining) == 3


# ---------------------------------------------------------------------------
# Analysis cycle (mocked)
# ---------------------------------------------------------------------------


class TestAnalysisCycle:
    """Test analysis cycle with mocked data loading and backtesting."""

    @pytest.mark.asyncio
    async def test_cycle_with_no_data_skips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EfficacyAgentConfig(
                results_dir=tmpdir,
                strategies=["bb_squeeze"],
            )
            agent = EfficacyAgent(config=cfg)

            with patch.object(
                agent, "_load_historical_data", return_value={}
            ):
                await agent._run_analysis_cycle()

            # Should not increment run count when no data
            assert agent.run_count == 0

    @pytest.mark.asyncio
    async def test_cycle_increments_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EfficacyAgentConfig(
                results_dir=tmpdir,
                strategies=["bb_squeeze"],
            )
            agent = EfficacyAgent(config=cfg)

            from agentic_trading.backtester.results import TradeDetail

            mock_trades = [
                TradeDetail(
                    strategy_id="bb_squeeze",
                    symbol="BTC/USDT",
                    direction="long",
                    entry_price=50000.0,
                    exit_price=50500.0,
                    return_pct=0.01,
                    fee_paid=10.0,
                    mae_pct=-0.005,
                    mfe_pct=0.015,
                    exit_reason="signal",
                    hold_seconds=3600.0,
                    qty=1.0,
                )
            ]

            with patch.object(
                agent,
                "_load_historical_data",
                return_value={"BTC/USDT": [MagicMock()]},
            ), patch.object(
                agent,
                "_backtest_strategy",
                return_value=mock_trades,
            ):
                await agent._run_analysis_cycle()

            assert agent.run_count == 1
            assert agent.last_run is not None
            assert agent.last_report is not None
            assert agent.last_report.total_trades == 1
