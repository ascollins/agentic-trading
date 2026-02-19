"""Tests for the OptimizerScheduler.

Tests config, lifecycle, persistence, and rotation without
requiring actual historical data or backtest runs.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_trading.core.config import OptimizerSchedulerConfig
from agentic_trading.core.enums import OptimizationRecommendation
from agentic_trading.optimizer.report import StrategyRecommendation
from agentic_trading.optimizer.scheduler import OptimizerScheduler


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestOptimizerSchedulerConfig:
    """Test OptimizerSchedulerConfig defaults and validation."""

    def test_default_values(self):
        cfg = OptimizerSchedulerConfig()
        assert cfg.enabled is False
        assert cfg.interval_hours == 24.0
        assert "multi_tf_ma" in cfg.strategies
        assert "bb_squeeze" in cfg.strategies
        assert len(cfg.strategies) == 8  # All 8 CMT strategies
        assert cfg.data_window_days == 90
        assert cfg.n_samples == 30
        assert cfg.auto_apply is False
        assert cfg.max_results_kept == 10
        assert cfg.initial_delay_minutes == 5.0

    def test_custom_values(self):
        cfg = OptimizerSchedulerConfig(
            enabled=True,
            interval_hours=12.0,
            strategies=["breakout"],
            n_samples=50,
            auto_apply=True,
        )
        assert cfg.enabled is True
        assert cfg.interval_hours == 12.0
        assert cfg.strategies == ["breakout"]
        assert cfg.n_samples == 50
        assert cfg.auto_apply is True


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestSchedulerLifecycle:
    """Test start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        cfg = OptimizerSchedulerConfig(
            enabled=True,
            initial_delay_minutes=0.0,
            interval_hours=1.0,
        )
        scheduler = OptimizerScheduler(cfg)

        assert not scheduler.is_running
        await scheduler.start()
        assert scheduler.is_running
        assert scheduler._task is not None

        await scheduler.stop()
        assert not scheduler.is_running
        assert scheduler._task is None

    @pytest.mark.asyncio
    async def test_double_start_is_idempotent(self):
        cfg = OptimizerSchedulerConfig(
            enabled=True,
            initial_delay_minutes=999.0,  # Won't run
        )
        scheduler = OptimizerScheduler(cfg)

        await scheduler.start()
        task1 = scheduler._task
        await scheduler.start()  # Should be no-op
        task2 = scheduler._task
        assert task1 is task2

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start(self):
        cfg = OptimizerSchedulerConfig()
        scheduler = OptimizerScheduler(cfg)
        # Should not raise
        await scheduler.stop()
        assert scheduler.run_count == 0

    @pytest.mark.asyncio
    async def test_introspection_properties(self):
        cfg = OptimizerSchedulerConfig()
        scheduler = OptimizerScheduler(cfg)

        assert scheduler.is_running is False
        assert scheduler.last_run is None
        assert scheduler.run_count == 0


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestResultsPersistence:
    """Test JSON result saving, loading, and rotation."""

    def test_save_and_load_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OptimizerSchedulerConfig(results_dir=tmpdir)
            scheduler = OptimizerScheduler(cfg)

            results = {
                "run_number": 1,
                "started_at": "2026-01-01T00:00:00+00:00",
                "strategies": {
                    "trend_following": {
                        "best_sharpe": 1.5,
                        "best_return": 12.3,
                    }
                },
            }

            timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
            filepath = scheduler._save_results(results, timestamp)
            assert filepath.exists()

            # Load back
            loaded = OptimizerScheduler.load_latest_results(tmpdir)
            assert loaded is not None
            assert loaded["run_number"] == 1
            assert loaded["strategies"]["trend_following"]["best_sharpe"] == 1.5

    def test_load_latest_returns_newest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OptimizerSchedulerConfig(results_dir=tmpdir)
            scheduler = OptimizerScheduler(cfg)

            # Save two results
            r1 = {"run_number": 1, "strategies": {}}
            r2 = {"run_number": 2, "strategies": {}}

            import time

            scheduler._save_results(
                r1, datetime(2026, 1, 1, tzinfo=timezone.utc)
            )
            time.sleep(0.01)  # Ensure different mtime
            scheduler._save_results(
                r2, datetime(2026, 1, 2, tzinfo=timezone.utc)
            )

            latest = OptimizerScheduler.load_latest_results(tmpdir)
            assert latest is not None
            assert latest["run_number"] == 2

    def test_load_all_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OptimizerSchedulerConfig(results_dir=tmpdir)
            scheduler = OptimizerScheduler(cfg)

            import time

            for i in range(3):
                scheduler._save_results(
                    {"run_number": i + 1, "strategies": {}},
                    datetime(2026, 1, i + 1, tzinfo=timezone.utc),
                )
                time.sleep(0.01)

            all_results = OptimizerScheduler.load_all_results(tmpdir)
            assert len(all_results) == 3
            # Should be newest first
            assert all_results[0]["run_number"] == 3
            assert all_results[2]["run_number"] == 1

    def test_rotation_keeps_only_max(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OptimizerSchedulerConfig(
                results_dir=tmpdir, max_results_kept=3
            )
            scheduler = OptimizerScheduler(cfg)

            import time

            for i in range(5):
                scheduler._save_results(
                    {"run_number": i + 1, "strategies": {}},
                    datetime(2026, 1, i + 1, tzinfo=timezone.utc),
                )
                time.sleep(0.01)

            scheduler._rotate_results()

            remaining = list(Path(tmpdir).glob("optimizer_*.json"))
            assert len(remaining) == 3

    def test_load_from_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert OptimizerScheduler.load_latest_results(tmpdir) is None
            assert OptimizerScheduler.load_all_results(tmpdir) == []

    def test_load_from_nonexistent_dir(self):
        assert (
            OptimizerScheduler.load_latest_results("/nonexistent/path")
            is None
        )
        assert (
            OptimizerScheduler.load_all_results("/nonexistent/path") == []
        )


# ---------------------------------------------------------------------------
# Callback tests
# ---------------------------------------------------------------------------


class TestResultsCallback:
    """Test the on_results_callback mechanism."""

    @pytest.mark.asyncio
    async def test_sync_callback_called(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = MagicMock()
            cfg = OptimizerSchedulerConfig(
                results_dir=tmpdir,
                strategies=["trend_following"],
                initial_delay_minutes=0.0,
            )
            scheduler = OptimizerScheduler(
                cfg,
                on_results_callback=callback,
            )

            mock_rec = StrategyRecommendation(
                strategy_id="trend_following",
                recommendation=OptimizationRecommendation.KEEP,
                rationale="test",
            )
            with patch.object(
                scheduler,
                "_optimize_and_recommend",
                return_value=mock_rec,
            ):
                await scheduler._run_optimization_cycle()

            callback.assert_called_once()
            call_args = callback.call_args[0][0]
            assert "strategies" in call_args
            assert call_args["run_number"] == 1

    @pytest.mark.asyncio
    async def test_async_callback_called(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = AsyncMock()
            cfg = OptimizerSchedulerConfig(
                results_dir=tmpdir,
                strategies=["trend_following"],
                initial_delay_minutes=0.0,
            )
            scheduler = OptimizerScheduler(
                cfg,
                on_results_callback=callback,
            )

            mock_rec = StrategyRecommendation(
                strategy_id="trend_following",
                recommendation=OptimizationRecommendation.KEEP,
                rationale="test",
            )
            with patch.object(
                scheduler,
                "_optimize_and_recommend",
                return_value=mock_rec,
            ):
                await scheduler._run_optimization_cycle()

            callback.assert_awaited_once()


# ---------------------------------------------------------------------------
# Optimization cycle tests
# ---------------------------------------------------------------------------


class TestOptimizationCycle:
    """Test optimization cycle with mocked optimizer."""

    @pytest.mark.asyncio
    async def test_cycle_increments_run_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OptimizerSchedulerConfig(
                results_dir=tmpdir,
                strategies=["trend_following"],
            )
            scheduler = OptimizerScheduler(cfg)

            mock_rec = StrategyRecommendation(
                strategy_id="trend_following",
                recommendation=OptimizationRecommendation.KEEP,
                rationale="test",
            )
            with patch.object(
                scheduler,
                "_optimize_and_recommend",
                return_value=mock_rec,
            ):
                await scheduler._run_optimization_cycle()

            assert scheduler.run_count == 1
            assert scheduler.last_run is not None

    @pytest.mark.asyncio
    async def test_cycle_handles_strategy_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OptimizerSchedulerConfig(
                results_dir=tmpdir,
                strategies=["trend_following", "mean_reversion"],
            )
            scheduler = OptimizerScheduler(cfg)

            async def mock_optimize(strategy_id):
                if strategy_id == "trend_following":
                    raise RuntimeError("No data!")
                return StrategyRecommendation(
                    strategy_id=strategy_id,
                    recommendation=OptimizationRecommendation.KEEP,
                    rationale="test",
                )

            with patch.object(
                scheduler, "_optimize_and_recommend", side_effect=mock_optimize
            ):
                await scheduler._run_optimization_cycle()

            # Should still complete and save results
            assert scheduler.run_count == 1
            latest = OptimizerScheduler.load_latest_results(tmpdir)
            assert latest is not None
            assert "error" in latest["strategies"]["trend_following"]

    @pytest.mark.asyncio
    async def test_cycle_persists_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OptimizerSchedulerConfig(
                results_dir=tmpdir,
                strategies=["breakout"],
            )
            scheduler = OptimizerScheduler(cfg)

            mock_rec = StrategyRecommendation(
                strategy_id="breakout",
                recommendation=OptimizationRecommendation.UPDATE,
                optimized_score=2.5,
                improvement_pct=50.0,
                optimized_params={"donchian_period": 25},
                rationale="test improvement",
            )
            with patch.object(
                scheduler,
                "_optimize_and_recommend",
                return_value=mock_rec,
            ):
                await scheduler._run_optimization_cycle()

            results = OptimizerScheduler.load_latest_results(tmpdir)
            assert results["strategies"]["breakout"]["optimized_score"] == 2.5

    @pytest.mark.asyncio
    async def test_multiple_cycles_rotate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OptimizerSchedulerConfig(
                results_dir=tmpdir,
                strategies=["trend_following"],
                max_results_kept=2,
            )
            scheduler = OptimizerScheduler(cfg)

            # Manually save 4 results with different timestamps to avoid collisions
            import time

            for i in range(4):
                scheduler._save_results(
                    {"run_number": i + 1, "strategies": {}},
                    datetime(2026, 1, i + 1, tzinfo=timezone.utc),
                )
                time.sleep(0.01)

            scheduler._rotate_results()

            files = list(Path(tmpdir).glob("optimizer_*.json"))
            assert len(files) == 2  # Rotated to max 2


# ---------------------------------------------------------------------------
# Config integration in Settings
# ---------------------------------------------------------------------------


class TestSettingsIntegration:
    """Test OptimizerSchedulerConfig integration in Settings."""

    def test_settings_has_optimizer_scheduler(self):
        from agentic_trading.core.config import Settings

        settings = Settings()
        assert hasattr(settings, "optimizer_scheduler")
        assert isinstance(settings.optimizer_scheduler, OptimizerSchedulerConfig)
        assert settings.optimizer_scheduler.enabled is False

    def test_settings_load_with_optimizer_config(self):
        from agentic_trading.core.config import Settings

        settings = Settings(
            optimizer_scheduler=OptimizerSchedulerConfig(
                enabled=True,
                interval_hours=6.0,
                n_samples=100,
            )
        )
        assert settings.optimizer_scheduler.enabled is True
        assert settings.optimizer_scheduler.interval_hours == 6.0
        assert settings.optimizer_scheduler.n_samples == 100
