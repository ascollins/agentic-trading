"""Tests for the IntelligenceManager facade.

Tests cover:
1. IntelligenceManager.from_config construction and wiring
2. Component accessors
3. Lifecycle (start/stop)
4. Delegated operations — feature computation
5. Delegated operations — historical data
6. Delegated operations — analysis
7. Delegated operations — data quality
8. Backtest mode (no event bus)
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_trading.core.enums import Exchange, Timeframe
from agentic_trading.core.models import Candle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candle(
    symbol: str = "BTC/USDT",
    close: float = 50000.0,
    volume: float = 100.0,
    ts: datetime | None = None,
) -> Candle:
    """Create a minimal Candle for testing."""
    return Candle(
        symbol=symbol,
        exchange=Exchange.BYBIT,
        timeframe=Timeframe.M1,
        timestamp=ts or datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        open=close - 10,
        high=close + 20,
        low=close - 30,
        close=close,
        volume=volume,
    )


def _make_mock_event_bus():
    """Create a mock event bus."""
    bus = MagicMock()
    bus.subscribe = AsyncMock()
    bus.publish = AsyncMock()
    return bus


# ---------------------------------------------------------------------------
# Factory — backtest mode (no event bus)
# ---------------------------------------------------------------------------

class TestIntelligenceManagerFactoryBacktest:
    """from_config with no event bus creates backtest-mode manager."""

    def test_from_config_backtest(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        assert mgr is not None
        assert mgr.feature_engine is not None

    def test_backtest_no_feed_manager(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        assert mgr.feed_manager is None

    def test_backtest_no_candle_builder(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        assert mgr.candle_builder is None

    def test_backtest_has_historical_loader(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        assert mgr.historical_loader is not None

    def test_backtest_has_data_qa(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        assert mgr.data_qa is not None

    def test_backtest_has_htf_analyzer(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        assert mgr.htf_analyzer is not None

    def test_backtest_has_smc_scorer(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        assert mgr.smc_scorer is not None

    def test_backtest_has_trade_plan_generator(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        assert mgr.trade_plan_generator is not None


# ---------------------------------------------------------------------------
# Factory — live/paper mode (with event bus)
# ---------------------------------------------------------------------------

class TestIntelligenceManagerFactoryLive:
    """from_config with event bus creates live-mode manager."""

    def test_from_config_live_has_candle_builder(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        bus = _make_mock_event_bus()
        mgr = IntelligenceManager.from_config(event_bus=bus)
        assert mgr.candle_builder is not None

    def test_from_config_no_feed_without_exchange_configs(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        bus = _make_mock_event_bus()
        mgr = IntelligenceManager.from_config(event_bus=bus)
        # No exchange configs or symbols → no feed manager
        assert mgr.feed_manager is None

    def test_custom_buffer_size(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config(buffer_size=200)
        assert mgr.feature_engine._buffer_size == 200


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestIntelligenceManagerLifecycle:
    """IntelligenceManager lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_stop_backtest(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        bus = _make_mock_event_bus()
        mgr = IntelligenceManager.from_config(event_bus=bus)
        await mgr.start()
        await mgr.stop()
        # Should not raise

    @pytest.mark.asyncio
    async def test_start_stop_backtest_no_bus(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        # Feature engine with no event bus — start/stop are no-ops
        await mgr.start()
        await mgr.stop()


# ---------------------------------------------------------------------------
# Component accessors
# ---------------------------------------------------------------------------

class TestComponentAccessors:
    """IntelligenceManager exposes sub-components."""

    def test_feature_engine_type(self):
        from agentic_trading.intelligence.features.engine import FeatureEngine
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        assert isinstance(mgr.feature_engine, FeatureEngine)

    def test_historical_loader_type(self):
        from agentic_trading.intelligence.historical import HistoricalDataLoader
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        assert isinstance(mgr.historical_loader, HistoricalDataLoader)

    def test_data_qa_type(self):
        from agentic_trading.intelligence.data_qa import DataQualityChecker
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        assert isinstance(mgr.data_qa, DataQualityChecker)

    def test_htf_analyzer_type(self):
        from agentic_trading.intelligence.analysis.htf_analyzer import HTFAnalyzer
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        assert isinstance(mgr.htf_analyzer, HTFAnalyzer)

    def test_smc_scorer_type(self):
        from agentic_trading.intelligence.analysis.smc_confluence import (
            SMCConfluenceScorer,
        )
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        assert isinstance(mgr.smc_scorer, SMCConfluenceScorer)

    def test_trade_plan_generator_type(self):
        from agentic_trading.intelligence.analysis.smc_trade_plan import (
            SMCTradePlanGenerator,
        )
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        assert isinstance(mgr.trade_plan_generator, SMCTradePlanGenerator)


# ---------------------------------------------------------------------------
# Delegated operations — feature computation
# ---------------------------------------------------------------------------

class TestDelegatedFeatureComputation:
    """IntelligenceManager delegates to FeatureEngine."""

    def test_add_candle_delegates(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        candle = _make_candle()
        mgr.add_candle(candle)
        buf = mgr.get_buffer("BTC/USDT", Timeframe.M1)
        assert len(buf) == 1

    def test_get_buffer_empty(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        buf = mgr.get_buffer("BTC/USDT", Timeframe.M1)
        assert buf == []

    def test_compute_features_with_candles(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        # Need enough candles for EMA computation
        candles = [
            _make_candle(
                close=50000 + i * 10,
                ts=datetime(2024, 1, 1, 12, i, 0, tzinfo=timezone.utc),
            )
            for i in range(30)
        ]
        for c in candles:
            mgr.add_candle(c)
        fv = mgr.compute_features("BTC/USDT", Timeframe.M1, candles)
        assert fv is not None
        assert fv.symbol == "BTC/USDT"


# ---------------------------------------------------------------------------
# Delegated operations — analysis
# ---------------------------------------------------------------------------

class TestDelegatedAnalysis:
    """IntelligenceManager delegates to analysis tools."""

    def test_analyze_htf_returns_assessment(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        # Minimal aligned features dict (will produce default assessment)
        features = {
            "1h_ema_21": 50000.0,
            "1h_ema_50": 49500.0,
            "1h_ema_200": 48000.0,
            "1h_close": 50100.0,
            "1h_rsi_14": 55.0,
            "1h_adx_14": 30.0,
            "1h_atr_14": 500.0,
            "4h_ema_21": 50000.0,
            "4h_ema_50": 49500.0,
            "4h_ema_200": 48000.0,
            "4h_close": 50100.0,
            "4h_rsi_14": 55.0,
            "4h_adx_14": 30.0,
            "4h_atr_14": 500.0,
        }
        result = mgr.analyze_htf("BTC/USDT", features)
        assert result is not None
        assert result.symbol == "BTC/USDT"

    def test_score_smc_confluence(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        # Minimal features — SMC scorer will produce default result
        features = {}
        result = mgr.score_smc_confluence("BTC/USDT", features)
        assert result is not None

    def test_analyze_htf_raises_without_analyzer(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        # Create manager without HTF analyzer
        fe_mock = MagicMock()
        mgr = IntelligenceManager(feature_engine=fe_mock, htf_analyzer=None)
        with pytest.raises(RuntimeError, match="No HTF analyzer"):
            mgr.analyze_htf("BTC/USDT", {})

    def test_score_smc_raises_without_scorer(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        fe_mock = MagicMock()
        mgr = IntelligenceManager(feature_engine=fe_mock, smc_scorer=None)
        with pytest.raises(RuntimeError, match="No SMC scorer"):
            mgr.score_smc_confluence("BTC/USDT", {})


# ---------------------------------------------------------------------------
# Delegated operations — data quality
# ---------------------------------------------------------------------------

class TestDelegatedDataQuality:
    """IntelligenceManager delegates to DataQualityChecker."""

    def test_check_data_quality_no_issues(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        mgr = IntelligenceManager.from_config()
        # Single candle — too few for meaningful checks but should not error
        candles = [_make_candle()]
        issues = mgr.check_data_quality(candles, Timeframe.M1)
        assert isinstance(issues, list)

    def test_check_data_quality_raises_without_checker(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        fe_mock = MagicMock()
        mgr = IntelligenceManager(feature_engine=fe_mock, data_qa=None)
        with pytest.raises(RuntimeError, match="No data QA"):
            mgr.check_data_quality([], Timeframe.M1)


# ---------------------------------------------------------------------------
# Package-level import
# ---------------------------------------------------------------------------

class TestPackageImport:
    """IntelligenceManager is importable from package __init__."""

    def test_import_from_package(self):
        from agentic_trading.intelligence import IntelligenceManager

        assert IntelligenceManager is not None

    def test_import_from_module(self):
        from agentic_trading.intelligence.manager import IntelligenceManager

        assert IntelligenceManager is not None

    def test_import_identity(self):
        from agentic_trading.intelligence import (
            IntelligenceManager as PkgIM,
        )
        from agentic_trading.intelligence.manager import (
            IntelligenceManager as ModIM,
        )
        assert PkgIM is ModIM
