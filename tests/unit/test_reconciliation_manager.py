"""Tests for the ReconciliationManager facade and journal/ → reconciliation/ shim layer.

Tests cover:
1. ReconciliationManager.from_config construction and wiring
2. Component accessors
3. Lifecycle (start/stop)
4. Delegated operations
5. Backward-compat shim imports (journal.* → reconciliation.journal.*)
6. Backward-compat shim imports (execution.reconciliation → reconciliation.loop)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_adapter():
    """Create a mock exchange adapter."""
    adapter = MagicMock()
    adapter.get_open_orders = AsyncMock(return_value=[])
    adapter.get_positions = AsyncMock(return_value=[])
    adapter.get_balances = AsyncMock(return_value=[])
    return adapter


def _make_mock_event_bus():
    """Create a mock event bus."""
    bus = MagicMock()
    bus.subscribe = AsyncMock()
    bus.publish = AsyncMock()
    return bus


def _make_mock_order_manager():
    """Create a mock order manager."""
    om = MagicMock()
    om.get_order = MagicMock(return_value=None)
    om.get_active_orders = MagicMock(return_value=[])
    return om


def _make_manager(**kwargs):
    """Build a ReconciliationManager via the factory with mocks."""
    from agentic_trading.reconciliation.manager import ReconciliationManager

    return ReconciliationManager.from_config(
        adapter=kwargs.get("adapter", _make_mock_adapter()),
        event_bus=kwargs.get("event_bus", _make_mock_event_bus()),
        order_manager=kwargs.get("order_manager", _make_mock_order_manager()),
    )


def _make_manager_backtest():
    """Build a ReconciliationManager without adapter (backtest mode)."""
    from agentic_trading.reconciliation.manager import ReconciliationManager

    return ReconciliationManager.from_config()


# ---------------------------------------------------------------------------
# ReconciliationManager factory
# ---------------------------------------------------------------------------

class TestReconciliationManagerFactory:
    """ReconciliationManager.from_config produces a fully wired instance."""

    def test_from_config_creates_manager(self):
        mgr = _make_manager()
        assert mgr is not None
        assert mgr.journal is not None
        assert mgr.rolling_tracker is not None

    def test_from_config_creates_quality_scorecard(self):
        mgr = _make_manager()
        assert mgr.quality_scorecard is not None

    def test_from_config_creates_recon_loop(self):
        mgr = _make_manager()
        assert mgr.recon_loop is not None

    def test_from_config_backtest_no_recon_loop(self):
        mgr = _make_manager_backtest()
        assert mgr.journal is not None
        assert mgr.recon_loop is None

    def test_from_config_custom_rolling_window(self):
        from agentic_trading.reconciliation.manager import ReconciliationManager

        mgr = ReconciliationManager.from_config(rolling_window=50)
        assert mgr.rolling_tracker is not None
        assert mgr.rolling_tracker._window_size == 50


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestReconciliationManagerLifecycle:
    """ReconciliationManager lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_starts_recon_loop(self):
        mgr = _make_manager()
        await mgr.start()
        assert mgr.recon_loop.is_running
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_stop_stops_recon_loop(self):
        mgr = _make_manager()
        await mgr.start()
        await mgr.stop()
        assert not mgr.recon_loop.is_running

    @pytest.mark.asyncio
    async def test_start_stop_backtest_no_error(self):
        mgr = _make_manager_backtest()
        await mgr.start()
        await mgr.stop()
        # Should not raise


# ---------------------------------------------------------------------------
# Component accessors
# ---------------------------------------------------------------------------

class TestComponentAccessors:
    """ReconciliationManager exposes sub-components."""

    def test_journal_accessor(self):
        from agentic_trading.reconciliation.journal.journal import TradeJournal

        mgr = _make_manager()
        assert isinstance(mgr.journal, TradeJournal)

    def test_rolling_tracker_accessor(self):
        from agentic_trading.reconciliation.journal.rolling_tracker import RollingTracker

        mgr = _make_manager()
        assert isinstance(mgr.rolling_tracker, RollingTracker)

    def test_quality_scorecard_accessor(self):
        from agentic_trading.reconciliation.journal.quality_scorecard import QualityScorecard

        mgr = _make_manager()
        assert isinstance(mgr.quality_scorecard, QualityScorecard)

    def test_recon_loop_accessor(self):
        from agentic_trading.reconciliation.loop import ReconciliationLoop

        mgr = _make_manager()
        assert isinstance(mgr.recon_loop, ReconciliationLoop)


# ---------------------------------------------------------------------------
# Delegated operations
# ---------------------------------------------------------------------------

class TestDelegatedOperations:
    """ReconciliationManager delegates to journal."""

    def test_get_open_trades_empty(self):
        mgr = _make_manager()
        assert mgr.get_open_trades() == []

    def test_get_closed_trades_empty(self):
        mgr = _make_manager()
        assert mgr.get_closed_trades() == []

    def test_get_trade_none(self):
        mgr = _make_manager()
        assert mgr.get_trade("nonexistent") is None

    @pytest.mark.asyncio
    async def test_reconcile_delegates(self):
        mgr = _make_manager()
        result = await mgr.reconcile()
        assert result is not None

    @pytest.mark.asyncio
    async def test_reconcile_backtest_returns_none(self):
        mgr = _make_manager_backtest()
        result = await mgr.reconcile()
        assert result is None


# ---------------------------------------------------------------------------
# Journal shim imports (journal.* → reconciliation.journal.*)
# ---------------------------------------------------------------------------

class TestJournalShimImports:
    """Verify all shim imports from journal.* resolve correctly."""

    def test_journal_init_imports(self):
        from agentic_trading.journal import (
            CoinFlipBaseline,
            ConfidenceCalibrator,
            CorrelationMatrix,
            Grade,
            MetricGrade,
            Mistake,
            MistakeDetector,
            MonteCarloProjector,
            OvertradingDetector,
            PortfolioQualityReport,
            QualityReport,
            QualityScorecard,
            RollingTracker,
            SessionAnalyser,
            StrategyType,
            TradeExporter,
            TradeJournal,
            TradeOutcome,
            TradePhase,
            TradeRecord,
            TradeReplayer,
        )
        # Verify identity with canonical classes
        from agentic_trading.reconciliation.journal import (
            TradeRecord as CanonicalTR,
            TradeJournal as CanonicalTJ,
        )
        assert TradeRecord is CanonicalTR
        assert TradeJournal is CanonicalTJ

    def test_record_shim(self):
        from agentic_trading.journal.record import TradeRecord as ShimTR
        from agentic_trading.reconciliation.journal.record import (
            TradeRecord as CanonicalTR,
        )
        assert ShimTR is CanonicalTR

    def test_journal_shim(self):
        from agentic_trading.journal.journal import TradeJournal as ShimTJ
        from agentic_trading.reconciliation.journal.journal import (
            TradeJournal as CanonicalTJ,
        )
        assert ShimTJ is CanonicalTJ

    def test_rolling_tracker_shim(self):
        from agentic_trading.journal.rolling_tracker import (
            RollingTracker as ShimRT,
        )
        from agentic_trading.reconciliation.journal.rolling_tracker import (
            RollingTracker as CanonicalRT,
        )
        assert ShimRT is CanonicalRT

    def test_confidence_shim(self):
        from agentic_trading.journal.confidence import (
            ConfidenceCalibrator as ShimCC,
        )
        from agentic_trading.reconciliation.journal.confidence import (
            ConfidenceCalibrator as CanonicalCC,
        )
        assert ShimCC is CanonicalCC

    def test_monte_carlo_shim(self):
        from agentic_trading.journal.monte_carlo import (
            MonteCarloProjector as ShimMC,
        )
        from agentic_trading.reconciliation.journal.monte_carlo import (
            MonteCarloProjector as CanonicalMC,
        )
        assert ShimMC is CanonicalMC

    def test_overtrading_shim(self):
        from agentic_trading.journal.overtrading import (
            OvertradingDetector as ShimOD,
        )
        from agentic_trading.reconciliation.journal.overtrading import (
            OvertradingDetector as CanonicalOD,
        )
        assert ShimOD is CanonicalOD

    def test_coin_flip_shim(self):
        from agentic_trading.journal.coin_flip import (
            CoinFlipBaseline as ShimCF,
        )
        from agentic_trading.reconciliation.journal.coin_flip import (
            CoinFlipBaseline as CanonicalCF,
        )
        assert ShimCF is CanonicalCF

    def test_mistakes_shim(self):
        from agentic_trading.journal.mistakes import (
            MistakeDetector as ShimMD,
            Mistake as ShimM,
        )
        from agentic_trading.reconciliation.journal.mistakes import (
            MistakeDetector as CanonicalMD,
            Mistake as CanonicalM,
        )
        assert ShimMD is CanonicalMD
        assert ShimM is CanonicalM

    def test_session_analysis_shim(self):
        from agentic_trading.journal.session_analysis import (
            SessionAnalyser as ShimSA,
        )
        from agentic_trading.reconciliation.journal.session_analysis import (
            SessionAnalyser as CanonicalSA,
        )
        assert ShimSA is CanonicalSA

    def test_correlation_shim(self):
        from agentic_trading.journal.correlation import (
            CorrelationMatrix as ShimCM,
        )
        from agentic_trading.reconciliation.journal.correlation import (
            CorrelationMatrix as CanonicalCM,
        )
        assert ShimCM is CanonicalCM

    def test_replay_shim(self):
        from agentic_trading.journal.replay import (
            TradeReplayer as ShimTRep,
        )
        from agentic_trading.reconciliation.journal.replay import (
            TradeReplayer as CanonicalTRep,
        )
        assert ShimTRep is CanonicalTRep

    def test_export_shim(self):
        from agentic_trading.journal.export import (
            TradeExporter as ShimTE,
        )
        from agentic_trading.reconciliation.journal.export import (
            TradeExporter as CanonicalTE,
        )
        assert ShimTE is CanonicalTE

    def test_quality_scorecard_shim(self):
        from agentic_trading.journal.quality_scorecard import (
            QualityScorecard as ShimQS,
            QualityReport as ShimQR,
            PortfolioQualityReport as ShimPQR,
            MetricGrade as ShimMG,
            Grade as ShimG,
            StrategyType as ShimST,
        )
        from agentic_trading.reconciliation.journal.quality_scorecard import (
            QualityScorecard as CanonicalQS,
            Grade as CanonicalG,
        )
        assert ShimQS is CanonicalQS
        assert ShimG is CanonicalG


# ---------------------------------------------------------------------------
# Execution reconciliation shim
# ---------------------------------------------------------------------------

class TestExecutionReconciliationShim:
    """Verify execution.reconciliation shim resolves correctly."""

    def test_reconciliation_loop_shim(self):
        from agentic_trading.execution.reconciliation import (
            ReconciliationLoop as ShimRL,
        )
        from agentic_trading.reconciliation.loop import (
            ReconciliationLoop as CanonicalRL,
        )
        assert ShimRL is CanonicalRL
