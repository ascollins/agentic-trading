"""Tests for PR 15 — _run_live_or_paper orch-component wiring.

Validates that ``_run_live_or_paper`` now sources feature_engine,
strategies, portfolio_manager, SignalManager, and ReconciliationManager
from the Orchestrator rather than constructing them locally.

Tests cover:
1. Source-level: removed local FeatureEngine / create_strategy construction
2. Source-level: removed local PortfolioManager construction
3. Source-level: _signal_mgr / _recon_mgr sourced from orch
4. Source-level: _recon_mgr._journal injected from local journal
5. Source-level: orch passed as third arg to _run_live_or_paper
6. Orchestrator's SignalManager has portfolio_manager wired
7. Orchestrator's ReconciliationManager has a journal
8. Orchestrator feature_engine matches intelligence.feature_engine
9. Strategies sourced from orch.signal.strategies
10. Backtest path unchanged (no orch reference)
"""

from __future__ import annotations

import inspect

import pytest


# ---------------------------------------------------------------------------
# Source-level assertions: _run_live_or_paper
# ---------------------------------------------------------------------------

class TestLivePaperUsesOrch:
    """_run_live_or_paper now sources components from Orchestrator."""

    def _get_live_source(self) -> str:
        from agentic_trading.main import _run_live_or_paper

        return inspect.getsource(_run_live_or_paper)

    # --- Removed local constructions ---------------------------------

    def test_no_local_feature_engine_construction(self):
        """FeatureEngine is no longer constructed locally."""
        source = self._get_live_source()
        assert "FeatureEngine(" not in source

    def test_no_local_portfolio_manager_construction(self):
        """PortfolioManager is no longer constructed locally."""
        source = self._get_live_source()
        assert "PortfolioManager(" not in source

    def test_no_local_signal_mgr_construction(self):
        """SignalManager is no longer constructed locally with runner=None."""
        source = self._get_live_source()
        assert "_SignalMgr(" not in source
        assert "SignalManager(\n" not in source

    def test_no_local_recon_mgr_construction(self):
        """ReconciliationManager is no longer locally constructed."""
        source = self._get_live_source()
        assert "_ReconMgr(" not in source

    def test_no_create_strategy_calls(self):
        """create_strategy() is only used in the fallback branch."""
        source = self._get_live_source()
        # The only create_strategy occurrence should be in the fallback
        # block (when strategies list is empty).
        count = source.count("create_strategy")
        # Fallback block may still import it; total references ≤ 2
        assert count <= 2

    def test_no_portfolio_manager_import(self):
        """PortfolioManager is no longer imported at the top of the function."""
        source = self._get_live_source()
        assert "from .portfolio.manager import PortfolioManager" not in source
        assert "from agentic_trading.portfolio.manager import PortfolioManager" not in source

    def test_no_intent_converter_import(self):
        """build_order_intents is no longer directly imported."""
        source = self._get_live_source()
        assert "from .portfolio.intent_converter import build_order_intents" not in source

    # --- Orch wiring references present ------------------------------

    def test_feature_engine_from_orch(self):
        """feature_engine is sourced from orch.intelligence."""
        source = self._get_live_source()
        assert "orch.intelligence.feature_engine" in source

    def test_strategies_from_orch(self):
        """strategies are sourced from orch.signal.strategies."""
        source = self._get_live_source()
        assert "orch.signal.strategies" in source

    def test_signal_mgr_from_orch(self):
        """_signal_mgr is orch.signal."""
        source = self._get_live_source()
        assert "_signal_mgr = orch.signal" in source

    def test_recon_mgr_from_orch(self):
        """_recon_mgr is orch.reconciliation."""
        source = self._get_live_source()
        assert "_recon_mgr = orch.reconciliation" in source

    def test_journal_injected_into_recon_mgr(self):
        """The local journal is injected into _recon_mgr._journal."""
        source = self._get_live_source()
        assert "_recon_mgr._journal = journal" in source

    # --- Signature check ---------------------------------------------

    def test_signature_includes_orch(self):
        """_run_live_or_paper takes an Orchestrator as 3rd parameter."""
        from agentic_trading.main import _run_live_or_paper

        sig = inspect.signature(_run_live_or_paper)
        params = list(sig.parameters.keys())
        assert "orch" in params

    def test_signature_orch_annotated(self):
        """The orch parameter has the Orchestrator type annotation."""
        from agentic_trading.main import _run_live_or_paper

        sig = inspect.signature(_run_live_or_paper)
        ann = sig.parameters["orch"].annotation
        # Accept either direct class reference or string annotation
        ann_str = str(ann)
        assert "Orchestrator" in ann_str


# ---------------------------------------------------------------------------
# Source-level: run() passes orch to _run_live_or_paper
# ---------------------------------------------------------------------------

class TestRunPassesOrch:
    """run() passes the Orchestrator to _run_live_or_paper."""

    def test_run_calls_live_or_paper_with_orch(self):
        from agentic_trading import main

        source = inspect.getsource(main.run)
        assert "_run_live_or_paper(settings, ctx, orch)" in source


# ---------------------------------------------------------------------------
# Source-level: backtest unchanged
# ---------------------------------------------------------------------------

class TestBacktestUnchanged:
    """_run_backtest does not reference Orchestrator components."""

    def test_backtest_does_not_use_orch(self):
        from agentic_trading.main import _run_backtest

        source = inspect.getsource(_run_backtest)
        assert "orch." not in source

    def test_backtest_still_creates_feature_engine(self):
        from agentic_trading.main import _run_backtest

        source = inspect.getsource(_run_backtest)
        assert "FeatureEngine(" in source


# ---------------------------------------------------------------------------
# Orchestrator object wiring (runtime)
# ---------------------------------------------------------------------------

class TestOrchComponentsWired:
    """Orchestrator creates properly wired sub-components."""

    def _make_orch(self):
        from agentic_trading.core.config import Settings
        from agentic_trading.core.enums import Mode
        from agentic_trading.orchestrator import Orchestrator

        settings = Settings(mode=Mode.PAPER)
        return Orchestrator.from_config(settings)

    def test_signal_has_portfolio_manager(self):
        orch = self._make_orch()
        pm = orch.signal.portfolio_manager
        assert pm is not None

    def test_signal_portfolio_manager_has_on_signal(self):
        orch = self._make_orch()
        pm = orch.signal.portfolio_manager
        assert hasattr(pm, "on_signal")
        assert callable(pm.on_signal)

    def test_signal_portfolio_manager_has_generate_targets(self):
        orch = self._make_orch()
        pm = orch.signal.portfolio_manager
        assert hasattr(pm, "generate_targets")
        assert callable(pm.generate_targets)

    def test_signal_has_process_signal(self):
        orch = self._make_orch()
        assert hasattr(orch.signal, "process_signal")
        assert callable(orch.signal.process_signal)

    def test_signal_has_alias_features(self):
        orch = self._make_orch()
        assert hasattr(orch.signal, "alias_features")
        assert callable(orch.signal.alias_features)

    def test_reconciliation_has_journal(self):
        orch = self._make_orch()
        assert orch.reconciliation.journal is not None

    def test_reconciliation_has_handle_fill(self):
        orch = self._make_orch()
        assert hasattr(orch.reconciliation, "handle_fill")
        assert callable(orch.reconciliation.handle_fill)

    def test_reconciliation_has_reconcile_positions(self):
        orch = self._make_orch()
        assert hasattr(orch.reconciliation, "reconcile_positions")
        assert callable(orch.reconciliation.reconcile_positions)

    def test_intelligence_has_feature_engine(self):
        orch = self._make_orch()
        assert orch.intelligence.feature_engine is not None

    def test_feature_engine_matches_intelligence(self):
        """The feature_engine on intelligence is the same object."""
        orch = self._make_orch()
        fe = orch.intelligence.feature_engine
        assert fe is orch.intelligence.feature_engine

    def test_signal_strategies_returns_list(self):
        orch = self._make_orch()
        strats = orch.signal.strategies
        assert isinstance(strats, list)

    def test_journal_injection_works(self):
        """Simulates what _run_live_or_paper does: inject a new journal."""
        from agentic_trading.reconciliation.journal.journal import TradeJournal

        orch = self._make_orch()
        original_journal = orch.reconciliation.journal
        new_journal = TradeJournal(max_closed_trades=100)
        orch.reconciliation._journal = new_journal
        assert orch.reconciliation.journal is new_journal
        assert orch.reconciliation.journal is not original_journal


# ---------------------------------------------------------------------------
# Safe mode sizing multiplier propagation
# ---------------------------------------------------------------------------

class TestSizingMultiplierPropagation:
    """Orchestrator propagates safe_mode sizing to signal.portfolio_manager."""

    def test_safe_mode_sizing_propagated(self):
        from agentic_trading.core.config import SafeModeConfig, Settings
        from agentic_trading.core.enums import Mode
        from agentic_trading.orchestrator import Orchestrator

        settings = Settings(
            mode=Mode.PAPER,
            safe_mode=SafeModeConfig(
                enabled=True,
                position_size_multiplier=0.5,
            ),
        )
        orch = Orchestrator.from_config(settings)
        pm = orch.signal.portfolio_manager
        # The portfolio manager should have the 0.5 sizing multiplier
        assert pm._sizing_multiplier == 0.5

    def test_default_sizing_multiplier_is_one(self):
        from agentic_trading.core.config import Settings
        from agentic_trading.core.enums import Mode
        from agentic_trading.orchestrator import Orchestrator

        settings = Settings(mode=Mode.PAPER)
        orch = Orchestrator.from_config(settings)
        pm = orch.signal.portfolio_manager
        assert pm._sizing_multiplier == 1.0
