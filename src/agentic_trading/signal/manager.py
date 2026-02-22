"""Unified SignalManager facade for the signal layer.

Composes the :class:`StrategyRunner`, :class:`PortfolioManager`,
:class:`PortfolioAllocator`, :class:`CorrelationRiskAnalyzer`, and the
:func:`build_order_intents` converter into a single entry point that
bridges feature vectors through to order intents.

Usage::

    from agentic_trading.signal.manager import SignalManager

    mgr = SignalManager.from_config(
        strategy_ids=["trend_following", "mean_reversion"],
        feature_engine=feature_engine,
        event_bus=event_bus,
    )
    await mgr.start(ctx)

    # Backtest / manual mode
    targets = mgr.generate_targets(ctx, capital=100_000)
    intents = mgr.build_intents(targets, Exchange.BYBIT, now)

Signal processing::

    result = mgr.process_signal(sig, journal, ctx, exchange, capital)
    # result.intents — order intents to publish
    # result.is_exit — True when a FLAT signal generated an exit intent
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from agentic_trading.core.ids import content_hash

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal processing result
# ---------------------------------------------------------------------------

@dataclass
class SignalResult:
    """Structured result returned by :meth:`SignalManager.process_signal`.

    Carries enough context for callers (e.g. main.py) to perform follow-up
    actions such as narration and metrics emission without re-deriving
    entry-vs-exit classification.
    """

    intents: list = field(default_factory=list)
    """Order intents generated (entry or exit)."""

    is_exit: bool = False
    """Whether the signal was a FLAT signal that triggered an exit."""

    exit_trace_id: str | None = None
    """The exit intent's trace_id (present only for FLAT signals)."""

    entry_trace_id: str | None = None
    """The original entry trade's trace_id (present only for FLAT exits)."""


class SignalManager:
    """Unified facade for the signal layer.

    Owns strategy dispatch (StrategyRunner), position sizing
    (PortfolioManager), allocation constraints (PortfolioAllocator),
    correlation analysis (CorrelationRiskAnalyzer), and the intent
    converter.  Provides a clean lifecycle (start) and accessor
    properties for each sub-component.

    Parameters
    ----------
    runner:
        StrategyRunner that dispatches feature vectors to strategies.
    portfolio_manager:
        PortfolioManager for signal aggregation and sizing.
    allocator:
        PortfolioAllocator for portfolio-level position constraints.
    correlation_analyzer:
        CorrelationRiskAnalyzer for cross-asset correlation tracking.
    """

    def __init__(
        self,
        runner: Any,
        portfolio_manager: Any,
        allocator: Any | None = None,
        correlation_analyzer: Any | None = None,
        bounds_calculator: Any | None = None,
    ) -> None:
        self._runner = runner
        self._portfolio_manager = portfolio_manager
        self._allocator = allocator
        self._correlation_analyzer = correlation_analyzer
        self._bounds_calculator = bounds_calculator

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        strategy_ids: list[str] | None = None,
        strategies: list | None = None,
        feature_engine: Any | None = None,
        event_bus: Any | None = None,
        on_signal: Callable | None = None,
        max_position_pct: float = 0.10,
        max_gross_exposure_pct: float = 1.0,
        sizing_multiplier: float = 1.0,
        governance_sizing_fn: Callable[[str], float] | None = None,
        max_single_position_pct: float = 0.10,
        max_correlated_exposure_pct: float = 0.25,
        correlation_lookback: int = 60,
        correlation_threshold: float = 0.7,
        max_concurrent_positions: int = 8,
        max_daily_entries: int = 10,
        max_leverage: float = 3.0,
    ) -> SignalManager:
        """Build a fully wired SignalManager from configuration.

        Parameters
        ----------
        strategy_ids:
            Strategy IDs to instantiate from the registry.  Mutually
            exclusive with *strategies*.
        strategies:
            Pre-built strategy instances.  Mutually exclusive with
            *strategy_ids*.
        feature_engine:
            Feature engine providing candle buffers.  Required for
            ``StrategyRunner``.
        event_bus:
            Event bus for subscribe/publish.  Required for
            ``StrategyRunner``.
        on_signal:
            Optional callback ``(signal, elapsed_seconds) -> None``
            invoked after each signal is produced.
        max_position_pct:
            Maximum single-position size as fraction of capital
            (default 0.10).
        max_gross_exposure_pct:
            Maximum gross portfolio exposure (default 1.0).
        sizing_multiplier:
            Global sizing multiplier (default 1.0).
        governance_sizing_fn:
            Optional function ``strategy_id -> float`` for governance-
            based sizing adjustments.
        max_single_position_pct:
            Allocator single-position limit (default 0.10).
        max_correlated_exposure_pct:
            Allocator correlated-cluster limit (default 0.25).
        correlation_lookback:
            Rolling window for correlation computation (default 60).
        correlation_threshold:
            Threshold for clustering correlated assets (default 0.7).

        Returns
        -------
        SignalManager
        """
        # --- Resolve strategies ---
        strat_list: list = []
        if strategies is not None:
            strat_list = list(strategies)
        elif strategy_ids:
            from agentic_trading.signal.strategies.registry import create_strategy

            strat_list = [create_strategy(sid) for sid in strategy_ids]

        # --- Strategy runner (requires feature engine + event bus) ---
        runner = None
        if feature_engine is not None and event_bus is not None:
            from agentic_trading.signal.runner import StrategyRunner

            runner = StrategyRunner(
                strategies=strat_list,
                feature_engine=feature_engine,
                event_bus=event_bus,
                on_signal=on_signal,
            )

        # --- Portfolio manager ---
        from agentic_trading.signal.portfolio.manager import PortfolioManager

        portfolio_manager = PortfolioManager(
            max_position_pct=max_position_pct,
            max_gross_exposure_pct=max_gross_exposure_pct,
            sizing_multiplier=sizing_multiplier,
            governance_sizing_fn=governance_sizing_fn,
        )

        # --- Portfolio allocator ---
        from agentic_trading.signal.portfolio.allocator import PortfolioAllocator

        allocator = PortfolioAllocator(
            max_gross_exposure_pct=max_gross_exposure_pct,
            max_single_position_pct=max_single_position_pct,
            max_correlated_exposure_pct=max_correlated_exposure_pct,
        )

        # --- Correlation analyzer ---
        from agentic_trading.signal.portfolio.correlation_risk import (
            CorrelationRiskAnalyzer,
        )

        correlation_analyzer = CorrelationRiskAnalyzer(
            lookback_periods=correlation_lookback,
            correlation_threshold=correlation_threshold,
        )

        # --- Position bounds calculator ---
        from agentic_trading.signal.portfolio.bounds import (
            PositionBoundsCalculator,
        )

        bounds_calculator = PositionBoundsCalculator(
            max_single_position_pct=max_single_position_pct,
            max_concurrent_positions=max_concurrent_positions,
            max_daily_entries=max_daily_entries,
            max_leverage=max_leverage,
            max_correlated_exposure_pct=max_correlated_exposure_pct,
        )

        return cls(
            runner=runner,
            portfolio_manager=portfolio_manager,
            allocator=allocator,
            correlation_analyzer=correlation_analyzer,
            bounds_calculator=bounds_calculator,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, ctx: Any) -> None:
        """Start the strategy runner (subscribe to feature.vector).

        Parameters
        ----------
        ctx:
            ``TradingContext`` passed through to the runner.
        """
        if self._runner is not None:
            await self._runner.start(ctx)
        logger.info("SignalManager started")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def runner(self):
        """The strategy runner (``None`` if no event bus provided)."""
        return self._runner

    @property
    def portfolio_manager(self):
        """The portfolio manager for signal aggregation and sizing."""
        return self._portfolio_manager

    @property
    def allocator(self):
        """The portfolio allocator for position constraints."""
        return self._allocator

    @property
    def correlation_analyzer(self):
        """The correlation risk analyzer."""
        return self._correlation_analyzer

    @property
    def strategies(self) -> list:
        """Registered strategies (from runner, or empty)."""
        if self._runner is not None:
            return self._runner.strategies
        return []

    @property
    def signal_count(self) -> int:
        """Total signals produced by the runner since start."""
        if self._runner is not None:
            return self._runner.signal_count
        return 0

    @property
    def bounds_calculator(self):
        """The position bounds calculator (or ``None``)."""
        return self._bounds_calculator

    def compute_bounds(self, ctx: Any, capital: float, prices: dict | None = None) -> Any:
        """Pre-compute position bounds for all symbols in context.

        Delegates to ``PositionBoundsCalculator.compute()``.
        Returns ``None`` if no calculator is configured.
        """
        if self._bounds_calculator is None:
            return None

        symbols = list(ctx.instruments.keys()) if hasattr(ctx, "instruments") else []
        if not symbols:
            return None

        portfolio = ctx.portfolio_state if hasattr(ctx, "portfolio_state") else None
        if portfolio is None:
            from agentic_trading.core.interfaces import PortfolioState
            portfolio = PortfolioState()

        clusters = None
        if self._correlation_analyzer is not None:
            clusters = self._correlation_analyzer.find_clusters()

        return self._bounds_calculator.compute(
            symbols=symbols,
            portfolio=portfolio,
            capital=capital,
            instruments=ctx.instruments if hasattr(ctx, "instruments") else None,
            prices=prices or {},
            correlation_clusters=clusters,
        )

    # ------------------------------------------------------------------
    # Delegated operations — signal collection
    # ------------------------------------------------------------------

    def on_signal(self, signal) -> None:
        """Forward a signal to the portfolio manager.

        Delegates to ``portfolio_manager.on_signal()``.
        """
        self._portfolio_manager.on_signal(signal)

    # ------------------------------------------------------------------
    # Delegated operations — target generation
    # ------------------------------------------------------------------

    def generate_targets(self, ctx, capital: float) -> list:
        """Generate target positions from pending signals.

        Delegates to ``portfolio_manager.generate_targets()``.
        """
        return self._portfolio_manager.generate_targets(ctx, capital)

    def allocate(
        self,
        targets: list,
        portfolio,
        capital: float,
    ) -> list:
        """Apply portfolio-level constraints to target positions.

        Delegates to ``allocator.allocate()``.  Optionally incorporates
        correlation clusters from the correlation analyzer.
        """
        if self._allocator is None:
            return targets

        clusters = None
        if self._correlation_analyzer is not None:
            clusters = self._correlation_analyzer.find_clusters()

        return self._allocator.allocate(
            targets, portfolio, capital, correlation_clusters=clusters,
        )

    # ------------------------------------------------------------------
    # Delegated operations — intent conversion
    # ------------------------------------------------------------------

    @staticmethod
    def build_intents(
        targets: list,
        exchange,
        timestamp: datetime,
        order_type=None,
        instruments: dict | None = None,
    ) -> list:
        """Convert target positions into order intents.

        Delegates to ``build_order_intents()``.
        """
        from agentic_trading.signal.portfolio.intent_converter import (
            build_order_intents,
        )

        kwargs: dict[str, Any] = {
            "targets": targets,
            "exchange": exchange,
            "timestamp": timestamp,
        }
        if order_type is not None:
            kwargs["order_type"] = order_type
        if instruments is not None:
            kwargs["instruments"] = instruments
        return build_order_intents(**kwargs)

    # ------------------------------------------------------------------
    # Delegated operations — correlation
    # ------------------------------------------------------------------

    def update_returns(self, symbol: str, ret: float) -> None:
        """Feed a return observation to the correlation analyzer.

        Delegates to ``correlation_analyzer.update_returns()``.
        """
        if self._correlation_analyzer is not None:
            self._correlation_analyzer.update_returns(symbol, ret)

    def find_correlation_clusters(self) -> list[list[str]]:
        """Find correlated asset clusters.

        Delegates to ``correlation_analyzer.find_clusters()``.
        """
        if self._correlation_analyzer is not None:
            return self._correlation_analyzer.find_clusters()
        return []

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def set_sizing_multiplier(self, multiplier: float) -> None:
        """Update the portfolio manager's global sizing multiplier."""
        self._portfolio_manager.set_sizing_multiplier(multiplier)

    # ------------------------------------------------------------------
    # Signal processing pipeline
    # ------------------------------------------------------------------

    def process_signal(
        self,
        signal: Any,
        journal: Any,
        ctx: Any,
        exchange: Any,
        capital: float,
        *,
        signal_cache: dict[str, Any] | None = None,
        exit_map: dict[str, str] | None = None,
    ) -> SignalResult:
        """Convert a strategy signal into order intents.

        Encapsulates the signal→portfolio→intent pipeline that was
        previously inline in ``main._run_live_or_paper.on_feature_vector``.
        Handles both directional entries and FLAT exits.

        This method does **not** publish intents to the event bus, emit
        Prometheus metrics, or trigger narration — those remain in the
        caller.

        Parameters
        ----------
        signal:
            A strategy ``Signal`` object.
        journal:
            ``TradeJournal`` (or ``ReconciliationManager``) for looking
            up open trades when processing FLAT exits.
        ctx:
            ``TradingContext`` providing the clock.
        exchange:
            ``Exchange`` enum value for intent routing.
        capital:
            Current capital for position sizing.
        signal_cache:
            Optional dict to populate with ``trace_id → signal``
            mapping for downstream fill-time lookups.
        exit_map:
            Optional dict to populate with ``exit_trace_id →
            entry_trace_id`` for exit fill classification.

        Returns
        -------
        SignalResult
            Structured result with generated intents and classification.
        """
        result = SignalResult()

        # Populate signal cache if provided
        if signal_cache is not None:
            signal_cache[signal.trace_id] = signal

        if signal.direction.value == "flat":
            # FLAT signal = exit any open position for this strategy+symbol
            _journal = (
                journal.journal
                if hasattr(journal, "journal")
                else journal
            )
            open_trade = _journal.get_trade_by_position(
                signal.strategy_id, signal.symbol,
            )
            if open_trade is not None and open_trade.entry_fills:
                from agentic_trading.core.enums import (
                    AssetClass,
                    OrderType,
                    QtyUnit,
                    Side,
                    TimeInForce,
                )
                from agentic_trading.core.events import OrderIntent

                exit_side = (
                    Side.SELL
                    if open_trade.direction == "long"
                    else Side.BUY
                )
                exit_qty = open_trade.remaining_qty
                if exit_qty > Decimal("0"):
                    ts_bucket = int(ctx.clock.now().timestamp()) // 60
                    raw_key = (
                        f"exit:{signal.strategy_id}:"
                        f"{signal.symbol}:{ts_bucket}"
                    )
                    dedupe_key = content_hash(raw_key)

                    # Enrich with instrument metadata
                    _inst = ctx.instruments.get(signal.symbol) if hasattr(ctx, "instruments") else None
                    _asset_class = _inst.asset_class if _inst is not None else AssetClass.CRYPTO
                    _inst_hash = _inst.instrument_hash if _inst is not None else ""

                    exit_intent = OrderIntent(
                        dedupe_key=dedupe_key,
                        strategy_id=signal.strategy_id,
                        symbol=signal.symbol,
                        exchange=exchange,
                        side=exit_side,
                        order_type=OrderType.MARKET,
                        time_in_force=TimeInForce.GTC,
                        qty=exit_qty,
                        price=None,
                        reduce_only=True,
                        trace_id=signal.trace_id,
                        asset_class=_asset_class,
                        qty_unit=QtyUnit.BASE,
                        instrument_hash=_inst_hash,
                    )

                    result.intents = [exit_intent]
                    result.is_exit = True
                    result.exit_trace_id = signal.trace_id
                    result.entry_trace_id = open_trade.trace_id

                    if exit_map is not None:
                        exit_map[signal.trace_id] = open_trade.trace_id

                    if signal_cache is not None:
                        signal_cache[signal.trace_id] = signal
        else:
            # Directional signal → entry via portfolio manager
            self._portfolio_manager.on_signal(signal)
            targets = self._portfolio_manager.generate_targets(ctx, capital)
            if targets:
                _instruments = ctx.instruments if hasattr(ctx, "instruments") else None
                result.intents = self.build_intents(
                    targets, exchange, ctx.clock.now(),
                    instruments=_instruments,
                )

        return result

    def process_signal_batch(
        self,
        signals: list[Any],
        journal: Any,
        ctx: Any,
        exchange: Any,
        capital: float,
        *,
        signal_cache: dict[str, Any] | None = None,
        exit_map: dict[str, str] | None = None,
    ) -> list[SignalResult]:
        """Process a batch of directional signals as a group.

        Unlike :meth:`process_signal` which handles one signal at a time,
        this method feeds ALL signals to the PortfolioManager before
        calling ``generate_targets()``.  This enables proper confidence-
        weighted voting when multiple strategies fire on the same symbol.

        **FLAT signals should NOT be included** — process those individually
        via :meth:`process_signal` before calling this method.

        Parameters
        ----------
        signals:
            List of directional ``Signal`` objects (LONG or SHORT only).
        journal:
            Trade journal (unused for directional batch, kept for API
            compatibility).
        ctx:
            ``TradingContext`` providing the clock and portfolio state.
        exchange:
            ``Exchange`` enum value for intent routing.
        capital:
            Current capital for position sizing.
        signal_cache:
            Optional dict to populate with ``trace_id → signal``
            mapping for downstream fill-time lookups.
        exit_map:
            Optional dict (unused for directional signals).

        Returns
        -------
        list[SignalResult]
            One ``SignalResult`` per emitted intent (may be fewer than
            input signals due to conflict resolution and suppression).
        """
        results: list[SignalResult] = []

        if not signals:
            return results

        # Populate signal cache for all signals in batch
        if signal_cache is not None:
            for sig in signals:
                signal_cache[sig.trace_id] = sig

        # Feed ALL directional signals to portfolio manager at once
        for sig in signals:
            self._portfolio_manager.on_signal(sig)

        # Generate targets — PortfolioManager now sees the full picture
        # and can do confidence-weighted voting across strategies
        targets = self._portfolio_manager.generate_targets(ctx, capital)

        if targets:
            _instruments = ctx.instruments if hasattr(ctx, "instruments") else None
            intents = self.build_intents(
                targets, exchange, ctx.clock.now(),
                instruments=_instruments,
            )
            # Create one SignalResult per intent
            for intent in intents:
                result = SignalResult()
                result.intents = [intent]
                results.append(result)

        return results

    # ------------------------------------------------------------------
    # Feature aliasing
    # ------------------------------------------------------------------

    _FEATURE_ALIASES: dict[str, str] = {
        "adx_14": "adx",
        "atr_14": "atr",
        "rsi_14": "rsi",
        "donchian_upper_20": "donchian_upper",
        "donchian_lower_20": "donchian_lower",
    }

    @staticmethod
    def alias_features(features: dict[str, Any]) -> dict[str, Any]:
        """Create a copy of *features* with canonical indicator aliases.

        Maps versioned indicator keys (e.g. ``adx_14``) to short aliases
        (``adx``) expected by strategies, without overwriting existing
        short-name keys.

        Parameters
        ----------
        features:
            Raw feature dict from a ``FeatureVector``.

        Returns
        -------
        dict[str, Any]
            Aliased copy (original is not mutated).
        """
        aliased = dict(features)
        for long_name, short_name in SignalManager._FEATURE_ALIASES.items():
            if long_name in aliased and short_name not in aliased:
                aliased[short_name] = aliased[long_name]
        return aliased
