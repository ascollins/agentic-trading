"""Application bootstrap and mode routing.

This is the main entry point that wires together all modules
and starts the trading loop for the selected mode.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Any

from .core.clock import SimClock, WallClock
from .core.config import Settings, load_settings
from .core.enums import Exchange, Mode, Timeframe
from .core.interfaces import PortfolioState, TradingContext
from .event_bus.bus import create_event_bus

logger = logging.getLogger(__name__)


async def run(
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> None:
    """Main entry point. Load config, validate safety, wire modules, run."""

    # 1. Load settings
    settings = load_settings(config_path=config_path, overrides=overrides)

    # 2. Safety gates
    settings.validate_live_mode()

    # 3. Set up logging
    _setup_logging(settings)

    logger.info(
        "Starting agentic-trading",
        extra={
            "mode": settings.mode.value,
            "read_only": settings.read_only,
            "safe_mode": settings.safe_mode.enabled,
        },
    )

    # 4. Create clock
    if settings.mode == Mode.BACKTEST:
        clock = SimClock()
    else:
        clock = WallClock()

    # 5. Create event bus
    event_bus = create_event_bus(settings.mode, settings.redis_url)

    # 6. Build trading context
    ctx = TradingContext(
        clock=clock,
        event_bus=event_bus,
        instruments={},  # Populated after exchange metadata fetch
        portfolio_state=PortfolioState(),
        risk_limits=settings.risk.model_dump(),
    )

    # 7. Start event bus
    await event_bus.start()

    # 7.5 Start Prometheus metrics server (all modes)
    try:
        from .observability.metrics import start_metrics_server

        metrics_port = settings.observability.metrics_port
        start_metrics_server(port=metrics_port, mode=settings.mode.value)
        logger.info("Prometheus metrics server started on port %d", metrics_port)
    except Exception:
        logger.warning("Failed to start metrics server", exc_info=True)

    # 8. Route to mode-specific loop
    try:
        if settings.mode == Mode.BACKTEST:
            await _run_backtest(settings, ctx)
        else:
            await _run_live_or_paper(settings, ctx)
    except KeyboardInterrupt:
        logger.info("Shutting down (keyboard interrupt)")
    finally:
        await event_bus.stop()
        logger.info("Shutdown complete")


async def _run_backtest(settings: Settings, ctx: TradingContext) -> None:
    """Run backtest mode with BacktestEngine."""
    from .backtester.engine import BacktestEngine
    from .data.historical import HistoricalDataLoader
    from .features.engine import FeatureEngine
    from .strategies.registry import create_strategy

    # Import strategy modules to trigger @register_strategy decorators
    import agentic_trading.strategies.trend_following  # noqa: F401
    import agentic_trading.strategies.mean_reversion  # noqa: F401
    import agentic_trading.strategies.breakout  # noqa: F401
    import agentic_trading.strategies.funding_arb  # noqa: F401

    bt = settings.backtest
    logger.info("Backtest mode: %s to %s", bt.start_date, bt.end_date)

    # Create strategies
    strategies = []
    for strat_cfg in settings.strategies:
        if strat_cfg.enabled:
            strategy = create_strategy(strat_cfg.strategy_id, strat_cfg.params)
            strategies.append(strategy)
            logger.info("Loaded strategy: %s", strat_cfg.strategy_id)

    # Fallback: if no strategies configured, use trend_following with defaults
    if not strategies:
        from .strategies.trend_following import TrendFollowingStrategy
        strategies = [TrendFollowingStrategy()]
        logger.info("No strategies configured, using default trend_following")

    # Create feature engine
    feature_engine = FeatureEngine()

    # Load historical data
    loader = HistoricalDataLoader(data_dir=bt.data_dir)
    symbols = settings.symbols.symbols or ["BTC/USDT"]
    exchange = Exchange.BINANCE

    start_dt = datetime.strptime(bt.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(bt.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    candles_by_symbol: dict[str, list] = {}
    for symbol in symbols:
        candles = loader.load_candles(
            exchange=exchange,
            symbol=symbol,
            timeframe=Timeframe.M1,
            start=start_dt,
            end=end_dt,
        )
        if candles:
            candles_by_symbol[symbol] = candles
            logger.info("Loaded %d candles for %s", len(candles), symbol)
        else:
            logger.warning("No data found for %s", symbol)

    if not candles_by_symbol:
        logger.error("No historical data found. Run download_historical.py first.")
        return

    # Create and run backtest engine
    engine = BacktestEngine(
        strategies=strategies,
        feature_engine=feature_engine,
        initial_capital=bt.initial_capital,
        slippage_model=bt.slippage_model,
        slippage_bps=bt.slippage_bps,
        fee_maker=bt.fee_maker,
        fee_taker=bt.fee_taker,
        funding_enabled=bt.funding_enabled,
        partial_fills=bt.partial_fills,
        latency_ms=bt.latency_ms,
        seed=bt.random_seed,
    )

    result = await engine.run(candles_by_symbol)

    # Print results
    summary = result.summary()
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key:30s}: {value!s:>12}")
    print("=" * 60)
    print(f"  {'deterministic_hash':30s}: {result.deterministic_hash:>12}")
    print()


async def _run_live_or_paper(settings: Settings, ctx: TradingContext) -> None:
    """Run paper or live trading loop.

    Wires: FeedManager -> FeatureEngine -> Strategies -> Execution.
    For paper mode, uses PaperAdapter for simulated fills.
    """
    from decimal import Decimal

    from .features.engine import FeatureEngine
    from .strategies.registry import create_strategy

    # Import strategy modules to trigger @register_strategy decorators
    import agentic_trading.strategies.trend_following  # noqa: F401
    import agentic_trading.strategies.mean_reversion  # noqa: F401
    import agentic_trading.strategies.breakout  # noqa: F401
    import agentic_trading.strategies.funding_arb  # noqa: F401

    logger.info("Starting %s trading loop", settings.mode.value)

    # Import metrics helpers (server already started in run())
    try:
        from .observability.metrics import (
            record_signal,
            record_candle_processed,
            update_equity,
            record_journal_trade,
            update_journal_rolling_metrics,
            update_journal_counts,
            update_journal_confidence,
            update_journal_overtrading,
            update_journal_edge,
            update_journal_monte_carlo,
            record_journal_mistake,
            update_journal_mistake_impact,
            update_journal_session_metrics,
            update_journal_correlation,
            update_journal_best_session,
        )
    except Exception:
        pass

    # Set up graceful shutdown
    stop_event = asyncio.Event()
    feed_manager = None

    def _signal_handler() -> None:
        logger.info("Received shutdown signal")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # Create feature engine with event bus integration
    feature_engine = FeatureEngine(event_bus=ctx.event_bus)
    await feature_engine.start()

    # Create strategies
    strategies = []
    for strat_cfg in settings.strategies:
        if strat_cfg.enabled:
            strategy = create_strategy(strat_cfg.strategy_id, strat_cfg.params)
            strategies.append(strategy)
            logger.info("Loaded strategy: %s", strat_cfg.strategy_id)

    if not strategies:
        from .strategies.trend_following import TrendFollowingStrategy
        strategies = [TrendFollowingStrategy()]
        logger.info("No strategies configured, using default trend_following")

    # Wire governance framework (if enabled)
    governance_gate = None
    governance_canary = None
    if settings.governance.enabled:
        from .governance.gate import GovernanceGate
        from .governance.maturity import MaturityManager
        from .governance.health_score import HealthTracker
        from .governance.impact_classifier import ImpactClassifier
        from .governance.drift_detector import DriftDetector
        from .governance.tokens import TokenManager
        from .governance.canary import GovernanceCanary

        maturity_mgr = MaturityManager(settings.governance.maturity)
        health_tracker = HealthTracker(settings.governance.health_score)
        impact_clf = ImpactClassifier(settings.governance.impact_classifier)
        drift_det = DriftDetector(settings.governance.drift_detector)
        token_mgr = TokenManager(settings.governance.execution_tokens)

        governance_gate = GovernanceGate(
            config=settings.governance,
            maturity=maturity_mgr,
            health=health_tracker,
            impact=impact_clf,
            drift=drift_det,
            tokens=(
                token_mgr
                if settings.governance.execution_tokens.require_tokens
                else None
            ),
            event_bus=ctx.event_bus,
        )

        # Start canary watchdog
        governance_canary = GovernanceCanary(
            settings.governance.canary,
            event_bus=ctx.event_bus,
        )
        await governance_canary.start_periodic(
            settings.governance.canary.check_interval_seconds
        )
        logger.info(
            "Governance framework enabled (maturity, health, canary, impact, drift)"
        )

    # Wire Trade Journal & Analytics (Edgewonk-inspired, Tiers 1-3)
    from .journal import (
        TradeJournal,
        RollingTracker,
        ConfidenceCalibrator,
        MonteCarloProjector,
        OvertradingDetector,
        CoinFlipBaseline,
        MistakeDetector,
        SessionAnalyser,
        CorrelationMatrix,
        TradeReplayer,
    )

    rolling_tracker = RollingTracker(window_size=100)
    confidence_cal = ConfidenceCalibrator(n_buckets=5)
    monte_carlo = MonteCarloProjector(n_simulations=1000, seed=42)
    overtrading_det = OvertradingDetector(lookback=50, threshold_z=2.0)
    coin_flip = CoinFlipBaseline(n_simulations=10_000, seed=42)

    # Tier 3 components
    mistake_detector = MistakeDetector()
    session_analyser = SessionAnalyser()
    correlation_matrix = CorrelationMatrix()
    trade_replayer = TradeReplayer()

    def _on_trade_closed(trade):
        """Post-close callback: feed all analytics components + emit metrics."""
        try:
            sid = trade.strategy_id
            pnl = float(trade.net_pnl)
            r = trade.r_multiple
            won = trade.outcome.value == "win"

            # Feed Tier 2 analytics
            rolling_tracker.add_trade(trade)
            confidence_cal.record(sid, trade.signal_confidence, won, r)
            monte_carlo.add_trade(sid, pnl, r)
            overtrading_det.record_trade(sid, trade.opened_at)
            coin_flip.add_trade(sid, pnl, r)

            # Feed Tier 3 analytics
            detected_mistakes = mistake_detector.analyse(trade)
            session_analyser.add_trade(trade)
            correlation_matrix.add_trade(trade)

            # Emit Prometheus metrics
            try:
                record_journal_trade(sid, trade.outcome.value)

                # Rolling metrics
                snap = rolling_tracker.snapshot(sid)
                if snap:
                    update_journal_rolling_metrics(sid, snap)

                # Journal counts
                update_journal_counts(
                    journal.open_trade_count, journal.closed_trade_count
                )

                # Confidence calibration
                report = confidence_cal.report(sid)
                if report["total_observations"] > 0:
                    update_journal_confidence(sid, report["brier_score"])

                # Overtrading
                update_journal_overtrading(
                    sid, overtrading_det.is_overtrading(sid)
                )

                # Edge test (only when enough data)
                if len(coin_flip._data.get(sid, coin_flip._data[sid]).pnl_series) >= 20:
                    edge = coin_flip.evaluate(sid)
                    update_journal_edge(sid, edge.get("p_value_bootstrap", 1.0))

                # Monte Carlo (only when enough data)
                mc_data = monte_carlo._data.get(sid)
                if mc_data and len(mc_data.pnl_series) >= 20:
                    proj = monte_carlo.project(sid, initial_equity=100_000.0)
                    kelly = monte_carlo.kelly_fraction(sid)
                    update_journal_monte_carlo(
                        sid,
                        proj.get("ruin_probability", 0.0),
                        kelly,
                    )

                # Tier 3 metrics — mistakes
                for m in detected_mistakes:
                    record_journal_mistake(sid, m.mistake_type)
                mistake_report = mistake_detector.report(sid)
                update_journal_mistake_impact(
                    sid, mistake_report.get("total_pnl_impact", 0.0)
                )

                # Tier 3 metrics — session analysis
                session_report = session_analyser.report(sid)
                if session_report.get("total_trades", 0) > 0:
                    update_journal_session_metrics(
                        sid, session_report.get("by_session", {})
                    )
                    update_journal_best_session(
                        sid,
                        session_report.get("best_session"),
                        session_report.get("best_hour"),
                    )

                # Tier 3 metrics — correlation (periodic, not every trade)
                corr_report = correlation_matrix.report()
                for pair_key, r_val in corr_report.get("strategy_correlation", {}).items():
                    parts = pair_key.split("|")
                    if len(parts) == 2:
                        update_journal_correlation(parts[0], parts[1], r_val)

            except Exception:
                logger.debug("Journal metrics emission failed", exc_info=True)

        except Exception:
            logger.warning("Journal on_trade_closed callback failed", exc_info=True)

    # Create journal with governance integration
    health_tracker_ref = None
    drift_detector_ref = None
    if settings.governance.enabled:
        health_tracker_ref = health_tracker
        drift_detector_ref = drift_det

    journal = TradeJournal(
        max_closed_trades=10_000,
        health_tracker=health_tracker_ref,
        drift_detector=drift_detector_ref,
        on_trade_closed=_on_trade_closed,
    )
    logger.info(
        "Trade Journal & Analytics wired — Tiers 1-3 "
        "(rolling, confidence, MC, overtrading, edge, mistakes, session, correlation, replay)"
    )

    # Wire Avatar Narration (if enabled)
    narration_service = None
    narration_store = None
    narration_runner = None
    if settings.narration.enabled:
        from .narration.service import NarrationService, Verbosity as NarrVerbosity
        from .narration.store import NarrationStore as NarrStore
        from .narration.schema import (
            DecisionExplanation,
            ConsideredSetup,
            RiskSummary,
            PositionSnapshot,
        )
        from .narration.server import start_narration_server

        _verb_map = {
            "quiet": NarrVerbosity.QUIET,
            "normal": NarrVerbosity.NORMAL,
            "detailed": NarrVerbosity.DETAILED,
            "presenter": NarrVerbosity.PRESENTER,
        }
        narration_service = NarrationService(
            verbosity=_verb_map.get(
                settings.narration.verbosity, NarrVerbosity.NORMAL
            ),
            heartbeat_seconds=settings.narration.heartbeat_seconds,
            dedupe_window_seconds=settings.narration.dedupe_window_seconds,
        )
        narration_store = NarrStore(max_items=settings.narration.max_stored_items)

        if settings.narration.tavus_mock:
            from .narration.tavus import MockTavusAdapter
            tavus_adapter = MockTavusAdapter(
                base_url=f"http://localhost:{settings.narration.server_port}"
            )
        else:
            from .narration.tavus import TavusAdapterHttp
            tavus_adapter = TavusAdapterHttp()

        try:
            narration_runner = await start_narration_server(
                store=narration_store,
                tavus=tavus_adapter,
                port=settings.narration.server_port,
                service=narration_service,
            )
            logger.info(
                "Narration server started on port %d (mock_tavus=%s)",
                settings.narration.server_port,
                settings.narration.tavus_mock,
            )
        except Exception:
            logger.warning("Failed to start narration server", exc_info=True)
            narration_runner = None

        # Seed narration store with sample data so dashboards aren't empty
        # while waiting for the first strategy signal
        try:
            from .narration.standalone import _seed_store
            _seed_store(narration_store, narration_service)
            logger.info(
                "Narration store seeded with %d sample items",
                narration_store.count,
            )
        except Exception:
            logger.debug("Narration seed failed (non-critical)", exc_info=True)

    # Paper adapter for simulated execution
    if settings.mode == Mode.PAPER:
        from .execution.adapters.paper import PaperAdapter

        # Use the first configured exchange, fallback to BINANCE
        paper_exchange = Exchange.BINANCE
        if settings.exchanges:
            paper_exchange = settings.exchanges[0].name

        adapter = PaperAdapter(
            exchange=paper_exchange,
            initial_balances={"USDT": Decimal("100000")},
        )
        logger.info("Paper adapter ready with 100,000 USDT on %s", paper_exchange.value)

        # Set initial portfolio equity metric
        try:
            update_equity(100_000.0)
        except Exception:
            pass

    # Wire feature vectors to strategy signals
    from .core.events import FeatureVector

    async def on_feature_vector(event):
        if not isinstance(event, FeatureVector):
            return

        # Track candle processing metrics
        try:
            record_candle_processed(event.symbol, event.timeframe.value if hasattr(event.timeframe, 'value') else str(event.timeframe))
        except Exception:
            pass

        candle_buffer = feature_engine.get_buffer(event.symbol, event.timeframe)
        if not candle_buffer:
            return

        # Alias indicator keys for strategy compatibility
        aliased = dict(event.features)
        if "adx_14" in aliased and "adx" not in aliased:
            aliased["adx"] = aliased["adx_14"]
        if "atr_14" in aliased and "atr" not in aliased:
            aliased["atr"] = aliased["atr_14"]

        patched_fv = FeatureVector(
            symbol=event.symbol,
            timeframe=event.timeframe,
            features=aliased,
            source_module=event.source_module,
        )

        latest_candle = candle_buffer[-1]
        for strategy in strategies:
            sig = strategy.on_candle(ctx, latest_candle, patched_fv)
            if sig is not None:
                logger.info(
                    "Signal: %s %s conf=%.2f | %s",
                    sig.direction.value,
                    sig.symbol,
                    sig.confidence,
                    sig.rationale,
                )
                await ctx.event_bus.publish("strategy.signal", sig)

                # Emit metrics
                try:
                    record_signal(sig.strategy_id, sig.symbol, sig.direction.value)
                except Exception:
                    pass

                # Feed narration service
                if narration_service is not None and narration_store is not None:
                    try:
                        from .narration.schema import (
                            DecisionExplanation,
                            RiskSummary,
                            PositionSnapshot,
                        )

                        action = "NO_TRADE" if sig.direction.value == "flat" else "ENTER"
                        regime_str = ""
                        if ctx.regime:
                            regime_str = getattr(ctx.regime, "regime", "unknown")
                            if hasattr(regime_str, "value"):
                                regime_str = regime_str.value

                        pos_snap = PositionSnapshot()
                        if ctx.portfolio_state:
                            pos_snap = PositionSnapshot(
                                open_positions=len(ctx.portfolio_state.positions),
                                gross_exposure_usd=float(ctx.portfolio_state.gross_exposure),
                                net_exposure_usd=float(ctx.portfolio_state.net_exposure),
                            )

                        risk_sum = RiskSummary(
                            health_score=1.0,
                            maturity_level="",
                        )
                        if governance_gate is not None:
                            try:
                                risk_sum.governance_action = "active"
                            except Exception:
                                pass

                        explanation = DecisionExplanation(
                            symbol=sig.symbol,
                            timeframe=event.timeframe.value if hasattr(event.timeframe, "value") else str(event.timeframe),
                            market_summary=f"{sig.symbol} is currently being analysed.",
                            active_strategy=sig.strategy_id,
                            active_regime=regime_str,
                            action=action,
                            reasons=[sig.rationale] if sig.rationale else [],
                            reason_confidences=[sig.confidence],
                            why_not=(
                                [sig.rationale or "No clear setup"] if action == "NO_TRADE" else []
                            ),
                            risk=risk_sum,
                            position=pos_snap,
                            trace_id=sig.trace_id,
                            signal_id=sig.event_id,
                        )

                        item = narration_service.generate(explanation)
                        if item is not None:
                            item.metadata = {
                                "action": action,
                                "symbol": sig.symbol,
                                "regime": regime_str,
                            }
                            narration_store.add(item, explanation=explanation)
                    except Exception:
                        logger.debug("Narration generation failed", exc_info=True)

    await ctx.event_bus.subscribe("feature.vector", "strategy_runner", on_feature_vector)

    # Start live market data feeds if exchange configs are available
    symbols = settings.symbols.symbols or ["BTC/USDT"]
    if settings.exchanges:
        try:
            from .data.feed_manager import FeedManager
            from .data.candle_builder import CandleBuilder

            candle_builder = CandleBuilder(event_bus=ctx.event_bus)
            feed_manager = FeedManager(
                event_bus=ctx.event_bus,
                candle_builder=candle_builder,
                exchange_configs=settings.exchanges,
                symbols=symbols,
            )
            await feed_manager.start()
            logger.info(
                "FeedManager started: %d feed tasks for %s",
                feed_manager.active_task_count,
                symbols,
            )
        except Exception:
            logger.warning("Failed to start FeedManager", exc_info=True)
            feed_manager = None
    else:
        logger.info(
            "No exchange configs — running without live feeds. "
            "Publish CandleEvent to 'market.candle' manually or configure exchanges."
        )

    logger.info(
        "Trading loop running (%s mode). %s. Press Ctrl+C to stop.",
        settings.mode.value,
        f"Receiving live feeds for {symbols}" if feed_manager else "Waiting for market data on event bus",
    )
    await stop_event.wait()

    # Graceful shutdown
    if narration_runner is not None:
        await narration_runner.cleanup()
        logger.info("Narration server stopped")
    if governance_canary is not None:
        await governance_canary.stop()
    if feed_manager:
        await feed_manager.stop()
    await feature_engine.stop()
    logger.info("Trading loop stopped")


async def run_walk_forward(
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
    n_folds: int = 5,
) -> None:
    """Run walk-forward validation across multiple folds.

    Splits historical data into train/test windows, runs backtest on each,
    and reports overfitting metrics.
    """
    from .backtester.engine import BacktestEngine
    from .core.config import load_settings
    from .data.historical import HistoricalDataLoader
    from .features.engine import FeatureEngine
    from .strategies.registry import create_strategy
    from .strategies.research.walk_forward import WalkForwardValidator, WalkForwardResult
    from .strategies.research.experiment_log import ExperimentLogger, ExperimentConfig, ExperimentResult

    # Import strategy modules to trigger @register_strategy decorators
    import agentic_trading.strategies.trend_following  # noqa: F401
    import agentic_trading.strategies.mean_reversion  # noqa: F401
    import agentic_trading.strategies.breakout  # noqa: F401
    import agentic_trading.strategies.funding_arb  # noqa: F401

    settings = load_settings(config_path=config_path, overrides=overrides)
    _setup_logging(settings)

    bt = settings.backtest
    logger.info("Walk-forward validation: %s to %s, %d folds", bt.start_date, bt.end_date, n_folds)

    # Create strategies
    strategies_cfg = []
    for strat_cfg in settings.strategies:
        if strat_cfg.enabled:
            strategies_cfg.append(strat_cfg)

    # Load historical data
    loader = HistoricalDataLoader(data_dir=bt.data_dir)
    symbols = settings.symbols.symbols or ["BTC/USDT"]
    exchange = Exchange.BINANCE

    start_dt = datetime.strptime(bt.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(bt.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    all_candles: dict[str, list] = {}
    for symbol in symbols:
        candles = loader.load_candles(
            exchange=exchange,
            symbol=symbol,
            timeframe=Timeframe.M1,
            start=start_dt,
            end=end_dt,
        )
        if candles:
            all_candles[symbol] = candles
            logger.info("Loaded %d candles for %s", len(candles), symbol)

    if not all_candles:
        logger.error("No data for walk-forward. Run download_historical.py first.")
        return

    # Collect all timestamps (from the first symbol)
    first_symbol = list(all_candles.keys())[0]
    timestamps = [c.timestamp for c in all_candles[first_symbol]]

    # Create walk-forward windows
    validator = WalkForwardValidator(n_folds=n_folds, train_pct=0.7, gap_periods=1)
    windows = validator.create_windows(timestamps)

    logger.info("Created %d walk-forward windows", len(windows))

    fold_results: list[WalkForwardResult] = []
    experiment_logger = ExperimentLogger()

    for window in windows:
        logger.info(
            "Fold %d: train %s→%s, test %s→%s",
            window.fold_index,
            window.train_start.strftime("%Y-%m-%d"),
            window.train_end.strftime("%Y-%m-%d"),
            window.test_start.strftime("%Y-%m-%d"),
            window.test_end.strftime("%Y-%m-%d"),
        )

        # Split candles for this fold
        train_candles: dict[str, list] = {}
        test_candles: dict[str, list] = {}
        for symbol, candles in all_candles.items():
            train_candles[symbol] = [
                c for c in candles
                if window.train_start <= c.timestamp <= window.train_end
            ]
            test_candles[symbol] = [
                c for c in candles
                if window.test_start <= c.timestamp <= window.test_end
            ]

        # Run backtest on train set
        def _make_strategies():
            strats = []
            for scfg in strategies_cfg:
                strats.append(create_strategy(scfg.strategy_id, scfg.params))
            if not strats:
                from .strategies.trend_following import TrendFollowingStrategy
                strats = [TrendFollowingStrategy()]
            return strats

        train_engine = BacktestEngine(
            strategies=_make_strategies(),
            feature_engine=FeatureEngine(),
            initial_capital=bt.initial_capital,
            slippage_bps=bt.slippage_bps,
            fee_maker=bt.fee_maker,
            fee_taker=bt.fee_taker,
            seed=bt.random_seed,
        )
        train_result = await train_engine.run(train_candles)

        # Run backtest on test set
        test_engine = BacktestEngine(
            strategies=_make_strategies(),
            feature_engine=FeatureEngine(),
            initial_capital=bt.initial_capital,
            slippage_bps=bt.slippage_bps,
            fee_maker=bt.fee_maker,
            fee_taker=bt.fee_taker,
            seed=bt.random_seed,
        )
        test_result = await test_engine.run(test_candles)

        fold_results.append(WalkForwardResult(
            fold_index=window.fold_index,
            train_sharpe=train_result.sharpe_ratio,
            test_sharpe=test_result.sharpe_ratio,
            train_return=train_result.total_return,
            test_return=test_result.total_return,
            train_max_dd=train_result.max_drawdown,
            test_max_dd=test_result.max_drawdown,
        ))

        # Log experiment
        exp_id = f"wf_fold_{window.fold_index}_{bt.start_date}_{bt.end_date}"
        experiment_logger.log_config(ExperimentConfig(
            experiment_id=exp_id,
            strategy_id=train_result.strategy_id or "trend_following",
            params={},
            symbols=symbols,
            start_date=window.train_start.isoformat(),
            end_date=window.test_end.isoformat(),
            timeframes=["1m"],
            slippage_model=bt.slippage_model,
            fee_maker=bt.fee_maker,
            fee_taker=bt.fee_taker,
            random_seed=bt.random_seed,
            notes=f"Walk-forward fold {window.fold_index}",
        ))
        experiment_logger.log_result(ExperimentResult(
            experiment_id=exp_id,
            total_return=test_result.total_return,
            sharpe_ratio=test_result.sharpe_ratio,
            max_drawdown=test_result.max_drawdown,
            win_rate=test_result.win_rate,
            profit_factor=test_result.profit_factor,
            total_trades=test_result.total_trades,
            avg_trade_return=test_result.avg_trade_return,
            metadata={"train_sharpe": train_result.sharpe_ratio, "test_sharpe": test_result.sharpe_ratio},
        ))

    # Aggregate report
    report = validator.evaluate(fold_results)

    # Print results
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 80)
    print(f"  {'Folds':30s}: {len(report.folds)}")
    print(f"  {'Avg Train Sharpe':30s}: {report.avg_train_sharpe:>12.4f}")
    print(f"  {'Avg Test Sharpe':30s}: {report.avg_test_sharpe:>12.4f}")
    print(f"  {'Overfit Score':30s}: {report.overfit_score:>12.4f}")
    print(f"  {'Degradation %':30s}: {report.degradation_pct:>12.2f}%")
    print(f"  {'Is Overfit':30s}: {'YES' if report.is_overfit else 'NO':>12}")
    print("-" * 80)
    print(f"  {'Fold':>4}  {'Train Sharpe':>12}  {'Test Sharpe':>12}  {'Train Ret':>10}  {'Test Ret':>10}  {'Overfit':>8}")
    for fr in report.folds:
        print(
            f"  {fr.fold_index:>4}  {fr.train_sharpe:>12.4f}  {fr.test_sharpe:>12.4f}  "
            f"{fr.train_return:>10.2%}  {fr.test_return:>10.2%}  {fr.overfit_ratio:>8.2f}"
        )
    print("=" * 80)
    print()


def _setup_logging(settings: Settings) -> None:
    """Configure structured logging."""
    level = getattr(logging, settings.observability.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
