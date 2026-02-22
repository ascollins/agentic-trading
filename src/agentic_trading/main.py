"""Application bootstrap and mode routing.

This is the main entry point that wires together all modules
and starts the trading loop for the selected mode.

Construction of the clock, event bus, TradingContext, and all layer
managers is delegated to :class:`Orchestrator`.  This module handles
settings loading, safety validation, Prometheus metrics, and
mode-specific runtime logic (backtest engine, live-trading event
handlers, narration, UI, etc.).
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Any

from .core.config import Settings, load_settings
from .core.enums import Exchange, Mode, OrderType, Side, Timeframe, TimeInForce
from .core.interfaces import PortfolioState, TradingContext
from .orchestrator import Orchestrator

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

    # 4. Import strategy modules to trigger @register_strategy decorators.
    # Must happen before Orchestrator.from_config() which calls create_strategy().
    import agentic_trading.strategies.bb_squeeze  # noqa: F401
    import agentic_trading.strategies.breakout  # noqa: F401
    import agentic_trading.strategies.fibonacci_confluence  # noqa: F401
    import agentic_trading.strategies.funding_arb  # noqa: F401
    import agentic_trading.strategies.mean_reversion  # noqa: F401
    import agentic_trading.strategies.mean_reversion_enhanced  # noqa: F401
    import agentic_trading.strategies.multi_tf_ma  # noqa: F401
    import agentic_trading.strategies.obv_divergence  # noqa: F401
    import agentic_trading.strategies.rsi_divergence  # noqa: F401
    import agentic_trading.strategies.stochastic_macd  # noqa: F401
    import agentic_trading.strategies.supply_demand  # noqa: F401
    import agentic_trading.strategies.trend_following  # noqa: F401
    import agentic_trading.strategies.prediction_consensus  # noqa: F401

    # 5. Build Orchestrator (clock, bus, context, all layer managers)
    orch = Orchestrator.from_config(settings)
    ctx = orch.ctx
    event_bus = orch.bus.legacy_bus

    # 6. Start event bus
    await orch.bus.start()

    # 6.5 Start Prometheus metrics server (all modes)
    try:
        from .observability.metrics import start_metrics_server

        metrics_port = settings.observability.metrics_port
        start_metrics_server(port=metrics_port, mode=settings.mode.value)
        logger.info("Prometheus metrics server started on port %d", metrics_port)
    except Exception:
        logger.warning("Failed to start metrics server", exc_info=True)

    # 6. Route to mode-specific loop
    try:
        if settings.mode == Mode.BACKTEST:
            await _run_backtest(settings, ctx)
        else:
            await _run_live_or_paper(settings, ctx, orch)
    except KeyboardInterrupt:
        logger.info("Shutting down (keyboard interrupt)")
    finally:
        await orch.bus.stop()
        logger.info("Shutdown complete")


def _aggregate_candles(candles: list, target_tf: Timeframe) -> list:
    """Aggregate 1m candles to a higher timeframe for backtesting.

    Groups candles by UTC-aligned timeframe windows and produces one
    aggregated OHLCV candle per window.
    """
    from .core.models import Candle
    from .intelligence.candle_builder import _align_timestamp

    if not candles:
        return []

    buckets: dict[datetime, list] = {}
    for c in candles:
        aligned = _align_timestamp(c.timestamp, target_tf)
        buckets.setdefault(aligned, []).append(c)

    result = []
    for open_time in sorted(buckets.keys()):
        group = buckets[open_time]
        result.append(Candle(
            symbol=group[0].symbol,
            exchange=group[0].exchange,
            timeframe=target_tf,
            timestamp=open_time,
            open=group[0].open,
            high=max(c.high for c in group),
            low=min(c.low for c in group),
            close=group[-1].close,
            volume=sum(c.volume for c in group),
            quote_volume=sum(c.quote_volume for c in group),
            trades=sum(c.trades for c in group),
            is_closed=True,
        ))
    return result


async def _run_backtest(settings: Settings, ctx: TradingContext) -> None:
    """Run backtest mode with BacktestEngine."""
    import agentic_trading.strategies.bb_squeeze  # noqa: F401
    import agentic_trading.strategies.breakout  # noqa: F401
    import agentic_trading.strategies.fibonacci_confluence  # noqa: F401
    import agentic_trading.strategies.funding_arb  # noqa: F401
    import agentic_trading.strategies.mean_reversion  # noqa: F401
    import agentic_trading.strategies.mean_reversion_enhanced  # noqa: F401
    import agentic_trading.strategies.multi_tf_ma  # noqa: F401
    import agentic_trading.strategies.obv_divergence  # noqa: F401
    import agentic_trading.strategies.rsi_divergence  # noqa: F401
    import agentic_trading.strategies.stochastic_macd  # noqa: F401
    import agentic_trading.strategies.supply_demand  # noqa: F401

    # Import strategy modules to trigger @register_strategy decorators
    import agentic_trading.strategies.trend_following  # noqa: F401

    from .backtester.engine import BacktestEngine
    from .data.historical import HistoricalDataLoader
    from .features.engine import FeatureEngine
    from .strategies.registry import create_strategy

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

    # Aggregate to target timeframe if configured (default: 1m = no aggregation)
    input_tf_str = getattr(bt, "input_timeframe", "1m")
    if input_tf_str != "1m":
        target_tf = Timeframe(input_tf_str)
        for symbol in list(candles_by_symbol.keys()):
            raw = candles_by_symbol[symbol]
            candles_by_symbol[symbol] = _aggregate_candles(raw, target_tf)
            logger.info(
                "Aggregated %s to %s: %d candles",
                symbol, target_tf.value, len(candles_by_symbol[symbol]),
            )

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
        max_concurrent_positions=4,
        max_daily_entries=6,
        portfolio_cooldown_seconds=3600,
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

    # Print per-strategy breakdown
    if result.per_strategy:
        print()
        print("PER-STRATEGY BREAKDOWN")
        print("-" * 76)
        print(
            f"  {'Strategy':<25s} {'Trades':>7s} {'WinRate':>8s} "
            f"{'PF':>6s} {'AvgRet':>8s} {'PnL%':>8s}"
        )
        print("-" * 76)
        for s in sorted(result.per_strategy, key=lambda x: x.total_pnl_pct, reverse=True):
            print(
                f"  {s.strategy_id:<25s} {s.total_trades:>7d} "
                f"{s.win_rate:>7.1%} {s.profit_factor:>6.2f} "
                f"{s.avg_return:>+7.2%} {s.total_pnl_pct:>+7.2f}%"
            )
        print("-" * 76)

    print()


async def _run_live_or_paper(
    settings: Settings,
    ctx: TradingContext,
    orch: Orchestrator,
) -> None:
    """Run paper or live trading loop.

    Wires: FeedManager -> FeatureEngine -> Strategies -> Execution.
    For paper mode, uses PaperAdapter for simulated fills.

    Component sourcing (PR 15):
    - Feature engine, strategies, portfolio manager, signal manager come
      from the Orchestrator's layer managers.
    - Adapter, execution engine, journal + analytics, narration, TP/SL,
      and governance are still constructed locally.
    """
    from decimal import Decimal

    logger.info("Starting %s trading loop", settings.mode.value)

    # Import metrics helpers (server already started in run())
    try:
        from .observability.metrics import (
            record_candle_processed,
            record_decision_latency,
            record_fill,
            record_governance_block,
            record_governance_decision,
            record_governance_latency,
            record_journal_mistake,
            record_journal_trade,
            record_order,
            record_signal,
            update_active_tokens,
            update_canary_status,
            update_daily_pnl,
            update_data_staleness,
            update_drawdown,
            update_equity,
            update_gross_exposure,
            update_health_score,
            update_journal_best_session,
            update_journal_confidence,
            update_journal_correlation,
            update_journal_counts,
            update_journal_edge,
            update_journal_mistake_impact,
            update_journal_monte_carlo,
            update_journal_overtrading,
            update_journal_rolling_metrics,
            update_journal_session_metrics,
            update_kill_switch,
            update_maturity_level,
            update_portfolio_quality,
            update_position,
            update_quality_scores,
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

    # Feature engine and strategies come from the Orchestrator.
    feature_engine = orch.intelligence.feature_engine
    await feature_engine.start()

    strategies = orch.signal.strategies
    if strategies:
        for strat in strategies:
            logger.info("Loaded strategy: %s", getattr(strat, "strategy_id", "?"))
    else:
        import agentic_trading.strategies.trend_following  # noqa: F401

        from .strategies.registry import create_strategy
        from .strategies.trend_following import TrendFollowingStrategy  # noqa: F401
        strategies = [TrendFollowingStrategy()]
        logger.info("No strategies configured, using default trend_following")

    # Wire governance framework (if enabled)
    governance_gate = None
    governance_canary = None
    if settings.governance.enabled:
        from .governance.canary import GovernanceCanary
        from .governance.drift_detector import DriftDetector
        from .governance.gate import GovernanceGate
        from .governance.health_score import HealthTracker
        from .governance.impact_classifier import ImpactClassifier
        from .governance.maturity import MaturityManager
        from .governance.tokens import TokenManager

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

        # Start canary watchdog (B17 FIX: wire kill_switch_fn)
        governance_canary = GovernanceCanary(
            settings.governance.canary,
            kill_switch_fn=risk_manager.kill_switch.activate,
            event_bus=ctx.event_bus,
        )
        await governance_canary.start_periodic(
            settings.governance.canary.check_interval_seconds
        )
        logger.info(
            "Governance framework enabled (maturity, health, canary, impact, drift)"
        )

        for strat in strategies:
            try:
                drift_det.set_baseline(strat.strategy_id, {
                    "win_rate": 0.50, "avg_rr": 1.5, "sharpe": 1.0, "profit_factor": 1.5,
                })
            except Exception:
                pass

    # Wire Trade Journal & Analytics (Edgewonk-inspired, Tiers 1-3)
    from .journal import (
        CoinFlipBaseline,
        ConfidenceCalibrator,
        CorrelationMatrix,
        MistakeDetector,
        MonteCarloProjector,
        OvertradingDetector,
        RollingTracker,
        SessionAnalyser,
        TradeJournal,
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

                # Quality scorecard evaluation
                try:
                    from .journal.quality_scorecard import QualityScorecard
                    qs = QualityScorecard()
                    stats = journal.get_strategy_stats(sid)
                    if stats and stats.get("total_trades", 0) >= 5:
                        edge_result = None
                        mc_result = None
                        try:
                            edge_result = coin_flip.evaluate(sid)
                        except Exception:
                            pass
                        try:
                            mc_data = monte_carlo._data.get(sid)
                            if mc_data and len(mc_data.pnl_series) >= 5:
                                mc_result = monte_carlo.project(sid, initial_equity=100_000.0)
                        except Exception:
                            pass
                        q_report = qs.evaluate(
                            strategy_id=sid,
                            stats=stats,
                            rolling_snapshot=snap,
                            edge_result=edge_result,
                            monte_carlo_result=mc_result,
                        )
                        update_quality_scores(sid, q_report.to_dict())

                    # Portfolio-level quality (across all strategies)
                    all_stats = journal.get_all_strategy_stats()
                    strategy_reports = []
                    for s_id, s_stats in all_stats.items():
                        if s_stats.get("total_trades", 0) >= 5:
                            try:
                                s_snap = rolling_tracker.snapshot(s_id)
                                r = qs.evaluate(strategy_id=s_id, stats=s_stats, rolling_snapshot=s_snap)
                                strategy_reports.append(r)
                            except Exception:
                                pass
                    if strategy_reports:
                        p_report = qs.evaluate_portfolio(strategy_reports)
                        update_portfolio_quality(p_report)
                except Exception:
                    logger.debug("Quality scorecard metrics failed", exc_info=True)

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

    # Wire journal persistence to PostgreSQL
    _journal_repo = None
    try:
        from .journal.persistence import JournalRepo
        from .storage.postgres.connection import get_session, init_engine
        _db_engine = await init_engine(
            settings.postgres_url, create_tables=True
        )
        _journal_repo = True  # sentinel — repo is created per-session
        logger.info("Journal persistence wired to PostgreSQL")
    except Exception:
        logger.warning("Journal persistence unavailable — trades will be in-memory only", exc_info=True)

    _original_on_trade_closed = _on_trade_closed

    def _on_trade_closed_with_persistence(trade):
        """Wrap the original callback to also persist to database."""
        _original_on_trade_closed(trade)
        if _journal_repo is not None:
            asyncio.ensure_future(_persist_trade(trade))

    async def _persist_trade(trade):
        """Persist a closed trade to PostgreSQL."""
        try:
            from .journal.persistence import JournalRepo
            from .storage.postgres.connection import get_session
            async with get_session() as session:
                repo = JournalRepo(session)
                await repo.save_trade(trade)
                logger.debug("Trade persisted: %s %s", trade.strategy_id, trade.symbol)
        except Exception:
            logger.warning("Failed to persist trade to database", exc_info=True)

    # Replace journal callback with persistence-wrapped version
    if _journal_repo is not None:
        journal._on_trade_closed = _on_trade_closed_with_persistence

    # Wire Avatar Narration (if enabled)
    narration_service = None
    narration_store = None
    narration_runner = None
    if settings.narration.enabled:
        from .narration.schema import (
            ConsideredSetup,
            DecisionExplanation,
            PositionSnapshot,
            RiskSummary,
        )
        from .narration.server import start_narration_server
        from .narration.service import NarrationService
        from .narration.service import Verbosity as NarrVerbosity
        from .narration.store import NarrationStore as NarrStore

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
        _narration_persist = None
        if settings.mode != Mode.BACKTEST:
            _narration_persist = "data/narration_history.jsonl"
        narration_store = NarrStore(
            max_items=settings.narration.max_stored_items,
            persistence_path=_narration_persist,
        )

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
                pipeline_log=orch.pipeline_log,
                context_manager=orch.context_manager,
            )
            logger.info(
                "Narration server started on port %d (mock_tavus=%s, reasoning=%s)",
                settings.narration.server_port,
                settings.narration.tavus_mock,
                orch.pipeline_log is not None,
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

    # ---------------------------------------------------------------
    # Exchange adapter + Execution pipeline
    # ---------------------------------------------------------------
    from .core.events import FillEvent, OrderAck
    from .execution.engine import ExecutionEngine
    from .risk.manager import RiskManager

    # Read-only guard: skip execution pipeline when observing only
    _read_only = settings.read_only
    if _read_only:
        logger.info("READ-ONLY mode: execution pipeline disabled, signals will be observed but not traded")

    if settings.mode == Mode.LIVE:
        logger.warning(
            "LIVE MODE ACTIVE — orders will be sent to %s (testnet=%s)",
            settings.exchanges[0].name.value if settings.exchanges else "?",
            settings.exchanges[0].testnet if settings.exchanges else "?",
        )

    # Determine exchange for routing
    active_exchange = Exchange.BINANCE
    if settings.exchanges:
        active_exchange = settings.exchanges[0].name

    adapter = None  # will be set below

    # Detect FX mode from exchange
    _is_fx = active_exchange in (Exchange.OANDA, Exchange.LMAX)
    _balance_ccy = "USD" if _is_fx else "USDT"

    if settings.mode == Mode.PAPER:
        from .execution.adapters.paper import PaperAdapter

        adapter = PaperAdapter(
            exchange=active_exchange,
            initial_balances={_balance_ccy: Decimal("100000")},
        )
        logger.info(
            "Paper adapter ready with 100,000 %s on %s",
            _balance_ccy, active_exchange.value,
        )

        try:
            update_equity(100_000.0)
        except Exception:
            pass

    elif settings.mode == Mode.LIVE:
        from .execution.adapters.ccxt_adapter import CCXTAdapter

        if not settings.exchanges:
            logger.error("Live mode requires at least one exchange config")
            return
        exc_cfg = settings.exchanges[0]
        adapter = CCXTAdapter(
            exchange_name=exc_cfg.name.value,
            api_key=exc_cfg.api_key,
            api_secret=exc_cfg.secret,
            sandbox=exc_cfg.testnet,
            demo=exc_cfg.demo,
            default_type="swap",
        )
        logger.info(
            "Live adapter ready: %s (testnet=%s, demo=%s)",
            exc_cfg.name.value, exc_cfg.testnet, exc_cfg.demo,
        )

    # Fetch instrument metadata from exchange and populate ctx.instruments
    _sym_list = settings.symbols.symbols or []

    # FX paper mode: pre-load hardcoded instrument definitions
    # (no exchange API to query for FX paper trading)
    if _is_fx and settings.mode == Mode.PAPER and _sym_list:
        from .core.fx_instruments import build_fx_instruments

        _fx_instruments = build_fx_instruments(_sym_list, exchange=active_exchange)
        for sym, inst in _fx_instruments.items():
            ctx.instruments[sym] = inst
            if adapter is not None:
                adapter.load_instrument(inst)
            logger.info(
                "  FX %s: pip=%s lot=%s tick=%s",
                sym, inst.pip_size, inst.lot_size, inst.tick_size,
            )
        logger.info(
            "FX instruments pre-loaded: %d/%d symbols",
            len(_fx_instruments), len(_sym_list),
        )

    if adapter is not None and _sym_list and not ctx.instruments:
        logger.info("Fetching instrument metadata for %d symbols...", len(_sym_list))
        for sym in _sym_list:
            try:
                inst = await adapter.get_instrument(sym)
                if inst is not None:
                    ctx.instruments[sym] = inst
                    logger.info(
                        "  %s: tick=%s step=%s min_qty=%s",
                        sym, inst.tick_size, inst.step_size, inst.min_qty,
                    )
            except Exception as e:
                logger.warning("Failed to fetch instrument for %s: %s", sym, e)
        logger.info(
            "Instruments loaded: %d/%d symbols",
            len(ctx.instruments), len(_sym_list),
        )

    # Wire instruments into FeatureEngine (loaded after orchestrator creation)
    if ctx.instruments:
        feature_engine._instruments = ctx.instruments

    # Risk manager
    risk_manager = RiskManager(
        config=settings.risk,
        event_bus=ctx.event_bus,
        instruments=ctx.instruments,
    )

    # Institutional control plane: ToolGateway
    # All exchange side effects MUST go through the ToolGateway.
    from agentic_trading.control_plane.audit_log import AuditLog
    from agentic_trading.control_plane.tool_gateway import ToolGateway

    audit_log = AuditLog()
    tool_gateway: ToolGateway | None = None
    if adapter is not None:
        tool_gateway = ToolGateway(
            adapter=adapter,
            audit_log=audit_log,
            event_bus=ctx.event_bus,
            kill_switch_fn=risk_manager.kill_switch.is_active,
        )
        logger.info("ToolGateway initialized (audit_log=memory)")

    # Execution engine
    execution_engine = ExecutionEngine(
        adapter=adapter,
        event_bus=ctx.event_bus,
        risk_manager=risk_manager,
        kill_switch=risk_manager.kill_switch.is_active,
        portfolio_state_provider=lambda: ctx.portfolio_state,
        governance_gate=governance_gate,
        tool_gateway=tool_gateway,
    )
    await execution_engine.start()

    # Signal cache — maps trace_id → Signal for fill-time narration
    _signal_cache: dict[str, Any] = {}
    _SIGNAL_CACHE_MAX = 500
    _SIGNAL_CACHE_TTL = 300  # 5 minutes — evict stale entries first

    # Track exit orders: maps exit trace_id → original entry trace_id
    _exit_map: dict[str, str] = {}

    # Capital for sizing (paper = 100k, live = from adapter later)
    _capital = 100_000.0
    _initial_equity = 0.0  # Set on startup for drawdown calculation
    _daily_equity_base = 0.0  # Set on startup for daily PnL calculation

    # Signal and reconciliation layer facades — sourced from Orchestrator.
    # Inject the locally-created journal (which carries the analytics
    # _on_trade_closed callback and DB persistence wrapper) into the
    # Orchestrator's ReconciliationManager so fills and reconciliation
    # operate on the same journal that drives Prometheus metrics.
    _signal_mgr = orch.signal
    _recon_mgr = orch.reconciliation
    _recon_mgr._journal = journal

    logger.info(
        "Execution pipeline wired: %s adapter → RiskManager → ExecutionEngine",
        settings.mode.value,
    )

    # ---------------------------------------------------------------
    # Consensus Gate: multi-agent consultation before trade execution
    # ---------------------------------------------------------------
    from .reasoning.consensus import ConsensusGate, ConsensusVerdict
    from .reasoning.message_bus import ReasoningMessageBus

    _reasoning_bus = ReasoningMessageBus()
    _consensus_store = None
    try:
        import os

        from .reasoning.conversation_store import JsonFileConversationStore
        _store_path = os.path.join(
            settings.backtest.data_dir if hasattr(settings.backtest, "data_dir") else "data",
            "conversations.jsonl",
        )
        _consensus_store = JsonFileConversationStore(path=_store_path)
        logger.info("Consensus conversation store: %s", _store_path)
    except Exception:
        logger.debug("Consensus store init failed, using in-memory", exc_info=True)

    _risk_params = {}
    if hasattr(settings, "risk") and settings.risk:
        _risk_params = {
            "max_position_pct": getattr(settings.risk, "max_position_pct", 0.10),
            "max_gross_exposure_pct": getattr(settings.risk, "max_gross_exposure_pct", 1.0),
            "max_drawdown_pct": getattr(settings.risk, "max_drawdown_pct", 0.15),
        }

    from .reasoning.consensus import (
        CMTTechnicianDesk,
        MarketStructureDesk,
        RiskManagerDesk,
        SMCAnalystDesk,
    )

    _consensus_gate = ConsensusGate(
        message_bus=_reasoning_bus,
        participants=[
            MarketStructureDesk(),
            SMCAnalystDesk(),
            CMTTechnicianDesk(),
            RiskManagerDesk(**_risk_params),
        ],
        min_approval_ratio=0.5,
        require_risk_approval=True,
        conversation_store=_consensus_store,
    )
    logger.info(
        "ConsensusGate initialized: 4 desk participants, "
        "min_approval=50%%, risk_veto=True"
    )

    # Inject consensus gate + reasoning bus into narration server app
    # (narration server starts earlier, consensus gate created later)
    if narration_runner is not None:
        try:
            narration_runner.app["consensus_gate"] = _consensus_gate
            narration_runner.app["reasoning_bus"] = _reasoning_bus
            logger.info("Consensus gate injected into narration server")
        except Exception:
            logger.debug("Could not inject consensus gate into narration server", exc_info=True)

    # Wire feature vectors to strategy signals → execution pipeline
    from .core.events import FeatureVector
    from .narration.humanizer import humanize_rationale

    async def on_feature_vector(event):
        if not isinstance(event, FeatureVector):
            return

        # Track candle processing metrics + data staleness
        try:
            record_candle_processed(event.symbol, event.timeframe.value if hasattr(event.timeframe, 'value') else str(event.timeframe))
            # Data staleness: seconds since the candle timestamp
            if hasattr(event, "timestamp") and event.timestamp:
                staleness = (ctx.clock.now() - event.timestamp).total_seconds()
                update_data_staleness(event.symbol, max(0.0, staleness))
        except Exception:
            pass

        # Update paper adapter with latest market price
        if settings.mode == Mode.PAPER and adapter is not None:
            close_price = event.features.get("close")
            if close_price and hasattr(adapter, "set_market_price"):
                adapter.set_market_price(event.symbol, Decimal(str(close_price)))

        candle_buffer = feature_engine.get_buffer(event.symbol, event.timeframe)
        if not candle_buffer:
            return

        # Alias indicator keys for strategy compatibility
        from .signal.manager import SignalManager as _SM
        aliased = _SM.alias_features(event.features)

        patched_fv = FeatureVector(
            symbol=event.symbol,
            timeframe=event.timeframe,
            features=aliased,
            source_module=event.source_module,
        )

        latest_candle = candle_buffer[-1]
        import time as _time_mod

        # ============================================================
        # Phase 1: Collect ALL strategy signals for this candle cycle
        # ============================================================
        _candle_flat_signals: list = []    # FLAT (exit) signals
        _candle_dir_signals: list = []     # Directional (LONG/SHORT) signals

        for strategy in strategies:
            _t_sig_start = _time_mod.monotonic()
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
                    _decision_elapsed = _time_mod.monotonic() - _t_sig_start
                    record_decision_latency(_decision_elapsed)
                except Exception:
                    pass

                # Cache signal for fill-time narration lookup
                _signal_cache[sig.trace_id] = sig

                if governance_gate is not None:
                    try:
                        governance_gate.record_signal(sig)
                    except Exception:
                        pass

                # Separate FLAT from directional
                if sig.direction.value == "flat":
                    _candle_flat_signals.append(sig)
                else:
                    _candle_dir_signals.append(sig)

        # Cache eviction — once per candle cycle, not per signal
        _cache_now = ctx.clock.now()
        _stale = [
            k for k, v in _signal_cache.items()
            if hasattr(v, "timestamp")
            and (_cache_now - v.timestamp).total_seconds() > _SIGNAL_CACHE_TTL
        ]
        for _k in _stale:
            _signal_cache.pop(_k, None)
        if len(_signal_cache) > _SIGNAL_CACHE_MAX:
            excess = len(_signal_cache) - _SIGNAL_CACHE_MAX
            for _k in list(_signal_cache)[:excess]:
                _signal_cache.pop(_k, None)

        # ============================================================
        # Phase 2a: Process FLAT (exit) signals — record exit + route
        # ============================================================
        for sig in _candle_flat_signals:
            # Record exit in consensus gate for cooldown tracking
            _consensus_gate.record_exit(
                sig.symbol,
                when=ctx.clock.now(),
            )
            # Log the exit conversation
            _consensus_gate.consult_exit(sig, {"now": ctx.clock.now()})

            sig_result = _signal_mgr.process_signal(
                sig, journal, ctx, active_exchange, _capital,
                signal_cache=_signal_cache,
                exit_map=_exit_map,
            )
            for intent in sig_result.intents:
                if not _read_only:
                    await ctx.event_bus.publish("execution", intent)
                else:
                    logger.info(
                        "[READ-ONLY] Exit intent: %s %s %s qty=%s (trace=%s)",
                        intent.strategy_id, intent.side.value,
                        intent.symbol, intent.qty,
                        intent.trace_id[:8] if intent.trace_id else "?",
                    )

        # ============================================================
        # Phase 2b: Directional signals → Consensus Gate → Execution
        # ============================================================
        # Instead of routing signals directly to PortfolioManager,
        # each signal goes through the desk consultation:
        #   Market Structure → SMC → CMT → Risk → Verdict
        # Only APPROVED signals proceed to sizing and execution.
        _approved_dir_signals: list = []
        for sig in _candle_dir_signals:
            # Build the context for desk consultation
            _desk_context: dict = {
                "features": patched_fv.features if patched_fv else {},
                "regime": (
                    getattr(ctx.regime, "regime", None)
                    if ctx.regime else None
                ),
                "portfolio_state": ctx.portfolio_state,
                "risk_state": None,
                "cmt_assessment": None,
                "kill_switch_active": (
                    await risk_manager.kill_switch.is_active()
                    if hasattr(risk_manager, "kill_switch")
                    else False
                ),
                "now": ctx.clock.now(),
            }

            consensus = _consensus_gate.consult(sig, _desk_context)

            if consensus.is_approved:
                _approved_dir_signals.append(sig)
                logger.info(
                    "Consensus APPROVED: %s %s %s (score=%.2f, %dms)",
                    sig.direction.value,
                    sig.symbol,
                    sig.strategy_id,
                    consensus.weighted_score,
                    consensus.elapsed_ms,
                )
            else:
                logger.info(
                    "Consensus %s: %s %s %s — %s",
                    consensus.verdict.value.upper(),
                    sig.direction.value,
                    sig.symbol,
                    sig.strategy_id,
                    consensus.reasoning,
                )

            if governance_gate is not None:
                try:
                    from .core.events import GovernanceDecision
                    from .core.enums import GovernanceAction
                    decision = GovernanceDecision(
                        strategy_id=sig.strategy_id,
                        symbol=sig.symbol,
                        action=GovernanceAction.ALLOW if consensus.is_approved else GovernanceAction.BLOCK,
                        reason=consensus.reasoning[:200] if consensus.reasoning else "",
                        sizing_multiplier=1.0,
                        timestamp=ctx.clock.now(),
                    )
                    governance_gate._recent_decisions.append(decision)
                except Exception:
                    pass

        # Process only approved signals through portfolio sizing
        if _approved_dir_signals:
            _batch_results = _signal_mgr.process_signal_batch(
                _approved_dir_signals, journal, ctx, active_exchange, _capital,
                signal_cache=_signal_cache,
                exit_map=_exit_map,
            )
            for sig_result in _batch_results:
                for intent in sig_result.intents:
                    if not _read_only:
                        await ctx.event_bus.publish("execution", intent)
                    else:
                        logger.info(
                            "[READ-ONLY] Would submit: %s %s %s qty=%s (trace=%s)",
                            intent.strategy_id, intent.side.value,
                            intent.symbol, intent.qty,
                            intent.trace_id[:8] if intent.trace_id else "?",
                        )

        # ============================================================
        # Phase 3: Narration for all collected signals
        # ============================================================
        # Build set of approved signal trace IDs for narration
        _approved_trace_ids = {s.trace_id for s in _approved_dir_signals}

        _all_candle_signals = _candle_flat_signals + _candle_dir_signals
        for sig in _all_candle_signals:
            if narration_service is not None and narration_store is not None:
                try:
                    from .narration.schema import (
                        DecisionExplanation,
                        PositionSnapshot,
                        RiskSummary,
                    )

                    # Determine action based on consensus outcome
                    if sig.direction.value == "flat":
                        action = "NO_TRADE"
                    elif sig.trace_id in _approved_trace_ids:
                        action = "HOLD"  # Will narrate ENTER on fill
                    else:
                        action = "NO_TRADE"  # Rejected by consensus

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

                    reasons = humanize_rationale(
                        sig.rationale, sig.features_used,
                    ) if sig.rationale else []

                    # Build market summary with consensus info
                    if sig.direction.value != "flat" and sig.trace_id not in _approved_trace_ids:
                        _market_summary = (
                            f"{sig.symbol}: {sig.direction.value.upper()} signal from "
                            f"{sig.strategy_id} was evaluated by the desk but not approved. "
                            f"The trading desk consulted Market Structure, SMC, CMT, and Risk "
                            f"before making this decision."
                        )
                    elif sig.trace_id in _approved_trace_ids:
                        _market_summary = (
                            f"{sig.symbol}: {sig.direction.value.upper()} signal from "
                            f"{sig.strategy_id} was approved by the desk after consultation "
                            f"with Market Structure, SMC, CMT, and Risk managers."
                        )
                    else:
                        _market_summary = f"{sig.symbol} is currently being analysed."

                    explanation = DecisionExplanation(
                        symbol=sig.symbol,
                        timeframe=event.timeframe.value if hasattr(event.timeframe, "value") else str(event.timeframe),
                        market_summary=_market_summary,
                        active_strategy=sig.strategy_id,
                        active_regime=regime_str,
                        action=action,
                        reasons=reasons,
                        reason_confidences=[sig.confidence],
                        why_not=(
                            reasons if action == "NO_TRADE" else []
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

    # _reconcile_journal_positions — now delegated to _recon_mgr.reconcile_positions()

    # ---------------------------------------------------------------
    # Fill handler: journal + fill-time narration
    # ---------------------------------------------------------------

    async def on_execution_event(event):
        """Handle fills from the execution engine → journal + narration."""
        if not isinstance(event, FillEvent):
            return

        # Record fill metric
        try:
            side_val = event.side.value if hasattr(event.side, "value") else str(event.side)
            record_fill(event.symbol, side_val)
        except Exception:
            pass

        # Classify and record the fill via ReconciliationManager
        fill_result = _recon_mgr.handle_fill(
            event,
            _signal_cache,
            _exit_map,
            fallback_strategy_ids=[
                "trend_following", "mean_reversion", "breakout", "funding_arb",
            ],
        )

        trace_id = event.trace_id
        cached_sig = _signal_cache.get(trace_id)
        strategy_id = fill_result.strategy_id
        is_exit = fill_result.is_exit
        entry_trace_id = fill_result.entry_trace_id
        direction = fill_result.direction

        if is_exit and entry_trace_id:
            # ---- EXIT FILL — narration (journal recording done by handle_fill) ----
            if narration_service is not None and narration_store is not None:
                try:
                    from .narration.schema import (
                        DecisionExplanation,
                        PositionSnapshot,
                        RiskSummary,
                    )

                    regime_str = ""
                    if ctx.regime:
                        regime_str = getattr(ctx.regime, "regime", "unknown")
                        if hasattr(regime_str, "value"):
                            regime_str = regime_str.value

                    reasons = ["Exit signal triggered"]
                    if cached_sig and cached_sig.rationale:
                        reasons = humanize_rationale(
                            cached_sig.rationale, cached_sig.features_used,
                        )

                    explanation = DecisionExplanation(
                        symbol=event.symbol,
                        timeframe="",
                        market_summary=f"Position closed on {event.symbol} at {event.price}.",
                        active_strategy=strategy_id,
                        active_regime=regime_str,
                        action="EXIT",
                        reasons=reasons,
                        reason_confidences=[cached_sig.confidence if cached_sig else 0.5],
                        risk=RiskSummary(health_score=1.0),
                        position=PositionSnapshot(
                            open_positions=journal.open_trade_count,
                        ),
                        trace_id=entry_trace_id,
                    )

                    item = narration_service.generate(explanation, force=True)
                    if item is not None:
                        item.metadata = {
                            "action": "EXIT",
                            "symbol": event.symbol,
                            "regime": regime_str,
                        }
                        narration_store.add(item, explanation=explanation)
                except Exception:
                    logger.debug("Exit narration failed", exc_info=True)

        else:
            # ---- ENTRY FILL — TP/SL + narration (journal recording done by handle_fill) ----

            # ---- Set server-side TP/SL on exchange ----
            _exit_cfg = settings.exits
            if (
                not _read_only
                and adapter is not None
                and _exit_cfg.enabled
            ):
                try:
                    _tp: Decimal | None = None
                    _sl: Decimal | None = None
                    _trail: Decimal | None = None

                    # Ensure direction is always "long" or "short".
                    # fill_result.direction is "" when the signal cache missed
                    # (strategy_id="unknown") and the recon manager couldn't
                    # classify the fill from the journal.  Fall back to the fill
                    # side: BUY → long entry, SELL → short entry.
                    if not direction:
                        _side_val = (
                            event.side.value
                            if hasattr(event.side, "value")
                            else str(event.side)
                        ).lower()
                        direction = "long" if _side_val == "buy" else "short"

                    if cached_sig is not None:
                        _tp = cached_sig.take_profit
                        _sl = cached_sig.stop_loss
                        _trail = cached_sig.trailing_stop

                    # Fallback 1: compute from risk_constraints ATR if signal lacks TP/SL
                    if (_tp is None or _sl is None) and cached_sig is not None:
                        _rc = cached_sig.risk_constraints
                        _fill_price = event.price
                        _atr_val = _rc.get("atr", 0)
                        if _atr_val and float(_atr_val) > 0:
                            _atr_d = Decimal(str(_atr_val))
                            _sl_dist = Decimal(str(
                                _rc.get("stop_distance_atr",
                                        float(_atr_d * Decimal(str(_exit_cfg.sl_atr_multiplier))))
                            ))
                            _tp_dist = _atr_d * Decimal(str(_exit_cfg.tp_atr_multiplier))
                            if direction == "long":
                                _sl = _sl or (_fill_price - _sl_dist)
                                _tp = _tp or (_fill_price + _tp_dist)
                            else:
                                _sl = _sl or (_fill_price + _sl_dist)
                                _tp = _tp or (_fill_price - _tp_dist)

                    # Fallback 2 (last resort): estimate ATR as 0.4% of fill price.
                    # Covers: signal cache miss (trace_id bug), strategies that
                    # produce signals without ATR (9 of 12), or any other path
                    # where both signal-level and risk_constraints TP/SL are absent.
                    # Uses the same formula as startup reconciliation.
                    if _tp is None or _sl is None:
                        _fill_price_fb = event.price
                        _atr_est = _fill_price_fb * Decimal("0.004")
                        _sl_dist_fb = _atr_est * Decimal(str(_exit_cfg.sl_atr_multiplier))
                        _tp_dist_fb = _atr_est * Decimal(str(_exit_cfg.tp_atr_multiplier))
                        if direction == "long":
                            _sl = _sl or (_fill_price_fb - _sl_dist_fb)
                            _tp = _tp or (_fill_price_fb + _tp_dist_fb)
                        else:
                            _sl = _sl or (_fill_price_fb + _sl_dist_fb)
                            _tp = _tp or (_fill_price_fb - _tp_dist_fb)
                        logger.info(
                            "TP/SL fallback (0.4%% ATR est) for %s: tp=%s sl=%s",
                            event.symbol, _tp, _sl,
                        )

                    # Trailing stop for trend strategies
                    _active_price: Decimal | None = None
                    if (
                        _trail is None
                        and strategy_id in _exit_cfg.trailing_strategies
                        and cached_sig is not None
                    ):
                        _rc2 = cached_sig.risk_constraints
                        _atr2 = _rc2.get("atr", 0)
                        if _atr2 and float(_atr2) > 0:
                            _trail = Decimal(str(float(_atr2))) * Decimal(
                                str(_exit_cfg.trailing_stop_atr_multiplier)
                            )

                    # Preserve existing trailing stop: Bybit's set-trading-stop
                    # API replaces ALL fields, so if we send TP/SL without
                    # trailing, any existing trailing stop gets wiped.  When
                    # _trail is still None (non-trailing strategy), query
                    # the exchange and carry forward the active trailing stop.
                    if _trail is None:
                        try:
                            _raw_pos = await adapter._ccxt.fetch_positions(
                                [event.symbol]
                            )
                            _pos_match = next(
                                (p for p in _raw_pos
                                 if (
                                     p.get("symbol") == event.symbol
                                     or p.get("symbol", "").split(":")[0] == event.symbol
                                 )),
                                None,
                            )
                            if _pos_match:
                                _pos_info = _pos_match.get("info") or {}
                                _existing_trail = float(
                                    _pos_info.get("trailingStop") or 0
                                )
                                _existing_active = float(
                                    _pos_info.get("activePrice") or 0
                                )
                                if _existing_trail > 0:
                                    _trail = Decimal(str(_existing_trail))
                                    if _existing_active > 0:
                                        _active_price = Decimal(
                                            str(_existing_active)
                                        )
                                    logger.debug(
                                        "Preserving existing trailing for %s: "
                                        "trail=%s active=%s",
                                        event.symbol, _trail, _active_price,
                                    )
                        except Exception:
                            logger.debug(
                                "Could not query existing trailing for %s",
                                event.symbol, exc_info=True,
                            )

                    # Breakeven activation: trailing stop only arms after price
                    # moves ≥1× SL distance in profit from the fill price.
                    # active_price = fill_price ± SL distance.
                    if _trail is not None and _sl is not None and _active_price is None:
                        _fill_price_for_trail = event.price
                        _sl_dist_for_trail = abs(_fill_price_for_trail - _sl)
                        if direction == "long":
                            _active_price = _fill_price_for_trail + _sl_dist_for_trail
                        else:
                            _active_price = _fill_price_for_trail - _sl_dist_for_trail
                        logger.debug(
                            "Trailing active_price computed for %s: direction=%s fill=%s sl=%s active=%s",
                            event.symbol, direction, _fill_price_for_trail, _sl, _active_price,
                        )

                    if _tp is not None or _sl is not None or _trail is not None:
                        _tpsl_ok = False
                        _tpsl_max_attempts = 5
                        # Brief initial delay: Bybit may not yet reflect the
                        # position server-side immediately after a fill ACK.
                        await asyncio.sleep(0.5)
                        for _tpsl_attempt in range(1, _tpsl_max_attempts + 1):
                            try:
                                if tool_gateway is not None:
                                    from agentic_trading.control_plane.action_types import (
                                        ActionScope as _AS,
                                    )
                                    from agentic_trading.control_plane.action_types import (
                                        ProposedAction as _PA,
                                    )
                                    from agentic_trading.control_plane.action_types import (
                                        ToolName as _TN,
                                    )
                                    _tp_params: dict[str, Any] = {"symbol": event.symbol}
                                    if _tp is not None:
                                        _tp_params["take_profit"] = str(_tp)
                                    if _sl is not None:
                                        _tp_params["stop_loss"] = str(_sl)
                                    if _trail is not None:
                                        _tp_params["trailing_stop"] = str(_trail)
                                    if _active_price is not None:
                                        _tp_params["active_price"] = str(_active_price)
                                    _tpsl_result = await tool_gateway.call(_PA(
                                        tool_name=_TN.SET_TRADING_STOP,
                                        scope=_AS(strategy_id=strategy_id, symbol=event.symbol, actor="main:fill_handler"),
                                        request_params=_tp_params,
                                    ))
                                    if _tpsl_result.success:
                                        _tpsl_ok = True
                                    else:
                                        logger.warning(
                                            "TP/SL attempt %d/%d FAILED on %s via tool_gateway: %s",
                                            _tpsl_attempt, _tpsl_max_attempts,
                                            event.symbol, _tpsl_result.error,
                                        )
                                else:
                                    await adapter.set_trading_stop(
                                        event.symbol,
                                        take_profit=_tp,
                                        stop_loss=_sl,
                                        trailing_stop=_trail,
                                        active_price=_active_price,
                                    )
                                    _tpsl_ok = True
                            except Exception:
                                logger.warning(
                                    "TP/SL attempt %d/%d raised for %s",
                                    _tpsl_attempt, _tpsl_max_attempts,
                                    event.symbol, exc_info=True,
                                )

                            if _tpsl_ok:
                                break
                            if _tpsl_attempt < _tpsl_max_attempts:
                                # Exponential backoff: 1s, 2s, 4s, 8s
                                _backoff = min(2 ** (_tpsl_attempt - 1), 8)
                                await asyncio.sleep(float(_backoff))

                        if _tpsl_ok:
                            logger.info(
                                "TP/SL set on %s: tp=%s sl=%s trail=%s active_price=%s (strategy=%s)",
                                event.symbol, _tp, _sl, _trail, _active_price, strategy_id,
                            )
                        else:
                            logger.error(
                                "TP/SL FAILED on %s after %d attempts "
                                "(tp=%s sl=%s trail=%s strategy=%s). "
                                "Position has NO stop protection!",
                                event.symbol, _tpsl_max_attempts,
                                _tp, _sl, _trail, strategy_id,
                            )

                        # Emit metric
                        try:
                            from .observability.metrics import TRADING_STOPS_SET
                            TRADING_STOPS_SET.labels(
                                symbol=event.symbol, stop_type="entry",
                            ).inc()
                        except Exception:
                            pass
                except Exception:
                    logger.exception(
                        "Failed to set TP/SL for %s after entry fill", event.symbol,
                    )

            # Narrate the confirmed entry
            if narration_service is not None and narration_store is not None and cached_sig:
                try:
                    from .narration.schema import (
                        DecisionExplanation,
                        PositionSnapshot,
                        RiskSummary,
                    )

                    regime_str = ""
                    if ctx.regime:
                        regime_str = getattr(ctx.regime, "regime", "unknown")
                        if hasattr(regime_str, "value"):
                            regime_str = regime_str.value

                    reasons = humanize_rationale(
                        cached_sig.rationale, cached_sig.features_used,
                    ) if cached_sig.rationale else ["Trade conditions met"]

                    explanation = DecisionExplanation(
                        symbol=event.symbol,
                        timeframe="",
                        market_summary=f"Position entered on {event.symbol} at {event.price}.",
                        active_strategy=strategy_id,
                        active_regime=regime_str,
                        action="ENTER",
                        reasons=reasons,
                        reason_confidences=[cached_sig.confidence],
                        risk=RiskSummary(health_score=1.0),
                        position=PositionSnapshot(
                            open_positions=journal.open_trade_count,
                        ),
                        trace_id=trace_id,
                    )

                    item = narration_service.generate(explanation, force=True)
                    if item is not None:
                        item.metadata = {
                            "action": "ENTER",
                            "symbol": event.symbol,
                            "regime": regime_str,
                        }
                        narration_store.add(item, explanation=explanation)
                except Exception:
                    logger.debug("Fill narration failed", exc_info=True)

        # Update journal counts metric
        try:
            update_journal_counts(
                journal.open_trade_count, journal.closed_trade_count,
            )
        except Exception:
            pass

        # Reconcile portfolio state (via ToolGateway when available)
        if adapter is not None:
            try:
                positions = await adapter.get_positions()  # TODO(Day6): route via ToolGateway reads
                balances = await adapter.get_balances()  # TODO(Day6): route via ToolGateway reads
                ctx.portfolio_state = PortfolioState(
                    positions={p.symbol: p for p in positions},
                    balances={b.currency: b for b in balances},
                )
                # Update equity and exposure metrics
                total_equity = sum(
                    float(b.total) for b in balances
                ) + sum(
                    float(p.unrealized_pnl) for p in positions
                )
                gross_exp = float(ctx.portfolio_state.gross_exposure)
                try:
                    update_equity(total_equity)
                    update_gross_exposure(gross_exp)

                    # Position-level metrics
                    for p in positions:
                        p_side = p.side.value if hasattr(p.side, "value") else str(p.side)
                        update_position(p.symbol, p_side, float(p.qty))

                    # Drawdown: compute from initial equity if available
                    if _initial_equity > 0 and total_equity < _initial_equity:
                        dd = ((_initial_equity - total_equity) / _initial_equity)
                        update_drawdown(dd)
                    else:
                        update_drawdown(0.0)

                    # Kill switch status
                    if hasattr(risk_manager, "kill_switch"):
                        ks_active = await risk_manager.kill_switch.is_active()
                    else:
                        ks_active = False
                    update_kill_switch(ks_active)

                    # Daily PnL: equity change since daily baseline
                    nonlocal _daily_equity_base
                    if _daily_equity_base > 0:
                        daily_pnl = total_equity - _daily_equity_base
                        update_daily_pnl(daily_pnl)
                except Exception:
                    pass

                # Reconcile journal: force-close open trades whose
                # positions no longer exist on the exchange.
                _recon_mgr.reconcile_positions(positions)
            except Exception:
                logger.debug("Portfolio state reconciliation failed", exc_info=True)

    await ctx.event_bus.subscribe("execution", "fill_handler", on_execution_event)

    # Start live market data feeds if exchange configs are available
    # Default to top USDT perpetuals by volume on Bybit
    _DEFAULT_SYMBOLS = [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "XRP/USDT",
        "BNB/USDT",
        "DOGE/USDT",
        "TRX/USDT",
        "SUI/USDT",
        "ADA/USDT",
        "AAVE/USDT",
        "ZEC/USDT",
        "PEPE/USDT",
    ]
    symbols = settings.symbols.symbols or _DEFAULT_SYMBOLS
    if settings.exchanges:
        try:
            from .data.candle_builder import CandleBuilder
            from .data.feed_manager import FeedManager

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

    # Start periodic optimizer scheduler (if enabled)
    optimizer_scheduler = None
    if settings.optimizer_scheduler.enabled:
        from .optimizer.scheduler import OptimizerScheduler

        optimizer_scheduler = OptimizerScheduler(
            config=settings.optimizer_scheduler,
            data_dir=settings.backtest.data_dir,
        )
        await optimizer_scheduler.start()
        logger.info(
            "Optimizer scheduler enabled (interval=%.1fh, strategies=%s)",
            settings.optimizer_scheduler.interval_hours,
            settings.optimizer_scheduler.strategies,
        )

    # Startup reconciliation: sync journal against exchange positions
    # so that stale open trades from prior sessions (or externally
    # closed positions) are cleaned up immediately.
    if adapter is not None:
        try:
            positions = await adapter.get_positions()  # TODO(Day6): route via ToolGateway reads
            balances = await adapter.get_balances()  # TODO(Day6): route via ToolGateway reads
            ctx.portfolio_state = PortfolioState(
                positions={p.symbol: p for p in positions},
                balances={b.currency: b for b in balances},
            )
            _recon_mgr.reconcile_positions(positions)

            # Set initial equity baseline for drawdown and daily PnL calculation
            _initial_equity = sum(
                float(b.total) for b in balances
            ) + sum(
                float(p.unrealized_pnl) for p in positions
            )
            _daily_equity_base = _initial_equity
            try:
                update_equity(_initial_equity)
                update_gross_exposure(float(ctx.portfolio_state.gross_exposure))
                update_drawdown(0.0)
                if hasattr(risk_manager, "kill_switch"):
                    ks_active = await risk_manager.kill_switch.is_active()
                else:
                    ks_active = False
                update_kill_switch(ks_active)
                update_daily_pnl(0.0)

                # Emit position-level metrics at startup
                for p in positions:
                    p_side = p.side.value if hasattr(p.side, "value") else str(p.side)
                    update_position(p.symbol, p_side, float(p.qty))
            except Exception:
                pass

            logger.info(
                "Startup reconciliation: %d exchange positions, "
                "%d journal open trades, equity=$%.2f",
                len(positions),
                journal.open_trade_count,
                _initial_equity,
            )

            # ---- Set TP/SL on existing positions that lack them ----
            _exit_cfg = settings.exits
            if _exit_cfg.enabled and not _read_only and positions:
                for _pos in positions:
                    if float(_pos.qty) == 0:
                        continue
                    try:
                        # Check if position already has TP/SL via CCXT normalized data.
                        # fetch_positions() returns CCXT-normalized objects where
                        # takeProfitPrice / stopLossPrice are top-level numeric fields
                        # (CCXT's parse_position maps Bybit's takeProfit/stopLoss to these).
                        # Symbol matching handles both unified CCXT format ('BTC/USDT:USDT')
                        # and plain format ('BTC/USDT').
                        _raw_positions = await adapter._ccxt.fetch_positions(  # TODO(Day6): abstract raw position query
                            [_pos.symbol]
                        )
                        _bybit_pos = next(
                            (p for p in _raw_positions
                             if (
                                 p.get("symbol") == _pos.symbol
                                 or p.get("symbol", "").split(":")[0] == _pos.symbol
                             )),
                            None,
                        )
                        # takeProfitPrice / stopLossPrice are CCXT-normalized top-level
                        # numeric fields (None or 0.0 when unset).
                        _has_tp = float((_bybit_pos or {}).get("takeProfitPrice") or 0) > 0
                        _has_sl = float((_bybit_pos or {}).get("stopLossPrice") or 0) > 0
                        _recon_info = (_bybit_pos or {}).get("info") or {}
                        _has_trail = float(_recon_info.get("trailingStop") or 0) > 0
                        _wants_trail = bool(_exit_cfg.trailing_strategies)

                        _needs_update = (
                            not _has_tp
                            or not _has_sl
                            or (_wants_trail and not _has_trail)
                        )
                        if _needs_update:
                            _entry = _pos.entry_price
                            # Estimate ATR as ~0.4% of price for startup fallback
                            _atr_est = _entry * Decimal("0.004")
                            _sl_d = _atr_est * Decimal(str(_exit_cfg.sl_atr_multiplier))
                            _tp_d = _atr_est * Decimal(str(_exit_cfg.tp_atr_multiplier))

                            _dir = "long" if _pos.side.value == "long" else "short"
                            if _dir == "long":
                                _sl_price = _entry - _sl_d
                                _tp_price = _entry + _tp_d
                            else:
                                _sl_price = _entry + _sl_d
                                _tp_price = _entry - _tp_d

                            # Trailing stop + breakeven active_price
                            _recon_trail: Decimal | None = None
                            _recon_active: Decimal | None = None
                            if _wants_trail and not _has_trail:
                                _trail_mult = Decimal(
                                    str(_exit_cfg.trailing_stop_atr_multiplier)
                                )
                                _recon_trail = _atr_est * _trail_mult
                                if _dir == "long":
                                    _recon_active = _entry + _sl_d
                                else:
                                    _recon_active = _entry - _sl_d

                            # Route TP/SL through ToolGateway (B3 fix)
                            if tool_gateway is not None:
                                from agentic_trading.control_plane.action_types import (
                                    ActionScope as _AS,
                                )
                                from agentic_trading.control_plane.action_types import (
                                    ProposedAction as _PA,
                                )
                                from agentic_trading.control_plane.action_types import (
                                    ToolName as _TN,
                                )
                                _stp_params: dict[str, Any] = {"symbol": _pos.symbol}
                                if not _has_tp:
                                    _stp_params["take_profit"] = str(_tp_price)
                                if not _has_sl:
                                    _stp_params["stop_loss"] = str(_sl_price)
                                if _recon_trail is not None:
                                    _stp_params["trailing_stop"] = str(_recon_trail)
                                if _recon_active is not None:
                                    _stp_params["active_price"] = str(_recon_active)
                                _stp_result = await tool_gateway.call(_PA(
                                    tool_name=_TN.SET_TRADING_STOP,
                                    scope=_AS(symbol=_pos.symbol, actor="main:startup_recon"),
                                    request_params=_stp_params,
                                ))
                                if not _stp_result.success:
                                    logger.error(
                                        "Startup TP/SL FAILED for %s: %s",
                                        _pos.symbol, _stp_result.error,
                                    )
                                else:
                                    logger.info(
                                        "Startup TP/SL set for %s (%s): "
                                        "tp=%s sl=%s trail=%s active=%s",
                                        _pos.symbol, _dir,
                                        _tp_price if not _has_tp else "kept",
                                        _sl_price if not _has_sl else "kept",
                                        _recon_trail or "none",
                                        _recon_active or "none",
                                    )
                            else:
                                await adapter.set_trading_stop(
                                    _pos.symbol,
                                    take_profit=_tp_price if not _has_tp else None,
                                    stop_loss=_sl_price if not _has_sl else None,
                                    trailing_stop=_recon_trail,
                                    active_price=_recon_active,
                                )
                                logger.info(
                                    "Startup TP/SL set for %s (%s): "
                                    "tp=%s sl=%s trail=%s active=%s",
                                    _pos.symbol, _dir,
                                    _tp_price if not _has_tp else "kept",
                                    _sl_price if not _has_sl else "kept",
                                    _recon_trail or "none",
                                    _recon_active or "none",
                                )
                    except Exception:
                        logger.warning(
                            "Failed to set startup TP/SL for %s",
                            _pos.symbol,
                            exc_info=True,
                        )

        except Exception:
            logger.warning("Startup position reconciliation failed", exc_info=True)

    # ---------------------------------------------------------------
    # Refurb components: AgentRegistry, ApprovalManager,
    # ExecutionQualityTracker, StrategyLifecycleManager,
    # IncidentManager, DailyEffectivenessScorecard
    # ---------------------------------------------------------------
    from .agents.registry import AgentRegistry
    from .execution.quality_tracker import ExecutionQualityTracker
    from .governance.approval_manager import ApprovalManager
    from .governance.incident_manager import IncidentManager
    from .governance.strategy_lifecycle import StrategyLifecycleManager
    from .observability.daily_scorecard import DailyEffectivenessScorecard

    agent_registry = AgentRegistry()

    # ApprovalManager (uses existing approval rules from governance config)
    approval_rules = []
    if settings.governance.enabled:
        try:
            from .governance.approval_models import ApprovalRule
            # Use governance-configured rules if available
            approval_cfg = getattr(settings.governance, "approval", None)
            if approval_cfg and hasattr(approval_cfg, "rules"):
                approval_rules = approval_cfg.rules
        except Exception:
            pass
    approval_manager = ApprovalManager(
        rules=approval_rules,
        event_bus=ctx.event_bus,
        auto_approve_l1=True,
    )

    # ExecutionQualityTracker
    quality_tracker = ExecutionQualityTracker(window_size=500)

    # IncidentManager (extends BaseAgent — subscribes to risk/system topics)
    incident_manager = IncidentManager(
        event_bus=ctx.event_bus,
        interval=30.0,
    )
    agent_registry.register(incident_manager)

    # StrategyLifecycleManager (extends BaseAgent — periodic evidence checks)
    lifecycle_manager = StrategyLifecycleManager(
        event_bus=ctx.event_bus,
        journal=journal,
        governance_gate=governance_gate,
        interval=60.0,
    )
    agent_registry.register(lifecycle_manager)

    # Register strategies with lifecycle manager
    for strat in strategies:
        try:
            lifecycle_manager.register_strategy(strat.strategy_id)
        except Exception:
            logger.debug(
                "Could not register strategy %s with lifecycle manager",
                getattr(strat, "strategy_id", "?"),
            )

    # DailyEffectivenessScorecard
    scorecard = DailyEffectivenessScorecard(
        journal=journal,
        quality_tracker=quality_tracker,
        risk_manager=risk_manager,
        agent_registry=agent_registry,
        event_bus=ctx.event_bus,
    )

    # TpSlWatchdog — runs every 5 minutes, re-applies missing TP/SL on open positions
    from .execution.tpsl_watchdog import TpSlWatchdog

    tpsl_watchdog = TpSlWatchdog(
        adapter=adapter,
        exit_cfg=settings.exits,
        interval=300.0,
        tool_gateway=tool_gateway if not _read_only else None,
        trailing_strategies=list(settings.exits.trailing_strategies),
        read_only=_read_only,
    )
    agent_registry.register(tpsl_watchdog)

    # PredictionMarketAgent — polls Polymarket Gamma API for consensus data
    if settings.prediction_market.enabled:
        from .agents.prediction_market import PredictionMarketAgent

        pm_agent = PredictionMarketAgent(
            event_bus=ctx.event_bus,
            config=settings.prediction_market,
            symbols=settings.symbols.symbols or [],
            agent_id="prediction-market",
        )
        agent_registry.register(pm_agent)

        # Wire PM agent into PortfolioManager for confidence adjustment
        _signal_mgr.portfolio_manager.set_prediction_market_agent(pm_agent)
        _signal_mgr.portfolio_manager.configure_pm(
            max_boost=settings.prediction_market.max_confidence_boost,
            shadow_mode=settings.prediction_market.shadow_mode,
        )
        logger.info(
            "PredictionMarketAgent created (max_boost=%.2f, shadow_mode=%s)",
            settings.prediction_market.max_confidence_boost,
            settings.prediction_market.shadow_mode,
        )

    # Start all registered agents
    await agent_registry.start_all()
    logger.info(
        "Refurb components wired: AgentRegistry(%d agents), ApprovalManager, "
        "ExecutionQualityTracker, StrategyLifecycleManager, IncidentManager, "
        "DailyEffectivenessScorecard, TpSlWatchdog",
        agent_registry.count,
    )

    # ---------------------------------------------------------------
    # Institutional components for Supervision UI
    # ---------------------------------------------------------------
    model_registry = None
    try:
        from .intelligence.model_registry import ModelRegistry

        model_registry = ModelRegistry()
        logger.info("ModelRegistry initialized (in-memory)")
    except Exception:
        logger.debug("ModelRegistry not available", exc_info=True)

    if model_registry is not None:
        from .intelligence.model_registry import ModelStage
        stage_map = {
            "candidate": ModelStage.RESEARCH, "backtest": ModelStage.RESEARCH,
            "eval_pack": ModelStage.RESEARCH, "paper": ModelStage.PAPER,
            "limited": ModelStage.LIMITED, "scale": ModelStage.PRODUCTION,
            "demoted": ModelStage.RETIRED,
        }
        for strat in strategies:
            try:
                sid = strat.strategy_id
                lc_stage = lifecycle_manager.get_all_stages().get(sid, "candidate")
                rec = model_registry.register(
                    name=sid,
                    description=f"Strategy: {sid}",
                    metrics={"win_rate": 0.0, "sharpe": 0.0},
                    tags=["strategy", "auto-registered"],
                )
                target = stage_map.get(lc_stage, ModelStage.RESEARCH)
                if target != ModelStage.RESEARCH:
                    model_registry.promote(rec.model_id, target, approved_by="system", reason="auto")
            except Exception:
                logger.debug("Could not register %s in ModelRegistry", getattr(strat, "strategy_id", "?"))

    case_manager = None
    try:
        from .compliance.case_manager import CaseManager

        case_manager = CaseManager()
        logger.info("CaseManager initialized (in-memory)")
    except Exception:
        logger.debug("CaseManager not available", exc_info=True)

    pre_trade_checker = getattr(risk_manager, "pre_trade", None)

    # ---------------------------------------------------------------
    # Supervision UI (HTMX dashboard)
    # ---------------------------------------------------------------
    ui_server = None
    ui_task = None
    if settings.ui.enabled:
        try:
            import uvicorn

            from .ui.app import create_ui_app

            ui_app = create_ui_app(
                journal=journal,
                agent_registry=agent_registry,
                governance_gate=governance_gate,
                approval_manager=approval_manager,
                incident_manager=incident_manager,
                scorecard=scorecard,
                lifecycle_manager=lifecycle_manager,
                quality_tracker=quality_tracker,
                event_bus=ctx.event_bus,
                settings=settings,
                risk_manager=risk_manager,
                adapter=adapter,
                trading_context=ctx,
                model_registry=model_registry,
                case_manager=case_manager,
                pre_trade_checker=pre_trade_checker,
            )

            ui_config = uvicorn.Config(
                ui_app,
                host=settings.ui.host,
                port=settings.ui.port,
                log_level="warning",  # Suppress uvicorn access logs
                access_log=False,
            )
            ui_server = uvicorn.Server(ui_config)

            # Run uvicorn in a background task so it doesn't block the trading loop
            ui_task = asyncio.create_task(
                ui_server.serve(), name="supervision-ui",
            )
            logger.info(
                "Supervision UI started on http://%s:%d",
                settings.ui.host, settings.ui.port,
            )
        except ImportError:
            logger.warning(
                "Supervision UI disabled — install fastapi + uvicorn: "
                "pip install fastapi jinja2 uvicorn"
            )
        except Exception:
            logger.warning("Failed to start supervision UI", exc_info=True)

    logger.info(
        "Trading loop running (%s mode). %s. Press Ctrl+C to stop.",
        settings.mode.value,
        f"Receiving live feeds for {symbols}" if feed_manager else "Waiting for market data on event bus",
    )
    await stop_event.wait()

    # Graceful shutdown
    await execution_engine.stop()
    if adapter is not None and hasattr(adapter, "close"):
        try:
            await adapter.close()
        except Exception:
            pass
    if ui_server is not None:
        ui_server.should_exit = True
        if ui_task is not None:
            try:
                await asyncio.wait_for(ui_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                ui_task.cancel()
        logger.info("Supervision UI stopped")
    if narration_runner is not None:
        await narration_runner.cleanup()
        logger.info("Narration server stopped")
    # Stop refurb agents (IncidentManager, StrategyLifecycleManager)
    try:
        await agent_registry.stop_all()
    except Exception:
        logger.warning("Agent registry shutdown failed", exc_info=True)
    if governance_canary is not None:
        await governance_canary.stop()
    if optimizer_scheduler is not None:
        await optimizer_scheduler.stop()
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
    import agentic_trading.strategies.bb_squeeze  # noqa: F401
    import agentic_trading.strategies.breakout  # noqa: F401
    import agentic_trading.strategies.fibonacci_confluence  # noqa: F401
    import agentic_trading.strategies.funding_arb  # noqa: F401
    import agentic_trading.strategies.mean_reversion  # noqa: F401
    import agentic_trading.strategies.mean_reversion_enhanced  # noqa: F401
    import agentic_trading.strategies.multi_tf_ma  # noqa: F401
    import agentic_trading.strategies.obv_divergence  # noqa: F401
    import agentic_trading.strategies.rsi_divergence  # noqa: F401
    import agentic_trading.strategies.stochastic_macd  # noqa: F401
    import agentic_trading.strategies.supply_demand  # noqa: F401

    # Import strategy modules to trigger @register_strategy decorators
    import agentic_trading.strategies.trend_following  # noqa: F401

    from .backtester.engine import BacktestEngine
    from .core.config import load_settings
    from .data.historical import HistoricalDataLoader
    from .features.engine import FeatureEngine
    from .strategies.registry import create_strategy
    from .strategies.research.experiment_log import (
        ExperimentConfig,
        ExperimentLogger,
        ExperimentResult,
    )
    from .strategies.research.walk_forward import WalkForwardResult, WalkForwardValidator

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


async def run_optimize(
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
    strategy_id: str | None = None,
    n_samples: int = 30,
) -> None:
    """Run one-off parameter optimization.

    Loads historical data and runs ParameterOptimizer for each enabled
    strategy (or a single specified strategy).
    """
    import agentic_trading.strategies.bb_squeeze  # noqa: F401
    import agentic_trading.strategies.breakout  # noqa: F401
    import agentic_trading.strategies.fibonacci_confluence  # noqa: F401
    import agentic_trading.strategies.funding_arb  # noqa: F401
    import agentic_trading.strategies.mean_reversion  # noqa: F401
    import agentic_trading.strategies.mean_reversion_enhanced  # noqa: F401
    import agentic_trading.strategies.multi_tf_ma  # noqa: F401
    import agentic_trading.strategies.obv_divergence  # noqa: F401
    import agentic_trading.strategies.rsi_divergence  # noqa: F401
    import agentic_trading.strategies.stochastic_macd  # noqa: F401
    import agentic_trading.strategies.supply_demand  # noqa: F401

    # Import strategy modules to trigger @register_strategy decorators
    import agentic_trading.strategies.trend_following  # noqa: F401

    from .core.config import load_settings
    from .data.historical import HistoricalDataLoader
    from .features.engine import FeatureEngine
    from .optimizer.engine import ParameterOptimizer
    from .optimizer.report import print_summary

    settings = load_settings(config_path=config_path, overrides=overrides)
    _setup_logging(settings)

    bt = settings.backtest
    logger.info("Optimizer mode: %s to %s", bt.start_date, bt.end_date)

    # Determine which strategies to optimize
    if strategy_id:
        strategy_ids = [strategy_id]
    else:
        strategy_ids = [
            s.strategy_id
            for s in settings.strategies
            if s.enabled
        ]

    if not strategy_ids:
        logger.error("No strategies to optimize.")
        return

    # Load historical data
    loader = HistoricalDataLoader(data_dir=bt.data_dir)
    symbols = settings.symbols.symbols or ["BTC/USDT"]

    # Try Binance first, then Bybit
    available_binance = loader.available_symbols(Exchange.BINANCE)
    available_bybit = loader.available_symbols(Exchange.BYBIT)

    if available_binance:
        exchange = Exchange.BINANCE
    elif available_bybit:
        exchange = Exchange.BYBIT
    else:
        logger.error("No historical data found. Run download_historical.py first.")
        return

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

    if not candles_by_symbol:
        logger.error("No historical data found. Run download_historical.py first.")
        return

    # Save results
    import json
    from pathlib import Path

    results_dir = Path(settings.optimizer_scheduler.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run optimizer for each strategy
    for sid in strategy_ids:
        logger.info("Optimizing: %s (%d samples)", sid, n_samples)

        feature_engine = FeatureEngine(indicator_config={"smc_enabled": False})
        optimizer = ParameterOptimizer(
            strategy_id=sid,
            candles_by_symbol=candles_by_symbol,
            feature_engine=feature_engine,
            initial_capital=bt.initial_capital,
            slippage_bps=bt.slippage_bps,
            fee_maker=bt.fee_maker,
            fee_taker=bt.fee_taker,
            seed=bt.random_seed,
        )

        report = await optimizer.run(n_samples=n_samples)
        print_summary(report)

    logger.info("Optimization complete.")


def _setup_logging(settings: Settings) -> None:
    """Configure structured logging."""
    level = getattr(logging, settings.observability.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
