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

    # Start Prometheus metrics server
    try:
        from .observability.metrics import (
            start_metrics_server,
            record_signal,
            record_candle_processed,
            update_equity,
        )

        metrics_port = settings.observability.metrics_port
        start_metrics_server(port=metrics_port, mode=settings.mode.value)
        logger.info("Prometheus metrics server started on port %d", metrics_port)
    except Exception:
        logger.warning("Failed to start metrics server", exc_info=True)

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

    # Paper adapter for simulated execution
    if settings.mode == Mode.PAPER:
        from .execution.adapters.paper import PaperAdapter

        adapter = PaperAdapter(
            exchange=Exchange.BINANCE,
            initial_balances={"USDT": Decimal("100000")},
        )
        logger.info("Paper adapter ready with 100,000 USDT")

    # Wire feature vectors to strategy signals
    from .core.events import FeatureVector

    async def on_feature_vector(event):
        if not isinstance(event, FeatureVector):
            return

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
