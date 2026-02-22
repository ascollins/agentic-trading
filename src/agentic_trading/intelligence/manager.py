"""Unified IntelligenceManager facade for the intelligence layer.

Composes market data ingestion (FeedManager, CandleBuilder),
feature computation (FeatureEngine), historical data loading, data
quality checking, and technical analysis (HTFAnalyzer, SMC) into a
single entry point.

Usage::

    from agentic_trading.intelligence.manager import IntelligenceManager

    # Live / paper mode — event-bus-driven
    mgr = IntelligenceManager.from_config(
        event_bus=event_bus,
        exchange_configs=settings.exchanges,
        symbols=["BTC/USDT"],
    )
    await mgr.start()

    # Backtest mode — direct compute, no event bus
    mgr = IntelligenceManager.from_config(
        data_dir="data/historical",
    )
    candles = mgr.load_candles(Exchange.BYBIT, "BTC/USDT", Timeframe.M1)
    for c in candles:
        mgr.feature_engine.add_candle(c)
    fv = mgr.compute_features("BTC/USDT", Timeframe.M1, candles)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class IntelligenceManager:
    """Unified facade for the intelligence layer.

    Owns data ingestion, feature computation, historical loading, data
    quality validation, and analysis tooling.  Provides a clean lifecycle
    (start / stop) and accessor properties for each sub-component.

    Parameters
    ----------
    feature_engine:
        FeatureEngine for indicator computation.
    candle_builder:
        CandleBuilder for multi-timeframe candle aggregation.
    feed_manager:
        FeedManager for live WebSocket feeds (``None`` in backtest).
    historical_loader:
        HistoricalDataLoader for reading Parquet data.
    data_qa:
        DataQualityChecker for data validation.
    htf_analyzer:
        HTFAnalyzer for higher-timeframe structure assessment.
    smc_scorer:
        SMCConfluenceScorer for multi-timeframe SMC alignment.
    trade_plan_generator:
        SMCTradePlanGenerator for automated trade plan creation.
    """

    def __init__(
        self,
        feature_engine: Any,
        candle_builder: Any | None = None,
        feed_manager: Any | None = None,
        historical_loader: Any | None = None,
        data_qa: Any | None = None,
        htf_analyzer: Any | None = None,
        smc_scorer: Any | None = None,
        trade_plan_generator: Any | None = None,
        open_interest_engine: Any | None = None,
        orderbook_engine: Any | None = None,
        correlation_tracker: Any | None = None,
        kline_provider: Any | None = None,
        bootstrap_config: Any | None = None,
    ) -> None:
        self._feature_engine = feature_engine
        self._candle_builder = candle_builder
        self._feed_manager = feed_manager
        self._historical_loader = historical_loader
        self._data_qa = data_qa
        self._htf_analyzer = htf_analyzer
        self._smc_scorer = smc_scorer
        self._trade_plan_generator = trade_plan_generator
        self._open_interest_engine = open_interest_engine
        self._orderbook_engine = orderbook_engine
        self._correlation_tracker = correlation_tracker
        self._kline_provider = kline_provider
        self._bootstrap_config = bootstrap_config

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        event_bus: Any | None = None,
        exchange_configs: list | None = None,
        symbols: list[str] | None = None,
        indicator_config: dict[str, Any] | None = None,
        buffer_size: int = 500,
        target_timeframes: list | None = None,
        data_dir: str | Path = "data/historical",
        htf_trend_ema_fast: int = 21,
        htf_trend_ema_slow: int = 50,
        htf_adx_threshold: float = 25.0,
        bootstrap_config: Any | None = None,
    ) -> IntelligenceManager:
        """Build a fully wired IntelligenceManager from configuration.

        Parameters
        ----------
        event_bus:
            Event bus (``IEventBus``).  Required for live / paper mode.
            When ``None``, creates a backtest-mode manager (no feeds).
        exchange_configs:
            Exchange connection configs for live data feeds.
        symbols:
            Symbols to subscribe to for live feeds.
        indicator_config:
            Custom indicator configuration for the FeatureEngine.
        buffer_size:
            FeatureEngine candle buffer size (default 500).
        target_timeframes:
            CandleBuilder target timeframes (default M5-D1).
        data_dir:
            Directory for historical Parquet data.
        htf_trend_ema_fast:
            Fast EMA period for HTF trend analysis (default 21).
        htf_trend_ema_slow:
            Slow EMA period for HTF trend analysis (default 50).
        htf_adx_threshold:
            ADX threshold for trend detection (default 25.0).

        Returns
        -------
        IntelligenceManager
        """
        from agentic_trading.intelligence.features.engine import FeatureEngine

        # --- Feature engine (always needed) ---
        feature_engine = FeatureEngine(
            event_bus=event_bus,
            buffer_size=buffer_size,
            indicator_config=indicator_config,
        )

        # --- Candle builder (always created, bus-wired if available) ---
        candle_builder = None
        if event_bus is not None:
            from agentic_trading.intelligence.candle_builder import CandleBuilder

            candle_builder = CandleBuilder(
                event_bus=event_bus,
                target_timeframes=target_timeframes,
            )

        # --- Feed manager (only in live / paper mode) ---
        feed_manager = None
        if (
            event_bus is not None
            and exchange_configs
            and symbols
        ):
            from agentic_trading.intelligence.feed_manager import FeedManager

            feed_manager = FeedManager(
                event_bus=event_bus,
                candle_builder=candle_builder,
                exchange_configs=exchange_configs,
                symbols=symbols,
            )

        # --- Historical data loader ---
        from agentic_trading.intelligence.historical import HistoricalDataLoader

        historical_loader = HistoricalDataLoader(data_dir=data_dir)

        # --- Data quality checker ---
        from agentic_trading.intelligence.data_qa import DataQualityChecker

        data_qa = DataQualityChecker()

        # --- Analysis tools ---
        from agentic_trading.intelligence.analysis.htf_analyzer import HTFAnalyzer
        from agentic_trading.intelligence.analysis.smc_confluence import (
            SMCConfluenceScorer,
        )
        from agentic_trading.intelligence.analysis.smc_trade_plan import (
            SMCTradePlanGenerator,
        )

        smc_scorer = SMCConfluenceScorer()
        htf_analyzer = HTFAnalyzer(
            trend_ema_fast=htf_trend_ema_fast,
            trend_ema_slow=htf_trend_ema_slow,
            adx_trend_threshold=htf_adx_threshold,
            smc_scorer=smc_scorer,
        )
        trade_plan_generator = SMCTradePlanGenerator()

        # --- Auxiliary data engines (event-bus-driven only) ---
        open_interest_engine = None
        orderbook_engine = None
        correlation_tracker = None

        if event_bus is not None:
            from agentic_trading.intelligence.features.open_interest import (
                OpenInterestEngine,
            )
            from agentic_trading.intelligence.features.orderbook import (
                OrderbookEngine,
            )
            from agentic_trading.intelligence.features.correlation import (
                LiveCorrelationTracker,
            )

            open_interest_engine = OpenInterestEngine(event_bus=event_bus)
            orderbook_engine = OrderbookEngine(event_bus=event_bus)
            correlation_tracker = LiveCorrelationTracker(event_bus=event_bus)

        # --- REST kline provider (bootstrap only, non-backtest) ---
        kline_provider = None
        if bootstrap_config is not None and bootstrap_config.enabled:
            from agentic_trading.intelligence.rest_kline_provider import (
                BinanceKlineProvider,
                MarketType,
            )

            mt = MarketType(bootstrap_config.default_market_type)
            kline_provider = BinanceKlineProvider(
                market_type=mt,
                max_limit=bootstrap_config.rest_limit,
                rate_limit_rpm=bootstrap_config.rate_limit_rpm,
                max_retries=bootstrap_config.max_retries,
                base_backoff=bootstrap_config.base_backoff,
                timeout=bootstrap_config.timeout,
            )

        return cls(
            feature_engine=feature_engine,
            candle_builder=candle_builder,
            feed_manager=feed_manager,
            historical_loader=historical_loader,
            data_qa=data_qa,
            htf_analyzer=htf_analyzer,
            smc_scorer=smc_scorer,
            trade_plan_generator=trade_plan_generator,
            open_interest_engine=open_interest_engine,
            orderbook_engine=orderbook_engine,
            correlation_tracker=correlation_tracker,
            kline_provider=kline_provider,
            bootstrap_config=bootstrap_config,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, *, symbols: list[str] | None = None) -> None:
        """Start event-driven components (feature engine, feed manager).

        If a bootstrap config is present, seeds FeatureEngine buffers
        from Binance REST klines **after** FeatureEngine starts but
        **before** FeedManager starts live streaming.

        Parameters
        ----------
        symbols:
            Symbols to bootstrap.  When ``None``, bootstrap is skipped.
        """
        if self._feature_engine is not None:
            await self._feature_engine.start()

        # Bootstrap FeatureEngine buffers before live feed starts
        if symbols and self._kline_provider is not None:
            await self.bootstrap(symbols)

        if self._open_interest_engine is not None:
            await self._open_interest_engine.start()
        if self._orderbook_engine is not None:
            await self._orderbook_engine.start()
        if self._correlation_tracker is not None:
            await self._correlation_tracker.start()
        if self._feed_manager is not None:
            await self._feed_manager.start()
        logger.info("IntelligenceManager started")

    async def bootstrap(self, symbols: list[str]) -> None:
        """Seed FeatureEngine ring buffers from Binance REST klines.

        For each symbol and each timeframe (M1 through D1), fetches
        ``backfill_days`` worth of historical candles and feeds them
        into the feature engine.  Runs gap detection for data quality
        logging.

        Parameters
        ----------
        symbols:
            List of unified symbols (e.g. ``["BTC/USDT", "ETH/USDT"]``).
        """
        if self._kline_provider is None or self._bootstrap_config is None:
            return

        from agentic_trading.core.enums import Timeframe
        from agentic_trading.intelligence.rest_kline_provider import MarketType

        config = self._bootstrap_config
        timeframes = [
            Timeframe.M1, Timeframe.M5, Timeframe.M15,
            Timeframe.H1, Timeframe.H4, Timeframe.D1,
        ]
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=config.backfill_days)

        total_candles = 0
        total_feeds = 0

        try:
            await self._kline_provider.open()

            for symbol in symbols:
                # Per-symbol market type override
                mt_str = config.symbol_market_types.get(
                    symbol, config.default_market_type
                )
                mt = MarketType(mt_str)

                for tf in timeframes:
                    try:
                        candles = await self._kline_provider.fetch_historical(
                            symbol, tf,
                            start_time=start_time,
                            end_time=end_time,
                            market_type=mt,
                        )

                        # Data quality check (log warnings only)
                        if candles and self._data_qa is not None:
                            issues = self._data_qa.check_gaps(candles, tf)
                            for issue in issues:
                                logger.warning(
                                    "Bootstrap data gap: %s:%s — %s",
                                    symbol, tf.value, issue,
                                )

                        # Feed into FeatureEngine
                        for candle in candles:
                            self._feature_engine.add_candle(candle)

                        total_candles += len(candles)
                        total_feeds += 1
                        logger.info(
                            "Bootstrap: loaded %d candles for %s:%s",
                            len(candles), symbol, tf.value,
                        )
                    except Exception:
                        logger.exception(
                            "Bootstrap failed for %s:%s — skipping",
                            symbol, tf.value,
                        )
        finally:
            await self._kline_provider.close()

        logger.info(
            "Bootstrap complete: %d total candles across %d feeds",
            total_candles, total_feeds,
        )

    async def stop(self) -> None:
        """Stop event-driven components."""
        if self._feed_manager is not None:
            await self._feed_manager.stop()
        if self._correlation_tracker is not None:
            await self._correlation_tracker.stop()
        if self._orderbook_engine is not None:
            await self._orderbook_engine.stop()
        if self._open_interest_engine is not None:
            await self._open_interest_engine.stop()
        if self._feature_engine is not None:
            await self._feature_engine.stop()
        logger.info("IntelligenceManager stopped")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def feature_engine(self):
        """The feature computation engine."""
        return self._feature_engine

    @property
    def candle_builder(self):
        """Multi-timeframe candle builder (``None`` in backtest)."""
        return self._candle_builder

    @property
    def feed_manager(self):
        """Live market-data feed manager (``None`` in backtest)."""
        return self._feed_manager

    @property
    def historical_loader(self):
        """Historical Parquet data loader."""
        return self._historical_loader

    @property
    def data_qa(self):
        """Data quality checker."""
        return self._data_qa

    @property
    def htf_analyzer(self):
        """Higher-timeframe structure analyzer."""
        return self._htf_analyzer

    @property
    def smc_scorer(self):
        """SMC multi-timeframe confluence scorer."""
        return self._smc_scorer

    @property
    def trade_plan_generator(self):
        """SMC-based trade plan generator."""
        return self._trade_plan_generator

    @property
    def open_interest_engine(self):
        """Open interest feature engine (``None`` in backtest)."""
        return self._open_interest_engine

    @property
    def orderbook_engine(self):
        """Orderbook depth feature engine (``None`` in backtest)."""
        return self._orderbook_engine

    @property
    def correlation_tracker(self):
        """Live BTC correlation tracker (``None`` in backtest)."""
        return self._correlation_tracker

    # ------------------------------------------------------------------
    # Delegated operations — feature computation
    # ------------------------------------------------------------------

    def compute_features(self, symbol: str, timeframe, candles: list):
        """Compute features for a batch of candles (backtest shortcut).

        Delegates to ``feature_engine.compute_features()``.
        """
        return self._feature_engine.compute_features(symbol, timeframe, candles)

    def add_candle(self, candle) -> None:
        """Add a candle to the feature engine buffer (backtest shortcut).

        Delegates to ``feature_engine.add_candle()``.
        """
        self._feature_engine.add_candle(candle)

    def get_buffer(self, symbol: str, timeframe) -> list:
        """Get candle buffer for a symbol/timeframe pair.

        Delegates to ``feature_engine.get_buffer()``.
        """
        return self._feature_engine.get_buffer(symbol, timeframe)

    # ------------------------------------------------------------------
    # Delegated operations — historical data
    # ------------------------------------------------------------------

    def load_candles(
        self,
        exchange,
        symbol: str,
        timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list:
        """Load historical candles from Parquet files.

        Delegates to ``historical_loader.load_candles()``.
        """
        if self._historical_loader is None:
            raise RuntimeError("No historical loader configured")
        return self._historical_loader.load_candles(
            exchange, symbol, timeframe, start=start, end=end,
        )

    # ------------------------------------------------------------------
    # Delegated operations — analysis
    # ------------------------------------------------------------------

    def analyze_htf(self, symbol: str, aligned_features: dict[str, float]):
        """Run higher-timeframe structure analysis.

        Delegates to ``htf_analyzer.analyze()``.
        """
        if self._htf_analyzer is None:
            raise RuntimeError("No HTF analyzer configured")
        return self._htf_analyzer.analyze(symbol, aligned_features)

    def score_smc_confluence(
        self, symbol: str, aligned_features: dict[str, float],
    ):
        """Score SMC confluence across timeframes.

        Delegates to ``smc_scorer.score()``.
        """
        if self._smc_scorer is None:
            raise RuntimeError("No SMC scorer configured")
        return self._smc_scorer.score(symbol, aligned_features)

    def generate_smc_report(
        self,
        symbol: str,
        current_price: float,
        aligned_features: dict[str, float],
    ):
        """Generate an SMC analysis report.

        Delegates to ``trade_plan_generator.generate_report()``.
        """
        if self._trade_plan_generator is None:
            raise RuntimeError("No trade plan generator configured")
        return self._trade_plan_generator.generate_report(
            symbol, current_price, aligned_features,
        )

    def generate_trade_plan(self, report):
        """Generate a trade plan from an SMC analysis report.

        Delegates to ``trade_plan_generator.generate_trade_plan()``.
        """
        if self._trade_plan_generator is None:
            raise RuntimeError("No trade plan generator configured")
        return self._trade_plan_generator.generate_trade_plan(report)

    def check_data_quality(self, candles, timeframe) -> list:
        """Run gap detection on a candle series.

        Returns a list of ``DataQualityIssue`` objects for any gaps
        detected in the series.  For per-candle checks (staleness,
        price sanity, volume anomaly), use ``data_qa`` directly.

        Parameters
        ----------
        candles:
            Sorted sequence of candles.
        timeframe:
            Expected candle timeframe.

        Returns
        -------
        list[DataQualityIssue]
        """
        if self._data_qa is None:
            raise RuntimeError("No data QA checker configured")
        return self._data_qa.check_gaps(candles, timeframe)
