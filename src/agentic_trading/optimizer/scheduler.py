"""Periodic optimizer scheduler.

Runs the ParameterOptimizer on a configurable interval to discover
improved strategy parameters.  Follows the GovernanceCanary asyncio
pattern: ``start()`` creates a background ``asyncio.Task`` that loops
with ``asyncio.sleep``, and ``stop()`` cancels it gracefully.

CPU-bound backtest work is offloaded via ``asyncio.to_thread()`` so the
main event loop remains responsive for trading.

Results are persisted as JSON files with automatic rotation (keep last N).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.config import OptimizerSchedulerConfig
from agentic_trading.core.enums import AgentType
from agentic_trading.core.events import AgentCapabilities

logger = logging.getLogger(__name__)


class OptimizerScheduler(BaseAgent):
    """Periodic background optimizer runner.

    Usage::

        scheduler = OptimizerScheduler(config, data_dir="data/historical")
        await scheduler.start()
        # ... runs in background ...
        await scheduler.stop()
    """

    def __init__(
        self,
        config: OptimizerSchedulerConfig,
        data_dir: str = "data/historical",
        on_results_callback: Any | None = None,
        *,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            interval=config.interval_hours * 3600,
        )
        self._config = config
        self._data_dir = data_dir
        self._on_results = on_results_callback
        self._last_run: datetime | None = None
        self._run_count = 0
        self._results_dir = Path(config.results_dir)

    # ------------------------------------------------------------------
    # IAgent
    # ------------------------------------------------------------------

    @property
    def agent_type(self) -> AgentType:
        return AgentType.OPTIMIZER

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=[],
            publishes_to=[],
            description="Periodic strategy parameter optimizer with walk-forward validation",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _on_start(self) -> None:
        self._results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "OptimizerScheduler started (interval=%.1fh, strategies=%s, "
            "initial_delay=%.1fm)",
            self._config.interval_hours,
            self._config.strategies,
            self._config.initial_delay_minutes,
        )

    async def _on_stop(self) -> None:
        logger.info("OptimizerScheduler stopped (ran %d times)", self._run_count)

    # ------------------------------------------------------------------
    # BaseAgent periodic work (overrides _loop for initial delay)
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """Internal scheduler loop with initial delay."""
        # Initial delay to let the trading system warm up
        delay_secs = self._config.initial_delay_minutes * 60
        if delay_secs > 0:
            logger.info(
                "OptimizerScheduler waiting %.0fs before first run...",
                delay_secs,
            )
            await asyncio.sleep(delay_secs)

        while self._running:
            try:
                await self._run_optimization_cycle()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._error_count += 1
                logger.error(
                    "OptimizerScheduler cycle failed", exc_info=True
                )
            await asyncio.sleep(self._interval)

    async def _run_optimization_cycle(self) -> None:
        """Run one full optimization cycle across configured strategies."""
        start_time = datetime.now(timezone.utc)
        self._run_count += 1
        logger.info(
            "OptimizerScheduler cycle #%d starting at %s",
            self._run_count,
            start_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

        cycle_results: dict[str, Any] = {
            "run_number": self._run_count,
            "started_at": start_time.isoformat(),
            "strategies": {},
        }

        for strategy_id in self._config.strategies:
            try:
                result = await self._optimize_strategy(strategy_id)
                cycle_results["strategies"][strategy_id] = result
            except Exception as e:
                logger.error(
                    "Optimizer failed for %s: %s", strategy_id, e,
                    exc_info=True,
                )
                cycle_results["strategies"][strategy_id] = {
                    "error": str(e),
                }

        end_time = datetime.now(timezone.utc)
        cycle_results["completed_at"] = end_time.isoformat()
        cycle_results["duration_seconds"] = (
            end_time - start_time
        ).total_seconds()

        # Persist results
        self._save_results(cycle_results, start_time)
        self._rotate_results()
        self._last_run = end_time

        logger.info(
            "OptimizerScheduler cycle #%d completed in %.1fs",
            self._run_count,
            cycle_results["duration_seconds"],
        )

        # Callback (e.g. for narration or event bus publishing)
        if self._on_results is not None:
            try:
                cb_result = self._on_results(cycle_results)
                if asyncio.iscoroutine(cb_result):
                    await cb_result
            except Exception:
                logger.warning(
                    "OptimizerScheduler results callback failed",
                    exc_info=True,
                )

    async def _optimize_strategy(
        self, strategy_id: str
    ) -> dict[str, Any]:
        """Optimise a single strategy, offloading CPU work to a thread."""
        logger.info("Optimizing strategy: %s", strategy_id)

        # Load historical data (synchronous I/O — run in thread)
        candles_by_symbol = await asyncio.to_thread(
            self._load_historical_data, strategy_id
        )

        if not candles_by_symbol:
            msg = f"No historical data available for {strategy_id}"
            logger.warning(msg)
            return {"error": msg, "best_params": {}}

        total_candles = sum(len(v) for v in candles_by_symbol.values())
        logger.info(
            "Loaded %d candles across %d symbols for %s",
            total_candles,
            len(candles_by_symbol),
            strategy_id,
        )

        # Run the optimizer in a thread (CPU-bound backtests)
        report = await asyncio.to_thread(
            self._run_optimizer_sync,
            strategy_id,
            candles_by_symbol,
        )

        result: dict[str, Any] = {
            "best_params": report.best_params,
            "best_sharpe": report.best_sharpe,
            "best_return": report.best_return,
            "samples_tested": report.samples_tested,
            "is_overfit": report.is_overfit,
            "data_period": report.data_period,
        }

        # Walk-forward details
        if report.walk_forward:
            wf = report.walk_forward
            result["walk_forward"] = {
                "avg_train_sharpe": wf.avg_train_sharpe,
                "avg_test_sharpe": wf.avg_test_sharpe,
                "overfit_score": wf.overfit_score,
                "degradation_pct": wf.degradation_pct,
                "is_overfit": wf.is_overfit,
            }

        # Top-5 results summary
        sorted_results = sorted(
            report.results, key=lambda r: r.sharpe_ratio, reverse=True
        )[:5]
        result["top_results"] = [
            {
                "params": r.params,
                "sharpe": r.sharpe_ratio,
                "return_pct": round(r.total_return * 100, 2),
                "max_dd": round(r.max_drawdown * 100, 2),
                "trades": r.total_trades,
                "win_rate": round(r.win_rate * 100, 1),
            }
            for r in sorted_results
        ]

        logger.info(
            "Optimization result for %s: sharpe=%.3f return=%.1f%% overfit=%s",
            strategy_id,
            report.best_sharpe,
            report.best_return,
            report.is_overfit,
        )

        return result

    # ------------------------------------------------------------------
    # Synchronous helpers (run in thread)
    # ------------------------------------------------------------------

    def _load_historical_data(
        self, strategy_id: str
    ) -> dict[str, list]:
        """Load historical candle data for optimization.

        Loads the most recent ``data_window_days`` of 1-minute candles
        from Parquet files on disk.
        """
        from agentic_trading.core.enums import Exchange, Timeframe
        from agentic_trading.data.historical import HistoricalDataLoader

        loader = HistoricalDataLoader(data_dir=self._data_dir)

        # Determine which symbols have data
        available = loader.available_symbols(Exchange.BINANCE)
        # Also check Bybit
        available_bybit = loader.available_symbols(Exchange.BYBIT)

        # Pick whichever exchange has data
        if available:
            exchange = Exchange.BINANCE
            symbols = available
        elif available_bybit:
            exchange = Exchange.BYBIT
            symbols = available_bybit
        else:
            logger.warning("No historical data found in %s", self._data_dir)
            return {}

        # Time window
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=self._config.data_window_days)

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

        return candles_by_symbol

    def _run_optimizer_sync(
        self,
        strategy_id: str,
        candles_by_symbol: dict[str, list],
    ) -> Any:
        """Run ParameterOptimizer synchronously (called from thread).

        Creates a new event loop in the thread because ParameterOptimizer.run()
        is async (it uses BacktestEngine which is async).
        """
        from agentic_trading.features.engine import FeatureEngine
        from agentic_trading.optimizer.engine import ParameterOptimizer

        feature_engine = FeatureEngine(
            indicator_config={"smc_enabled": False}
        )
        optimizer = ParameterOptimizer(
            strategy_id=strategy_id,
            candles_by_symbol=candles_by_symbol,
            feature_engine=feature_engine,
        )

        # Run async optimizer in a new event loop (we're in a thread)
        loop = asyncio.new_event_loop()
        try:
            report = loop.run_until_complete(
                optimizer.run(
                    n_samples=self._config.n_samples,
                    top_n_for_wf=self._config.top_n_for_wf,
                    wf_folds=self._config.wf_folds,
                )
            )
        finally:
            loop.close()

        return report

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_results(
        self, results: dict[str, Any], timestamp: datetime
    ) -> Path:
        """Save optimization results to a JSON file."""
        filename = f"optimizer_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self._results_dir / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("Optimizer results saved to %s", filepath)
        return filepath

    def _rotate_results(self) -> None:
        """Remove old result files beyond max_results_kept."""
        if not self._results_dir.exists():
            return

        files = sorted(
            self._results_dir.glob("optimizer_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for old_file in files[self._config.max_results_kept :]:
            try:
                old_file.unlink()
                logger.debug("Rotated old optimizer result: %s", old_file.name)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_run(self) -> datetime | None:
        return self._last_run

    @property
    def run_count(self) -> int:
        return self._run_count

    @staticmethod
    def load_latest_results(
        results_dir: str = "data/optimizer_results",
    ) -> dict[str, Any] | None:
        """Load the most recent optimizer results file.

        Convenience method for CLI / dashboards.
        """
        results_path = Path(results_dir)
        if not results_path.exists():
            return None

        files = sorted(
            results_path.glob("optimizer_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not files:
            return None

        with open(files[0]) as f:
            return json.load(f)

    @staticmethod
    def load_all_results(
        results_dir: str = "data/optimizer_results",
    ) -> list[dict[str, Any]]:
        """Load all optimizer result files (newest first).

        Convenience method for CLI / dashboards.
        """
        results_path = Path(results_dir)
        if not results_path.exists():
            return []

        files = sorted(
            results_path.glob("optimizer_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        results = []
        for f in files:
            with open(f) as fp:
                results.append(json.load(fp))
        return results
