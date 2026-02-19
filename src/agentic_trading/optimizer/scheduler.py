"""Periodic optimizer scheduler — CMT strategy improvement engine.

Runs the ParameterOptimizer on a configurable interval across all CMT
strategies, producing institutional-grade recommendations (KEEP / UPDATE /
DISABLE) based on multi-objective composite scoring (Sortino, Calmar, max
drawdown, profit factor, expectancy, Sharpe).

Results are published to the ``optimizer.result`` event bus topic and
persisted as JSON files with automatic rotation.

CPU-bound backtest work is offloaded via ``asyncio.to_thread()`` so the
main event loop remains responsive for trading.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.config import OptimizerSchedulerConfig, StrategyParamConfig
from agentic_trading.core.enums import AgentType, OptimizationRecommendation
from agentic_trading.core.events import AgentCapabilities
from agentic_trading.optimizer.report import (
    CompositeScoreWeights,
    OptimizationCycleReport,
    StrategyRecommendation,
    StrategyResult,
)

logger = logging.getLogger(__name__)


def _metrics_from_result(result: StrategyResult) -> dict[str, float]:
    """Extract a flat metrics dict from a StrategyResult."""
    loss_rate = 1.0 - result.win_rate
    expectancy = (
        result.win_rate * result.avg_win
        - loss_rate * abs(result.avg_loss)
    )
    return {
        "sharpe": result.sharpe_ratio,
        "sortino": result.sortino_ratio,
        "calmar": result.calmar_ratio,
        "max_drawdown": result.max_drawdown,
        "profit_factor": result.profit_factor,
        "win_rate": result.win_rate,
        "avg_win": result.avg_win,
        "avg_loss": result.avg_loss,
        "expectancy": round(expectancy, 4),
        "total_trades": float(result.total_trades),
        "annualized_return": result.annualized_return,
        "total_return": result.total_return,
    }


class OptimizerScheduler(BaseAgent):
    """Periodic CMT strategy improvement engine.

    Discovers CMT strategies, runs multi-objective optimisation, compares
    against current params, and produces actionable recommendations.

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
        event_bus: Any | None = None,
        strategy_config: list[StrategyParamConfig] | None = None,
        governance_gate: Any | None = None,
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
        self._event_bus = event_bus
        self._strategy_config = list(strategy_config) if strategy_config else []
        self._governance_gate = governance_gate
        self._last_run: datetime | None = None
        self._run_count = 0
        self._results_dir = Path(config.results_dir)
        self._score_weights = CompositeScoreWeights(
            sortino_weight=config.sortino_weight,
            calmar_weight=config.calmar_weight,
            max_drawdown_penalty=config.max_drawdown_penalty,
            profit_factor_weight=config.profit_factor_weight,
            expectancy_weight=config.expectancy_weight,
            sharpe_weight=config.sharpe_weight,
        )
        self._history: list[OptimizationCycleReport] = []

    # ------------------------------------------------------------------
    # IAgent
    # ------------------------------------------------------------------

    @property
    def agent_type(self) -> AgentType:
        return AgentType.OPTIMIZER

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=[],
            publishes_to=["optimizer.result"],
            description=(
                "Periodic CMT strategy optimizer with multi-objective scoring, "
                "walk-forward validation, and auto-apply guardrails"
            ),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _on_start(self) -> None:
        self._results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "OptimizerScheduler started (interval=%.1fh, initial_delay=%.1fm)",
            self._config.interval_hours,
            self._config.initial_delay_minutes,
        )

    async def _on_stop(self) -> None:
        logger.info("OptimizerScheduler stopped (ran %d times)", self._run_count)

    # ------------------------------------------------------------------
    # BaseAgent periodic work (overrides _loop for initial delay)
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """Internal scheduler loop with initial delay."""
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

    # ------------------------------------------------------------------
    # Strategy discovery
    # ------------------------------------------------------------------

    def _discover_strategies(self) -> list[str]:
        """Discover which strategies to optimize.

        Default: uses the config strategies list (all 8 CMT strategies).
        When ``discover_all_strategies`` is True, queries the strategy
        registry and filters to those with param grids.
        """
        if not self._config.discover_all_strategies:
            strategies = list(self._config.strategies)
        else:
            from agentic_trading.optimizer.param_grid import list_strategies_with_grids
            from agentic_trading.signal.strategies.registry import list_strategies

            registered = list_strategies()
            with_grids = set(list_strategies_with_grids())
            strategies = [s for s in registered if s in with_grids]

        # Filter to those with grids
        from agentic_trading.optimizer.param_grid import strategies_missing_grids

        missing = strategies_missing_grids(strategies)
        if missing:
            logger.info(
                "OptimizerScheduler: skipping %d strategies without param grids: %s",
                len(missing),
                missing,
            )
        optimizable = [s for s in strategies if s not in missing]

        logger.info(
            "OptimizerScheduler: will optimise %d strategies: %s",
            len(optimizable),
            optimizable,
        )
        return optimizable

    def _get_current_params(self, strategy_id: str) -> dict[str, Any]:
        """Retrieve the currently-configured params for a strategy."""
        for scfg in self._strategy_config:
            if scfg.strategy_id == strategy_id:
                return dict(scfg.params)
        return {}

    # ------------------------------------------------------------------
    # Main optimisation cycle
    # ------------------------------------------------------------------

    async def _run_optimization_cycle(self) -> None:
        """Run one full optimization cycle with recommendations."""
        start_time = datetime.now(timezone.utc)
        self._run_count += 1
        logger.info(
            "OptimizerScheduler cycle #%d starting at %s",
            self._run_count,
            start_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

        strategies = self._discover_strategies()
        cycle_report = OptimizationCycleReport(run_number=self._run_count)

        cycle_results: dict[str, Any] = {
            "run_number": self._run_count,
            "started_at": start_time.isoformat(),
            "strategies": {},
        }

        for strategy_id in strategies:
            try:
                recommendation = await self._optimize_and_recommend(strategy_id)
                cycle_report.recommendations.append(recommendation)
                cycle_results["strategies"][strategy_id] = {
                    "recommendation": recommendation.recommendation.value,
                    "current_score": recommendation.current_score,
                    "optimized_score": recommendation.optimized_score,
                    "improvement_pct": recommendation.improvement_pct,
                    "is_overfit": recommendation.is_overfit,
                    "optimized_params": recommendation.optimized_params,
                    "rationale": recommendation.rationale,
                    "metrics_optimized": recommendation.metrics_optimized,
                }

                # Publish per-strategy result event
                await self._publish_strategy_result(recommendation)

                # Auto-apply if enabled and meets guardrails
                if self._config.auto_apply:
                    await self._try_auto_apply(recommendation)

                cycle_report.strategies_optimized += 1
            except Exception as e:
                logger.error(
                    "Optimizer failed for %s: %s", strategy_id, e,
                    exc_info=True,
                )
                cycle_results["strategies"][strategy_id] = {"error": str(e)}
                cycle_report.strategies_failed += 1

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        cycle_report.duration_seconds = duration
        cycle_results["completed_at"] = end_time.isoformat()
        cycle_results["duration_seconds"] = duration

        # Persist and track
        self._save_results(cycle_results, start_time)
        self._rotate_results()
        self._history.append(cycle_report)
        self._last_run = end_time

        # Publish cycle-level event
        await self._publish_cycle_completed(cycle_report)

        # Legacy callback
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

        logger.info(
            "OptimizerScheduler cycle #%d completed in %.1fs: "
            "%d optimised, %d skipped, %d failed",
            self._run_count,
            duration,
            cycle_report.strategies_optimized,
            cycle_report.strategies_skipped,
            cycle_report.strategies_failed,
        )

    # ------------------------------------------------------------------
    # Per-strategy optimisation + recommendation
    # ------------------------------------------------------------------

    async def _optimize_and_recommend(
        self, strategy_id: str
    ) -> StrategyRecommendation:
        """Optimise a strategy and produce a recommendation.

        Steps:
        1. Load historical data
        2. Run optimizer with multi-objective scoring
        3. Backtest current params for apples-to-apples comparison
        4. Compare composite scores
        5. Generate recommendation (KEEP / UPDATE / DISABLE)
        """
        logger.info("Optimising strategy: %s", strategy_id)

        # Load data
        candles_by_symbol = await asyncio.to_thread(
            self._load_historical_data, strategy_id
        )
        if not candles_by_symbol:
            return StrategyRecommendation(
                strategy_id=strategy_id,
                recommendation=OptimizationRecommendation.SKIP,
                rationale="No historical data available",
            )

        total_candles = sum(len(v) for v in candles_by_symbol.values())
        logger.info(
            "Loaded %d candles across %d symbols for %s",
            total_candles,
            len(candles_by_symbol),
            strategy_id,
        )

        # Run optimizer (CPU-bound, in thread)
        report = await asyncio.to_thread(
            self._run_optimizer_sync,
            strategy_id,
            candles_by_symbol,
        )

        if not report.results:
            return StrategyRecommendation(
                strategy_id=strategy_id,
                recommendation=OptimizationRecommendation.SKIP,
                rationale="Optimizer produced no valid results",
            )

        # Score the best optimised result
        best_result = report.results[0]  # Already sorted by composite score
        optimized_score = self._score_weights.compute(best_result)
        optimized_metrics = _metrics_from_result(best_result)

        # Backtest current params for apples-to-apples comparison
        current_params = self._get_current_params(strategy_id)
        current_score = 0.0
        current_metrics: dict[str, float] = {}

        if current_params:
            try:
                current_result = await asyncio.to_thread(
                    self._backtest_params_sync,
                    strategy_id,
                    current_params,
                    candles_by_symbol,
                )
                current_score = self._score_weights.compute(current_result)
                current_metrics = _metrics_from_result(current_result)
            except Exception:
                logger.warning(
                    "Failed to backtest current params for %s, using score=0",
                    strategy_id,
                    exc_info=True,
                )

        # Calculate improvement
        if current_score != 0:
            improvement_pct = (
                (optimized_score - current_score) / abs(current_score)
            ) * 100
        elif optimized_score > 0:
            improvement_pct = 100.0
        else:
            improvement_pct = 0.0

        # Determine recommendation
        is_overfit = report.is_overfit
        wf_passed = (
            report.walk_forward is not None
            and not report.walk_forward.is_overfit
        )

        recommendation, rationale = self._determine_recommendation(
            strategy_id=strategy_id,
            optimized_score=optimized_score,
            current_score=current_score,
            improvement_pct=improvement_pct,
            is_overfit=is_overfit,
            wf_passed=wf_passed,
            best_result=best_result,
            current_metrics=current_metrics,
            optimized_metrics=optimized_metrics,
        )

        return StrategyRecommendation(
            strategy_id=strategy_id,
            recommendation=recommendation,
            current_params=current_params,
            optimized_params=best_result.params,
            current_score=round(current_score, 4),
            optimized_score=round(optimized_score, 4),
            improvement_pct=round(improvement_pct, 2),
            rationale=rationale,
            is_overfit=is_overfit,
            walk_forward_passed=wf_passed,
            metrics_current=current_metrics,
            metrics_optimized=optimized_metrics,
        )

    # ------------------------------------------------------------------
    # Recommendation decision logic
    # ------------------------------------------------------------------

    def _determine_recommendation(
        self,
        strategy_id: str,
        optimized_score: float,
        current_score: float,
        improvement_pct: float,
        is_overfit: bool,
        wf_passed: bool,
        best_result: StrategyResult,
        current_metrics: dict[str, float],
        optimized_metrics: dict[str, float],
    ) -> tuple[OptimizationRecommendation, str]:
        """Determine KEEP / UPDATE / DISABLE with professional rationale."""
        cfg = self._config

        # Compute expectancy of best result
        loss_rate = 1.0 - best_result.win_rate
        best_expectancy = (
            best_result.win_rate * best_result.avg_win
            - loss_rate * abs(best_result.avg_loss)
        )

        # 1. DISABLE: terrible metrics across all combos
        if (
            best_result.sortino_ratio < 0
            and best_result.max_drawdown < cfg.disable_max_drawdown
            and best_expectancy < 0
        ):
            return (
                OptimizationRecommendation.DISABLE,
                (
                    f"Strategy is not viable across all param combos: "
                    f"Sortino={best_result.sortino_ratio:.3f}, "
                    f"Calmar={best_result.calmar_ratio:.3f}, "
                    f"max DD={best_result.max_drawdown:.1%}, "
                    f"expectancy={best_expectancy:.4f} (negative)"
                ),
            )

        # 2. KEEP: insufficient improvement
        if improvement_pct < cfg.min_improvement_pct:
            return (
                OptimizationRecommendation.KEEP,
                (
                    f"Improvement {improvement_pct:.1f}% below threshold "
                    f"{cfg.min_improvement_pct:.1f}%: "
                    f"current_score={current_score:.4f}, "
                    f"optimised_score={optimized_score:.4f}"
                ),
            )

        # 3. KEEP: overfit detected
        if is_overfit and cfg.require_walk_forward_pass:
            return (
                OptimizationRecommendation.KEEP,
                (
                    f"Optimised params show {improvement_pct:.1f}% improvement "
                    f"but walk-forward validation detected overfitting"
                ),
            )

        # 4. UPDATE: meaningful improvement
        if improvement_pct >= cfg.min_improvement_pct:
            wf_note = (
                "walk-forward passed" if wf_passed else "walk-forward not required"
            )
            # Build professional rationale with metric deltas
            parts = [
                f"Composite score improved {improvement_pct:.1f}%: "
                f"{current_score:.4f} → {optimized_score:.4f} ({wf_note})"
            ]
            # Add specific metric improvements
            m = optimized_metrics
            parts.append(
                f"Sortino={m.get('sortino', 0):.3f}, "
                f"Calmar={m.get('calmar', 0):.3f}, "
                f"max DD={m.get('max_drawdown', 0):.1%}, "
                f"PF={m.get('profit_factor', 0):.2f}, "
                f"expectancy={m.get('expectancy', 0):.4f}"
            )
            if current_metrics:
                cm = current_metrics
                parts.append(
                    f"vs current: Sortino={cm.get('sortino', 0):.3f}, "
                    f"Calmar={cm.get('calmar', 0):.3f}, "
                    f"max DD={cm.get('max_drawdown', 0):.1%}, "
                    f"PF={cm.get('profit_factor', 0):.2f}"
                )
            return (
                OptimizationRecommendation.UPDATE,
                ". ".join(parts),
            )

        # 5. Default: keep
        return (
            OptimizationRecommendation.KEEP,
            f"No significant improvement found (improvement={improvement_pct:.1f}%)",
        )

    # ------------------------------------------------------------------
    # Auto-apply with guardrails
    # ------------------------------------------------------------------

    async def _try_auto_apply(
        self, recommendation: StrategyRecommendation
    ) -> None:
        """Attempt to auto-apply optimised params if guardrails pass."""
        if recommendation.recommendation != OptimizationRecommendation.UPDATE:
            return

        if recommendation.is_overfit:
            logger.info(
                "Auto-apply skipped for %s: flagged as overfit",
                recommendation.strategy_id,
            )
            return

        if (
            self._config.require_walk_forward_pass
            and not recommendation.walk_forward_passed
        ):
            logger.info(
                "Auto-apply skipped for %s: walk-forward validation failed",
                recommendation.strategy_id,
            )
            return

        if self._config.require_governance_approval and self._governance_gate is not None:
            logger.info(
                "Auto-apply for %s requires governance approval; "
                "recording recommendation for manual review",
                recommendation.strategy_id,
            )
            return

        applied = self._apply_params(
            recommendation.strategy_id,
            recommendation.optimized_params,
        )

        if applied:
            logger.info(
                "Auto-applied optimised params for %s (improvement=%.1f%%)",
                recommendation.strategy_id,
                recommendation.improvement_pct,
            )
            await self._publish_param_change(recommendation)

    def _apply_params(
        self, strategy_id: str, new_params: dict[str, Any]
    ) -> bool:
        """Apply new params to the strategy config in memory."""
        for scfg in self._strategy_config:
            if scfg.strategy_id == strategy_id:
                scfg.params.update(new_params)
                logger.info(
                    "Updated in-memory params for %s: %s",
                    strategy_id,
                    new_params,
                )
                return True

        logger.warning(
            "Cannot apply params for %s: not found in strategy config",
            strategy_id,
        )
        return False

    # ------------------------------------------------------------------
    # Synchronous helpers (run in thread)
    # ------------------------------------------------------------------

    def _load_historical_data(
        self, strategy_id: str
    ) -> dict[str, list]:
        """Load historical candle data for optimization."""
        from agentic_trading.core.enums import Exchange, Timeframe
        from agentic_trading.data.historical import HistoricalDataLoader

        loader = HistoricalDataLoader(data_dir=self._data_dir)

        available = loader.available_symbols(Exchange.BINANCE)
        available_bybit = loader.available_symbols(Exchange.BYBIT)

        if available:
            exchange = Exchange.BINANCE
            symbols = available
        elif available_bybit:
            exchange = Exchange.BYBIT
            symbols = available_bybit
        else:
            logger.warning("No historical data found in %s", self._data_dir)
            return {}

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
        """Run ParameterOptimizer synchronously (called from thread)."""
        from agentic_trading.features.engine import FeatureEngine
        from agentic_trading.optimizer.engine import ParameterOptimizer

        feature_engine = FeatureEngine(
            indicator_config={"smc_enabled": False}
        )
        optimizer = ParameterOptimizer(
            strategy_id=strategy_id,
            candles_by_symbol=candles_by_symbol,
            feature_engine=feature_engine,
            score_weights=self._score_weights,
        )

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

    def _backtest_params_sync(
        self,
        strategy_id: str,
        params: dict[str, Any],
        candles_by_symbol: dict[str, list],
    ) -> StrategyResult:
        """Backtest a specific set of params synchronously (in thread)."""
        from agentic_trading.backtester.engine import BacktestEngine
        from agentic_trading.features.engine import FeatureEngine
        from agentic_trading.signal.strategies.registry import create_strategy

        strategy = create_strategy(strategy_id, params)
        fe = FeatureEngine(indicator_config={"smc_enabled": False})

        engine = BacktestEngine(
            strategies=[strategy],
            feature_engine=fe,
            initial_capital=100_000.0,
            slippage_bps=5.0,
            fee_maker=0.0002,
            fee_taker=0.0004,
            seed=42,
        )

        loop = asyncio.new_event_loop()
        try:
            bt_result = loop.run_until_complete(
                engine.run(candles_by_symbol)
            )
        finally:
            loop.close()

        return StrategyResult(
            params=params,
            total_return=bt_result.total_return,
            sharpe_ratio=bt_result.sharpe_ratio,
            sortino_ratio=bt_result.sortino_ratio,
            calmar_ratio=bt_result.calmar_ratio,
            max_drawdown=bt_result.max_drawdown,
            total_trades=bt_result.total_trades,
            win_rate=bt_result.win_rate,
            profit_factor=bt_result.profit_factor,
            avg_win=bt_result.avg_win,
            avg_loss=bt_result.avg_loss,
            annualized_return=bt_result.annualized_return,
            total_fees=bt_result.total_fees,
        )

    # ------------------------------------------------------------------
    # Event publishing
    # ------------------------------------------------------------------

    async def _publish_strategy_result(
        self, recommendation: StrategyRecommendation
    ) -> None:
        """Publish a per-strategy optimization result event."""
        if self._event_bus is None:
            return

        from agentic_trading.core.events import StrategyOptimizationResult

        event = StrategyOptimizationResult(
            strategy_id=recommendation.strategy_id,
            recommendation=recommendation.recommendation.value,
            current_composite_score=recommendation.current_score,
            optimized_composite_score=recommendation.optimized_score,
            improvement_pct=recommendation.improvement_pct,
            current_params=recommendation.current_params,
            optimized_params=recommendation.optimized_params,
            is_overfit=recommendation.is_overfit,
            walk_forward_passed=recommendation.walk_forward_passed,
            auto_applied=False,
            rationale=recommendation.rationale,
            metrics=recommendation.metrics_optimized,
        )
        try:
            await self._event_bus.publish("optimizer.result", event)
        except Exception:
            logger.error(
                "Failed to publish strategy optimization result",
                exc_info=True,
            )

    async def _publish_cycle_completed(
        self, cycle_report: OptimizationCycleReport
    ) -> None:
        """Publish cycle-completed event."""
        if self._event_bus is None:
            return

        from agentic_trading.core.events import OptimizationCompleted

        event = OptimizationCompleted(
            run_number=cycle_report.run_number,
            strategies_optimized=cycle_report.strategies_optimized,
            strategies_skipped=cycle_report.strategies_skipped,
            strategies_failed=cycle_report.strategies_failed,
            duration_seconds=cycle_report.duration_seconds,
            recommendations={
                r.strategy_id: r.recommendation.value
                for r in cycle_report.recommendations
            },
        )
        try:
            await self._event_bus.publish("optimizer.result", event)
        except Exception:
            logger.error(
                "Failed to publish optimization completed event",
                exc_info=True,
            )

    async def _publish_param_change(
        self, recommendation: StrategyRecommendation
    ) -> None:
        """Publish parameter change event."""
        if self._event_bus is None:
            return

        from agentic_trading.core.events import ParameterChangeApplied

        event = ParameterChangeApplied(
            strategy_id=recommendation.strategy_id,
            old_params=recommendation.current_params,
            new_params=recommendation.optimized_params,
            improvement_pct=recommendation.improvement_pct,
        )
        try:
            await self._event_bus.publish("optimizer.result", event)
        except Exception:
            logger.error(
                "Failed to publish parameter change event",
                exc_info=True,
            )

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

        for old_file in files[self._config.max_results_kept:]:
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

    @property
    def history(self) -> list[OptimizationCycleReport]:
        """Return historical optimization cycle reports."""
        return list(self._history)

    def get_latest_recommendation(
        self, strategy_id: str
    ) -> StrategyRecommendation | None:
        """Get the most recent recommendation for a strategy."""
        for cycle in reversed(self._history):
            for rec in cycle.recommendations:
                if rec.strategy_id == strategy_id:
                    return rec
        return None

    def get_improvement_trajectory(
        self, strategy_id: str
    ) -> list[dict[str, Any]]:
        """Track how a strategy's scores have changed over optimization runs."""
        trajectory: list[dict[str, Any]] = []
        for cycle in self._history:
            for rec in cycle.recommendations:
                if rec.strategy_id == strategy_id:
                    trajectory.append({
                        "run_number": cycle.run_number,
                        "current_score": rec.current_score,
                        "optimized_score": rec.optimized_score,
                        "recommendation": rec.recommendation.value,
                    })
        return trajectory

    @staticmethod
    def load_latest_results(
        results_dir: str = "data/optimizer_results",
    ) -> dict[str, Any] | None:
        """Load the most recent optimizer results file."""
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
        """Load all optimizer result files (newest first)."""
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
