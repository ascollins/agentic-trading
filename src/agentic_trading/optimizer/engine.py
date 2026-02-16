"""Parameter optimization engine.

Wraps the BacktestEngine in a parameter search loop with walk-forward
validation to find optimal strategy parameters while detecting overfitting.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from agentic_trading.backtester.engine import BacktestEngine
from agentic_trading.core.models import Candle
from agentic_trading.strategies.registry import create_strategy

# Import strategy modules so their @register_strategy decorators fire
import agentic_trading.strategies.trend_following  # noqa: F401
import agentic_trading.strategies.mean_reversion  # noqa: F401
import agentic_trading.strategies.breakout  # noqa: F401
import agentic_trading.strategies.funding_arb  # noqa: F401
from agentic_trading.strategies.research.walk_forward import (
    WalkForwardReport,
    WalkForwardResult,
    WalkForwardValidator,
)

from .param_grid import random_sample
from .report import OptimizationReport, StrategyResult

logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """Automated parameter optimization with walk-forward validation.

    Runs a grid/random search over strategy parameters, evaluating each
    combination via the BacktestEngine. The top candidates are then
    validated with walk-forward analysis to detect overfitting.

    Usage::

        optimizer = ParameterOptimizer(
            strategy_id="trend_following",
            candles_by_symbol=candles,
            feature_engine=feature_engine,
        )
        report = await optimizer.run(n_samples=30)
        print_summary(report)
    """

    def __init__(
        self,
        strategy_id: str,
        candles_by_symbol: dict[str, list[Candle]],
        feature_engine: Any,
        initial_capital: float = 100_000.0,
        slippage_bps: float = 5.0,
        fee_maker: float = 0.0002,
        fee_taker: float = 0.0004,
        seed: int = 42,
    ) -> None:
        self._strategy_id = strategy_id
        self._candles = candles_by_symbol
        self._feature_engine = feature_engine
        self._initial_capital = initial_capital
        self._slippage_bps = slippage_bps
        self._fee_maker = fee_maker
        self._fee_taker = fee_taker
        self._seed = seed

    async def run(
        self,
        n_samples: int = 50,
        top_n_for_wf: int = 3,
        wf_folds: int = 3,
        param_overrides: dict[str, list[Any]] | None = None,
    ) -> OptimizationReport:
        """Run the optimization loop.

        Args:
            n_samples: Number of parameter combinations to test.
            top_n_for_wf: How many top results to walk-forward validate.
            wf_folds: Number of walk-forward folds.
            param_overrides: Optional grid overrides per parameter.

        Returns:
            OptimizationReport with all results and best params.
        """
        # Generate parameter samples
        param_combos = random_sample(
            self._strategy_id,
            n=n_samples,
            seed=self._seed,
            overrides=param_overrides,
        )

        logger.info(
            "Starting optimization: %s with %d parameter combinations",
            self._strategy_id,
            len(param_combos),
        )

        # Run backtest for each parameter combination
        results: list[StrategyResult] = []

        for i, params in enumerate(param_combos):
            try:
                result = await self._evaluate_params(params)
                results.append(result)

                if (i + 1) % 10 == 0 or i == 0:
                    logger.info(
                        "  Progress: %d/%d (latest: return=%.2f%%, sharpe=%.3f)",
                        i + 1,
                        len(param_combos),
                        result.total_return * 100,
                        result.sharpe_ratio,
                    )
            except Exception as e:
                logger.warning(
                    "  Failed params %d/%d: %s — %s",
                    i + 1, len(param_combos), params, e,
                )

        if not results:
            return OptimizationReport(
                strategy_id=self._strategy_id,
                samples_tested=0,
            )

        # Sort by Sharpe ratio (descending)
        results.sort(key=lambda r: r.sharpe_ratio, reverse=True)

        best = results[0]
        report = OptimizationReport(
            strategy_id=self._strategy_id,
            results=results,
            best_params=best.params,
            best_sharpe=best.sharpe_ratio,
            best_return=best.total_return * 100,
            samples_tested=len(results),
            data_period=self._get_data_period(),
        )

        # Walk-forward validation on top candidates
        if top_n_for_wf > 0 and len(results) >= top_n_for_wf:
            top_results = results[:top_n_for_wf]
            wf_report = await self._walk_forward_validate(
                top_results[0].params, wf_folds
            )
            if wf_report:
                report.walk_forward = wf_report
                report.is_overfit = wf_report.is_overfit

        return report

    async def _evaluate_params(
        self, params: dict[str, Any]
    ) -> StrategyResult:
        """Run a single backtest with the given parameters."""
        # Create fresh strategy and feature engine instances
        strategy = create_strategy(self._strategy_id, params)

        # Create a fresh feature engine with SMC disabled for speed.
        # SMC features are informational and don't affect strategy signals,
        # so skipping them during optimisation is safe and much faster.
        from agentic_trading.features.engine import FeatureEngine
        fe = FeatureEngine(indicator_config={"smc_enabled": False})

        engine = BacktestEngine(
            strategies=[strategy],
            feature_engine=fe,
            initial_capital=self._initial_capital,
            slippage_bps=self._slippage_bps,
            fee_maker=self._fee_maker,
            fee_taker=self._fee_taker,
            seed=self._seed,
        )

        bt_result = await engine.run(self._candles)

        return StrategyResult(
            params=params,
            total_return=bt_result.total_return,
            sharpe_ratio=bt_result.sharpe_ratio,
            sortino_ratio=bt_result.sortino_ratio,
            max_drawdown=bt_result.max_drawdown,
            total_trades=bt_result.total_trades,
            win_rate=bt_result.win_rate,
            profit_factor=bt_result.profit_factor,
            total_fees=bt_result.total_fees,
        )

    async def _walk_forward_validate(
        self, params: dict[str, Any], n_folds: int
    ) -> WalkForwardReport | None:
        """Run walk-forward validation on the best parameters.

        Splits the data into train/test windows and evaluates
        in-sample vs out-of-sample performance.
        """
        # Get all timestamps for windowing
        all_timestamps: list[datetime] = []
        for candles in self._candles.values():
            all_timestamps.extend(c.timestamp for c in candles)
        all_timestamps.sort()
        unique_timestamps = sorted(set(all_timestamps))

        if len(unique_timestamps) < 200:
            logger.warning(
                "Insufficient data for walk-forward (%d timestamps, need 200+)",
                len(unique_timestamps),
            )
            return None

        validator = WalkForwardValidator(
            n_folds=n_folds,
            train_pct=0.7,
            gap_periods=1,
        )

        try:
            windows = validator.create_windows(unique_timestamps)
        except ValueError as e:
            logger.warning("Cannot create walk-forward windows: %s", e)
            return None

        fold_results: list[WalkForwardResult] = []

        for window in windows:
            # Split candles into train and test sets
            train_candles = self._filter_candles(
                window.train_start, window.train_end
            )
            test_candles = self._filter_candles(
                window.test_start, window.test_end
            )

            if not train_candles or not test_candles:
                continue

            # Run backtest on train period
            train_result = await self._run_backtest_on(params, train_candles)
            test_result = await self._run_backtest_on(params, test_candles)

            fold_results.append(
                WalkForwardResult(
                    fold_index=window.fold_index,
                    train_sharpe=train_result.sharpe_ratio,
                    test_sharpe=test_result.sharpe_ratio,
                    train_return=train_result.total_return,
                    test_return=test_result.total_return,
                    train_max_dd=train_result.max_drawdown,
                    test_max_dd=test_result.max_drawdown,
                    params=params,
                )
            )

        if not fold_results:
            return None

        return validator.evaluate(fold_results)

    async def _run_backtest_on(
        self,
        params: dict[str, Any],
        candles_by_symbol: dict[str, list[Candle]],
    ) -> Any:
        """Run backtest on a specific candle subset."""
        strategy = create_strategy(self._strategy_id, params)
        from agentic_trading.features.engine import FeatureEngine
        fe = FeatureEngine(indicator_config={"smc_enabled": False})

        engine = BacktestEngine(
            strategies=[strategy],
            feature_engine=fe,
            initial_capital=self._initial_capital,
            slippage_bps=self._slippage_bps,
            fee_maker=self._fee_maker,
            fee_taker=self._fee_taker,
            seed=self._seed,
        )

        return await engine.run(candles_by_symbol)

    def _filter_candles(
        self, start: datetime, end: datetime
    ) -> dict[str, list[Candle]]:
        """Filter candles to a specific time window."""
        filtered: dict[str, list[Candle]] = {}
        for symbol, candles in self._candles.items():
            subset = [c for c in candles if start <= c.timestamp <= end]
            if subset:
                filtered[symbol] = subset
        return filtered

    def _get_data_period(self) -> str:
        """Get a human-readable data period string."""
        all_ts = []
        for candles in self._candles.values():
            if candles:
                all_ts.append(candles[0].timestamp)
                all_ts.append(candles[-1].timestamp)
        if len(all_ts) >= 2:
            start = min(all_ts).strftime("%Y-%m-%d")
            end = max(all_ts).strftime("%Y-%m-%d")
            return f"{start} to {end}"
        return "unknown"
