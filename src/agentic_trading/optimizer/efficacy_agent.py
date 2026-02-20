"""Efficacy analysis agent — periodic trade loss diagnosis.

Extends BaseAgent to run the EfficacyAnalyzer on a schedule, producing
diagnostic reports that identify why trades are losing and recommending
corrective actions in priority order (costs → exits → regime → signal).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.backtester.engine import BacktestEngine
from agentic_trading.backtester.results import TradeDetail
from agentic_trading.core.enums import AgentType
from agentic_trading.core.events import AgentCapabilities

from .efficacy import EfficacyAnalyzer
from .efficacy_models import EfficacyReport

logger = logging.getLogger(__name__)


class EfficacyAgent(BaseAgent):
    """Periodic trade efficacy analysis agent.

    Runs backtests with current strategy parameters, collects per-trade
    details, and diagnoses loss drivers using the EfficacyAnalyzer.

    Usage::

        from agentic_trading.core.config import EfficacyAgentConfig

        config = EfficacyAgentConfig(enabled=True, interval_hours=24.0)
        agent = EfficacyAgent(config=config, data_dir="data/historical")
        await agent.start()
        # ... runs analysis on schedule ...
        await agent.stop()
    """

    def __init__(
        self,
        config: Any,
        data_dir: str = "data/historical",
        event_bus: Any | None = None,
        *,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            interval=config.interval_hours * 3600,
        )
        self._config = config
        self._data_dir = data_dir
        self._event_bus = event_bus
        self._analyzer = EfficacyAnalyzer(
            min_trades=config.min_trades_per_segment,
        )
        self._results_dir = Path(config.results_dir)
        self._run_count = 0
        self._last_run: datetime | None = None
        self._last_report: EfficacyReport | None = None

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
            description="Periodic trade efficacy analysis and loss diagnosis",
        )

    @property
    def last_run(self) -> datetime | None:
        return self._last_run

    @property
    def run_count(self) -> int:
        return self._run_count

    @property
    def last_report(self) -> EfficacyReport | None:
        return self._last_report

    # ------------------------------------------------------------------
    # Work loop
    # ------------------------------------------------------------------

    async def _work(self) -> None:
        """Single cycle: run backtests, collect trades, analyse."""
        logger.info("EfficacyAgent: starting analysis cycle %d", self._run_count + 1)

        try:
            await self._run_analysis_cycle()
        except Exception:
            logger.exception("EfficacyAgent: analysis cycle failed")
            raise

    async def _run_analysis_cycle(self) -> None:
        """Run the full efficacy analysis cycle."""
        # 1. Load historical data
        candles_by_symbol = await asyncio.to_thread(
            self._load_historical_data,
        )
        if not candles_by_symbol:
            logger.warning("EfficacyAgent: no historical data available, skipping")
            return

        total_candles = sum(len(v) for v in candles_by_symbol.values())
        logger.info(
            "EfficacyAgent: loaded %d candles across %d symbols",
            total_candles,
            len(candles_by_symbol),
        )

        # 2. Run backtests for each strategy and collect trade details
        all_trades: list[TradeDetail] = []
        for strategy_id in self._config.strategies:
            trades = await self._backtest_strategy(strategy_id, candles_by_symbol)
            all_trades.extend(trades)

        logger.info(
            "EfficacyAgent: collected %d trades across %d strategies",
            len(all_trades),
            len(self._config.strategies),
        )

        # 3. Run efficacy analysis
        report = await asyncio.to_thread(
            self._analyzer.analyze,
            all_trades,
            strategy_id="all",
        )

        # 4. Save results
        self._run_count += 1
        self._last_run = datetime.now(timezone.utc)
        self._last_report = report

        await asyncio.to_thread(self._save_report, report)

        # 5. Publish event if bus available
        if self._event_bus is not None:
            await self._publish_event(report)

        # 6. Log summary
        self._log_summary(report)

    async def _backtest_strategy(
        self,
        strategy_id: str,
        candles_by_symbol: dict[str, list],
    ) -> list[TradeDetail]:
        """Run a backtest for a single strategy and return trade details."""
        try:
            from agentic_trading.features.engine import FeatureEngine
            from agentic_trading.strategies.registry import create_strategy

            strategy = create_strategy(strategy_id)
            fe = FeatureEngine(indicator_config={"smc_enabled": False})

            engine = BacktestEngine(
                strategies=[strategy],
                feature_engine=fe,
                initial_capital=100_000.0,
                seed=42,
            )

            result = await engine.run(candles_by_symbol)
            return result.trade_details

        except Exception:
            logger.exception(
                "EfficacyAgent: backtest failed for %s", strategy_id
            )
            return []

    def _load_historical_data(self) -> dict[str, list]:
        """Load historical candle data (same pattern as OptimizerScheduler)."""
        from datetime import timedelta

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

    def _save_report(self, report: EfficacyReport) -> Path:
        """Save report as JSON file."""
        self._results_dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        filepath = self._results_dir / f"efficacy_{ts_str}.json"
        filepath.write_text(json.dumps(report.to_dict(), indent=2))
        logger.info("EfficacyAgent: saved report to %s", filepath)

        # Rotate old results
        self._rotate_results()
        return filepath

    def _rotate_results(self, max_keep: int = 10) -> None:
        """Remove old result files, keeping only the newest N."""
        if not self._results_dir.exists():
            return
        files = sorted(
            self._results_dir.glob("efficacy_*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        for f in files[max_keep:]:
            f.unlink()

    async def _publish_event(self, report: EfficacyReport) -> None:
        """Publish efficacy analysis completed event."""
        try:
            from agentic_trading.core.events import EfficacyAnalysisCompleted

            event = EfficacyAnalysisCompleted(
                strategy_id=report.strategy_id,
                total_trades=report.total_trades,
                win_rate=report.win_rate,
                profit_factor=report.profit_factor,
                loss_driver_count=len(report.loss_drivers),
                critical_drivers=sum(
                    1 for d in report.loss_drivers if d.severity == "critical"
                ),
                top_recommendation=(
                    report.recommendations[0] if report.recommendations else ""
                ),
            )
            await self._event_bus.publish("optimizer.result", event)
        except Exception:
            logger.exception("EfficacyAgent: failed to publish event")

    def _log_summary(self, report: EfficacyReport) -> None:
        """Log a concise summary of the analysis."""
        logger.info(
            "EfficacyAgent: cycle %d complete — %d trades, WR=%.1f%%, PF=%.2f",
            self._run_count,
            report.total_trades,
            report.win_rate * 100,
            report.profit_factor,
        )
        for rec in report.recommendations[:3]:
            logger.info("  → %s", rec)

    # ------------------------------------------------------------------
    # Static loaders
    # ------------------------------------------------------------------

    @staticmethod
    def load_latest_report(results_dir: str) -> EfficacyReport | None:
        """Load the most recent efficacy report from disk."""
        path = Path(results_dir)
        if not path.exists():
            return None
        files = sorted(
            path.glob("efficacy_*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if not files:
            return None
        data = json.loads(files[0].read_text())
        # Return raw dict wrapped in report (simplified load)
        report = EfficacyReport(
            timestamp=data.get("timestamp", ""),
            strategy_id=data.get("strategy_id", ""),
            total_trades=data.get("total_trades", 0),
            win_rate=data.get("win_rate", 0.0),
            profit_factor=data.get("profit_factor", 0.0),
            recommendations=data.get("recommendations", []),
        )
        return report
