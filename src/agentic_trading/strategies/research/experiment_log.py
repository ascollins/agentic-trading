"""Experiment config and result tracking.

Logs all backtest experiment configurations and results
for reproducibility and analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration snapshot for a single experiment run."""

    experiment_id: str
    strategy_id: str
    params: dict[str, Any]
    symbols: list[str]
    start_date: str
    end_date: str
    timeframes: list[str]
    slippage_model: str
    fee_maker: float
    fee_taker: float
    random_seed: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str = ""


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    experiment_id: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    daily_returns: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperimentLogger:
    """Logs experiment configs and results to JSON files.

    Structure:
      experiments/
        {experiment_id}/
          config.json
          result.json
    """

    def __init__(self, base_dir: str = "data/experiments") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def log_config(self, config: ExperimentConfig) -> None:
        """Save experiment configuration."""
        exp_dir = self._base_dir / config.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "experiment_id": config.experiment_id,
            "strategy_id": config.strategy_id,
            "params": config.params,
            "symbols": config.symbols,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "timeframes": config.timeframes,
            "slippage_model": config.slippage_model,
            "fee_maker": config.fee_maker,
            "fee_taker": config.fee_taker,
            "random_seed": config.random_seed,
            "created_at": config.created_at.isoformat(),
            "notes": config.notes,
        }

        path = exp_dir / "config.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Experiment config saved: %s", path)

    def log_result(self, result: ExperimentResult) -> None:
        """Save experiment results."""
        exp_dir = self._base_dir / result.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "experiment_id": result.experiment_id,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "total_trades": result.total_trades,
            "avg_trade_return": result.avg_trade_return,
            "calmar_ratio": result.calmar_ratio,
            "sortino_ratio": result.sortino_ratio,
            "metadata": result.metadata,
        }

        path = exp_dir / "result.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Experiment result saved: %s", path)

    def load_config(self, experiment_id: str) -> ExperimentConfig | None:
        """Load a saved experiment config."""
        path = self._base_dir / experiment_id / "config.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return ExperimentConfig(**data)

    def load_result(self, experiment_id: str) -> ExperimentResult | None:
        """Load a saved experiment result."""
        path = self._base_dir / experiment_id / "result.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return ExperimentResult(**data)

    def list_experiments(self) -> list[str]:
        """List all experiment IDs."""
        if not self._base_dir.exists():
            return []
        return sorted(
            d.name for d in self._base_dir.iterdir() if d.is_dir()
        )
