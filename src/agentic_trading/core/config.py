"""Configuration management.

Loads from TOML config files + environment variables.
Uses pydantic-settings for validation and env var overriding.
"""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from .enums import Exchange, Mode, Timeframe


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class ExchangeConfig(BaseModel):
    name: Exchange
    api_key_env: str = ""  # Name of env var holding the API key
    secret_env: str = ""  # Name of env var holding the secret
    testnet: bool = True
    rate_limit: int = 1200  # ms between requests
    timeout: int = 30000  # ms

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")

    @property
    def secret(self) -> str:
        return os.environ.get(self.secret_env, "")


class InstrumentFilter(BaseModel):
    min_daily_volume_usd: float = 1_000_000
    min_liquidity_score: float = 0.0
    max_spread_bps: float = 50.0
    min_notional_usd: float = 10.0
    allowed_quotes: list[str] = Field(default_factory=lambda: ["USDT", "USDC"])


class SymbolConfig(BaseModel):
    symbols: list[str] = Field(default_factory=list)  # Explicit whitelist
    filters: InstrumentFilter = Field(default_factory=InstrumentFilter)
    max_symbols: int = 200


class StrategyParamConfig(BaseModel):
    strategy_id: str
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    timeframes: list[Timeframe] = Field(
        default_factory=lambda: [Timeframe.M5, Timeframe.H1]
    )
    max_position_pct: float = 0.05  # Max 5% of portfolio per position


class RegimeConfig(BaseModel):
    hysteresis_count: int = 3  # Consecutive signals before switch
    max_switches_per_day: int = 4
    cooldown_minutes: int = 60
    hmm_lookback_days: int = 30


class RiskConfig(BaseModel):
    max_portfolio_leverage: float = 3.0
    max_single_position_pct: float = 0.10  # 10% of portfolio
    max_correlated_exposure_pct: float = 0.25
    max_daily_loss_pct: float = 0.05  # 5% daily loss limit
    max_drawdown_pct: float = 0.15  # 15% max drawdown
    var_confidence: float = 0.95
    var_lookback_days: int = 30
    circuit_breaker_cooldown_seconds: int = 300
    reconciliation_interval_seconds: int = 30
    kill_switch_cancel_all: bool = True


class SafeModeConfig(BaseModel):
    enabled: bool = False
    max_symbols: int = 5  # Only top N liquid
    max_leverage: int = 2
    position_size_multiplier: float = 0.5  # Half size


class BacktestConfig(BaseModel):
    start_date: str = "2024-01-01"
    end_date: str = "2024-06-30"
    initial_capital: float = 100_000.0
    slippage_model: str = "volatility_based"  # fixed_bps, volatility_based, impact
    slippage_bps: float = 5.0
    fee_maker: float = 0.0002
    fee_taker: float = 0.0004
    funding_enabled: bool = True
    partial_fills: bool = True
    latency_ms: int = 50
    random_seed: int = 42
    data_dir: str = "data/historical"


class ObservabilityConfig(BaseModel):
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "console"
    metrics_port: int = 9090
    decision_audit: bool = True


# --- Governance sub-configs (Soteria-inspired) ---


class MaturityConfig(BaseModel):
    default_level: str = "L1_paper"
    promotion_min_trades: int = 50
    promotion_min_win_rate: float = 0.45
    promotion_min_profit_factor: float = 1.1
    demotion_drawdown_pct: float = 0.10
    demotion_loss_streak: int = 10
    l3_sizing_cap: float = 0.25


class HealthScoreConfig(BaseModel):
    window_size: int = 50
    debt_per_loss: float = 1.0
    credit_per_win: float = 0.5
    max_debt: float = 10.0
    min_sizing_multiplier: float = 0.1
    recovery_rate: float = 0.8


class CanaryConfig(BaseModel):
    check_interval_seconds: int = 30
    components: list[str] = Field(
        default_factory=lambda: ["event_bus", "redis", "kill_switch"]
    )
    failure_threshold: int = 3
    action_on_failure: str = "kill"


class ImpactClassifierConfig(BaseModel):
    high_notional_usd: float = 50_000.0
    critical_notional_usd: float = 200_000.0
    concentration_threshold_pct: float = 0.15


class DriftDetectorConfig(BaseModel):
    baseline_window_trades: int = 200
    deviation_threshold_pct: float = 30.0
    pause_threshold_pct: float = 50.0
    metrics_tracked: list[str] = Field(
        default_factory=lambda: [
            "win_rate",
            "avg_rr",
            "sharpe",
            "profit_factor",
        ]
    )


class ExecutionTokenConfig(BaseModel):
    default_ttl_seconds: int = 300
    max_active_tokens: int = 10
    require_tokens: bool = False


class GovernanceConfig(BaseModel):
    """Master governance configuration (Soteria-inspired)."""

    enabled: bool = False
    maturity: MaturityConfig = Field(default_factory=MaturityConfig)
    health_score: HealthScoreConfig = Field(default_factory=HealthScoreConfig)
    canary: CanaryConfig = Field(default_factory=CanaryConfig)
    impact_classifier: ImpactClassifierConfig = Field(
        default_factory=ImpactClassifierConfig
    )
    drift_detector: DriftDetectorConfig = Field(
        default_factory=DriftDetectorConfig
    )
    execution_tokens: ExecutionTokenConfig = Field(
        default_factory=ExecutionTokenConfig
    )


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Top-level application settings.

    Loaded from TOML config files, overridden by environment variables.
    """

    mode: Mode = Mode.BACKTEST
    read_only: bool = False  # Observe only, no orders

    # Safety gates for live trading
    i_understand_live_trading: bool = False
    live_flag: bool = False  # Set by --live CLI flag

    # Sub-configs
    exchanges: list[ExchangeConfig] = Field(default_factory=list)
    symbols: SymbolConfig = Field(default_factory=SymbolConfig)
    strategies: list[StrategyParamConfig] = Field(default_factory=list)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    safe_mode: SafeModeConfig = Field(default_factory=SafeModeConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)

    # Infrastructure
    redis_url: str = "redis://localhost:6379/0"
    postgres_url: str = "postgresql+asyncpg://trading:trading@localhost:5432/trading"
    data_dir: str = "data"

    model_config = {"env_prefix": "TRADING_", "env_nested_delimiter": "__"}

    def validate_live_mode(self) -> None:
        """Enforce safety gates for live trading."""
        from .errors import SafetyGateError

        if self.mode != Mode.LIVE:
            return

        if not self.i_understand_live_trading:
            raise SafetyGateError(
                "Live trading requires I_UNDERSTAND_LIVE_TRADING=true "
                "environment variable."
            )
        if not self.live_flag:
            raise SafetyGateError(
                "Live trading requires the --live CLI flag."
            )


def load_settings(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load settings from TOML file + env vars.

    Args:
        config_path: Path to TOML config file (optional).
        overrides: Dict of overrides to apply on top.
    """
    data: dict[str, Any] = {}

    if config_path:
        path = Path(config_path)
        if path.exists():
            import tomli

            with open(path, "rb") as f:
                data = tomli.load(f)

    if overrides:
        data.update(overrides)

    return Settings(**data)
