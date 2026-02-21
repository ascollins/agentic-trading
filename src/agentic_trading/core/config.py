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
    demo: bool = False  # Bybit demo trading (production URL, virtual funds)
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
    max_concurrent_positions: int = 8  # Max open positions at any time
    max_daily_entries: int = 10  # Max new entries per calendar day
    portfolio_cooldown_seconds: int = 3600  # Min seconds between any new entry

    # Pre-trade controls (institutional spec §4.5)
    price_collar_bps: float = 200.0  # Max deviation from ref price in bps (2%)
    max_messages_per_minute_per_strategy: int = 60  # Message throttle per strategy
    max_messages_per_minute_per_symbol: int = 30  # Message throttle per symbol


class FXRiskConfig(BaseModel):
    """FX-specific risk configuration."""

    max_leverage: int = 50
    max_notional_per_order_usd: float = 1_000_000.0
    max_spread_pips: float = 5.0
    max_daily_rollover_cost_usd: float = 500.0
    max_slippage_pips: float = 3.0
    allowed_sessions: list[str] = Field(
        default_factory=lambda: ["london", "new_york", "tokyo"]
    )
    block_weekend_orders: bool = True
    major_pairs_only: bool = True
    allowed_pairs: list[str] = Field(
        default_factory=lambda: [
            "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
            "AUD/USD", "USD/CAD", "NZD/USD",
        ]
    )


class ExitConfig(BaseModel):
    """Server-side TP/SL defaults when strategy doesn't provide explicit levels."""

    enabled: bool = True  # Master switch for TP/SL
    sl_atr_multiplier: float = 2.5  # SL = entry ± ATR × this
    tp_atr_multiplier: float = 5.0  # TP = entry ± ATR × this (2:1 R:R)
    trailing_stop_atr_multiplier: float = 2.0  # Trailing distance = ATR × this
    trailing_strategies: list[str] = Field(
        default_factory=lambda: [
            "trend_following",
            "breakout",
            "multi_tf_ma",
            "bb_squeeze",
        ]
    )


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
    input_timeframe: str = "1m"  # Aggregate 1m candles to this TF (1m, 5m, 15m, 1h, 4h)


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


# --- Narration sub-config ---


class NarrationConfig(BaseModel):
    """Avatar narration configuration."""

    enabled: bool = False
    server_port: int = 8099
    verbosity: str = "normal"  # quiet / normal / detailed / presenter
    heartbeat_seconds: float = 60.0
    dedupe_window_seconds: float = 30.0
    max_stored_items: int = 200
    tavus_mock: bool = True  # Use mock Tavus adapter by default


class OptimizerSchedulerConfig(BaseModel):
    """Periodic strategy optimizer configuration."""

    enabled: bool = False
    interval_hours: float = 24.0  # How often to run optimizer
    strategies: list[str] = Field(
        default_factory=lambda: [
            "multi_tf_ma", "rsi_divergence", "stochastic_macd", "bb_squeeze",
            "mean_reversion_enhanced", "supply_demand", "fibonacci_confluence",
            "obv_divergence",
        ]
    )
    discover_all_strategies: bool = False  # Auto-discover from registry
    data_window_days: int = 90  # How many days of historical data to use
    n_samples: int = 30  # Parameter combinations to test per run
    top_n_for_wf: int = 3  # Top results for walk-forward validation
    wf_folds: int = 3  # Walk-forward validation folds
    auto_apply: bool = False  # Automatically apply best params to live strategies
    max_results_kept: int = 10  # Rotate result files, keep last N
    results_dir: str = "data/optimizer_results"  # Where to store results JSON
    initial_delay_minutes: float = 5.0  # Delay before first run (allow warmup)

    # Scoring weights (institutional priority ordering)
    sortino_weight: float = 0.25
    calmar_weight: float = 0.20
    max_drawdown_penalty: float = 0.20
    profit_factor_weight: float = 0.15
    expectancy_weight: float = 0.10
    sharpe_weight: float = 0.10

    # Auto-apply guardrails
    min_improvement_pct: float = 10.0  # Min composite score improvement for UPDATE
    require_walk_forward_pass: bool = True
    require_governance_approval: bool = True

    # DISABLE thresholds
    disable_max_drawdown: float = -0.30
    disable_min_sharpe: float = -0.5


class EfficacyAgentConfig(BaseModel):
    """Periodic trade efficacy analysis configuration."""

    enabled: bool = False
    interval_hours: float = 24.0
    strategies: list[str] = Field(
        default_factory=lambda: [
            "multi_tf_ma", "rsi_divergence", "stochastic_macd", "bb_squeeze",
            "mean_reversion_enhanced", "supply_demand", "fibonacci_confluence",
            "obv_divergence",
        ]
    )
    data_window_days: int = 90
    min_trades_per_segment: int = 50
    results_dir: str = "data/efficacy_results"


class CMTConfig(BaseModel):
    """CMT Autonomous Strategist Agent configuration."""

    enabled: bool = False
    api_key_env: str = "ANTHROPIC_API_KEY"  # Env var holding API key
    model: str = "claude-sonnet-4-5-20250929"
    analysis_interval_seconds: int = 14400  # 4 hours
    min_confluence_score: int = 5  # Minimum CMT confluence for trade signal
    max_daily_api_calls: int = 50  # Budget guard
    skill_path: str = "skills/cmt-analyst"  # Path to CMT skill bundle
    timeframes: list[str] = Field(
        default_factory=lambda: ["5m", "15m", "1h", "4h", "1d"]
    )


class PredictionMarketConfig(BaseModel):
    """Prediction market integration configuration."""

    enabled: bool = False
    poll_interval_seconds: float = 300.0  # 5 minutes
    min_volume_usd: float = 100_000.0  # Ignore markets below this
    max_confidence_boost: float = 0.15  # Max confidence adjustment +/-
    max_sizing_multiplier: float = 1.5  # Max sizing boost from PM data
    min_sizing_multiplier: float = 0.5  # Min sizing reduction from PM data
    twap_window_hours: float = 2.0  # Time-weighted average window
    consensus_threshold: float = 0.3  # Min |consensus| to act
    event_risk_hours: float = 4.0  # Hours before event to reduce exposure
    event_risk_uncertainty_range: list[float] = Field(
        default_factory=lambda: [0.35, 0.65]
    )  # PM probability range considered "uncertain"
    event_risk_sizing_multiplier: float = 0.25  # Sizing during uncertain events
    keywords: list[str] = Field(
        default_factory=lambda: [
            "bitcoin", "btc", "ethereum", "eth", "crypto",
            "fed", "rate", "inflation", "sec", "regulation",
        ]
    )
    shadow_mode: bool = True  # Log PM signals without affecting trades


class UIConfig(BaseModel):
    """Supervision dashboard configuration."""

    enabled: bool = True  # Start UI server alongside trading engine
    host: str = "0.0.0.0"  # Bind address
    port: int = 8080  # HTTP port for the supervision UI


class ContextConfig(BaseModel):
    """Context manager configuration."""

    memory_ttl_hours: float = 24.0
    max_memory_entries: int = 10_000
    pipeline_log_dir: str = "data/pipeline_logs"
    memory_store_path: str = "data/memory_store.jsonl"
    enable_reasoning_capture: bool = True
    enable_extended_thinking: bool = True


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
    exits: ExitConfig = Field(default_factory=ExitConfig)
    safe_mode: SafeModeConfig = Field(default_factory=SafeModeConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    narration: NarrationConfig = Field(default_factory=NarrationConfig)
    optimizer_scheduler: OptimizerSchedulerConfig = Field(
        default_factory=OptimizerSchedulerConfig
    )
    efficacy_agent: EfficacyAgentConfig = Field(
        default_factory=EfficacyAgentConfig
    )
    ui: UIConfig = Field(default_factory=UIConfig)
    cmt: CMTConfig = Field(default_factory=CMTConfig)
    prediction_market: PredictionMarketConfig = Field(
        default_factory=PredictionMarketConfig
    )
    context: ContextConfig = Field(default_factory=ContextConfig)
    fx_risk: FXRiskConfig = Field(default_factory=FXRiskConfig)

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
        _deep_merge(data, overrides)

    return Settings(**data)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Deep merge *overrides* into *base*, modifying *base* in place.

    Only dicts are merged recursively; other types are replaced.
    """
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
