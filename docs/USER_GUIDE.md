# Agentic Trading Platform - User Guide

A complete guide to installing, configuring, and operating the Agentic Trading platform for automated cryptocurrency trading.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Configuration](#3-configuration)
4. [Downloading Market Data](#4-downloading-market-data)
5. [Running a Backtest](#5-running-a-backtest)
6. [Walk-Forward Validation](#6-walk-forward-validation)
7. [Paper Trading](#7-paper-trading)
8. [Live Trading](#8-live-trading)
9. [Strategies](#9-strategies)
10. [Analysis Tools](#10-analysis-tools)
11. [Position Sizing](#11-position-sizing)
12. [Risk Management](#12-risk-management)
13. [Kill Switch](#13-kill-switch)
14. [Monitoring and Observability](#14-monitoring-and-observability)
15. [Docker Deployment](#15-docker-deployment)
16. [Extending the Platform](#16-extending-the-platform)
17. [Troubleshooting](#17-troubleshooting)
18. [Quick Reference](#18-quick-reference)

---

## 1. Overview

Agentic Trading is an event-driven, multi-agent cryptocurrency trading platform. It supports backtesting against historical data, paper trading on exchange testnets, and live trading on production exchanges.

### How It Works

The platform operates as an automated trading pipeline:

1. **Market data** streams in via WebSocket (live/paper) or is replayed from Parquet files (backtest).
2. The **Feature Engine** computes 40+ technical indicators (EMAs, RSI, Bollinger Bands, ATR, ADX, volume profiles, etc.).
3. **Strategies** evaluate features and produce `Signal` events when they detect opportunities.
4. The **Risk Engine** validates each signal against portfolio limits, drawdown thresholds, and exposure caps.
5. The **Portfolio Manager** sizes positions using configurable methods (volatility-adjusted, Kelly, stop-loss-based, etc.).
6. The **Execution Engine** routes approved orders to exchanges and manages fills.
7. A **Reconciliation Loop** continuously syncs local state against the exchange.

The same strategy code runs unchanged across all three modes. You develop and validate in backtest, confirm with paper trading, then deploy live.

### Does It Trade Automatically?

**Yes.** Once running in paper or live mode, the platform autonomously:

- Streams real-time market data from exchanges (Binance, Bybit).
- Computes technical indicators on every new candle.
- Evaluates all enabled strategies and generates trade signals.
- Validates signals through pre-trade risk checks.
- Sizes and submits orders automatically.
- Manages open positions, partial fills, and reconciliation.
- Enforces circuit breakers and drawdown limits.

**You configure the strategies, risk limits, and symbol universe. The platform executes trades on its own.** You can monitor it through Grafana dashboards, Prometheus metrics, and structured logs.

For safety, live trading requires two explicit safety gates (an environment variable and a CLI flag) and starts in safe mode with reduced position sizes. A kill switch allows instant emergency halts.

---

## 2. Installation

### Prerequisites

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Runtime |
| Docker | 24+ | Postgres, Redis, Prometheus, Grafana |
| Git | 2.x | Source control |

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/ascollins/agentic-trading.git
cd agentic-trading

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install with dev dependencies
pip install -e ".[dev]"

# 4. Verify installation
python3 -c "import agentic_trading; print('OK')"
```

### Start Infrastructure

Postgres and Redis are required for paper/live trading and integration tests. Backtesting can run without them.

```bash
# Start Postgres and Redis
docker compose up -d postgres redis

# Verify services are healthy
docker compose ps

# Run database migrations
alembic upgrade head
```

### Configure Environment

```bash
# Copy the example env file
cp .env.example .env
```

Edit `.env` with your exchange API keys:

```bash
# Binance testnet keys (for paper trading)
TRADING_BINANCE_API_KEY=your_api_key_here
TRADING_BINANCE_SECRET=your_secret_here
TRADING_BINANCE_TESTNET=true

# Bybit testnet keys (for paper trading)
TRADING_BYBIT_API_KEY=your_api_key_here
TRADING_BYBIT_SECRET=your_secret_here
TRADING_BYBIT_TESTNET=true

# Infrastructure (defaults work with Docker Compose)
TRADING_REDIS_URL=redis://localhost:6379/0
TRADING_POSTGRES_URL=postgresql+asyncpg://trading:trading@localhost:5432/trading
```

Get testnet API keys from:
- **Binance Futures Testnet**: https://testnet.binancefuture.com
- **Bybit Testnet**: https://testnet.bybit.com

---

## 3. Configuration

All configuration uses TOML files in the `configs/` directory. Settings are layered in order of precedence:

1. `configs/default.toml` -- Base settings for all modes
2. Mode-specific file (`backtest.toml`, `paper.toml`, `live.toml`)
3. `configs/strategies.toml` -- Strategy selection and parameters
4. `configs/instruments.toml` -- Symbol universe
5. Environment variables (prefix: `TRADING_`)
6. CLI flags

### 3.1 Core Settings (`configs/default.toml`)

```toml
mode = "backtest"  # backtest | paper | live

[symbols]
symbols = ["BTC/USDT", "ETH/USDT"]
max_symbols = 200

[symbols.filters]
min_daily_volume_usd = 1_000_000
max_spread_bps = 50.0
min_notional_usd = 10.0
allowed_quotes = ["USDT", "USDC"]

[regime]
hysteresis_count = 3           # Consecutive regime signals before switch
max_switches_per_day = 4
cooldown_minutes = 60
hmm_lookback_days = 30

[risk]
max_portfolio_leverage = 3.0
max_single_position_pct = 0.10  # 10% max per position
max_correlated_exposure_pct = 0.25
max_daily_loss_pct = 0.05       # 5% daily loss circuit breaker
max_drawdown_pct = 0.15         # 15% max drawdown
var_confidence = 0.95
kill_switch_cancel_all = true

[safe_mode]
enabled = false
max_symbols = 5
max_leverage = 2
position_size_multiplier = 0.5

[observability]
log_level = "INFO"        # DEBUG | INFO | WARNING | ERROR
log_format = "json"       # json | console
metrics_port = 9090
decision_audit = true
```

### 3.2 Backtest Settings (`configs/backtest.toml`)

```toml
mode = "backtest"

[backtest]
start_date = "2024-01-01"
end_date = "2024-06-30"
initial_capital = 100_000.0
slippage_model = "volatility_based"  # fixed_bps | volatility_based
slippage_bps = 5.0
fee_maker = 0.0002    # 0.02%
fee_taker = 0.0004    # 0.04%
funding_enabled = true
partial_fills = true
latency_ms = 50
random_seed = 42
data_dir = "data/historical"
```

### 3.3 Paper Settings (`configs/paper.toml`)

```toml
mode = "paper"

[[exchanges]]
name = "binance"
api_key_env = "TRADING_BINANCE_API_KEY"
secret_env = "TRADING_BINANCE_SECRET"
testnet = true
rate_limit = 1200

[[exchanges]]
name = "bybit"
api_key_env = "TRADING_BYBIT_API_KEY"
secret_env = "TRADING_BYBIT_SECRET"
testnet = true
rate_limit = 1200
```

### 3.4 Live Settings (`configs/live.toml`)

```toml
mode = "live"

[[exchanges]]
name = "binance"
api_key_env = "TRADING_BINANCE_API_KEY"
secret_env = "TRADING_BINANCE_SECRET"
testnet = false  # Production!
rate_limit = 1200

# Conservative defaults for live
[risk]
max_portfolio_leverage = 2.0
max_single_position_pct = 0.05
max_daily_loss_pct = 0.03
max_drawdown_pct = 0.10
reconciliation_interval_seconds = 15

# Safe mode ON by default
[safe_mode]
enabled = true
max_symbols = 3
max_leverage = 2
position_size_multiplier = 0.25  # 25% of normal sizing
```

### 3.5 Strategy Configuration (`configs/strategies.toml`)

Each strategy gets its own `[[strategies]]` block:

```toml
[[strategies]]
strategy_id = "trend_following"
enabled = true
timeframes = ["5m", "1h", "4h"]
max_position_pct = 0.05

[strategies.params]
fast_ema = 12
slow_ema = 26
adx_threshold = 25
atr_multiplier = 1.5
volume_filter = true
min_confidence = 0.3

[[strategies]]
strategy_id = "mean_reversion"
enabled = true
timeframes = ["5m", "15m"]
max_position_pct = 0.04

[strategies.params]
bb_period = 20
bb_std = 2.0
rsi_oversold = 30
rsi_overbought = 70
require_range_regime = true

[[strategies]]
strategy_id = "breakout"
enabled = true
timeframes = ["15m", "1h"]
max_position_pct = 0.04

[strategies.params]
donchian_period = 20
volume_confirmation_multiplier = 1.5
min_liquidity_score = 0.5

[[strategies]]
strategy_id = "funding_arb"
enabled = true
timeframes = ["1h"]
max_position_pct = 0.05

[strategies.params]
funding_threshold = 0.0001
high_funding_threshold = 0.0005
position_size_pct = 0.05
```

### 3.6 Symbol Universe (`configs/instruments.toml`)

```toml
[filters]
min_daily_volume_usd = 1_000_000
max_spread_bps = 50.0
allowed_quotes = ["USDT", "USDC"]

[whitelist]
spot = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "XRP/USDT", "ADA/USDT", "AVAX/USDT", "DOGE/USDT",
    "LINK/USDT", "DOT/USDT",
]
perp = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
]

[blacklist]
symbols = []  # Symbols to never trade
```

### 3.7 Environment Variable Overrides

Any config key can be overridden via environment variables with the `TRADING_` prefix. Use double underscores for nested keys:

```bash
TRADING_RISK__MAX_DAILY_LOSS_PCT=0.03
TRADING_OBSERVABILITY__LOG_LEVEL=DEBUG
TRADING_RISK__MAX_PORTFOLIO_LEVERAGE=2.0
```

---

## 4. Downloading Market Data

Before running backtests, download historical OHLCV candle data:

```bash
# Single symbol
python scripts/download_historical.py --symbols BTC/USDT --since 2024-01-01

# Multiple symbols
python scripts/download_historical.py \
    --symbols BTC/USDT,ETH/USDT,SOL/USDT \
    --since 2024-01-01

# Custom timeframe and date range
python scripts/download_historical.py \
    --symbols BTC/USDT \
    --timeframe 5m \
    --since 2024-01-01 \
    --until 2024-06-30

# Different exchange
python scripts/download_historical.py \
    --symbols BTC/USDT \
    --exchange bybit \
    --since 2024-01-01
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--symbols` | Required | Comma-separated trading pairs |
| `--since` | Required | Start date (YYYY-MM-DD) |
| `--until` | Now | End date (YYYY-MM-DD) |
| `--exchange` | `binance` | Exchange to download from |
| `--timeframe` | `1m` | Candle timeframe |
| `--output-dir` | `data/historical` | Output directory |

Data is saved as Parquet files in `data/historical/`. Verify with:

```bash
ls -la data/historical/
```

---

## 5. Running a Backtest

### Basic Usage

```bash
# Run with default config
python scripts/run_backtest.py --config configs/backtest.toml --symbols BTC/USDT

# Override date range
python scripts/run_backtest.py --symbols BTC/USDT --start 2024-01-01 --end 2024-06-30

# Multiple symbols
python scripts/run_backtest.py --symbols BTC/USDT,ETH/USDT,SOL/USDT
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/backtest.toml` | Config file path |
| `--symbols` | From config | Comma-separated symbols |
| `--start` | From config | Start date override |
| `--end` | From config | End date override |

### Understanding Results

After a backtest completes, you will see:

- **Summary statistics**: total return, Sharpe ratio, max drawdown, win rate, profit factor
- **Trade log**: individual trades with entry/exit prices, P&L, holding time
- **Decision audit**: every signal, risk check, and order (when `decision_audit = true`)

Key metrics to evaluate:

| Metric | Target | What It Tells You |
|--------|--------|-------------------|
| Total Return | Positive | Net portfolio change over the period |
| Sharpe Ratio | > 1.0 | Risk-adjusted return quality |
| Max Drawdown | < 15% | Worst peak-to-trough loss |
| Win Rate | > 45% | Percentage of profitable trades |
| Profit Factor | > 1.5 | Gross profit / gross loss |
| Avg Trade Duration | Varies | Holding period per trade |

### Backtest Features

The backtester includes realistic simulation:

- **Slippage models**: Fixed basis points or volatility-based
- **Fee simulation**: Separate maker/taker fees
- **Funding rates**: Perpetual futures funding simulation
- **Partial fills**: Realistic order filling
- **Latency simulation**: Configurable execution delay
- **Deterministic replay**: Set `random_seed` for reproducible results

---

## 6. Walk-Forward Validation

Walk-forward validation tests whether your strategy generalizes to unseen data by splitting the backtest period into training and testing windows.

### Running Walk-Forward

```bash
# 5-fold walk-forward validation
python -m agentic_trading.cli walk-forward \
    --config configs/backtest.toml \
    --symbols BTC/USDT \
    --folds 5

# Custom date range with more folds
python -m agentic_trading.cli walk-forward \
    --symbols BTC/USDT \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --folds 10
```

### Interpreting Results

The walk-forward report includes:

| Metric | Good Value | Bad Value | Meaning |
|--------|-----------|-----------|---------|
| Overfit Score | < 0.3 | > 0.5 | How much performance degrades out-of-sample |
| Degradation % | < 20% | > 50% | Percentage drop from train to test |
| Is Overfit | NO | YES | Overall assessment |

**Recommendations:**
- **Overfit Score < 0.3**: Strategy generalizes well, safe to paper trade
- **Overfit Score 0.3-0.5**: Moderate overfitting, consider simplifying parameters
- **Overfit Score > 0.5**: Significant overfitting, refit with fewer parameters

---

## 7. Paper Trading

Paper trading connects to exchange testnets with real market data but simulated execution. This is the essential bridge between backtesting and live trading.

### Starting Paper Trading

```bash
python scripts/run_paper.py --config configs/paper.toml
```

**What happens:**

1. Connects to Binance and/or Bybit testnets via WebSocket
2. Streams live candles and aggregates to higher timeframes
3. Runs all enabled strategies from `configs/strategies.toml`
4. Executes simulated orders on the testnet
5. Runs position reconciliation every 30 seconds
6. Exposes Prometheus metrics on port 9090

### Monitoring Paper Trading

```bash
# Watch logs
tail -f data/logs/paper.log

# Check kill switch status
python scripts/killswitch.py --status

# View Grafana dashboards (if running)
# Open http://localhost:3001 in your browser

# View raw metrics
curl http://localhost:9090/metrics
```

**Run paper trading for at least several days** before considering live deployment. Validate:

- Signals fire at expected times
- Position sizes are reasonable
- Drawdown stays within limits
- No unexpected errors or disconnections

---

## 8. Live Trading

### Safety Gates

Live trading is protected by **two independent safety gates** that must both be present:

1. **Environment variable**: `I_UNDERSTAND_LIVE_TRADING=true`
2. **CLI flag**: `--live`

If either is missing, the system refuses to start. This prevents accidental live execution.

### Starting Live Trading

```bash
I_UNDERSTAND_LIVE_TRADING=true python scripts/run_live.py --live --config configs/live.toml
```

### Safe Mode

Live trading starts in **safe mode by default** (`configs/live.toml`):

- Maximum 3 symbols
- Maximum 2x leverage
- 25% of normal position sizes

This lets you observe the system with real money at reduced risk. Once confident, disable safe mode by editing `configs/live.toml`:

```toml
[safe_mode]
enabled = false
```

### Recommended Deployment Path

1. **Backtest** - Validate strategy logic against historical data
2. **Walk-Forward** - Confirm the strategy generalizes (overfit score < 0.3)
3. **Paper Trade** - Run for 1-2 weeks with real market data
4. **Live (Safe Mode)** - Deploy with reduced sizing for 1-2 weeks
5. **Live (Full)** - Gradually increase sizing as confidence grows

### Production Checklist

Before going live:

- [ ] All backtest metrics are acceptable (Sharpe > 1, max DD < 15%)
- [ ] Walk-forward validation passes (overfit score < 0.5)
- [ ] Paper trading ran successfully for 1+ weeks
- [ ] Exchange API keys are for **production** (not testnet)
- [ ] `configs/live.toml` has `testnet = false`
- [ ] Risk limits are set conservatively
- [ ] Kill switch is tested and functional
- [ ] Monitoring stack is running (Prometheus + Grafana)
- [ ] Incident runbooks have been reviewed (`runbooks/incident_*.md`)
- [ ] You have a plan for when things go wrong

---

## 9. Strategies

The platform ships with four built-in strategies. All run in parallel; the portfolio manager resolves conflicting signals via confidence-weighted voting.

### 9.1 Trend Following

**ID**: `trend_following`

Identifies directional trends using EMA crossovers with ADX confirmation and volume filtering.

**Signal logic:**
- **LONG**: Fast EMA > Slow EMA, ADX above threshold, volume confirmed
- **SHORT**: Fast EMA < Slow EMA, ADX above threshold, volume confirmed

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fast_ema` | 12 | Fast EMA period |
| `slow_ema` | 26 | Slow EMA period |
| `adx_threshold` | 25 | Minimum ADX for trend entry |
| `atr_multiplier` | 1.5 | ATR multiplier for stop distance |
| `volume_filter` | true | Require volume confirmation |

**Sizing method**: Volatility-adjusted (ATR-based stop)

### 9.2 Mean Reversion

**ID**: `mean_reversion`

Trades price extremes at Bollinger Band boundaries when RSI confirms oversold/overbought conditions.

**Signal logic:**
- **LONG**: Price below lower Bollinger Band, RSI oversold
- **SHORT**: Price above upper Bollinger Band, RSI overbought

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bb_period` | 20 | Bollinger Band period |
| `bb_std` | 2.0 | Standard deviations |
| `rsi_oversold` | 30 | RSI oversold threshold |
| `rsi_overbought` | 70 | RSI overbought threshold |
| `require_range_regime` | true | Only trade in range regime |

**Sizing method**: Fixed fractional

### 9.3 Breakout

**ID**: `breakout`

Detects Donchian channel breakouts confirmed by volume spikes and expanding volatility.

**Signal logic:**
- **LONG**: Price breaks above Donchian upper channel with volume confirmation
- **SHORT**: Price breaks below Donchian lower channel with volume confirmation

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `donchian_period` | 20 | Channel lookback period |
| `volume_confirmation_multiplier` | 1.5 | Required volume spike |
| `min_liquidity_score` | 0.5 | Minimum liquidity for entry |

**Sizing method**: Liquidity-adjusted

### 9.4 Funding Rate Arbitrage

**ID**: `funding_arb`

Captures perpetual futures funding rate payments by taking the opposite side when funding is extreme.

**Signal logic:**
- **SHORT**: Funding rate significantly positive (shorts earn funding)
- **LONG**: Funding rate significantly negative (longs earn funding)

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `funding_threshold` | 0.0001 | 1 bps minimum funding |
| `high_funding_threshold` | 0.0005 | 5 bps high funding |
| `position_size_pct` | 0.05 | 5% position size |
| `atr_stop_multiplier` | 3.0 | Wide stop for funding trades |

**Sizing method**: Fixed fractional

### Regime-Aware Trading

All strategies interact with the **HMM-based regime detector**, which classifies market conditions as trending, ranging, or volatile. Strategies can adjust behavior based on regime:

- **Trend following**: Full confidence in trend regimes, halved in range
- **Mean reversion**: Only fires in range regimes (when `require_range_regime = true`)
- **Breakout**: Requires ATR expansion to confirm genuine breakouts

---

## 10. Analysis Tools

The `analysis` module provides pre-trade assessment tools that complement the automated strategy pipeline.

### 10.1 Risk-Reward Calculator

Evaluate setups before committing to them:

```python
from agentic_trading.analysis.rr_calculator import calculate_rr, project_pnl

# Basic R:R calculation
result = calculate_rr(
    entry=67000.0,
    stop_loss=65000.0,
    targets=[69000.0, 72000.0, 75000.0],
)

print(f"Direction: {result.direction.value}")
print(f"Risk per unit: ${result.risk_per_unit}")
print(f"Blended R:R: {result.blended_rr}")
print(f"Setup Grade: {result.setup_grade.value}")

for t in result.targets:
    print(f"  Target ${t.price}: {t.rr_ratio}R ({t.scale_out_pct:.0%} scale-out)")

# Project PnL scenarios
pnl = project_pnl(account_size=100_000, risk_pct=0.01, rr_result=result)
print(f"\nRisk amount: ${pnl['risk_amount']}")
print(f"Max loss: ${pnl['max_loss']}")
print(f"Full target profit: ${pnl['full_target_profit']}")
```

**Setup grades:**

| Grade | Criteria |
|-------|----------|
| A+ | Blended R:R >= 4.0 and Expected R >= 1.5 |
| A | Blended R:R >= 3.0 and Expected R >= 1.0 |
| B | Blended R:R >= 2.0 and Expected R >= 0.5 |
| C | Blended R:R >= 1.5 and Expected R >= 0.0 |
| D | Blended R:R >= 1.0 |
| F | Blended R:R < 1.0 |

### 10.2 Structured Trade Plans

Build comprehensive trade plans that bridge to the automated pipeline:

```python
from agentic_trading.analysis.trade_plan import TradePlan, EntryZone, TargetSpec
from agentic_trading.core.enums import (
    SignalDirection, ConvictionLevel, SetupGrade, MarketStructureBias,
)

plan = TradePlan(
    strategy_id="manual_setup",
    symbol="BTC/USDT",
    direction=SignalDirection.LONG,
    conviction=ConvictionLevel.HIGH,
    setup_grade=SetupGrade.A,
    confidence=0.8,
    htf_bias=MarketStructureBias.BULLISH,
    entry=EntryZone(
        primary_entry=67000,
        entry_low=66500,
        entry_high=67200,
        scaled_entries=[(67000, 0.5), (66500, 0.3), (66000, 0.2)],
    ),
    stop_loss=64000,
    targets=[
        TargetSpec(price=70000, rr_ratio=1.0, scale_out_pct=0.4, rationale="Previous resistance"),
        TargetSpec(price=73000, rr_ratio=2.0, scale_out_pct=0.35, rationale="Fib extension"),
        TargetSpec(price=76000, rr_ratio=3.0, scale_out_pct=0.25, rationale="ATH target"),
    ],
    blended_rr=1.85,
    rationale="D1 bullish structure, H4 pullback to demand zone, H1 BOS confirmation",
)

# Convert to Signal for the automated pipeline
signal_kwargs = plan.to_signal()

# Or get risk constraints for the portfolio manager
constraints = plan.to_signal_risk_constraints()
```

### 10.3 Higher-Timeframe Analyzer

Assess multi-timeframe market structure with weighted bias:

```python
from agentic_trading.analysis.htf_analyzer import HTFAnalyzer
from agentic_trading.core.enums import Timeframe

analyzer = HTFAnalyzer()

# Use aligned features from the multi-timeframe aligner
features = {
    "1d_close": 67500, "1d_ema_21": 66800, "1d_ema_50": 65500,
    "1d_sma_200": 58000, "1d_adx_14": 32, "1d_rsi_14": 58,
    "4h_close": 67500, "4h_ema_21": 67200, "4h_ema_50": 66800,
    "4h_adx_14": 28, "4h_rsi_14": 55,
    "1h_close": 67500, "1h_ema_21": 67400, "1h_ema_50": 67100,
    "1h_adx_14": 22, "1h_rsi_14": 52,
}

assessment = analyzer.analyze(
    "BTC/USDT", features,
    available_timeframes=[Timeframe.D1, Timeframe.H4, Timeframe.H1],
)

print(f"Overall bias: {assessment.overall_bias.value}")
print(f"Alignment score: {assessment.bias_alignment_score}")
print(f"Regime suggestion: {assessment.regime_suggestion.value}")
print(f"Confluences: {assessment.confluences}")
print(f"Conflicts: {assessment.conflicts}")
```

**Timeframe hierarchy** (highest weight first): D1 > H4 > H1 > M15 > M5 > M1. Higher timeframes receive more weight for directional bias, following the principle that structure on higher timeframes takes precedence.

### 10.4 Market Context and Macro Regimes

Frame your trading within the macro environment:

```python
from agentic_trading.analysis.market_context import assess_macro_regime

context = assess_macro_regime({
    "dxy_trend": "down",           # Dollar weakening
    "yields_10y_trend": "down",    # Yields falling
    "sp500_vs_200sma": 1.05,       # S&P above 200 SMA
    "funding_rates_avg": 0.005,    # Moderate funding
    "stablecoin_supply_trend": "up",  # Liquidity expanding
    "btc_dominance": 52.0,
})

print(f"Regime: {context.regime_key}")
print(f"Risk: {context.risk_regime}")
print(f"Dollar: {context.dollar_trend}")
print(f"Impact: {context.impact_on_crypto}")
print(f"Guidance: {context.positioning_guidance}")
```

**Five macro regimes:**

| Regime | Crypto Impact | Positioning |
|--------|---------------|-------------|
| `risk_on_easing` | Strongly bullish | Full sizing, wide stops, trend-follow |
| `risk_on_neutral` | Moderately bullish | Standard sizing, BTC-heavy |
| `risk_off_tightening` | Bearish | 50% sizing, tight stops |
| `risk_off_crisis` | Highly bearish | Minimal exposure, cash preservation |
| `neutral_ranging` | Neutral | Selective entries, mean-reversion |

### 10.5 Structural Correlation Check

Instantly detect correlated positions without historical data:

```python
from agentic_trading.portfolio.correlation_risk import quick_correlation_check

result = quick_correlation_check(
    ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"],
    threshold="high",
)

print(f"Diversification score: {result['diversification_score']}")
print(f"Warnings: {result['warnings']}")
print(f"Correlated clusters: {result['correlated_clusters']}")
```

This uses predefined crypto correlation tiers (very_high, high, moderate) to flag concentration risk instantly, complementing the dynamic rolling-correlation analyzer.

---

## 11. Position Sizing

Six sizing methods are available, each suited to different strategies:

### Volatility-Adjusted (ATR-Based)

Best for: Trend following strategies

```
Size = (Capital x Risk%) / (ATR x Multiplier)
```

Higher volatility automatically reduces position size.

### Fixed Fractional

Best for: Mean reversion, funding arbitrage

```
Size = (Capital x Fraction) / Price
```

Simple percentage-of-capital allocation.

### Kelly Criterion

Best for: Strategies with known statistical edge

```
Kelly% = WinRate - (1 - WinRate) / (AvgWin / AvgLoss)
```

Uses fractional Kelly (25% by default) for safety. Capped at 20% of capital.

### Stop-Loss Based

Best for: Trade plans with explicit stop levels

```
Size = (Capital x Risk%) / |Entry - StopLoss|
```

Sizes position so that hitting the stop loss results in exactly the intended dollar risk.

### Liquidity-Adjusted

Best for: Breakout strategies on less liquid pairs

Scales base position size by the pair's liquidity score to prevent excessive market impact.

### Scaled Entry (Laddered)

Best for: Averaging into positions at multiple price levels

Distributes total risk across weighted entry levels. For example, 50% at $67,000, 30% at $66,500, and 20% at $66,000.

---

## 12. Risk Management

### Pre-Trade Checks

Every signal is validated before execution:

- **Position limit**: No single position exceeds `max_single_position_pct` of capital
- **Exposure limit**: Total gross exposure stays below `max_gross_exposure_pct`
- **Correlation check**: Correlated exposure stays below `max_correlated_exposure_pct`
- **Drawdown limit**: Trading halts if `max_drawdown_pct` is breached
- **Daily loss limit**: Trading halts if `max_daily_loss_pct` is breached
- **VaR/ES check**: Position passes Value-at-Risk screening

### Circuit Breakers

Automatic protection against abnormal conditions:

- **Error rate breaker**: Trips when exchange errors exceed threshold
- **Staleness breaker**: Trips when candle data goes stale
- **Drawdown breaker**: Trips on excessive portfolio drawdown
- **Cooldown period**: Configurable recovery time before re-enabling

### Reconciliation

Continuous sync between local state and exchange state:

- **Position reconciliation**: Compares local vs exchange positions
- **Order reconciliation**: Detects stuck or stale orders
- **Balance reconciliation**: Validates capital calculations

Runs every 30 seconds (paper) or 15 seconds (live). Discrepancies are auto-repaired when possible and logged for review.

---

## 13. Kill Switch

The kill switch is a global emergency halt. When activated, all new order submissions are rejected and open orders are cancelled.

### CLI Usage

```bash
# Check current status
python scripts/killswitch.py --status

# Activate (emergency halt)
python scripts/killswitch.py --activate --reason "Manual halt: investigating anomaly"

# Deactivate (resume trading)
python scripts/killswitch.py --deactivate
```

### Automatic Activation

The kill switch activates automatically when:

- Daily loss exceeds `max_daily_loss_pct`
- Portfolio drawdown exceeds `max_drawdown_pct`

### Behavior When Active

- All new orders are **rejected**
- Open orders are **cancelled** (when `kill_switch_cancel_all = true`)
- Strategies keep running and generating signals (but nothing executes)
- Market data and reconciliation continue
- Existing positions are **not** auto-closed (requires manual intervention)
- The process stays running

### Recovery

1. Investigate the issue
2. Deactivate: `python scripts/killswitch.py --deactivate`
3. Monitor that trading resumes normally

---

## 14. Monitoring and Observability

### Starting the Monitoring Stack

```bash
# Start Prometheus + Grafana alongside the infrastructure
docker compose up -d prometheus grafana
```

**Access points:**

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3001 | admin / admin |
| Prometheus | http://localhost:9091 | None |
| App Metrics | http://localhost:9090/metrics | None |

### Grafana Dashboard

The pre-configured trading dashboard shows:

- Portfolio equity over time
- Active positions and exposure
- Signal generation rate by strategy
- Order fill rates and errors
- Circuit breaker status
- Exchange connectivity

### Prometheus Metrics

Key metrics exposed:

| Metric | Type | Description |
|--------|------|-------------|
| `trading_signals_total` | Counter | Signals by strategy, symbol, direction |
| `candles_processed_total` | Counter | Candles by symbol, timeframe |
| `portfolio_equity` | Gauge | Current portfolio value |
| `order_manager_active_count` | Gauge | Open orders |
| `order_manager_filled_total` | Counter | Filled orders |
| `circuit_breaker_trips_total` | Counter | Breaker trips by type |
| `exchange_request_errors_total` | Counter | Exchange errors |
| `feed_manager_active_tasks` | Gauge | Active data feeds |

### Structured Logging

All logs are JSON-formatted (configurable) with structured fields:

- Set log level: `TRADING_OBSERVABILITY__LOG_LEVEL=DEBUG`
- Set format: `TRADING_OBSERVABILITY__LOG_FORMAT=console` for development

When `decision_audit = true` (default), every signal, risk check, order, and reconciliation event is logged.

---

## 15. Docker Deployment

### Full Stack

```bash
# Start everything (app + Postgres + Redis + Prometheus + Grafana)
docker compose up -d

# Run a backtest in the container
docker compose run trading backtest --config configs/backtest.toml --symbols BTC/USDT

# View logs
docker compose logs -f trading

# Stop everything
docker compose down
```

### Service Ports

| Service | Host Port | Internal Port |
|---------|-----------|---------------|
| Postgres | 5433 | 5432 |
| Redis | 6379 | 6379 |
| App Metrics | 9090 | 9090 |
| Prometheus | 9091 | 9090 |
| Grafana | 3001 | 3000 |

### Data Persistence

Docker volumes persist data across restarts:

- `pgdata`: Postgres database
- `redisdata`: Redis state (kill switch, circuit breakers)
- `grafanadata`: Grafana dashboards and settings

To reset all data: `docker compose down -v` (destructive).

---

## 16. Extending the Platform

### Adding a New Strategy

1. Create a file in `src/agentic_trading/strategies/`:

```python
from agentic_trading.core.events import Signal
from agentic_trading.core.interfaces import TradingContext
from agentic_trading.core.models import Candle
from agentic_trading.strategies.base import BaseStrategy
from agentic_trading.strategies.registry import register_strategy


@register_strategy("my_strategy")
class MyStrategy(BaseStrategy):
    def on_candle(self, ctx: TradingContext, candle: Candle, features) -> Signal | None:
        # Your logic here
        rsi = features.get("rsi_14", 50)

        if rsi < 25:
            return Signal(
                strategy_id="my_strategy",
                symbol=candle.symbol,
                direction=SignalDirection.LONG,
                confidence=0.7,
                risk_constraints={"sizing_method": "fixed_fractional"},
            )
        return None
```

2. Add configuration in `configs/strategies.toml`:

```toml
[[strategies]]
strategy_id = "my_strategy"
enabled = true
timeframes = ["5m", "1h"]
max_position_pct = 0.05

[strategies.params]
my_param = 42
```

3. Write tests, run a backtest, then validate with paper trading.

### Key Design Rules

- Strategies interact with the platform **only** through `TradingContext`
- The `@register_strategy` decorator handles all registration
- Compute features in the Feature Engine, not inside strategy code
- Return `None` from `on_candle()` to pass on a given candle

---

## 17. Troubleshooting

### Common Issues

**"No historical data found"**

Download data first:
```bash
python scripts/download_historical.py --symbols BTC/USDT --since 2024-01-01
```

**"Cannot connect to Redis/Postgres"**

Start infrastructure:
```bash
docker compose up -d postgres redis
docker compose ps  # Verify "healthy" status
```

**"Live trading refused to start"**

Both safety gates are required:
```bash
I_UNDERSTAND_LIVE_TRADING=true python scripts/run_live.py --live
```

**"WebSocket disconnected"**

Check exchange status pages. Circuit breakers will prevent stale trading. If feeds don't recover:
```bash
python scripts/killswitch.py --activate --reason "Feed outage"
# Investigate, then:
python scripts/killswitch.py --deactivate
```

**"Position mismatch in reconciliation"**

Reconciliation auto-repairs most discrepancies. If persistent:
1. Activate kill switch
2. Check positions on exchange directly
3. Restart the application
4. Deactivate kill switch

### Incident Runbooks

Detailed operational runbooks are available in `runbooks/`:

| Runbook | Scenario |
|---------|----------|
| `incident_exchange_outage.md` | Exchange goes down or rate limits |
| `incident_desync.md` | Local state diverges from exchange |
| `incident_stuck_orders.md` | Orders stuck in submitted state |
| `incident_feed_outage.md` | Market data feeds stop |

### Running Tests

```bash
# Unit tests (fast, no external dependencies)
pytest tests/unit/

# Integration tests (requires Postgres + Redis)
pytest tests/integration/

# Property-based tests (Hypothesis)
pytest tests/property/

# Golden tests (deterministic reproducibility)
pytest tests/golden/

# Full suite with coverage
pytest --cov=agentic_trading --cov-report=term-missing

# Lint check
ruff check src/ tests/
```

---

## 18. Quick Reference

### Essential Commands

```bash
# Installation
pip install -e ".[dev]"

# Infrastructure
docker compose up -d postgres redis
alembic upgrade head

# Download Data
python scripts/download_historical.py --symbols BTC/USDT --since 2024-01-01

# Backtest
python scripts/run_backtest.py --config configs/backtest.toml --symbols BTC/USDT

# Walk-Forward Validation
python -m agentic_trading.cli walk-forward --folds 5 --symbols BTC/USDT

# Paper Trading
python scripts/run_paper.py --config configs/paper.toml

# Live Trading (BOTH safety gates required)
I_UNDERSTAND_LIVE_TRADING=true python scripts/run_live.py --live

# Kill Switch
python scripts/killswitch.py --status
python scripts/killswitch.py --activate --reason "Emergency"
python scripts/killswitch.py --deactivate

# Monitoring
docker compose up -d prometheus grafana
# Grafana: http://localhost:3001 (admin/admin)

# Tests
pytest tests/unit/
```

### Directory Structure

```
agentic-trading/
  src/agentic_trading/
    core/          # Config, enums, events, models, interfaces
    analysis/      # R:R calculator, trade plans, HTF analyzer, macro context
    strategies/    # Built-in strategies + regime detection
    portfolio/     # Sizing, correlation risk, portfolio management
    risk/          # Pre/post-trade checks, circuit breakers, kill switch
    execution/     # Order management, exchange adapters, reconciliation
    features/      # 40+ technical indicators
    data/          # Market data feeds, candle builder
    backtester/    # Backtest runner, slippage/fee models
    storage/       # Database persistence
    observability/ # Logging, metrics, health checks
  configs/         # TOML configuration files
  scripts/         # CLI scripts
  tests/           # Unit, integration, property, golden tests
  runbooks/        # Operational incident playbooks
  docs/            # Reference documentation
  data/            # Historical data + logs
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADING_MODE` | `backtest` | Operating mode |
| `I_UNDERSTAND_LIVE_TRADING` | unset | Live trading safety gate |
| `TRADING_BINANCE_API_KEY` | unset | Binance API key |
| `TRADING_BINANCE_SECRET` | unset | Binance secret |
| `TRADING_BYBIT_API_KEY` | unset | Bybit API key |
| `TRADING_BYBIT_SECRET` | unset | Bybit secret |
| `TRADING_REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `TRADING_POSTGRES_URL` | `postgresql+asyncpg://...` | Postgres connection |
| `TRADING_OBSERVABILITY__LOG_LEVEL` | `INFO` | Log verbosity |
| `TRADING_OBSERVABILITY__METRICS_PORT` | `9090` | Prometheus port |
