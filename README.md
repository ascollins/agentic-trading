# Agentic Trading

An event-driven, multi-agent crypto trading platform supporting backtesting, paper trading, and live execution across multiple exchanges.

## Architecture Overview

Agentic Trading is built around an **event-driven architecture** where independent agents (strategies, risk managers, execution engines) communicate through an in-process event bus. The same strategy code runs unchanged across three modes:

- **Backtest** -- replay historical candles with realistic slippage, fees, and latency simulation.
- **Paper** -- connect to exchange testnets via WebSocket for real-time data without risking capital.
- **Live** -- trade real money on production exchange APIs (gated behind explicit safety checks).

```
 Market Data (CCXT Pro WS)
        |
        v
  +-----------+     +----------------+     +-------------+
  | Feed Mgr  | --> | Feature Engine | --> | Strategies  |
  +-----------+     +----------------+     +-------------+
                                                  |
                                           Signal |
                                                  v
  +-------------+     +-----------+     +------------------+
  | Risk Engine | <-- | Portfolio | <-- | Regime Detector  |
  +-------------+     +-----------+     +------------------+
        |
        v
  +------------------+     +---------------------+
  | Execution Engine | --> | Exchange Adapters    |
  +------------------+     | (CCXT / Paper / BT)  |
                           +---------------------+
```

Strategies produce `Signal` events. The risk engine validates signals against pre-trade checks (exposure limits, drawdown, VaR). Approved signals become `OrderIntent` events handled by the execution engine. The reconciliation loop periodically syncs local state against exchange state.

## Key Features

- **Multi-exchange** -- Binance, Bybit (spot and perpetual futures) via CCXT/CCXT Pro.
- **Multi-strategy** -- Trend following, mean reversion, breakout. Run any combination in parallel.
- **Regime detection** -- HMM-based market regime classifier (trending, ranging, volatile) that gates which strategies may trade.
- **Risk management** -- Pre-trade checks, portfolio-level exposure limits, VaR/ES, drawdown limits, circuit breakers, kill switch.
- **Reconciliation** -- Continuous position, order, and balance reconciliation against the exchange.
- **Observability** -- Structured JSON logging, Prometheus metrics, Grafana dashboards, decision audit trail.
- **Deterministic backtesting** -- Golden tests ensure backtest results are bit-for-bit reproducible.
- **Safe mode** -- Reduced position sizing and symbol limits for initial live deployment.

## Quick Start

```bash
# Clone and install
git clone https://github.com/your-org/agentic-trading.git && cd agentic-trading
pip install -e ".[dev]"

# Start infrastructure (Postgres, Redis, Prometheus, Grafana)
docker compose up -d postgres redis

# Run database migrations
alembic upgrade head

# Download historical data
python scripts/download_historical.py --symbols BTC/USDT --since 2024-01-01

# Run a backtest
python scripts/run_backtest.py --config configs/backtest.toml --symbols BTC/USDT

# Run paper trading (requires exchange testnet API keys in .env)
python scripts/run_paper.py --config configs/paper.toml

# Run live trading (requires BOTH the env var AND the --live flag)
I_UNDERSTAND_LIVE_TRADING=true python scripts/run_live.py --live --config configs/live.toml
```

See `runbooks/quickstart.md` for a detailed first-time setup walkthrough.

## Project Structure

```
agentic-trading/
  src/agentic_trading/
    core/            # Config, enums, events, models, interfaces, clock
    event_bus/       # In-process async event bus
    data/            # Feed manager (CCXT Pro WS), candle builder, normalizer, historical loader, data QA
    features/        # Feature computation (indicators, volume profiles)
    strategies/      # Base class, registry, built-in strategies
      regime/        # HMM-based regime detection
      research/      # Research notebooks and experimental strategies
    portfolio/       # Portfolio state, position sizing
    risk/            # Pre/post-trade checks, VaR/ES, drawdown, circuit breakers, kill switch, alerts
    execution/       # Execution engine, order manager, reconciliation
      adapters/      # Exchange adapters: ccxt_adapter (live), paper, backtest
    backtester/      # Backtest runner, slippage/fee models
    storage/         # Postgres persistence (SQLAlchemy + asyncpg)
    observability/   # Structured logging, Prometheus metrics, health checks, decision audit
    cli.py           # Click CLI entry point
    main.py          # Application bootstrap
  configs/
    default.toml     # Base configuration (inherited by all modes)
    backtest.toml    # Backtest-specific settings
    paper.toml       # Paper trading settings (testnet=true)
    live.toml        # Live trading settings (conservative risk defaults)
    strategies.toml  # Strategy parameters
    instruments.toml # Symbol universe, filters, whitelist/blacklist
    prometheus.yml   # Prometheus scrape config
  scripts/
    download_historical.py   # Fetch OHLCV history via CCXT
    run_backtest.py          # Run backtest
    run_paper.py             # Run paper trading
    run_live.py              # Run live trading (safety-gated)
    killswitch.py            # Emergency kill switch CLI
  tests/
    unit/            # Fast unit tests
    integration/     # Tests requiring Redis/Postgres
    property/        # Hypothesis property-based tests
    golden/          # Deterministic backtest reproducibility tests
  runbooks/          # Operational runbooks and incident playbooks
  alembic/           # Database migrations
  data/
    historical/      # Downloaded OHLCV Parquet files
    logs/            # Runtime log output
  Dockerfile
  docker-compose.yml
  pyproject.toml
```

## Configuration

All configuration uses TOML files in `configs/`. Settings are layered:

1. `configs/default.toml` -- base settings shared by all modes.
2. Mode-specific file (`backtest.toml`, `paper.toml`, `live.toml`) -- overrides mode, exchange connections, and risk parameters.
3. `configs/strategies.toml` -- strategy selection and tuning parameters.
4. `configs/instruments.toml` -- symbol universe, filters, whitelist, blacklist.

Key configuration sections in `default.toml`:

| Section          | Purpose                                              |
|------------------|------------------------------------------------------|
| `mode`           | Operating mode: `backtest`, `paper`, or `live`       |
| `[symbols]`      | Default symbols and filters (volume, spread, quotes) |
| `[regime]`       | HMM regime detector tuning                           |
| `[risk]`         | Leverage, exposure, drawdown, VaR, circuit breakers   |
| `[safe_mode]`    | Reduced limits for initial live deployment           |
| `[observability]`| Log level, format, metrics port, audit toggle        |

Exchange API keys are **never stored in config files**. They are referenced by environment variable name (e.g., `api_key_env = "TRADING_BINANCE_API_KEY"`) and read from the environment or a `.env` file at runtime.

## Safety Gates

Live trading is protected by two independent safety gates that must both be present:

1. **Environment variable**: `I_UNDERSTAND_LIVE_TRADING=true`
2. **CLI flag**: `--live`

If either is missing, the system refuses to start in live mode. This prevents accidental live execution from scripts, CI, or misconfigured cron jobs.

Additionally, `live.toml` ships with **safe mode enabled by default**:

```toml
[safe_mode]
enabled = true
max_symbols = 3
max_leverage = 2
position_size_multiplier = 0.25
```

This limits the system to 3 symbols at 25% of normal position size until you explicitly disable it.

## Kill Switch

The kill switch is a global halt mechanism. When activated, all new order submissions are rejected and (optionally) all open orders are cancelled.

**Activate (emergency halt):**

```bash
python scripts/killswitch.py --activate --reason "Manual halt: investigating anomaly"
```

**Check status:**

```bash
python scripts/killswitch.py --status
```

**Deactivate (resume trading):**

```bash
python scripts/killswitch.py --deactivate
```

The kill switch state is stored in Redis (`kill_switch:active` key) so it works across multiple processes. It can also be triggered automatically by the risk engine when drawdown or loss limits are breached.

## Docker Setup

The full stack (app + Postgres + Redis + Prometheus + Grafana) runs via Docker Compose:

```bash
# Start everything
docker compose up -d

# Run a backtest inside the container
docker compose run trading backtest --config configs/backtest.toml --symbols BTC/USDT

# View logs
docker compose logs -f trading

# Grafana dashboards at http://localhost:3000 (admin/admin)
# Prometheus at http://localhost:9091
# App metrics at http://localhost:9090/metrics
```

Ports:

| Service     | Port  |
|-------------|-------|
| Postgres    | 5432  |
| Redis       | 6379  |
| App metrics | 9090  |
| Prometheus  | 9091  |
| Grafana     | 3000  |

## Running Tests

```bash
# All unit tests (fast, no external deps)
pytest tests/unit/

# Integration tests (requires running Postgres and Redis)
docker compose up -d postgres redis
pytest tests/integration/

# Property-based tests (Hypothesis)
pytest tests/property/

# Golden tests (deterministic backtest reproducibility)
pytest tests/golden/

# Full suite with coverage
pytest --cov=agentic_trading --cov-report=term-missing

# Skip integration tests
pytest -m "not integration"

# Lint
ruff check src/ tests/
```

## Contributing / Extending

### Adding a New Strategy

1. Create a new file in `src/agentic_trading/strategies/`, e.g., `my_strategy.py`.

2. Subclass `BaseStrategy` and implement `on_candle()`:

```python
from agentic_trading.core.events import FeatureVector, Signal
from agentic_trading.core.interfaces import TradingContext
from agentic_trading.core.models import Candle
from agentic_trading.strategies.base import BaseStrategy
from agentic_trading.strategies.registry import register_strategy


@register_strategy("my_strategy")
class MyStrategy(BaseStrategy):
    def on_candle(
        self,
        ctx: TradingContext,
        candle: Candle,
        features: FeatureVector,
    ) -> Signal | None:
        # Your logic here. Return a Signal to trade, or None to pass.
        ...
```

3. Add configuration in `configs/strategies.toml`:

```toml
[[strategies]]
strategy_id = "my_strategy"
enabled = true
timeframes = ["5m", "1h"]
max_position_pct = 0.05

[strategies.params]
my_param = 42
```

4. Write tests in `tests/unit/` to validate signal generation.

5. Run a backtest to verify behavior before paper or live trading.

### Key design rules

- Strategies interact with the platform **only** through `TradingContext`. They never import mode-specific modules.
- The `@register_strategy` decorator auto-registers the strategy with the factory; no other wiring needed.
- `on_candle()` receives pre-computed features. Compute custom features in the features module, not inside strategy code.

## License

TBD
