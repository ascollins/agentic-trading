# Quickstart Guide

Step-by-step guide for first-time setup, from installation through your first backtest and into paper trading.

## Prerequisites

| Dependency   | Version  | Purpose                          |
|-------------|----------|----------------------------------|
| Python      | 3.11+    | Runtime                          |
| Docker      | 24+      | Postgres, Redis, Prometheus, Grafana |
| Git         | 2.x      | Source control                   |

Optional for live/paper: exchange API keys for Binance and/or Bybit.

## 1. Clone and Install

```bash
git clone https://github.com/your-org/agentic-trading.git
cd agentic-trading

# Create a virtual environment (recommended)
python3.11 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
```

Verify the installation:

```bash
python -c "import agentic_trading; print('OK')"
```

## 2. Start Infrastructure

Postgres and Redis are required for integration tests, paper trading, and live trading. Backtesting can run without them (in-memory mode).

```bash
docker compose up -d postgres redis
```

Verify services are healthy:

```bash
docker compose ps
# Both should show "healthy"
```

Run database migrations:

```bash
alembic upgrade head
```

## 3. Configure Exchange API Keys

Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Binance (testnet keys for paper trading)
TRADING_BINANCE_API_KEY=your_binance_api_key
TRADING_BINANCE_SECRET=your_binance_secret

# Bybit (testnet keys for paper trading)
TRADING_BYBIT_API_KEY=your_bybit_api_key
TRADING_BYBIT_SECRET=your_bybit_secret

# Infrastructure
TRADING_REDIS_URL=redis://localhost:6379/0
TRADING_POSTGRES_URL=postgresql+asyncpg://trading:trading@localhost:5432/trading
```

For paper trading, use **testnet** API keys. The paper config (`configs/paper.toml`) already sets `testnet = true` for all exchanges.

Exchange testnet registration:

- Binance Futures Testnet: https://testnet.binancefuture.com
- Bybit Testnet: https://testnet.bybit.com

## 4. Download Historical Data

Download OHLCV candle data for backtesting:

```bash
# Single symbol
python scripts/download_historical.py --symbols BTC/USDT --since 2024-01-01

# Multiple symbols
python scripts/download_historical.py --symbols BTC/USDT ETH/USDT SOL/USDT --since 2024-01-01
```

Data is saved as Parquet files in `data/historical/`. Verify:

```bash
ls -la data/historical/
```

You should see files like `binance_BTC_USDT_1m.parquet`.

## 5. Run Your First Backtest

Run using the default backtest configuration:

```bash
python scripts/run_backtest.py --config configs/backtest.toml --symbols BTC/USDT
```

The default backtest config (`configs/backtest.toml`) runs from 2024-01-01 to 2024-06-30 with $100,000 initial capital, volatility-based slippage, and maker/taker fees.

To customize the date range or capital, edit `configs/backtest.toml`:

```toml
[backtest]
start_date = "2024-01-01"
end_date = "2024-06-30"
initial_capital = 100_000.0
slippage_bps = 5.0
fee_maker = 0.0002
fee_taker = 0.0004
```

## 6. Interpret Results

After a backtest completes, look for:

- **Summary stats** printed to stdout: total return, Sharpe ratio, max drawdown, win rate, profit factor.
- **Decision audit log**: every signal, risk check, and order is logged when `decision_audit = true` (enabled by default). Check logs in `data/logs/` or stdout.
- **Trade log**: individual trades with entry/exit prices, P&L, holding time.

Key metrics to evaluate:

| Metric            | What it tells you                                   |
|-------------------|-----------------------------------------------------|
| Total Return      | Net portfolio change over the backtest period       |
| Sharpe Ratio      | Risk-adjusted return (target > 1.0)                 |
| Max Drawdown      | Worst peak-to-trough loss (check against `max_drawdown_pct` limit) |
| Win Rate          | Percentage of profitable trades                     |
| Profit Factor     | Gross profit / gross loss (target > 1.5)            |
| Avg Trade Duration| Holding period per trade                            |

If results look reasonable, proceed to paper trading.

## 7. Move to Paper Trading

Paper trading connects to exchange testnets with real market data but simulated execution.

```bash
python scripts/run_paper.py --config configs/paper.toml
```

This will:

1. Connect to Binance and Bybit testnets via WebSocket (CCXT Pro).
2. Stream live 1m candles and aggregate to higher timeframes.
3. Run all enabled strategies from `configs/strategies.toml`.
4. Execute simulated orders on the testnet.
5. Run reconciliation every 30 seconds.
6. Expose Prometheus metrics on port 9090.

Monitor paper trading:

```bash
# Check kill switch status
python scripts/killswitch.py --status

# Watch logs
tail -f data/logs/paper.log

# View metrics (if Prometheus/Grafana are running)
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9091
```

Run paper trading for at least a few days to validate strategy behavior with real market data before considering live deployment.

## 8. Next Steps

- **Tune strategies**: Edit `configs/strategies.toml` to adjust parameters. Run backtests to compare.
- **Add symbols**: Update `configs/instruments.toml` to expand the trading universe.
- **Monitor risk**: Review `configs/default.toml` risk settings. Tighten limits before going live.
- **Incident runbooks**: Familiarize yourself with `runbooks/incident_*.md` for operational procedures.
- **Live trading**: See the Safety Gates section in `README.md`. Start with safe mode enabled (the default in `configs/live.toml`).
