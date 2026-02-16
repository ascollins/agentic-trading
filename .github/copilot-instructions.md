# Agentic Trading Platform — AI Coding Agent Instructions

## Big Picture Architecture

This is an **event-driven, multi-agent crypto trading platform** with three independent execution modes (backtest, paper, live) that all run the same strategy code. The architecture revolves around an **in-process event bus** where independent components publish events and react to them asynchronously.

### Core Data Flow

```
Market Data (CCXT/WS) → Feed Manager → Feature Engine → Strategies
                                                             ↓
                                                        Risk Engine
                                                             ↓
                                                       Execution Engine → Exchange Adapters
```

**Key insight**: Strategy code must never import mode-specific code. Strategies only interact via `TradingContext` (clock, event bus, portfolio state) and produce `Signal` events. The execution engine and risk systems are mode-agnostic.

### Three Modes, One Codebase
- **Backtest**: `SimClock` + `MemoryEventBus` + backtest adapter → deterministic, reproducible results
- **Paper**: `WallClock` + `RedisStreamsBus` + paper adapter → real-time but safe (testnet)
- **Live**: `WallClock` + `RedisStreamsBus` + CCXT adapter → production trading (gated by `I_UNDERSTAND_LIVE_TRADING=true` env var)

The mode routing happens in [main.py](src/agentic_trading/main.py#L20-L50); event bus creation is in [bus.py](src/agentic_trading/event_bus/bus.py).

## Critical Components & Their Boundaries

### Event Bus (Topic-Based Pub/Sub)
**Files**: [event_bus/](src/agentic_trading/event_bus/)

All inter-component communication goes through 7 topics (market, features, signals, execution, fills, risk, governance). Implementations:
- `MemoryEventBus`: synchronous in-memory queue for backtests
- `RedisStreamsBus`: persistent Redis-backed for paper/live

**Pattern**: Components subscribe to topics and publish events asynchronously. No tight coupling.

### Strategies (Signal Producers)
**Files**: [strategies/base.py](src/agentic_trading/strategies/base.py), [strategies/*.py](src/agentic_trading/strategies/)

All strategies inherit `BaseStrategy` and implement `on_candle(ctx, candle, features)` → returns `Signal | None`.

**Critical rule**: Strategy code MUST NOT:
- Import execution adapters, backtester, or storage modules
- Access `settings` directly (get params via constructor)
- Assume backtest vs. live (use `TradingContext.clock` instead)

Strategies get feature vectors pre-computed by [FeatureEngine](src/agentic_trading/features/engine.py) and can request regime state via `on_regime_change()`.

### Risk Manager (Pre/Post-Trade Gatekeeper)
**Files**: [risk/manager.py](src/agentic_trading/risk/manager.py), [risk/*.py](src/agentic_trading/risk/)

Single facade orchestrating:
- **Pre-trade**: position size, notional, leverage, exposure checks (see [pre_trade.py](src/agentic_trading/risk/pre_trade.py))
- **Post-trade**: fill price deviation, leverage spike, PnL sanity (see [post_trade.py](src/agentic_trading/risk/post_trade.py))
- **Circuit breakers**: volatility, spread, staleness thresholds
- **Kill switch**: global halt backed by Redis or in-memory
- **Drawdown monitor**: daily loss and peak drawdown limits
- **VaR/ES**: value-at-risk and expected shortfall

Called by execution engine AFTER deduplication, BEFORE order submission.

### Governance Layer (Strategy Lifecycle & Maturity)
**Files**: [governance/](src/agentic_trading/governance/)

**Key concept**: Governance sits ABOVE risk. It controls *which strategies may trade* and *at what sizing*, not individual order limits.

**Components**:
- `MaturityManager`: five-level ladder (L0=shadow, L1=paper, L2=gated, L3=constrained, L4=autonomous). Slow promotion (50+ trades, win rate > 0.45), fast demotion (drawdown > 10%).
- `HealthTracker`: rolling epistemic debt/credit model producing 0.0–1.0 health score that drives position sizing.
- `ImpactClassifier`: scores orders on irreversibility, blast radius, concentration → `ImpactTier` (LOW/MEDIUM/HIGH/CRITICAL).
- `DriftDetector`: compares live vs. backtest performance; deviations > 30% trigger REDUCE_SIZE, > 50% trigger PAUSE.
- `TokenManager`: time-bounded, revocable execution tokens with TTL and audit binding.
- `GovernanceGate`: orchestrator that composes all checks into a `GovernanceDecision` (ALLOW/REDUCE_SIZE/BLOCK/DEMOTE/PAUSE/KILL).

See [docs/state.md](docs/state.md) for detailed governance architecture. Governance is **optional** — gated by `settings.governance.enabled` (default `false`).

### Execution Engine (Order Lifecycle)
**Files**: [execution/engine.py](src/agentic_trading/execution/engine.py)

Orchestrates full order lifecycle:
1. Listen for `OrderIntent` events
2. Check kill switch
3. Run pre-trade risk checks via `RiskManager`
4. Deduplicate via `OrderManager` (prevents duplicate fills)
5. Submit to exchange adapter
6. Publish `OrderAck`, `FillEvent`, `PositionUpdate`
7. Reconciliation loop syncs local state against exchange

Critical: execution engine is **not** a trading algorithm—it's a state machine that enforces invariants (no partial fills, no orphaned orders).

### Exchange Adapters (Mode-Specific Bridge)
**Files**: [execution/adapters/](src/agentic_trading/execution/adapters/)

Each adapter implements `IExchangeAdapter`:
- `ccxt_adapter.py`: CCXT for live/paper (Binance, Bybit)
- `backtest.py`: in-memory order matching with slippage/fees
- `paper.py`: mock adapter for testing

Adapters handle symbol normalization, order format translation, and exchange-specific quirks (e.g., Bybit vs. Binance fee structures).

## Developer Workflows

### Running a Backtest
```bash
python scripts/run_backtest.py --config configs/backtest.toml --symbols BTC/USDT --start 2024-01-01 --end 2024-06-30
```

This invokes [cli.py](src/agentic_trading/cli.py) → `main.run()` with `Mode.BACKTEST`. Results are deterministic if `backtest.random_seed` is fixed.

### Running Paper Trading
```bash
# Requires testnet API keys in .env
python scripts/run_paper.py --config configs/paper.toml
```

Uses real-time candle updates from CCXT Pro WebSocket but routes orders to exchange testnets.

### Running Live Trading
```bash
# Requires BOTH env var AND --live flag for safety
I_UNDERSTAND_LIVE_TRADING=true python scripts/run_live.py --config configs/live.toml --live
```

Safety gates:
- `I_UNDERSTAND_LIVE_TRADING=true` env var check
- `--live` CLI flag required
- `settings.validate_live_mode()` enforces safe mode defaults if not explicitly overridden

### Testing

**Test structure** (see [tests/](tests/)):
- `unit/`: isolated component tests (no I/O)
- `integration/`: full pipeline tests (requires Redis/Postgres, deselect with `-m 'not integration'`)
- `property/`: hypothesis-based property tests
- `golden/`: deterministic backtest reproducibility tests

**Run all tests**:
```bash
pytest tests/
```

**Run only unit tests** (fast, no infrastructure):
```bash
pytest tests/unit/ -m 'not integration'
```

**Run a specific test**:
```bash
pytest tests/unit/test_backtest_engine.py::TestPositionLifecycle::test_no_stacking -v
```

### Database Migrations
```bash
docker compose up -d postgres
alembic upgrade head
```

Migrations live in [alembic/versions/](alembic/versions/). Always create a new version file for schema changes (managed by Alembic).

### Observability
- **Logs**: structured JSON via [structlog](src/agentic_trading/observability/) at `settings.observability.log_level`
- **Metrics**: Prometheus on port `settings.observability.metrics_port` (default 9090)
- **Grafana**: visualizations in [configs/grafana/dashboards/](configs/grafana/dashboards/)

## Project-Specific Patterns & Conventions

### Configuration & Pydantic-Settings
- Base config in [configs/default.toml](configs/default.toml) inherited by all modes
- Mode-specific overrides in [configs/backtest.toml](configs/backtest.toml), [configs/paper.toml](configs/paper.toml), [configs/live.toml](configs/live.toml)
- Environment variables override TOML via pydantic-settings (prefix: `TRADING_`)
- Runtime overrides passed as dicts to `main.run(overrides={...})`

### Events & Tracing
All events inherit `BaseEvent` and include:
- `event_id`: UUID for identity
- `timestamp`: UTC datetime
- `trace_id`: UUID for distributed tracing across async tasks
- `source_module`: string tag (e.g., "data", "strategy", "execution")

See [core/events.py](src/agentic_trading/core/events.py) for all 24 event types across 7 topics.

### Async/Await
The entire platform is asyncio-based (Python 3.11+). Use `async def`, `await`, and prefer `asyncio.create_task()` for concurrent work. The event bus `subscribe()` method is async and returns an async generator.

### Type Hints
All code uses strict type hints (Python 3.11+ syntax, no `typing.Optional`, use `X | None`). Enable Pylance or mypy for validation.

### Position Math & Reconciliation
Position state is deterministic and reconstructible from a fill stream. After every fill, the [PortfolioState](src/agentic_trading/core/interfaces.py) is updated and published as a `PositionUpdate` event. The reconciliation loop in `ExecutionEngine` periodically compares local state against the exchange to catch drifts.

**Key invariant**: `sum(positions) == net_notional` always holds. Violations trigger risk alerts.

### Strategy Parameter Tuning
Strategies get parameters from config via [StrategyParamConfig](src/agentic_trading/core/config.py#L45):
```toml
[[strategies]]
strategy_id = "trend_following"
enabled = true
params = { ema_fast = 10, ema_slow = 50 }
timeframes = ["5m", "1h"]
max_position_pct = 0.05
```

Access via `self._get_param(key, default)` in strategy code. Do NOT hardcode values.

### Backtest Determinism (Golden Tests)
Golden tests verify bit-for-bit reproducibility. Use `random_seed` in config and check that two runs with identical seed + config produce identical output hashes. See [tests/golden/](tests/golden/).

### Kill Switch & Circuit Breakers
The kill switch is a global halt mechanism (Redis-backed or in-memory). When active, all order submissions are blocked and a `KillSwitchEvent` is published. Circuit breakers (volatility, spread, staleness) auto-trigger the kill switch on threshold breach.

## External Dependencies & Integration Points

### CCXT/CCXT Pro
- **Live/Paper**: CCXT adapters fetch instrument metadata and submit orders; CCXT Pro WebSocket streams candles in real-time
- **Backtest**: historical candles loaded from parquet files in `data/historical/`

### Redis
- **Event bus**: RedisStreamsBus uses Redis Streams for persistent, observable event log
- **Kill switch**: global on/off state persisted in Redis
- **Paper/Live**: required for multi-instance coordination

### PostgreSQL
- **Trade journal**: filled orders, positions, and performance metrics stored via SQLAlchemy + asyncpg
- **Migrations**: Alembic manages schema versions

### Prometheus & Grafana
- **Metrics**: agentic-trading exposes Prometheus metrics on port 9090
- **Dashboards**: pre-built in `configs/grafana/dashboards/`

## Code Style & Linting

**Python version**: 3.11+  
**Linter**: ruff (configured in [pyproject.toml](pyproject.toml#L80-L85))  
**Line length**: 99 chars  
**Imports**: sorted with `I` rule (isort-style)

Run linter:
```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Adding a New Strategy

1. Create [src/agentic_trading/strategies/my_strategy.py](src/agentic_trading/strategies/)
2. Inherit `BaseStrategy`, implement `on_candle(ctx, candle, features)` → `Signal | None`
3. Add to [configs/strategies.toml](configs/strategies.toml) with a `[[strategies]]` block
4. Import and register in the strategy registry (see [strategies/__init__.py](src/agentic_trading/strategies/__init__.py))
5. Add unit tests in [tests/unit/test_my_strategy.py](tests/unit/)
6. Backtest: `python scripts/run_backtest.py --config configs/backtest.toml --symbols BTC/USDT`

## Adding a New Risk Check

1. Create a checker class in [src/agentic_trading/risk/](src/agentic_trading/risk/) (e.g., `my_check.py`)
2. Implement `async check(portfolio, order, ...) → bool` and publish `RiskAlert` on failure
3. Wire into `RiskManager.__init__()` and call in `pre_trade_check()` or `post_trade_check()`
4. Add unit tests; test failure scenarios in [tests/unit/test_risk_*.py](tests/unit/)

## Debugging & Common Issues

### Backtest Not Reproducible
- Check `backtest.random_seed` is set (see [configs/backtest.toml](configs/backtest.toml))
- Verify no stochastic operations outside the strategy (e.g., dict iteration order)
- Use [tests/golden/test_deterministic_backtest.py](tests/golden/test_deterministic_backtest.py) as a reference

### Kill Switch Stuck Active
- Manually clear Redis: `redis-cli DEL agentic:kill_switch`
- Check kill switch trigger logs: `grep KillSwitchEvent logs/`

### Order Not Filling
- Check reconciliation loop logs for exchange sync errors
- Verify instrument metadata was fetched: `grep "Fetched.*instruments" logs/`
- For paper trading, confirm API keys are correct and testnet is enabled

### Performance Regression
- Compare backtest equity curves using `drift_detector` thresholds (see [governance/drift_detector.py](src/agentic_trading/governance/drift_detector.py))
- Check if regime detection is blocking trades: `grep RegimeState logs/`

---

**Last updated**: 2026-02-14  
**Python version**: 3.11+  
**Platform**: macOS / Linux
