# CLAUDE.md - Agentic Trading Platform

## Project Overview

Event-driven crypto trading platform with 3 modes (backtest/paper/live), Soteria-inspired governance, and comprehensive risk management. Target: institutional-grade multi-agent architecture.

## Commands

- **Run tests**: `python3 -m pytest tests/ -q`
- **Run specific test**: `python3 -m pytest tests/unit/test_file.py -v`
- **Backtest**: `python3 -m agentic_trading backtest --config configs/backtest.toml`
- **Paper trade**: `python3 -m agentic_trading paper --config configs/paper.toml`
- **Live trade**: `python3 -m agentic_trading live --config configs/live.toml --live`
- **Docker**: `docker-compose up` (Redis, Postgres, Prometheus, Grafana)

## Architecture

### Event Flow
```
FeedManager -> CandleBuilder -> FeatureEngine -> Strategy -> PortfolioManager
-> ExecutionEngine -> Adapter -> FillEvent -> Journal
```

### Event Bus Topics
`strategy.signal`, `execution`, `feature.vector`, `state`, `system`, `market.candle`, `risk`, `governance`

### Modes
- **Backtest**: `SimClock` + `MemoryEventBus` + `BacktestAdapter` (deterministic)
- **Paper**: `WallClock` + `RedisStreamsBus` + `PaperAdapter` (simulated fills)
- **Live**: `WallClock` + `RedisStreamsBus` + `CCXTAdapter` (Bybit)

## Key File Locations

| Component | Path |
|---|---|
| Bootstrap | `src/agentic_trading/main.py` |
| Interfaces/Protocols | `src/agentic_trading/core/interfaces.py` |
| Events | `src/agentic_trading/core/events.py` |
| Enums | `src/agentic_trading/core/enums.py` |
| Errors | `src/agentic_trading/core/errors.py` |
| Config | `src/agentic_trading/core/config.py` |
| Execution engine | `src/agentic_trading/execution/engine.py` |
| Paper adapter | `src/agentic_trading/execution/adapters/paper.py` |
| CCXT adapter | `src/agentic_trading/execution/adapters/ccxt_adapter.py` |
| Portfolio manager | `src/agentic_trading/portfolio/manager.py` |
| Risk manager | `src/agentic_trading/risk/manager.py` |
| Governance gate | `src/agentic_trading/governance/gate.py` |
| Trade journal | `src/agentic_trading/journal/journal.py` |
| Feature engine | `src/agentic_trading/features/engine.py` |
| DB models | `src/agentic_trading/storage/postgres/models.py` |

## Coding Conventions

### Imports
```python
from __future__ import annotations  # Always first

import logging                      # stdlib
from datetime import datetime, timezone

from pydantic import BaseModel      # third-party

from agentic_trading.core.enums import OrderStatus  # absolute cross-package
from .order_manager import OrderManager              # relative same-package
```

### Type Hints (Python 3.10+ style)
```python
result: OrderAck | None = None        # Not Optional[OrderAck]
items: list[str] = []                  # Not List[str]
mapping: dict[str, Any] = {}           # Not Dict[str, Any]
callback: Callable[[str], float] | None = None
```

### Async Patterns

**Agent/service lifecycle** (start/stop/loop with graceful shutdown):
```python
async def start(self) -> None:
    self._running = True
    self._task = asyncio.create_task(self._loop(), name="descriptive-name")

async def stop(self) -> None:
    self._running = False
    if self._task is not None:
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

async def _loop(self) -> None:
    while self._running:
        try:
            await self._work()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Work failed")
        await asyncio.sleep(self._interval)
```

**Sync/async compatibility** (when caller might be sync or async):
```python
_result = self._checker.pre_trade_check(intent, state)
if inspect.isawaitable(_result):
    _result = await _result
result: RiskCheckResult = _result
```

**CPU-bound offload**:
```python
result = await asyncio.to_thread(self._sync_work, arg1, arg2)
```

### Pydantic Models (v2)
```python
class MyConfig(BaseModel):
    name: str
    items: list[str] = Field(default_factory=list)

    model_config = {"env_prefix": "TRADING_"}

# Immutable updates
updated = original.model_copy(update={"field": new_value})
```

### Logging
```python
logger = logging.getLogger(__name__)

logger.info("Order submitted: order_id=%s symbol=%s", ack.order_id, ack.symbol)
logger.warning("Rejected: dedupe_key=%s", intent.dedupe_key, exc_info=True)
logger.exception("Cycle failed")  # Only inside except blocks
```
Use `%s` format strings (not f-strings) for lazy evaluation.

### Error Handling
- All custom exceptions inherit from `TradingError` (see `core/errors.py`)
- Catch specific exceptions at handler level, not deep in call stacks
- Use `DuplicateOrderError` -> debug log, not warning
- Event bus handlers catch all exceptions to prevent consumer death

### Testing
```python
def _make_ctx(bus=None) -> TradingContext:
    """Factory fixture for test data."""
    ...

class TestMyComponent:
    def test_action_expected_result(self):
        ...

    def test_raises_on_invalid_input(self):
        with pytest.raises(OrderError, match="pattern"):
            ...
```
- Name: `test_<action>_<expected>`
- Factory functions for test data (prefix `_make_`)
- No magic numbers in assertions

### Naming
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants/enum members: `UPPER_CASE`
- Private: `_leading_underscore`
- Protocols: `I` prefix (`IEventBus`, `IStrategy`, `IExchangeAdapter`)

## Known Issues

- Redis consumer loop silently swallows exceptions at `redis_streams.py` line 151
- `IRiskChecker` Protocol says sync but `RiskManager` impl is async -- handled via `inspect.isawaitable()`
- One pre-existing property test failure: PaperAdapter balance math with large Decimals
- Strategies need 26+ candles (EMA26) before first signal in live mode
