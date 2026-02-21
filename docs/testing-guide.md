# Testing Guide

## 1. Quick Start

```bash
# Run unit + property tests (fast, no external services)
make test

# Run all tests including integration
make test-all

# Run a specific test file
python3 -m pytest tests/unit/test_feature_engine.py -v

# Run a specific test class or method
python3 -m pytest tests/unit/test_policy_engine.py::TestPolicyEngine::test_rule_passes -v

# Run with coverage report
make test-cov

# Full local CI (lint + test + typecheck)
make ci
```

## 2. Test Organization

```
tests/
├── conftest.py                  # Shared fixtures (instruments, candles, bus, etc.)
├── unit/                        # Fast, isolated tests (no I/O, no external services)
│   ├── governance/              # GovernanceGate subsystem tests
│   │   ├── conftest.py          # Governance-specific fixtures
│   │   ├── test_maturity.py
│   │   ├── test_health_score.py
│   │   ├── test_canary.py
│   │   ├── test_impact_classifier.py
│   │   ├── test_drift_detector.py
│   │   ├── test_tokens.py
│   │   └── test_gate.py
│   ├── journal/                 # TradeJournal subsystem tests
│   │   ├── conftest.py          # Journal-specific fixtures + factories
│   │   ├── test_record.py
│   │   ├── test_journal.py
│   │   ├── test_rolling_tracker.py
│   │   ├── test_confidence.py
│   │   ├── test_monte_carlo.py
│   │   ├── test_overtrading.py
│   │   ├── test_mistakes.py
│   │   ├── test_session_analysis.py
│   │   ├── test_correlation.py
│   │   ├── test_replay.py
│   │   └── test_export.py
│   ├── narration/               # Narration subsystem tests
│   │   ├── conftest.py
│   │   ├── test_store.py
│   │   ├── test_schema.py
│   │   ├── test_service.py
│   │   └── test_presenter.py
│   ├── test_control_plane_*.py  # ToolGateway, policy, approval (day1-day4)
│   ├── test_policy_engine.py    # PolicyEngine rule evaluation
│   ├── test_agent_framework.py  # BaseAgent, AgentRegistry
│   ├── test_feature_engine.py   # FeatureEngine core indicators
│   ├── test_smc_features.py     # Smart Money Concepts indicators
│   ├── test_strategies.py       # Strategy signal generation
│   └── ...
├── integration/                 # Cross-component tests (MemoryEventBus, no Docker)
│   ├── test_control_plane_pipeline.py   # ToolGateway end-to-end flow
│   ├── test_pre_trade_safety.py         # Price collar, self-match, throttle
│   ├── test_surveillance_detection.py   # Wash trade + spoofing detection
│   ├── test_feature_pipeline_extended.py # ARIMA, FFT, event bus integration
│   ├── test_model_registry_lifecycle.py # ModelRegistry + AuditBundle
│   ├── test_kill_switch_propagation.py  # Kill switch -> ExecutionEngine
│   ├── test_full_pipeline.py            # Signal -> Execution end-to-end
│   ├── test_order_dedupe.py             # Deduplication via event bus
│   ├── test_event_bus_roundtrip.py      # Pub/sub roundtrip
│   └── test_strategy_parity.py          # Strategy consistency
├── property/                    # Hypothesis property-based tests
│   ├── test_risk_invariants.py  # Sizing always non-negative, within bounds
│   ├── test_no_lookahead.py     # Strategy doesn't use future data
│   ├── test_order_state_machine.py # Order status transitions
│   └── test_position_math.py    # Position PnL calculations
└── golden/                      # Deterministic regression tests
    └── test_deterministic_backtest.py  # Same input -> same output (placeholder)
```

### Markers

Defined in `pyproject.toml`:

| Marker | Usage | Command |
|--------|-------|---------|
| `integration` | Tests requiring MemoryEventBus or multi-component wiring | `pytest -m integration` |
| `property` | Hypothesis property-based tests | `pytest -m property` |

## 3. Fixture Catalog

### Root `tests/conftest.py`

| Fixture | Type | Description |
|---------|------|-------------|
| `sample_instrument` | `Instrument` | BTC/USDT perp on Binance (tick 0.01, step 0.001, 125x max) |
| `sample_candle` | `Candle` | Single BTC/USDT 1m candle at 67000 |
| `sample_candles` | Factory `(n=100) -> list[Candle]` | Seeded random walk from 67000 (seed=42) |
| `sim_clock` | `SimClock` | Fixed start 2024-06-01 UTC |
| `memory_bus` | `MemoryEventBus` | Fresh in-memory event bus |
| `sample_order_intent` | `OrderIntent` | BUY 0.01 BTC/USDT LIMIT @ 67000 |
| `sample_fill` | `Fill` | BUY fill at 67005.50 |
| `sample_position` | `Position` | LONG 0.5 BTC entry=67000 mark=67500 |
| `sample_balance` | `Balance` | 100k USDT (65k free, 35k used) |

### Governance `tests/unit/governance/conftest.py`

| Fixture | Type | Description |
|---------|------|-------------|
| `maturity_config` | `MaturityConfig` | Default maturity config |
| `health_config` | `HealthScoreConfig` | Default health score config |
| `canary_config` | `CanaryConfig` | Default canary config |
| `impact_config` | `ImpactClassifierConfig` | Default impact classifier config |
| `drift_config` | `DriftDetectorConfig` | Default drift detector config |
| `token_config` | `ExecutionTokenConfig` | Default token config |
| `governance_config` | `GovernanceConfig` | Governance enabled |
| `governance_gate` | `GovernanceGate` | Fully wired gate (tokens=None, bus=None) |

### Journal `tests/unit/journal/conftest.py`

| Fixture / Factory | Type | Description |
|-------------------|------|-------------|
| `journal` | `TradeJournal` | max_closed_trades=100 |
| `rolling_tracker` | `RollingTracker` | window_size=50 |
| `confidence_calibrator` | `ConfidenceCalibrator` | 5 buckets, 500 max |
| `monte_carlo` | `MonteCarloProjector` | 500 sims, seed=42 |
| `overtrading_detector` | `OvertradingDetector` | lookback=20, z=2.0 |
| `coin_flip` | `CoinFlipBaseline` | 5000 sims, seed=42 |
| `make_fill()` | Free function | Creates `FillLeg` with defaults |
| `make_winning_trade()` | Free function | Complete winning `TradeRecord` |
| `make_losing_trade()` | Free function | Complete losing `TradeRecord` |

## 4. Writing Unit Tests

### Conventions

```python
from __future__ import annotations  # Always first

import pytest

from agentic_trading.core.enums import Side, OrderType
from agentic_trading.execution.risk.pre_trade import PreTradeChecker


class TestPreTradeChecker:
    """Group related tests by component."""

    def test_price_collar_passes_within_band(self):
        """Verb phrase describing expected behavior."""
        checker = PreTradeChecker(price_collar_bps=200.0)
        # ... setup ...
        results = checker.check(intent, portfolio)
        assert collar.passed is True

    def test_price_collar_rejects_outside_band(self):
        """Negative case — rejection path."""
        # ...
        assert collar.passed is False
        assert "collar" in collar.reason.lower()
```

### Rules

1. **Naming**: `test_<action>_<expected_result>` (e.g., `test_promote_research_to_paper`)
2. **Factory functions**: Prefix with `_make_` (e.g., `_make_intent()`, `_make_portfolio()`)
3. **No magic numbers**: Use named constants or clear inline values with comments
4. **One assertion focus**: Each test should verify one behavior, even if multiple `assert` statements
5. **Test both paths**: Always test the success path AND the rejection/failure path
6. **Use `pytest.raises`** for expected exceptions:
   ```python
   with pytest.raises(OrderError, match="duplicate"):
       engine.handle_intent(duplicate_intent)
   ```

### Async Tests

All async tests use `@pytest.mark.asyncio` (auto mode is configured in `pyproject.toml`):

```python
class TestAsyncComponent:
    @pytest.mark.asyncio
    async def test_event_handler_processes_fill(self):
        bus = MemoryEventBus()
        # ... setup ...
        await bus.publish("execution.fill", fill_event)
        await asyncio.sleep(0.05)  # Let handlers process
        assert captured[0].event_type == "fill"
```

## 5. Writing Integration Tests

### MemoryEventBus Pattern

All integration tests use `MemoryEventBus` — no Redis, Postgres, or Docker required. This is the established pattern from `test_kill_switch_propagation.py`:

```python
from agentic_trading.event_bus.memory_bus import MemoryEventBus

class TestMyIntegration:
    @pytest.mark.asyncio
    async def test_event_flows_through_components(self):
        bus = MemoryEventBus()
        captured: list[SomeEvent] = []

        async def capture(event):
            captured.append(event)

        await bus.subscribe("output.topic", "test", capture)

        # Wire up components with the bus
        component = MyComponent(event_bus=bus)
        await component.start()

        # Publish input events
        await bus.publish("input.topic", input_event)
        await asyncio.sleep(0.05)  # Settle time for handlers

        # Assert output
        assert len(captured) >= 1
        assert captured[0].field == expected_value

        await component.stop()
```

### Stub Catalog

Integration tests use inline stubs rather than shared mocks:

| Stub | Used In | Purpose |
|------|---------|---------|
| `_AlwaysPassRiskChecker` | kill switch, pipeline | Passes all pre/post trade checks |
| `_DummyAdapter` | kill switch, pipeline | Records submissions, returns FILLED |
| `_StubAdapter` | control plane | Records `submit_order` calls for ToolGateway |
| `_make_intent()` | pre-trade, surveillance | Factory for `OrderIntent` |
| `_make_portfolio()` | pre-trade | Factory for `PortfolioState` with positions/balances |
| `_make_fill()` | surveillance | Factory for `FillEvent` |
| `_make_resting_order()` | pre-trade | `SimpleNamespace` stub for open order |

### Tips

- **Settle time**: After publishing events, use `await asyncio.sleep(0.05)` to let async handlers process
- **Agent lifecycle**: Always `await agent.start()` before publishing and `await agent.stop()` after assertions
- **Capture pattern**: Use a `captured: list = []` with an `async def capture(event)` subscriber
- **Isolation**: Each test creates its own `MemoryEventBus` and components — no shared state

## 6. Property Tests with Hypothesis

Property tests live in `tests/property/` and use the `hypothesis` library to generate random inputs that must satisfy invariants.

```python
from hypothesis import given, settings, strategies as st

@given(
    capital=st.floats(min_value=100, max_value=1e8),
    risk_per_trade=st.floats(min_value=0.001, max_value=0.1),
    atr=st.floats(min_value=0.01, max_value=1000),
    price=st.floats(min_value=0.01, max_value=100000),
)
def test_vol_adjusted_size_non_negative(capital, risk_per_trade, atr, price):
    """Size is always >= 0 regardless of input."""
    size = volatility_adjusted_size(capital, risk_per_trade, atr, price)
    assert size >= 0
```

### Existing Property Tests

| File | Invariants Tested |
|------|-------------------|
| `test_risk_invariants.py` | Sizing always non-negative, fixed fractional within bounds, Kelly clamped |
| `test_no_lookahead.py` | Strategy never accesses future candle data |
| `test_order_state_machine.py` | Order status transitions follow allowed paths |
| `test_position_math.py` | Position PnL calculations consistent |

## 7. Running CI Locally

```bash
# Full CI pipeline (matches GitHub Actions)
make ci

# Individual steps:
make lint          # ruff check src/ tests/
make test          # unit + property tests
make typecheck     # mypy on core/governance/event_bus modules

# With coverage:
make test-cov      # Generates htmlcov/ directory

# Integration tests (uses MemoryEventBus, no Docker needed for most)
make test-integration
```

### GitHub Actions Workflow

The CI pipeline (`.github/workflows/ci.yml`) runs:

1. **lint** — `ruff check` + `ruff format --check`
2. **typecheck** — `mypy` on core modules (non-blocking warnings)
3. **test-unit** — unit + property tests on Python 3.11 & 3.12 with coverage
4. **test-integration** — integration tests with Redis + Postgres services
5. **docker-build** — verify Docker image builds
6. **ci-complete** — summary gate requiring all jobs to pass

## 8. Known Issues

| Issue | Workaround |
|-------|------------|
| 1 pre-existing property test failure: PaperAdapter balance math with large Decimals | Marked as known; does not block CI |
| `IRiskChecker` Protocol says sync but `RiskManager` is async | Handled via `inspect.isawaitable()` in ExecutionEngine |
| Strategies need 26+ candles (EMA26) before first signal in live mode | Expected — not a test issue |
| Golden tests are placeholders (require historical data fixture) | Skipped via `@pytest.mark.skip` |
| Redis consumer loop silently swallows exceptions at `redis_streams.py:151` | Integration tests use MemoryEventBus instead |
