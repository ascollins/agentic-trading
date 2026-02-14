"""Shared fixtures for journal tests."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from agentic_trading.journal.record import (
    FillLeg,
    MarkSample,
    TradePhase,
    TradeOutcome,
    TradeRecord,
)
from agentic_trading.journal.journal import TradeJournal
from agentic_trading.journal.rolling_tracker import RollingTracker
from agentic_trading.journal.confidence import ConfidenceCalibrator
from agentic_trading.journal.monte_carlo import MonteCarloProjector
from agentic_trading.journal.overtrading import OvertradingDetector
from agentic_trading.journal.coin_flip import CoinFlipBaseline


@pytest.fixture
def base_time():
    return datetime(2024, 1, 1, 12, 0, 0)


@pytest.fixture
def journal():
    return TradeJournal(max_closed_trades=100)


@pytest.fixture
def rolling_tracker():
    return RollingTracker(window_size=50)


@pytest.fixture
def confidence_calibrator():
    return ConfidenceCalibrator(n_buckets=5, max_observations=500)


@pytest.fixture
def monte_carlo():
    return MonteCarloProjector(n_simulations=500, seed=42)


@pytest.fixture
def overtrading_detector():
    return OvertradingDetector(
        lookback=20, threshold_z=2.0, cooldown_minutes=5, min_samples=5
    )


@pytest.fixture
def coin_flip():
    return CoinFlipBaseline(n_simulations=5000, seed=42)


def make_fill(
    fill_id: str = "fill_1",
    order_id: str = "order_1",
    side: str = "buy",
    price: float = 100.0,
    qty: float = 1.0,
    fee: float = 0.1,
    timestamp: datetime | None = None,
) -> FillLeg:
    """Helper to create a FillLeg."""
    return FillLeg(
        fill_id=fill_id,
        order_id=order_id,
        side=side,
        price=Decimal(str(price)),
        qty=Decimal(str(qty)),
        fee=Decimal(str(fee)),
        fee_currency="USDT",
        is_maker=False,
        timestamp=timestamp or datetime(2024, 1, 1, 12, 0, 0),
    )


def make_winning_trade(
    strategy_id: str = "trend",
    entry_price: float = 100.0,
    exit_price: float = 110.0,
    qty: float = 1.0,
    base_time: datetime | None = None,
) -> TradeRecord:
    """Create a complete winning trade record."""
    bt = base_time or datetime(2024, 1, 1, 12, 0, 0)
    trade = TradeRecord(
        trace_id="trace_win",
        strategy_id=strategy_id,
        symbol="BTC/USDT",
        direction="long",
        signal_confidence=0.8,
        initial_risk_price=Decimal(str(entry_price * 0.95)),
    )
    trade.add_entry_fill(make_fill(
        price=entry_price, qty=qty, timestamp=bt
    ))
    trade.compute_initial_risk()
    trade.add_mark_sample(MarkSample(
        timestamp=bt + timedelta(minutes=30),
        mark_price=Decimal(str(exit_price * 0.98)),
        unrealized_pnl=Decimal(str((exit_price * 0.98 - entry_price) * qty)),
    ))
    trade.add_mark_sample(MarkSample(
        timestamp=bt + timedelta(minutes=45),
        mark_price=Decimal(str(exit_price * 1.02)),
        unrealized_pnl=Decimal(str((exit_price * 1.02 - entry_price) * qty)),
    ))
    trade.add_exit_fill(make_fill(
        fill_id="exit_1",
        order_id="exit_order",
        side="sell",
        price=exit_price,
        qty=qty,
        timestamp=bt + timedelta(hours=1),
    ))
    return trade


def make_losing_trade(
    strategy_id: str = "trend",
    entry_price: float = 100.0,
    exit_price: float = 95.0,
    qty: float = 1.0,
    base_time: datetime | None = None,
) -> TradeRecord:
    """Create a complete losing trade record."""
    bt = base_time or datetime(2024, 1, 1, 12, 0, 0)
    trade = TradeRecord(
        trace_id="trace_loss",
        strategy_id=strategy_id,
        symbol="BTC/USDT",
        direction="long",
        signal_confidence=0.5,
        initial_risk_price=Decimal(str(entry_price * 0.95)),
    )
    trade.add_entry_fill(make_fill(
        price=entry_price, qty=qty, timestamp=bt
    ))
    trade.compute_initial_risk()
    trade.add_exit_fill(make_fill(
        fill_id="exit_1",
        order_id="exit_order",
        side="sell",
        price=exit_price,
        qty=qty,
        timestamp=bt + timedelta(hours=1),
    ))
    return trade
