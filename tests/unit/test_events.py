"""Test event creation, serialization, and deserialization."""

import json
from datetime import datetime, timezone
from decimal import Decimal

from agentic_trading.core.enums import (
    Exchange,
    OrderType,
    Side,
    SignalDirection,
    Timeframe,
    TimeInForce,
)
from agentic_trading.core.events import (
    BaseEvent,
    CandleEvent,
    OrderIntent,
    Signal,
)


class TestBaseEvent:
    def test_creation_defaults(self):
        event = BaseEvent()
        assert event.event_id is not None
        assert len(event.event_id) > 0
        assert event.timestamp is not None
        assert event.trace_id is not None
        assert event.source_module == ""

    def test_unique_event_ids(self):
        e1 = BaseEvent()
        e2 = BaseEvent()
        assert e1.event_id != e2.event_id

    def test_timestamp_is_utc(self):
        event = BaseEvent()
        assert event.timestamp.tzinfo is not None

    def test_serialization(self):
        event = BaseEvent(source_module="test")
        json_str = event.model_dump_json()
        data = json.loads(json_str)
        assert "event_id" in data
        assert "timestamp" in data
        assert "trace_id" in data
        assert data["source_module"] == "test"

    def test_deserialization(self):
        event = BaseEvent(source_module="test")
        json_str = event.model_dump_json()
        restored = BaseEvent.model_validate_json(json_str)
        assert restored.event_id == event.event_id
        assert restored.source_module == "test"


class TestCandleEvent:
    def test_creation(self):
        event = CandleEvent(
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            timeframe=Timeframe.M5,
            open=67000.0,
            high=67200.0,
            low=66900.0,
            close=67100.0,
            volume=25.0,
        )
        assert event.symbol == "BTC/USDT"
        assert event.source_module == "data"
        assert event.is_closed is True

    def test_serialization_roundtrip(self):
        event = CandleEvent(
            symbol="ETH/USDT",
            exchange=Exchange.BINANCE,
            timeframe=Timeframe.H1,
            open=2000.0,
            high=2050.0,
            low=1990.0,
            close=2030.0,
            volume=500.0,
            trades=1200,
        )
        json_str = event.model_dump_json()
        restored = CandleEvent.model_validate_json(json_str)
        assert restored.symbol == "ETH/USDT"
        assert restored.close == 2030.0
        assert restored.trades == 1200


class TestSignal:
    def test_creation(self):
        signal = Signal(
            strategy_id="trend_following",
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            confidence=0.75,
            rationale="EMA crossover",
            timeframe=Timeframe.M5,
        )
        assert signal.strategy_id == "trend_following"
        assert signal.direction == SignalDirection.LONG
        assert signal.confidence == 0.75
        assert signal.source_module == "strategy"

    def test_serialization_roundtrip(self):
        signal = Signal(
            strategy_id="mean_reversion",
            symbol="ETH/USDT",
            direction=SignalDirection.SHORT,
            confidence=0.55,
            features_used={"rsi": 78.5, "bb_upper": 2050.0},
        )
        json_str = signal.model_dump_json()
        restored = Signal.model_validate_json(json_str)
        assert restored.strategy_id == "mean_reversion"
        assert restored.direction == SignalDirection.SHORT
        assert restored.features_used["rsi"] == 78.5


class TestOrderIntent:
    def test_creation_with_dedupe_key(self):
        intent = OrderIntent(
            dedupe_key="tf-btc-12345-001",
            strategy_id="trend_following",
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            qty=Decimal("0.01"),
            price=Decimal("67000.00"),
        )
        assert intent.dedupe_key == "tf-btc-12345-001"
        assert intent.source_module == "execution"

    def test_dedupe_key_is_required(self):
        """OrderIntent requires a dedupe_key."""
        intent = OrderIntent(
            dedupe_key="my-unique-key",
            strategy_id="test",
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            side=Side.BUY,
            qty=Decimal("1"),
        )
        assert intent.dedupe_key == "my-unique-key"

    def test_serialization_roundtrip(self, sample_order_intent):
        json_str = sample_order_intent.model_dump_json()
        restored = OrderIntent.model_validate_json(json_str)
        assert restored.dedupe_key == sample_order_intent.dedupe_key
        assert restored.symbol == "BTC/USDT"
        assert restored.side == Side.BUY
        assert restored.qty == Decimal("0.01")

    def test_different_dedupe_keys(self):
        intent1 = OrderIntent(
            dedupe_key="key-aaa",
            strategy_id="s1",
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            side=Side.BUY,
            qty=Decimal("1"),
        )
        intent2 = OrderIntent(
            dedupe_key="key-bbb",
            strategy_id="s1",
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            side=Side.BUY,
            qty=Decimal("1"),
        )
        assert intent1.dedupe_key != intent2.dedupe_key
