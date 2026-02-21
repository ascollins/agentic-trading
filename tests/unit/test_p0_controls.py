"""Tests for P0 pre-trade controls: price collars, self-match prevention,
message throttles, and feature snapshot persistence.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_trading.core.enums import Exchange, OrderType, Side
from agentic_trading.core.events import FeatureVector, OrderIntent, RiskCheckResult
from agentic_trading.core.interfaces import PortfolioState
from agentic_trading.execution.risk.pre_trade import PreTradeChecker
from agentic_trading.intelligence.feature_snapshot import (
    FeatureSnapshot,
    FeatureSnapshotStore,
)
from agentic_trading.policy.default_policies import build_pre_trade_control_policies
from agentic_trading.policy.models import PolicyType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intent(
    symbol: str = "BTC/USDT",
    side: Side = Side.BUY,
    price: Decimal | None = Decimal("50000"),
    qty: Decimal = Decimal("0.1"),
    strategy_id: str = "test_strategy",
    reduce_only: bool = False,
) -> OrderIntent:
    return OrderIntent(
        dedupe_key=f"test-{symbol}-{side.value}",
        strategy_id=strategy_id,
        symbol=symbol,
        exchange=Exchange.BYBIT,
        side=side,
        order_type=OrderType.LIMIT if price else OrderType.MARKET,
        qty=qty,
        price=price,
        reduce_only=reduce_only,
    )


def _make_portfolio(
    positions: dict | None = None,
    balances: dict | None = None,
) -> PortfolioState:
    return PortfolioState(
        positions=positions or {},
        balances=balances or {},
    )


def _make_position(
    symbol: str = "BTC/USDT",
    qty: Decimal = Decimal("0.1"),
    entry_price: Decimal = Decimal("50000"),
    mark_price: Decimal = Decimal("50000"),
    side: str = "long",
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        qty=qty,
        entry_price=entry_price,
        mark_price=mark_price,
        side=SimpleNamespace(value=side),
        notional=qty * mark_price,
        is_open=qty > 0,
    )


def _make_order(
    symbol: str = "BTC/USDT",
    side: str = "sell",
    price: float = 49500.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        side=SimpleNamespace(value=side),
        price=Decimal(str(price)),
    )


def _first_failed(results: list[RiskCheckResult], check_name: str) -> RiskCheckResult | None:
    return next(
        (r for r in results if r.check_name == check_name and not r.passed),
        None,
    )


def _first_result(results: list[RiskCheckResult], check_name: str) -> RiskCheckResult | None:
    return next(
        (r for r in results if r.check_name == check_name),
        None,
    )


# ===========================================================================
# Test: Price Collar
# ===========================================================================


class TestPriceCollar:
    def test_pass_within_collar(self):
        """Order within 200 bps of reference price passes."""
        checker = PreTradeChecker(price_collar_bps=200.0)
        portfolio = _make_portfolio(
            positions={"BTC/USDT": _make_position(mark_price=Decimal("50000"))},
        )
        intent = _make_intent(price=Decimal("50500"))  # 100 bps off
        results = checker.check(intent, portfolio)
        collar_result = _first_result(results, "price_collar")
        assert collar_result is not None
        assert collar_result.passed

    def test_fail_outside_collar(self):
        """Order beyond 200 bps of reference price fails."""
        checker = PreTradeChecker(price_collar_bps=200.0)
        portfolio = _make_portfolio(
            positions={"BTC/USDT": _make_position(mark_price=Decimal("50000"))},
        )
        intent = _make_intent(price=Decimal("55000"))  # 1000 bps off
        results = checker.check(intent, portfolio)
        collar_result = _first_failed(results, "price_collar")
        assert collar_result is not None
        assert "1000" in collar_result.reason  # 1000 bps deviation

    def test_market_order_exempt(self):
        """Market orders (no price) are exempt from price collar."""
        checker = PreTradeChecker(price_collar_bps=200.0)
        portfolio = _make_portfolio(
            positions={"BTC/USDT": _make_position(mark_price=Decimal("50000"))},
        )
        intent = _make_intent(price=None)
        results = checker.check(intent, portfolio)
        collar_result = _first_result(results, "price_collar")
        assert collar_result is not None
        assert collar_result.passed

    def test_no_reference_price_passes(self):
        """When no reference price is available, pass conservatively."""
        checker = PreTradeChecker(price_collar_bps=200.0)
        portfolio = _make_portfolio()  # No positions
        intent = _make_intent(price=Decimal("50000"))
        results = checker.check(intent, portfolio)
        collar_result = _first_result(results, "price_collar")
        assert collar_result is not None
        assert collar_result.passed


# ===========================================================================
# Test: Self-Match Prevention
# ===========================================================================


class TestSelfMatchPrevention:
    def test_pass_no_open_orders(self):
        """No resting orders means no self-match possible."""
        checker = PreTradeChecker()
        intent = _make_intent(side=Side.BUY, price=Decimal("50000"))
        results = checker.check(intent, _make_portfolio())
        sm_result = _first_result(results, "self_match_prevention")
        assert sm_result is not None
        assert sm_result.passed

    def test_pass_same_side(self):
        """Buy order with resting buy does not self-match."""
        checker = PreTradeChecker()
        resting = [_make_order(side="buy", price=49000.0)]
        intent = _make_intent(side=Side.BUY, price=Decimal("50000"))
        results = checker.check(intent, _make_portfolio(), open_orders=resting)
        sm_result = _first_result(results, "self_match_prevention")
        assert sm_result is not None
        assert sm_result.passed

    def test_fail_buy_crosses_sell(self):
        """Buy at 50000 crosses resting sell at 49500."""
        checker = PreTradeChecker()
        resting = [_make_order(side="sell", price=49500.0)]
        intent = _make_intent(side=Side.BUY, price=Decimal("50000"))
        results = checker.check(intent, _make_portfolio(), open_orders=resting)
        sm_result = _first_failed(results, "self_match_prevention")
        assert sm_result is not None
        assert "self-match" in sm_result.reason

    def test_fail_sell_crosses_buy(self):
        """Sell at 49000 crosses resting buy at 50000."""
        checker = PreTradeChecker()
        resting = [_make_order(side="buy", price=50000.0)]
        intent = _make_intent(side=Side.SELL, price=Decimal("49000"))
        results = checker.check(intent, _make_portfolio(), open_orders=resting)
        sm_result = _first_failed(results, "self_match_prevention")
        assert sm_result is not None

    def test_pass_no_cross(self):
        """Buy at 49000 does not cross resting sell at 50000."""
        checker = PreTradeChecker()
        resting = [_make_order(side="sell", price=50000.0)]
        intent = _make_intent(side=Side.BUY, price=Decimal("49000"))
        results = checker.check(intent, _make_portfolio(), open_orders=resting)
        sm_result = _first_result(results, "self_match_prevention")
        assert sm_result is not None
        assert sm_result.passed

    def test_different_symbol_ignored(self):
        """Resting orders on different symbols are ignored."""
        checker = PreTradeChecker()
        resting = [_make_order(symbol="ETH/USDT", side="sell", price=49500.0)]
        intent = _make_intent(symbol="BTC/USDT", side=Side.BUY, price=Decimal("50000"))
        results = checker.check(intent, _make_portfolio(), open_orders=resting)
        sm_result = _first_result(results, "self_match_prevention")
        assert sm_result is not None
        assert sm_result.passed


# ===========================================================================
# Test: Message Throttle
# ===========================================================================


class TestMessageThrottle:
    def test_pass_within_limits(self):
        """Normal message rate passes."""
        checker = PreTradeChecker(
            max_messages_per_minute_per_strategy=10,
            max_messages_per_minute_per_symbol=10,
        )
        intent = _make_intent()
        results = checker.check(intent, _make_portfolio())
        throttle_result = _first_result(results, "message_throttle")
        assert throttle_result is not None
        assert throttle_result.passed

    def test_fail_strategy_throttle(self):
        """Exceeding strategy message rate triggers throttle."""
        checker = PreTradeChecker(
            max_messages_per_minute_per_strategy=3,
            max_messages_per_minute_per_symbol=100,
        )
        portfolio = _make_portfolio()
        # Fire 4 checks (3 is the max, so the 4th should fail)
        for _ in range(3):
            checker.check(_make_intent(), portfolio)

        results = checker.check(_make_intent(), portfolio)
        throttle_result = _first_failed(results, "message_throttle")
        assert throttle_result is not None
        assert "per_strategy" in throttle_result.details.get("throttle_type", "")

    def test_fail_symbol_throttle(self):
        """Exceeding per-symbol message rate triggers throttle."""
        checker = PreTradeChecker(
            max_messages_per_minute_per_strategy=100,
            max_messages_per_minute_per_symbol=3,
        )
        portfolio = _make_portfolio()
        # Use different strategy IDs so strategy throttle doesn't trip
        for i in range(3):
            checker.check(
                _make_intent(strategy_id=f"strat_{i}"),
                portfolio,
            )

        results = checker.check(
            _make_intent(strategy_id="strat_final"),
            portfolio,
        )
        throttle_result = _first_failed(results, "message_throttle")
        assert throttle_result is not None
        assert "per_symbol" in throttle_result.details.get("throttle_type", "")


# ===========================================================================
# Test: Feature Snapshot
# ===========================================================================


class TestFeatureSnapshot:
    def test_create_and_hash(self):
        """Snapshot creation computes a deterministic hash."""
        snapshot = FeatureSnapshot(
            symbol="BTC/USDT",
            feature_vector={"rsi": 45.0, "ema_20": 50100.0},
            strategy_id="trend_following",
            signal_direction="long",
            signal_confidence=0.75,
        )
        h = snapshot.compute_hash()
        assert isinstance(h, str)
        assert len(h) == 16

        # Same data produces same hash
        snapshot2 = FeatureSnapshot(
            symbol="BTC/USDT",
            feature_vector={"rsi": 45.0, "ema_20": 50100.0},
            strategy_id="trend_following",
            signal_direction="long",
            signal_confidence=0.75,
        )
        assert snapshot2.compute_hash() == h

    def test_different_data_different_hash(self):
        """Different feature vectors produce different hashes."""
        s1 = FeatureSnapshot(
            symbol="BTC/USDT",
            feature_vector={"rsi": 45.0},
            strategy_id="test",
        )
        s2 = FeatureSnapshot(
            symbol="BTC/USDT",
            feature_vector={"rsi": 55.0},
            strategy_id="test",
        )
        assert s1.compute_hash() != s2.compute_hash()


class TestFeatureSnapshotStore:
    def test_store_and_retrieve(self):
        """Store a snapshot in-memory and retrieve by ID."""
        store = FeatureSnapshotStore()
        snapshot = FeatureSnapshot(
            symbol="BTC/USDT",
            feature_vector={"rsi": 45.0},
            strategy_id="test",
        )
        snapshot_id = store.store(snapshot)
        assert store.count == 1

        retrieved = store.get(snapshot_id)
        assert retrieved is not None
        assert retrieved.symbol == "BTC/USDT"
        assert retrieved.content_hash != ""

    def test_retrieve_by_dedupe_key(self):
        """Retrieve snapshot by dedupe_key."""
        store = FeatureSnapshotStore()
        snapshot = FeatureSnapshot(
            symbol="BTC/USDT",
            dedupe_key="dk-123",
            feature_vector={"rsi": 45.0},
            strategy_id="test",
        )
        store.store(snapshot)

        found = store.get_by_dedupe_key("dk-123")
        assert found is not None
        assert found.dedupe_key == "dk-123"

        not_found = store.get_by_dedupe_key("dk-999")
        assert not_found is None

    def test_retrieve_by_trace(self):
        """Retrieve all snapshots for a trace_id."""
        store = FeatureSnapshotStore()
        for i in range(3):
            store.store(FeatureSnapshot(
                symbol=f"SYM{i}",
                trace_id="trace-abc",
                feature_vector={},
                strategy_id="test",
            ))
        store.store(FeatureSnapshot(
            symbol="OTHER",
            trace_id="trace-xyz",
            feature_vector={},
            strategy_id="test",
        ))

        abc_snaps = store.get_by_trace("trace-abc")
        assert len(abc_snaps) == 3
        xyz_snaps = store.get_by_trace("trace-xyz")
        assert len(xyz_snaps) == 1

    def test_jsonl_persistence(self):
        """Snapshots persist to JSONL and reload on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snapshots.jsonl"

            # Write
            store1 = FeatureSnapshotStore(persistence_path=path)
            store1.store(FeatureSnapshot(
                symbol="BTC/USDT",
                feature_vector={"rsi": 45.0},
                strategy_id="test",
                dedupe_key="dk-1",
            ))
            store1.store(FeatureSnapshot(
                symbol="ETH/USDT",
                feature_vector={"rsi": 55.0},
                strategy_id="test",
                dedupe_key="dk-2",
            ))
            assert store1.count == 2

            # Reload into new store
            store2 = FeatureSnapshotStore(persistence_path=path)
            assert store2.count == 2
            assert store2.get_by_dedupe_key("dk-1") is not None
            assert store2.get_by_dedupe_key("dk-2") is not None

    def test_max_memory_eviction(self):
        """Buffer evicts oldest entries when max_memory is exceeded."""
        store = FeatureSnapshotStore(max_memory=3)
        ids = []
        for i in range(5):
            sid = store.store(FeatureSnapshot(
                symbol=f"SYM{i}",
                feature_vector={},
                strategy_id="test",
            ))
            ids.append(sid)

        assert store.count == 3
        # Oldest 2 should be evicted from buffer
        assert store.get(ids[0]) is None
        assert store.get(ids[1]) is None
        assert store.get(ids[4]) is not None


# ===========================================================================
# Test: Pre-Trade Control Policy Set
# ===========================================================================


class TestPreTradeControlPolicies:
    def test_policy_set_has_correct_rules(self):
        """The policy set should contain 4 rules with correct types."""
        ps = build_pre_trade_control_policies()
        assert ps.set_id == "pre_trade_controls"
        assert len(ps.rules) == 4

        rule_ids = {r.rule_id for r in ps.rules}
        assert "price_collar" in rule_ids
        assert "self_match_prevention" in rule_ids
        assert "message_throttle_strategy" in rule_ids
        assert "message_throttle_symbol" in rule_ids

        for rule in ps.rules:
            assert rule.policy_type == PolicyType.PRE_TRADE_CONTROL

    def test_custom_config(self):
        """Policy rules should use config values."""
        from agentic_trading.core.config import RiskConfig

        cfg = RiskConfig(
            price_collar_bps=100.0,
            max_messages_per_minute_per_strategy=20,
            max_messages_per_minute_per_symbol=10,
        )
        ps = build_pre_trade_control_policies(cfg)
        collar_rule = next(r for r in ps.rules if r.rule_id == "price_collar")
        assert collar_rule.threshold == 100.0

        strat_throttle = next(
            r for r in ps.rules if r.rule_id == "message_throttle_strategy"
        )
        assert strat_throttle.threshold == 20.0


# ===========================================================================
# Test: FeatureVector version field
# ===========================================================================


class TestFeatureVectorVersion:
    def test_default_empty(self):
        """FeatureVector should have empty feature_version by default."""
        from agentic_trading.core.enums import Timeframe

        fv = FeatureVector(
            symbol="BTC/USDT",
            timeframe=Timeframe.M5,
            features={"rsi": 45.0},
        )
        assert fv.feature_version == ""

    def test_set_version(self):
        """FeatureVector should accept a feature_version."""
        from agentic_trading.core.enums import Timeframe

        fv = FeatureVector(
            symbol="BTC/USDT",
            timeframe=Timeframe.M5,
            features={"rsi": 45.0},
            feature_version="abc123",
        )
        assert fv.feature_version == "abc123"
