"""Tests for signal cache TTL-based eviction in main.py.

Verifies:
- Stale entries (older than TTL) are evicted.
- Recent entries (within TTL) are preserved.
- Count-cap safety still removes oldest entries when exceeding max.
- Mix of stale and fresh entries only evicts stale ones.
- Entries without a timestamp attribute are left alone (defensive).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Reproduce the eviction logic from main.py as a standalone function so we
# can unit-test it without bootstrapping the full trading context.
# ---------------------------------------------------------------------------

_SIGNAL_CACHE_MAX = 500
_SIGNAL_CACHE_TTL = 300  # seconds


def _evict_signal_cache(
    cache: dict[str, Any],
    now: datetime,
    *,
    max_size: int = _SIGNAL_CACHE_MAX,
    ttl: int = _SIGNAL_CACHE_TTL,
) -> None:
    """Mirror of the eviction block in main.py (lines ~807-819)."""
    stale = [
        k
        for k, v in cache.items()
        if hasattr(v, "timestamp")
        and (now - v.timestamp).total_seconds() > ttl
    ]
    for k in stale:
        cache.pop(k, None)
    if len(cache) > max_size:
        excess = len(cache) - max_size
        for k in list(cache)[:excess]:
            cache.pop(k, None)


def _make_signal(trace_id: str, timestamp: datetime) -> MagicMock:
    sig = MagicMock()
    sig.trace_id = trace_id
    sig.timestamp = timestamp
    return sig


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSignalCacheEviction:
    """Tests for TTL-based signal cache eviction."""

    def test_stale_entries_evicted(self) -> None:
        """Entries older than TTL are removed."""
        now = datetime.now(tz=timezone.utc)
        old_ts = now - timedelta(seconds=_SIGNAL_CACHE_TTL + 60)

        cache: dict[str, Any] = {}
        for i in range(5):
            sig = _make_signal(f"stale-{i}", old_ts)
            cache[f"stale-{i}"] = sig

        _evict_signal_cache(cache, now)
        assert len(cache) == 0

    def test_recent_entries_preserved(self) -> None:
        """Entries within TTL are kept."""
        now = datetime.now(tz=timezone.utc)
        recent_ts = now - timedelta(seconds=60)

        cache: dict[str, Any] = {}
        for i in range(5):
            sig = _make_signal(f"recent-{i}", recent_ts)
            cache[f"recent-{i}"] = sig

        _evict_signal_cache(cache, now)
        assert len(cache) == 5

    def test_mix_of_stale_and_fresh(self) -> None:
        """Only stale entries are evicted; fresh entries survive."""
        now = datetime.now(tz=timezone.utc)
        old_ts = now - timedelta(seconds=_SIGNAL_CACHE_TTL + 10)
        fresh_ts = now - timedelta(seconds=30)

        cache: dict[str, Any] = {}
        for i in range(3):
            cache[f"stale-{i}"] = _make_signal(f"stale-{i}", old_ts)
        for i in range(4):
            cache[f"fresh-{i}"] = _make_signal(f"fresh-{i}", fresh_ts)

        _evict_signal_cache(cache, now)
        assert len(cache) == 4
        for i in range(4):
            assert f"fresh-{i}" in cache
        for i in range(3):
            assert f"stale-{i}" not in cache

    def test_count_cap_after_ttl_eviction(self) -> None:
        """Count cap kicks in when TTL eviction is not sufficient."""
        now = datetime.now(tz=timezone.utc)
        fresh_ts = now - timedelta(seconds=10)
        max_size = 5

        cache: dict[str, Any] = {}
        for i in range(8):
            cache[f"sig-{i}"] = _make_signal(f"sig-{i}", fresh_ts)

        _evict_signal_cache(cache, now, max_size=max_size)
        assert len(cache) == max_size
        # FIFO: first 3 removed (sig-0, sig-1, sig-2)
        assert "sig-0" not in cache
        assert "sig-1" not in cache
        assert "sig-2" not in cache
        # Last 5 remain
        for i in range(3, 8):
            assert f"sig-{i}" in cache

    def test_ttl_eviction_prevents_count_cap(self) -> None:
        """When TTL eviction brings count under max, count cap is not needed."""
        now = datetime.now(tz=timezone.utc)
        old_ts = now - timedelta(seconds=_SIGNAL_CACHE_TTL + 10)
        fresh_ts = now - timedelta(seconds=10)
        max_size = 5

        cache: dict[str, Any] = {}
        # 4 stale + 3 fresh = 7 entries (> max_size)
        for i in range(4):
            cache[f"stale-{i}"] = _make_signal(f"stale-{i}", old_ts)
        for i in range(3):
            cache[f"fresh-{i}"] = _make_signal(f"fresh-{i}", fresh_ts)

        _evict_signal_cache(cache, now, max_size=max_size)
        # TTL removes 4 stale, leaving 3 fresh (< max_size), so count-cap is no-op
        assert len(cache) == 3
        for i in range(3):
            assert f"fresh-{i}" in cache

    def test_entries_without_timestamp_preserved(self) -> None:
        """Entries lacking a timestamp attribute are never evicted by TTL."""
        now = datetime.now(tz=timezone.utc)
        cache: dict[str, Any] = {}
        # Plain string value — no .timestamp
        cache["no-ts"] = "raw_value"
        # Dict — no .timestamp
        cache["dict-val"] = {"data": 123}

        _evict_signal_cache(cache, now)
        assert len(cache) == 2
        assert "no-ts" in cache
        assert "dict-val" in cache

    def test_entry_exactly_at_ttl_boundary_preserved(self) -> None:
        """An entry exactly at TTL age is NOT evicted (> not >=)."""
        now = datetime.now(tz=timezone.utc)
        boundary_ts = now - timedelta(seconds=_SIGNAL_CACHE_TTL)

        cache: dict[str, Any] = {}
        cache["boundary"] = _make_signal("boundary", boundary_ts)

        _evict_signal_cache(cache, now)
        assert "boundary" in cache

    def test_entry_one_second_past_ttl_evicted(self) -> None:
        """An entry 1 second past TTL IS evicted."""
        now = datetime.now(tz=timezone.utc)
        past_ts = now - timedelta(seconds=_SIGNAL_CACHE_TTL + 1)

        cache: dict[str, Any] = {}
        cache["expired"] = _make_signal("expired", past_ts)

        _evict_signal_cache(cache, now)
        assert "expired" not in cache

    def test_empty_cache_no_error(self) -> None:
        """Eviction on empty cache doesn't raise."""
        now = datetime.now(tz=timezone.utc)
        cache: dict[str, Any] = {}
        _evict_signal_cache(cache, now)
        assert len(cache) == 0

    def test_real_signal_object_eviction(self) -> None:
        """Test with actual Signal objects (integration-like)."""
        # Import real Signal class
        from agentic_trading.core.events import Signal
        from agentic_trading.core.enums import SignalDirection, Timeframe

        now = datetime.now(tz=timezone.utc)
        old_ts = now - timedelta(seconds=_SIGNAL_CACHE_TTL + 60)
        fresh_ts = now - timedelta(seconds=10)

        old_sig = Signal(
            strategy_id="test_strat",
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            confidence=0.8,
            timestamp=old_ts,
            trace_id="old-trace",
        )
        fresh_sig = Signal(
            strategy_id="test_strat",
            symbol="ETH/USDT",
            direction=SignalDirection.SHORT,
            confidence=0.7,
            timestamp=fresh_ts,
            trace_id="fresh-trace",
        )

        cache: dict[str, Any] = {
            "old-trace": old_sig,
            "fresh-trace": fresh_sig,
        }

        _evict_signal_cache(cache, now)
        assert "old-trace" not in cache
        assert "fresh-trace" in cache
        assert cache["fresh-trace"].symbol == "ETH/USDT"
