"""Test CircuitBreaker trips and resets, with hysteresis behavior."""

import time
from unittest.mock import patch

from agentic_trading.core.enums import CircuitBreakerType
from agentic_trading.risk.circuit_breakers import CircuitBreaker, CircuitBreakerManager


class TestCircuitBreaker:
    def test_not_tripped_below_threshold(self):
        cb = CircuitBreaker(
            breaker_type=CircuitBreakerType.VOLATILITY,
            threshold=3.0,
            hysteresis=3,
        )
        result = cb.check(2.0)
        assert result is False
        assert cb.tripped is False

    def test_single_violation_not_enough(self):
        cb = CircuitBreaker(
            breaker_type=CircuitBreakerType.VOLATILITY,
            threshold=3.0,
            hysteresis=3,
        )
        cb.check(5.0)  # First violation
        assert cb.tripped is False

    def test_trips_after_hysteresis_violations(self):
        cb = CircuitBreaker(
            breaker_type=CircuitBreakerType.VOLATILITY,
            threshold=3.0,
            hysteresis=3,
            window_seconds=300.0,
        )
        cb.check(5.0)  # Violation 1
        cb.check(5.0)  # Violation 2
        result = cb.check(5.0)  # Violation 3 -> trips
        assert result is True
        assert cb.tripped is True
        assert cb.trip_count == 1

    def test_stays_tripped_during_cooldown(self):
        cb = CircuitBreaker(
            breaker_type=CircuitBreakerType.VOLATILITY,
            threshold=3.0,
            hysteresis=1,
            cooldown_seconds=300.0,
        )
        cb.check(5.0)  # Trip
        assert cb.tripped is True
        # Check again with value below threshold - still tripped (cooldown)
        result = cb.check(1.0)
        assert result is True

    def test_resets_after_cooldown_with_low_value(self):
        cb = CircuitBreaker(
            breaker_type=CircuitBreakerType.VOLATILITY,
            threshold=3.0,
            hysteresis=1,
            cooldown_seconds=0.01,  # Very short cooldown for testing
        )
        cb.check(5.0)  # Trip
        assert cb.tripped is True

        # Wait for cooldown
        time.sleep(0.02)
        result = cb.check(1.0)  # Below threshold + cooldown elapsed
        assert result is False
        assert cb.tripped is False

    def test_force_trip(self):
        cb = CircuitBreaker(
            breaker_type=CircuitBreakerType.SPREAD,
            threshold=10.0,
            hysteresis=5,
        )
        cb.force_trip("manual test")
        assert cb.tripped is True
        assert cb.trip_count == 1

    def test_force_reset(self):
        cb = CircuitBreaker(
            breaker_type=CircuitBreakerType.SPREAD,
            threshold=10.0,
            hysteresis=1,
        )
        cb.check(20.0)  # Trip
        assert cb.tripped is True
        cb.force_reset()
        assert cb.tripped is False


class TestCircuitBreakerHysteresis:
    def test_violations_expire_outside_window(self):
        cb = CircuitBreaker(
            breaker_type=CircuitBreakerType.VOLATILITY,
            threshold=3.0,
            hysteresis=3,
            window_seconds=0.01,  # Very short window
        )
        cb.check(5.0)  # Violation 1
        cb.check(5.0)  # Violation 2
        time.sleep(0.02)  # Wait for window to expire
        result = cb.check(5.0)  # Only 1 violation in window
        assert result is False
        assert cb.tripped is False


class TestCircuitBreakerManager:
    def test_add_and_evaluate(self):
        mgr = CircuitBreakerManager()
        mgr.add_breaker(CircuitBreaker(
            breaker_type=CircuitBreakerType.VOLATILITY,
            threshold=3.0,
            hysteresis=1,
        ))
        events = mgr.evaluate("volatility", 5.0)
        assert len(events) == 1
        assert events[0].tripped is True

    def test_evaluate_no_state_change_no_event(self):
        mgr = CircuitBreakerManager()
        mgr.add_breaker(CircuitBreaker(
            breaker_type=CircuitBreakerType.VOLATILITY,
            threshold=3.0,
            hysteresis=3,
        ))
        events = mgr.evaluate("volatility", 5.0)  # 1 violation, need 3
        assert len(events) == 0

    def test_is_any_tripped(self):
        mgr = CircuitBreakerManager()
        mgr.add_breaker(CircuitBreaker(
            breaker_type=CircuitBreakerType.VOLATILITY,
            threshold=3.0,
            hysteresis=1,
        ))
        assert mgr.is_any_tripped() is False
        mgr.evaluate("volatility", 5.0)
        assert mgr.is_any_tripped() is True

    def test_reset_all(self):
        mgr = CircuitBreakerManager()
        mgr.add_breaker(CircuitBreaker(
            breaker_type=CircuitBreakerType.VOLATILITY,
            threshold=3.0,
            hysteresis=1,
        ))
        mgr.evaluate("volatility", 5.0)
        assert mgr.is_any_tripped() is True
        mgr.reset_all()
        assert mgr.is_any_tripped() is False
