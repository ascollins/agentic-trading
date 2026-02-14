"""Circuit breaker system.

Provides a :class:`CircuitBreakerManager` that owns a set of
:class:`CircuitBreaker` instances, each monitoring a different market
condition (volatility, spread, liquidity, data staleness, API errors).

Breakers trip after *N* consecutive violations within a time window
(hysteresis) and reset automatically after a cooldown period with no
new violations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agentic_trading.core.enums import CircuitBreakerType
from agentic_trading.core.events import CircuitBreakerEvent

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Single breaker
# ----------------------------------------------------------------------

@dataclass
class CircuitBreaker:
    """State machine for a single circuit breaker.

    Args:
        breaker_type: The category this breaker monitors.
        threshold: The value above which a violation is counted.
        window_seconds: Sliding window for counting violations.
        cooldown_seconds: How long the breaker stays tripped before
            auto-resetting (provided no new violations occur).
        hysteresis: Number of violations within the window required
            before the breaker actually trips.
        symbol: Optional symbol scope (empty string = global).
    """

    breaker_type: CircuitBreakerType
    threshold: float
    window_seconds: float = 300.0
    cooldown_seconds: float = 300.0
    hysteresis: int = 3
    symbol: str = ""

    # ---- internal mutable state ----
    tripped: bool = field(init=False, default=False)
    last_trip_time: float = field(init=False, default=0.0)
    trip_count: int = field(init=False, default=0)
    _violations: list[float] = field(init=False, default_factory=list)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def check(self, current_value: float) -> bool:
        """Evaluate the breaker against a new observation.

        Args:
            current_value: The measured value (e.g. ATR ratio, spread
                in bps, seconds since last data, error count).

        Returns:
            ``True`` if the breaker **is tripped** (trading should be
            paused), ``False`` if clear.
        """
        now = time.monotonic()

        # If already tripped, check cooldown
        if self.tripped:
            elapsed = now - self.last_trip_time
            if elapsed >= self.cooldown_seconds and current_value <= self.threshold:
                self._reset(now)
                return False
            # Still tripped (cooldown not elapsed or value still above threshold)
            return True

        # Record violation if above threshold
        if current_value > self.threshold:
            self._violations.append(now)

        # Prune violations outside the window
        cutoff = now - self.window_seconds
        self._violations = [t for t in self._violations if t >= cutoff]

        # Trip if hysteresis count reached
        if len(self._violations) >= self.hysteresis:
            self._trip(now, current_value)
            return True

        return False

    def force_trip(self, reason: str = "manual") -> None:
        """Manually trip the breaker (e.g. from CLI or alert engine)."""
        now = time.monotonic()
        self._trip(now, self.threshold)
        logger.warning(
            "Circuit breaker %s FORCE-TRIPPED: %s",
            self.breaker_type.value,
            reason,
        )

    def force_reset(self) -> None:
        """Manually reset the breaker."""
        self._reset(time.monotonic())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trip(self, now: float, current_value: float) -> None:
        self.tripped = True
        self.last_trip_time = now
        self.trip_count += 1
        self._violations.clear()
        logger.warning(
            "Circuit breaker TRIPPED: type=%s, symbol=%s, value=%.4f, "
            "threshold=%.4f, trip_count=%d",
            self.breaker_type.value,
            self.symbol or "*",
            current_value,
            self.threshold,
            self.trip_count,
        )

    def _reset(self, now: float) -> None:
        self.tripped = False
        self._violations.clear()
        logger.info(
            "Circuit breaker RESET: type=%s, symbol=%s, trip_count=%d",
            self.breaker_type.value,
            self.symbol or "*",
            self.trip_count,
        )


# ----------------------------------------------------------------------
# Manager
# ----------------------------------------------------------------------

class CircuitBreakerManager:
    """Manages a collection of :class:`CircuitBreaker` instances and
    produces :class:`CircuitBreakerEvent` objects on state transitions.

    Usage::

        mgr = CircuitBreakerManager()
        mgr.add_breaker(CircuitBreaker(
            breaker_type=CircuitBreakerType.VOLATILITY,
            threshold=3.0,
            window_seconds=300,
            cooldown_seconds=600,
            hysteresis=3,
        ))

        events = mgr.evaluate("volatility", current_atr_ratio, symbol="BTC/USDT")
    """

    # Map user-facing names to enum values for convenience
    _TYPE_ALIASES: dict[str, CircuitBreakerType] = {
        "volatility": CircuitBreakerType.VOLATILITY,
        "spread": CircuitBreakerType.SPREAD,
        "liquidity": CircuitBreakerType.LIQUIDITY,
        "staleness": CircuitBreakerType.STALENESS,
        "error_rate": CircuitBreakerType.ERROR_RATE,
    }

    def __init__(self) -> None:
        # Keyed by (breaker_type, symbol)
        self._breakers: dict[tuple[CircuitBreakerType, str], CircuitBreaker] = {}
        # Track previous tripped state for edge detection
        self._prev_state: dict[tuple[CircuitBreakerType, str], bool] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def add_breaker(self, breaker: CircuitBreaker) -> None:
        """Register a circuit breaker.

        Args:
            breaker: A configured :class:`CircuitBreaker` instance.
        """
        key = (breaker.breaker_type, breaker.symbol)
        self._breakers[key] = breaker
        self._prev_state[key] = breaker.tripped
        logger.info(
            "Registered circuit breaker: type=%s symbol=%s threshold=%.4f "
            "window=%.0fs cooldown=%.0fs hysteresis=%d",
            breaker.breaker_type.value,
            breaker.symbol or "*",
            breaker.threshold,
            breaker.window_seconds,
            breaker.cooldown_seconds,
            breaker.hysteresis,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        breaker_type: str | CircuitBreakerType,
        current_value: float,
        symbol: str = "",
    ) -> list[CircuitBreakerEvent]:
        """Evaluate a single breaker and return events on state transitions.

        Args:
            breaker_type: The breaker category (string alias or enum).
            current_value: The measured value.
            symbol: Scope symbol (empty = global).

        Returns:
            List of :class:`CircuitBreakerEvent`.  Empty if no state change.
        """
        if isinstance(breaker_type, str):
            bt = self._TYPE_ALIASES.get(breaker_type.lower())
            if bt is None:
                logger.error("Unknown breaker type alias: %s", breaker_type)
                return []
        else:
            bt = breaker_type

        key = (bt, symbol)
        breaker = self._breakers.get(key)
        if breaker is None:
            logger.debug("No breaker registered for %s/%s", bt.value, symbol or "*")
            return []

        was_tripped = self._prev_state.get(key, False)
        is_tripped = breaker.check(current_value)
        self._prev_state[key] = is_tripped

        events: list[CircuitBreakerEvent] = []

        # Emit event only on state transitions
        if is_tripped and not was_tripped:
            events.append(CircuitBreakerEvent(
                breaker_type=bt,
                tripped=True,
                symbol=symbol,
                reason=(
                    f"{bt.value} breaker tripped: value={current_value:.4f} "
                    f"threshold={breaker.threshold:.4f}"
                ),
                threshold=breaker.threshold,
                current_value=current_value,
            ))
        elif not is_tripped and was_tripped:
            events.append(CircuitBreakerEvent(
                breaker_type=bt,
                tripped=False,
                symbol=symbol,
                reason=f"{bt.value} breaker reset after cooldown",
                threshold=breaker.threshold,
                current_value=current_value,
            ))

        return events

    def evaluate_all(
        self,
        values: dict[str, float],
        symbol: str = "",
    ) -> list[CircuitBreakerEvent]:
        """Evaluate multiple breaker types at once.

        Args:
            values: Mapping of breaker type name -> measured value.
            symbol: Scope symbol.

        Returns:
            Combined list of events from all evaluations.
        """
        events: list[CircuitBreakerEvent] = []
        for breaker_name, value in values.items():
            events.extend(self.evaluate(breaker_name, value, symbol=symbol))
        return events

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def is_any_tripped(self, symbol: str = "") -> bool:
        """Return ``True`` if any breaker for the given symbol is currently tripped.

        If *symbol* is empty, checks all breakers regardless of scope.
        """
        for key, breaker in self._breakers.items():
            if symbol and key[1] != symbol:
                continue
            if breaker.tripped:
                return True
        return False

    def get_tripped_breakers(self) -> list[CircuitBreaker]:
        """Return all currently tripped breakers."""
        return [b for b in self._breakers.values() if b.tripped]

    def get_breaker(
        self,
        breaker_type: CircuitBreakerType,
        symbol: str = "",
    ) -> CircuitBreaker | None:
        """Look up a specific breaker by type and symbol."""
        return self._breakers.get((breaker_type, symbol))

    def reset_all(self) -> None:
        """Force-reset every breaker."""
        for breaker in self._breakers.values():
            breaker.force_reset()
        self._prev_state = {k: False for k in self._breakers}
        logger.info("All circuit breakers force-reset")
