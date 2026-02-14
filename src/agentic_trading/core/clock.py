"""Clock abstraction for mode-agnostic time.

WallClock: real wall-clock time (paper/live)
SimClock: deterministic simulated time (backtest)

Strategies never call datetime.now() directly — they use ctx.clock.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol


class IClock(Protocol):
    """Clock interface used by all time-dependent code."""

    def now(self) -> datetime:
        """Current time as timezone-aware UTC datetime."""
        ...

    def now_ms(self) -> int:
        """Current time as milliseconds since epoch."""
        ...


class WallClock:
    """Real wall-clock time. Used in paper and live modes."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def now_ms(self) -> int:
        return int(self.now().timestamp() * 1000)


class SimClock:
    """Simulated clock for deterministic backtesting.

    Time advances only when explicitly set by the backtest engine.
    """

    def __init__(self, start: datetime | None = None) -> None:
        self._time = start or datetime(2024, 1, 1, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self._time

    def now_ms(self) -> int:
        return int(self._time.timestamp() * 1000)

    def set_time(self, t: datetime) -> None:
        """Advance time. Must be monotonically increasing."""
        if t < self._time:
            raise ValueError(
                f"SimClock cannot go backwards: {t} < {self._time}"
            )
        self._time = t

    def advance_ms(self, ms: int) -> None:
        """Advance time by milliseconds."""
        from datetime import timedelta

        self.set_time(self._time + timedelta(milliseconds=ms))
