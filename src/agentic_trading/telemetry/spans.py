"""Lightweight local span tree for trace context.

No OpenTelemetry dependency -- just a simple stack of span IDs scoped
to a single trace.  Push on entry, pop on exit.
"""

from __future__ import annotations

from agentic_trading.core.ids import new_id


class SpanContext:
    """Thread-local span tree within one asyncio task."""

    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id
        self._stack: list[str] = []

    # -- mutators -----------------------------------------------------------

    def push_span(self) -> str:
        """Create a new child span, return its span_id."""
        span_id = new_id()
        self._stack.append(span_id)
        return span_id

    def pop_span(self) -> str | None:
        """Close current span, return its span_id (or ``None`` if empty)."""
        if self._stack:
            return self._stack.pop()
        return None

    # -- read-only ----------------------------------------------------------

    @property
    def current_span_id(self) -> str:
        """Return the innermost (current) span_id, or ``""`` if empty."""
        return self._stack[-1] if self._stack else ""

    @property
    def parent_span_id(self) -> str:
        """Return the parent span_id, or ``""`` if depth < 2."""
        return self._stack[-2] if len(self._stack) >= 2 else ""

    @property
    def depth(self) -> int:
        """Number of open spans."""
        return len(self._stack)
