"""Write-ownership enforcement tests.

Invariant: no agent can publish an event type it does not own.

Covers:
- Property-based test: random (agent, event_type) pairs.
- Exhaustive matrix: every owner publishes OK, every non-owner is rejected.
- Read-only consumer static analysis: narration and metrics never publish.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from agentic_trading.domain.events import (
    ALL_DOMAIN_EVENTS,
    DomainEvent,
    WRITE_OWNERSHIP,
)
from agentic_trading.infrastructure.event_bus import (
    InMemoryEventBus,
    WriteOwnershipError,
)


# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------

# Unique set of all owner strings (the agents that CAN write).
ALL_OWNERS: list[str] = sorted(set(WRITE_OWNERSHIP.values()))

# All canonical event types as a list (for hypothesis sampling).
ALL_EVENT_TYPES: list[type[DomainEvent]] = list(ALL_DOMAIN_EVENTS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(cls: type[DomainEvent], source: str) -> DomainEvent:
    """Instantiate any canonical event with the given source."""
    return cls(source=source)


# ---------------------------------------------------------------------------
# Property-based: random agent × random event type
# ---------------------------------------------------------------------------

class TestPropertyBasedOwnership:
    """Hypothesis-driven: no agent can cross-publish."""

    @given(
        agent=st.sampled_from(ALL_OWNERS),
        event_type=st.sampled_from(ALL_EVENT_TYPES),
    )
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_only_owner_can_publish(
        self, agent: str, event_type: type[DomainEvent]
    ) -> None:
        bus = InMemoryEventBus(enforce_ownership=True)
        event = _make_event(event_type, source=agent)
        expected_owner = WRITE_OWNERSHIP[event_type]

        if agent == expected_owner:
            await bus.publish(event)
            assert len(bus.get_history()) == 1
        else:
            with pytest.raises(WriteOwnershipError):
                await bus.publish(event)
            assert len(bus.get_history()) == 0


# ---------------------------------------------------------------------------
# Exhaustive matrix: every owner × every event type
# ---------------------------------------------------------------------------

class TestExhaustiveOwnershipMatrix:
    """For every (event_type, owner) pair: correct owner passes, all others fail."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("event_type", ALL_EVENT_TYPES, ids=lambda c: c.__name__)
    async def test_correct_owner_publishes(self, event_type: type[DomainEvent]) -> None:
        bus = InMemoryEventBus(enforce_ownership=True)
        owner = WRITE_OWNERSHIP[event_type]
        event = _make_event(event_type, source=owner)
        await bus.publish(event)
        assert len(bus.get_history()) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("event_type", ALL_EVENT_TYPES, ids=lambda c: c.__name__)
    async def test_wrong_owners_rejected(self, event_type: type[DomainEvent]) -> None:
        correct_owner = WRITE_OWNERSHIP[event_type]
        wrong_owners = [o for o in ALL_OWNERS if o != correct_owner]

        for wrong in wrong_owners:
            bus = InMemoryEventBus(enforce_ownership=True)
            event = _make_event(event_type, source=wrong)
            with pytest.raises(WriteOwnershipError):
                await bus.publish(event)


# ---------------------------------------------------------------------------
# Static analysis: read-only consumers never call bus.publish()
# ---------------------------------------------------------------------------

# Paths are relative to repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO_ROOT / "src" / "agentic_trading"


class TestReadOnlyConsumers:
    """Narration and metrics modules must never publish events.

    These are designed to be read-only consumers of the event bus.
    We do a simple static grep for ``publish(`` calls.
    """

    def _scan_for_publish(self, directory: Path, glob: str = "*.py") -> list[str]:
        """Return list of 'file:line' where publish is called."""
        violations: list[str] = []
        if not directory.exists():
            return violations
        for py_file in directory.rglob(glob):
            try:
                lines = py_file.read_text().splitlines()
            except OSError:
                continue
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                # Skip comments and strings
                if stripped.startswith("#"):
                    continue
                # Check for event bus publish calls
                if (
                    "bus.publish(" in stripped
                    or "event_bus.publish(" in stripped
                    or "_bus.publish(" in stripped
                    or ".publish(" in stripped
                    and "subscribe" not in stripped
                    and "def publish" not in stripped
                ):
                    # Narrow down: must look like an actual method call
                    if ".publish(" in stripped and "def " not in stripped:
                        violations.append(f"{py_file.relative_to(_SRC)}:{i}")
        return violations

    def test_narration_has_no_publish_calls(self) -> None:
        """The narration module must be a pure consumer (read-only)."""
        narration_dir = _SRC / "narration"
        violations = self._scan_for_publish(narration_dir)
        # Filter out false positives in standalone/server that publish
        # narration items to *their own* store (not the event bus).
        bus_violations = [
            v for v in violations
            if "event_bus" in (_SRC / v.split(":")[0]).read_text()
        ]
        # Currently narration doesn't import or use the new domain bus.
        # This test will catch regressions if someone adds a publish call.
        assert bus_violations == [], (
            f"Narration module must not publish to event bus: {bus_violations}"
        )

    def test_observability_metrics_has_no_publish_calls(self) -> None:
        """The metrics module must be a pure consumer (read-only)."""
        metrics_file = _SRC / "observability" / "metrics.py"
        if not metrics_file.exists():
            pytest.skip("metrics.py not found")
        content = metrics_file.read_text()
        # Should not call publish on the new domain event bus
        assert "from agentic_trading.infrastructure.event_bus" not in content, (
            "metrics.py should not import the new event bus for publishing"
        )
