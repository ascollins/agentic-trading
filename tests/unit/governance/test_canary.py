"""Tests for governance.canary — Infrastructure Safety Watchdog."""

import pytest

from agentic_trading.core.config import CanaryConfig
from agentic_trading.core.enums import GovernanceAction
from agentic_trading.governance.canary import GovernanceCanary


@pytest.fixture
def canary():
    cfg = CanaryConfig(failure_threshold=2, action_on_failure="kill")
    return GovernanceCanary(cfg)


class TestCanaryRegistration:
    """Component registration."""

    def test_register_component(self, canary):
        canary.register_component("redis", lambda: True)
        assert "redis" in canary.registered_components

    def test_multiple_components(self, canary):
        canary.register_component("redis", lambda: True)
        canary.register_component("event_bus", lambda: True)
        assert len(canary.registered_components) == 2


class TestCanaryChecks:
    """Health check execution."""

    @pytest.mark.asyncio
    async def test_all_healthy(self, canary):
        canary.register_component("redis", lambda: True)
        canary.register_component("bus", lambda: True)
        result = await canary.run_checks()
        assert result.all_healthy is True
        assert result.components_checked == 2
        assert result.failed_components == []

    @pytest.mark.asyncio
    async def test_single_failure(self, canary):
        canary.register_component("redis", lambda: False)
        canary.register_component("bus", lambda: True)
        result = await canary.run_checks()
        assert result.all_healthy is False
        assert "redis" in result.failed_components

    @pytest.mark.asyncio
    async def test_exception_treated_as_failure(self, canary):
        def bad_check():
            raise RuntimeError("connection refused")

        canary.register_component("redis", bad_check)
        result = await canary.run_checks()
        assert result.all_healthy is False
        assert "redis" in result.failed_components

    @pytest.mark.asyncio
    async def test_consecutive_failures_tracked(self, canary):
        canary.register_component("redis", lambda: False)
        await canary.run_checks()
        assert canary.get_failure_count("redis") == 1
        await canary.run_checks()
        assert canary.get_failure_count("redis") == 2

    @pytest.mark.asyncio
    async def test_recovery_resets_counter(self, canary):
        call_count = [0]

        def flaky():
            call_count[0] += 1
            return call_count[0] > 1  # Fails first, succeeds after

        canary.register_component("redis", flaky)
        await canary.run_checks()
        assert canary.get_failure_count("redis") == 1
        await canary.run_checks()
        assert canary.get_failure_count("redis") == 0


class TestCanaryKillSwitch:
    """Kill switch integration."""

    @pytest.mark.asyncio
    async def test_kill_switch_activated_on_threshold(self):
        kill_switch_calls = []

        def mock_kill(reason="", triggered_by=""):
            kill_switch_calls.append((reason, triggered_by))

        cfg = CanaryConfig(failure_threshold=2, action_on_failure="kill")
        canary = GovernanceCanary(cfg, kill_switch_fn=mock_kill)
        canary.register_component("redis", lambda: False)

        await canary.run_checks()  # 1st failure
        assert len(kill_switch_calls) == 0

        await canary.run_checks()  # 2nd failure → threshold
        assert len(kill_switch_calls) == 1
        assert "redis" in kill_switch_calls[0][0]

    @pytest.mark.asyncio
    async def test_alert_only_mode(self):
        """action_on_failure='alert' should not trigger kill switch."""
        kill_switch_calls = []

        def mock_kill(reason="", triggered_by=""):
            kill_switch_calls.append(reason)

        cfg = CanaryConfig(failure_threshold=1, action_on_failure="alert")
        canary = GovernanceCanary(cfg, kill_switch_fn=mock_kill)
        canary.register_component("redis", lambda: False)

        await canary.run_checks()
        assert len(kill_switch_calls) == 0

    @pytest.mark.asyncio
    async def test_no_components_is_healthy(self, canary):
        result = await canary.run_checks()
        assert result.all_healthy is True
        assert result.components_checked == 0
