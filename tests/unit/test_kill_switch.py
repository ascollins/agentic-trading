"""Test KillSwitch activate/deactivate in memory mode (no Redis)."""

import pytest

from agentic_trading.risk.kill_switch import KillSwitch


class TestKillSwitchMemoryMode:
    async def test_initially_inactive(self):
        ks = KillSwitch()  # No Redis -> memory mode
        assert await ks.is_active() is False

    async def test_activate(self):
        ks = KillSwitch()
        event = await ks.activate(reason="drawdown limit", triggered_by="risk_engine")
        assert await ks.is_active() is True
        assert event.activated is True
        assert event.reason == "drawdown limit"
        assert event.triggered_by == "risk_engine"

    async def test_deactivate(self):
        ks = KillSwitch()
        await ks.activate(reason="test", triggered_by="test")
        assert await ks.is_active() is True

        event = await ks.deactivate()
        assert await ks.is_active() is False
        assert event.activated is False

    async def test_get_status(self):
        ks = KillSwitch()
        status = await ks.get_status()
        assert status["active"] is False
        assert status["backend"] == "memory"

    async def test_get_status_after_activate(self):
        ks = KillSwitch()
        await ks.activate(reason="big loss", triggered_by="risk_engine")
        status = await ks.get_status()
        assert status["active"] is True
        assert status["reason"] == "big loss"
        assert status["triggered_by"] == "risk_engine"
        assert status["activated_at"] > 0

    async def test_multiple_activations(self):
        ks = KillSwitch()
        await ks.activate(reason="reason1", triggered_by="a")
        await ks.activate(reason="reason2", triggered_by="b")
        status = await ks.get_status()
        assert status["active"] is True
        assert status["reason"] == "reason2"

    async def test_close_is_safe(self):
        ks = KillSwitch()
        await ks.close()  # Should not raise even with no Redis

    async def test_deactivate_reason_includes_previous(self):
        ks = KillSwitch()
        await ks.activate(reason="max drawdown", triggered_by="monitor")
        event = await ks.deactivate()
        assert "max drawdown" in event.reason
