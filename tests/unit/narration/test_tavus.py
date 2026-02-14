"""Tests for Tavus adapter — mock and contract tests."""

from __future__ import annotations

import pytest

from agentic_trading.narration.tavus import MockTavusAdapter, TavusSession


class TestMockTavusAdapter:
    @pytest.mark.asyncio
    async def test_create_briefing(self):
        adapter = MockTavusAdapter()
        session = await adapter.create_briefing("Hello, this is a test briefing.")

        assert isinstance(session, TavusSession)
        assert session.status == "created"
        assert session.session_id.startswith("mock-")
        assert "mock-avatar" in session.playback_url
        assert session.script_text == "Hello, this is a test briefing."

    @pytest.mark.asyncio
    async def test_get_session_status(self):
        adapter = MockTavusAdapter()
        session = await adapter.create_briefing("Test script.")
        status = await adapter.get_session_status(session.session_id)

        assert status.session_id == session.session_id
        assert status.status == "created"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self):
        adapter = MockTavusAdapter()
        status = await adapter.get_session_status("nonexistent")
        assert status.status == "not_found"

    @pytest.mark.asyncio
    async def test_close(self):
        adapter = MockTavusAdapter()
        await adapter.create_briefing("Test.")
        assert len(adapter.sessions) == 1
        await adapter.close()
        assert len(adapter.sessions) == 0

    @pytest.mark.asyncio
    async def test_multiple_sessions(self):
        adapter = MockTavusAdapter()
        s1 = await adapter.create_briefing("First briefing.")
        s2 = await adapter.create_briefing("Second briefing.")
        assert s1.session_id != s2.session_id
        assert len(adapter.sessions) == 2

    @pytest.mark.asyncio
    async def test_custom_base_url(self):
        adapter = MockTavusAdapter(base_url="http://custom:9999")
        session = await adapter.create_briefing("Test.")
        assert "custom:9999" in session.playback_url
