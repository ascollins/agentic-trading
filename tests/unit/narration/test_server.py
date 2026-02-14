"""Tests for narration HTTP server — endpoints and integration."""

from __future__ import annotations

import pytest
from aiohttp.test_utils import AioHTTPTestCase, TestClient

from agentic_trading.narration.schema import NarrationItem
from agentic_trading.narration.server import create_narration_app
from agentic_trading.narration.store import NarrationStore
from agentic_trading.narration.tavus import MockTavusAdapter


@pytest.fixture
def store_with_items() -> NarrationStore:
    store = NarrationStore()
    item1 = NarrationItem(
        script_id="item-1",
        script_text="Market is trending up. Entering BTC/USDT position.",
        verbosity="normal",
        decision_ref="trace-1",
        published_text=True,
        metadata={"action": "ENTER", "symbol": "BTC/USDT", "regime": "trend"},
    )
    item2 = NarrationItem(
        script_id="item-2",
        script_text="No trade on ETH right now. Spread is too wide.",
        verbosity="normal",
        decision_ref="trace-2",
        published_text=True,
        metadata={"action": "NO_TRADE", "symbol": "ETH/USDT", "regime": "range"},
    )
    store.add(item1)
    store.add(item2)
    return store


@pytest.fixture
def mock_tavus() -> MockTavusAdapter:
    return MockTavusAdapter(base_url="http://localhost:8099")


@pytest.fixture
def app(store_with_items, mock_tavus):
    return create_narration_app(store=store_with_items, tavus=mock_tavus)


# ===========================================================================
# GET /narration/latest
# ===========================================================================

class TestNarrationLatest:
    @pytest.mark.asyncio
    async def test_returns_items(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/narration/latest")
        assert resp.status == 200
        data = await resp.json()
        assert isinstance(data, list)
        assert len(data) == 2
        # Newest first
        assert data[0]["script_id"] == "item-2"
        assert data[1]["script_id"] == "item-1"

    @pytest.mark.asyncio
    async def test_limit_param(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/narration/latest?limit=1")
        data = await resp.json()
        assert len(data) == 1

    @pytest.mark.asyncio
    async def test_cors_headers(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/narration/latest")
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    @pytest.mark.asyncio
    async def test_empty_store(self, mock_tavus, aiohttp_client):
        empty_store = NarrationStore()
        empty_app = create_narration_app(store=empty_store, tavus=mock_tavus)
        client = await aiohttp_client(empty_app)
        resp = await client.get("/narration/latest")
        data = await resp.json()
        assert data == []


# ===========================================================================
# GET /narration/health
# ===========================================================================

class TestNarrationHealth:
    @pytest.mark.asyncio
    async def test_health_check(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/narration/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"
        assert data["narration_count"] == 2


# ===========================================================================
# POST /avatar/briefing
# ===========================================================================

class TestAvatarBriefing:
    @pytest.mark.asyncio
    async def test_create_briefing(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.post("/avatar/briefing")
        assert resp.status == 200
        data = await resp.json()
        assert "playback_url" in data
        assert "script_text" in data
        assert data["status"] == "created"
        assert data["script_text"] == "No trade on ETH right now. Spread is too wide."

    @pytest.mark.asyncio
    async def test_briefing_no_items(self, mock_tavus, aiohttp_client):
        empty_store = NarrationStore()
        empty_app = create_narration_app(store=empty_store, tavus=mock_tavus)
        client = await aiohttp_client(empty_app)
        resp = await client.post("/avatar/briefing")
        assert resp.status == 404


# ===========================================================================
# GET /avatar/watch
# ===========================================================================

class TestAvatarWatch:
    @pytest.mark.asyncio
    async def test_watch_page_returns_html(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/avatar/watch")
        assert resp.status == 200
        assert "text/html" in resp.content_type
        text = await resp.text()
        assert "Avatar Narration" in text
        assert "NOT financial advice" in text

    @pytest.mark.asyncio
    async def test_watch_page_shows_script(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.get("/avatar/watch")
        text = await resp.text()
        assert "No trade on ETH right now" in text

    @pytest.mark.asyncio
    async def test_watch_page_empty_store(self, mock_tavus, aiohttp_client):
        empty_store = NarrationStore()
        empty_app = create_narration_app(store=empty_store, tavus=mock_tavus)
        client = await aiohttp_client(empty_app)
        resp = await client.get("/avatar/watch")
        text = await resp.text()
        assert "No narration available" in text
