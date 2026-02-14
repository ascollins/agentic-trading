"""Narration HTTP server — serves both Channel 1 (text stream) and Channel 2 (avatar).

Endpoints:
  GET  /narration/latest?limit=50  — JSON list of recent narrations (Grafana datasource)
  GET  /narration/health            — Health check
  POST /avatar/briefing             — Generate & trigger avatar for latest narration
  GET  /avatar/watch                — HTML page with avatar player + latest script

Runs as a lightweight aiohttp server alongside the main trading loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from aiohttp import web

from .presenter import BloombergPresenter
from .schema import NarrationItem
from .service import NarrationService
from .store import NarrationStore
from .tavus import TavusAdapter

logger = logging.getLogger(__name__)

_DISCLAIMER = (
    "This is an automated system behavior description. "
    "It is NOT financial advice. Trade at your own risk."
)


def create_narration_app(
    store: NarrationStore,
    tavus: TavusAdapter,
    service: NarrationService | None = None,
) -> web.Application:
    """Create the aiohttp web application for narration endpoints."""
    app = web.Application()
    app["store"] = store
    app["tavus"] = tavus
    app["service"] = service

    app.router.add_get("/narration/latest", handle_narration_latest)
    app.router.add_get("/narration/health", handle_health)
    app.router.add_post("/avatar/briefing", handle_avatar_briefing)
    app.router.add_get("/avatar/watch", handle_avatar_watch)

    # CORS headers for Grafana Infinity plugin
    app.middlewares.append(cors_middleware)

    return app


@web.middleware
async def cors_middleware(request: web.Request, handler):
    """Add CORS headers so Grafana can query the endpoint."""
    response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# ---------------------------------------------------------------------------
# Channel 1: Text Stream
# ---------------------------------------------------------------------------

async def handle_narration_latest(request: web.Request) -> web.Response:
    """GET /narration/latest — return recent narration items as JSON.

    Query params:
      limit (int): Max items to return (default 50)
    """
    store: NarrationStore = request.app["store"]
    limit = int(request.query.get("limit", "50"))
    items = store.to_json_list(limit=limit)
    return web.json_response(items)


async def handle_health(request: web.Request) -> web.Response:
    """GET /narration/health — simple health check."""
    store: NarrationStore = request.app["store"]
    return web.json_response({
        "status": "ok",
        "narration_count": store.count,
    })


# ---------------------------------------------------------------------------
# Channel 2: Avatar
# ---------------------------------------------------------------------------

async def handle_avatar_briefing(request: web.Request) -> web.Response:
    """POST /avatar/briefing — generate avatar session from latest narration.

    Returns JSON with playback_url, script_text, session_id.
    When a NarrationService with a Bloomberg Presenter is available, the
    Tavus conversational context includes the full presenter persona prompt.
    """
    store: NarrationStore = request.app["store"]
    tavus: TavusAdapter = request.app["tavus"]
    service: NarrationService | None = request.app.get("service")

    latest = store.latest_one()
    if latest is None:
        return web.json_response(
            {"error": "No narration available yet"}, status=404
        )

    # Build context — include presenter persona when available
    context: dict[str, str] = {
        "script_id": latest.script_id,
        "timestamp": latest.timestamp.isoformat(),
        "decision_ref": latest.decision_ref,
    }
    if service is not None:
        context["presenter_persona"] = service.presenter.system_prompt

    # Create Tavus session
    session = await tavus.create_briefing(
        latest.script_text,
        context=context,
    )

    # Update store with playback info
    latest.playback_url = session.playback_url
    latest.tavus_session_id = session.session_id
    latest.published_avatar = True

    return web.json_response({
        "script_id": latest.script_id,
        "script_text": latest.script_text,
        "playback_url": session.playback_url,
        "session_id": session.session_id,
        "status": session.status,
    })


async def handle_avatar_watch(request: web.Request) -> web.Response:
    """GET /avatar/watch — HTML page for avatar playback."""
    store: NarrationStore = request.app["store"]
    latest = store.latest_one()

    script_text = latest.script_text if latest else "No narration available yet."
    playback_url = latest.playback_url if latest else ""
    session_id = latest.tavus_session_id if latest else ""
    timestamp = latest.timestamp.isoformat() if latest else ""

    html = _avatar_watch_html(
        script_text=script_text,
        playback_url=playback_url,
        session_id=session_id,
        timestamp=timestamp,
    )
    return web.Response(text=html, content_type="text/html")


def _avatar_watch_html(
    script_text: str,
    playback_url: str,
    session_id: str,
    timestamp: str,
) -> str:
    """Generate the avatar watch page HTML."""
    escaped_script = script_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    escaped_url = playback_url.replace('"', "&quot;")

    player_section = ""
    if playback_url:
        player_section = f"""
        <div class="player-section">
            <a href="{escaped_url}" target="_blank" class="play-button">
                &#9654; Watch Avatar Briefing
            </a>
            <p class="session-id">Session: {session_id}</p>
            <iframe src="{escaped_url}" width="100%" height="400"
                    style="border:1px solid #333; border-radius:8px; margin-top:12px;"
                    allow="camera; microphone; autoplay" allowfullscreen></iframe>
        </div>
        """
    else:
        player_section = """
        <div class="player-section">
            <button class="play-button" id="generateBtn" onclick="generateBriefing()">
                &#9654; Generate Avatar Briefing
            </button>
            <p class="hint">Click to create a new avatar video from the latest narration.</p>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Avatar Narration - Trading Platform</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e; color: #e0e0e0;
            padding: 24px; max-width: 800px; margin: 0 auto;
        }}
        h1 {{ color: #00d4aa; font-size: 1.5rem; margin-bottom: 4px; }}
        .subtitle {{ color: #888; font-size: 0.85rem; margin-bottom: 20px; }}
        .card {{
            background: #16213e; border: 1px solid #333;
            border-radius: 12px; padding: 20px; margin-bottom: 16px;
        }}
        .script-text {{
            font-size: 1.1rem; line-height: 1.6; color: #fff;
            white-space: pre-wrap;
        }}
        .timestamp {{ color: #666; font-size: 0.8rem; margin-top: 8px; }}
        .play-button {{
            display: inline-block; background: #00d4aa; color: #1a1a2e;
            padding: 12px 28px; border-radius: 8px; text-decoration: none;
            font-weight: 600; font-size: 1rem; border: none; cursor: pointer;
            transition: background 0.2s;
        }}
        .play-button:hover {{ background: #00f0c0; }}
        .player-section {{ margin-top: 16px; text-align: center; }}
        .session-id {{ color: #666; font-size: 0.75rem; margin-top: 6px; }}
        .hint {{ color: #888; font-size: 0.8rem; margin-top: 8px; }}
        .disclaimer {{
            background: #2a1a1a; border: 1px solid #553333;
            border-radius: 8px; padding: 12px; margin-top: 20px;
            font-size: 0.75rem; color: #cc8888;
        }}
        .loading {{ display: none; color: #00d4aa; margin-top: 8px; }}
    </style>
</head>
<body>
    <h1>Avatar Narration</h1>
    <p class="subtitle">Live trading platform commentary</p>

    <div class="card">
        <h3 style="color:#00d4aa;margin-bottom:8px;">Latest Narration</h3>
        <p class="script-text">{escaped_script}</p>
        <p class="timestamp">{timestamp}</p>
    </div>

    {player_section}

    <p class="loading" id="loadingMsg">Generating avatar briefing...</p>

    <div class="disclaimer">
        {_DISCLAIMER}
    </div>

    <script>
    async function generateBriefing() {{
        const btn = document.getElementById('generateBtn');
        const loading = document.getElementById('loadingMsg');
        btn.disabled = true;
        btn.textContent = 'Generating...';
        loading.style.display = 'block';

        try {{
            const resp = await fetch('/avatar/briefing', {{ method: 'POST' }});
            const data = await resp.json();
            if (data.playback_url) {{
                window.location.reload();
            }} else {{
                btn.textContent = 'Retry';
                btn.disabled = false;
                loading.textContent = data.error || 'Failed to generate. Try again.';
            }}
        }} catch (e) {{
            btn.textContent = 'Retry';
            btn.disabled = false;
            loading.textContent = 'Network error. Try again.';
        }}
    }}
    </script>
</body>
</html>"""


async def start_narration_server(
    store: NarrationStore,
    tavus: TavusAdapter,
    host: str = "0.0.0.0",
    port: int = 8099,
    service: NarrationService | None = None,
) -> web.AppRunner:
    """Start the narration HTTP server.

    Returns the runner for lifecycle management (call runner.cleanup() to stop).
    """
    app = create_narration_app(store, tavus, service=service)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info("Narration server started on http://%s:%d", host, port)
    return runner
