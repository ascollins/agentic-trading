"""Tavus.io avatar adapter — isolated integration for avatar video narration.

Provides:
  - ``TavusAdapter`` — abstract protocol
  - ``TavusAdapterHttp`` — real Tavus API client (env-var configured)
  - ``MockTavusAdapter`` — dev/test mock that returns placeholder URLs

All secrets come from environment variables:
  - ``TAVUS_API_KEY``
  - ``TAVUS_REPLICA_ID``
  - ``TAVUS_ENDPOINT`` (optional, defaults to Tavus API)
"""

from __future__ import annotations

import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# Default Tavus API endpoint
_DEFAULT_TAVUS_ENDPOINT = "https://tavusapi.com/v2"


@dataclass
class TavusSession:
    """Result of creating a Tavus conversation/video session."""
    session_id: str = ""
    conversation_id: str = ""
    playback_url: str = ""
    status: str = ""  # "created" / "ready" / "error"
    script_text: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------

class TavusAdapter(ABC):
    """Protocol for Tavus avatar integration."""

    @abstractmethod
    async def create_briefing(
        self, script_text: str, *, context: dict[str, Any] | None = None
    ) -> TavusSession:
        """Create a new avatar briefing session from the given script."""
        ...

    @abstractmethod
    async def get_session_status(self, session_id: str) -> TavusSession:
        """Check the status of an existing session."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...


# ---------------------------------------------------------------------------
# Real HTTP adapter
# ---------------------------------------------------------------------------

class TavusAdapterHttp(TavusAdapter):
    """Real Tavus API client using environment variables for secrets.

    Environment variables:
      - ``TAVUS_API_KEY``: API key for authentication
      - ``TAVUS_REPLICA_ID``: Avatar replica to use
      - ``TAVUS_ENDPOINT``: API base URL (optional)
    """

    def __init__(
        self,
        api_key: str | None = None,
        replica_id: str | None = None,
        endpoint: str | None = None,
        timeout_seconds: int = 30,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key or os.environ.get("TAVUS_API_KEY", "")
        self._replica_id = replica_id or os.environ.get("TAVUS_REPLICA_ID", "")
        self._endpoint = (
            endpoint
            or os.environ.get("TAVUS_ENDPOINT", _DEFAULT_TAVUS_ENDPOINT)
        ).rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._max_retries = max_retries
        self._session: aiohttp.ClientSession | None = None

        if not self._api_key:
            logger.warning("TAVUS_API_KEY not set — Tavus calls will fail.")
        if not self._replica_id:
            logger.warning("TAVUS_REPLICA_ID not set — Tavus calls will fail.")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                headers={
                    "x-api-key": self._api_key,
                    "Content-Type": "application/json",
                },
            )
        return self._session

    async def create_briefing(
        self, script_text: str, *, context: dict[str, Any] | None = None
    ) -> TavusSession:
        """Create a Tavus conversation session with the avatar reading the script."""
        session = await self._get_session()

        payload = {
            "replica_id": self._replica_id,
            "conversational_context": script_text,
            "custom_greeting": script_text,
            "properties": {
                "max_call_duration": 120,
                "enable_recording": True,
                "language": "english",
            },
        }
        if context:
            payload["properties"]["metadata"] = context

        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                async with session.post(
                    f"{self._endpoint}/conversations",
                    json=payload,
                ) as resp:
                    data = await resp.json()

                    if resp.status >= 400:
                        logger.warning(
                            "Tavus API error (attempt %d): %s %s",
                            attempt, resp.status, data,
                        )
                        last_error = data
                        continue

                    conversation_id = data.get("conversation_id", "")
                    conversation_url = data.get("conversation_url", "")

                    logger.info(
                        "Tavus session created: id=%s url=%s",
                        conversation_id, conversation_url,
                    )
                    return TavusSession(
                        session_id=conversation_id,
                        conversation_id=conversation_id,
                        playback_url=conversation_url,
                        status="created",
                        script_text=script_text,
                        raw_response=data,
                    )
            except Exception as exc:
                logger.warning(
                    "Tavus API request failed (attempt %d): %s", attempt, exc
                )
                last_error = str(exc)

        return TavusSession(
            status="error",
            script_text=script_text,
            raw_response={"error": str(last_error)},
        )

    async def get_session_status(self, session_id: str) -> TavusSession:
        """Check Tavus conversation status."""
        session = await self._get_session()

        try:
            async with session.get(
                f"{self._endpoint}/conversations/{session_id}",
            ) as resp:
                data = await resp.json()
                return TavusSession(
                    session_id=session_id,
                    conversation_id=data.get("conversation_id", session_id),
                    playback_url=data.get("conversation_url", ""),
                    status=data.get("status", "unknown"),
                    raw_response=data,
                )
        except Exception as exc:
            logger.warning("Tavus status check failed: %s", exc)
            return TavusSession(
                session_id=session_id,
                status="error",
                raw_response={"error": str(exc)},
            )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("Tavus HTTP adapter closed")


# ---------------------------------------------------------------------------
# Mock adapter for dev/testing
# ---------------------------------------------------------------------------

class MockTavusAdapter(TavusAdapter):
    """Mock Tavus adapter for development and testing.

    Returns placeholder URLs and tracks created sessions.
    """

    def __init__(self, base_url: str = "http://localhost:8099") -> None:
        self._base_url = base_url.rstrip("/")
        self.sessions: dict[str, TavusSession] = {}

    async def create_briefing(
        self, script_text: str, *, context: dict[str, Any] | None = None
    ) -> TavusSession:
        session_id = f"mock-{uuid.uuid4().hex[:8]}"
        playback_url = f"{self._base_url}/mock-avatar/{session_id}"

        result = TavusSession(
            session_id=session_id,
            conversation_id=session_id,
            playback_url=playback_url,
            status="created",
            script_text=script_text,
        )
        self.sessions[session_id] = result

        logger.info("Mock Tavus session created: id=%s", session_id)
        return result

    async def get_session_status(self, session_id: str) -> TavusSession:
        if session_id in self.sessions:
            return self.sessions[session_id]
        return TavusSession(session_id=session_id, status="not_found")

    async def close(self) -> None:
        self.sessions.clear()
