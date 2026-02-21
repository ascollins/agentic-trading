"""Canonical ID and timestamp factories for the platform.

All modules import from here instead of defining local _uuid()/_now() copies.

ID Categories
-------------
1. Internal IDs: UUID v4 strings (event_id, trade_id, trace_id, etc.)
2. External IDs: Exchange-assigned, opaque strings (order_id, fill_id)
3. Content-derived IDs: SHA256[:N] deterministic hashes (dedupe_key, script_id)
4. Correlation IDs: UUID v4 strings linking a decision chain (trace_id)

Timestamp Rule
--------------
All timestamps are ``datetime`` with ``tzinfo=timezone.utc`` — never naive.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any


def new_id() -> str:
    """Generate a new UUID v4 string.  Use for all internal entity IDs."""
    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def content_hash(*parts: str, length: int = 16) -> str:
    """Generate a deterministic SHA256-based ID from content strings.

    Use for deduplication keys, script IDs, and idempotency keys.
    Concatenates all *parts* with ``':'`` before hashing.

    Parameters
    ----------
    *parts:
        Strings to hash together.
    length:
        Number of hex characters to return (default 16).
    """
    raw = ":".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:length]


def payload_hash(payload: dict[str, Any], *, length: int = 16) -> str:
    """Generate a deterministic hash from a JSON-serializable dict.

    Use for audit integrity hashes (request_hash, response_hash).

    Parameters
    ----------
    payload:
        Dict to hash.  Serialized with sorted keys and ``default=str``.
    length:
        Number of hex characters to return (default 16).
    """
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:length]


def instrument_hash(instrument_dict: dict[str, Any], *, length: int = 16) -> str:
    """Hash instrument specification for version pinning in decisions.

    Excludes the ``instrument_hash`` field itself to avoid circularity.

    Parameters
    ----------
    instrument_dict:
        Instrument model dumped to dict (via ``model_dump()``).
    length:
        Number of hex characters to return (default 16).
    """
    filtered = {k: v for k, v in instrument_dict.items() if k != "instrument_hash"}
    return payload_hash(filtered, length=length)
