"""Scoped execution tokens.

Replaces the simple dedupe_key with structured, time-limited,
revocable trade authorisation tokens.

Inspired by Soteria's Scoped Execution Tokens / Policy Decision
Service (C5): each action requires an explicit token that encodes
scope, TTL, and audit linkage.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from agentic_trading.core.config import ExecutionTokenConfig
from agentic_trading.core.ids import utc_now as _utcnow

logger = logging.getLogger(__name__)


@dataclass
class ExecutionToken:
    """A scoped, time-limited trade authorisation token."""

    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: str = ""
    scope: str = ""  # e.g. "order:BTC/USDT" or "strategy:trend_following"
    issued_at: datetime = field(default_factory=_utcnow)
    ttl_seconds: int = 300
    used: bool = False
    revoked: bool = False
    trace_id: str = ""

    @property
    def expires_at(self) -> datetime:
        """When this token expires."""
        from datetime import timedelta

        return self.issued_at + timedelta(seconds=self.ttl_seconds)

    @property
    def is_expired(self) -> bool:
        """Whether the token has expired."""
        return _utcnow() > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Whether the token can be consumed."""
        return not self.used and not self.revoked and not self.is_expired


class TokenManager:
    """Manages execution token lifecycle.

    Usage::

        mgr = TokenManager(config)
        token = mgr.issue("trend_following", scope="order:BTC/USDT")
        if mgr.validate(token.token_id):
            mgr.consume(token.token_id)
    """

    def __init__(self, config: ExecutionTokenConfig) -> None:
        self._config = config
        self._tokens: dict[str, ExecutionToken] = {}

    # ------------------------------------------------------------------
    # Issuance
    # ------------------------------------------------------------------

    def issue(
        self,
        strategy_id: str,
        scope: str = "",
        ttl_seconds: int | None = None,
        trace_id: str = "",
    ) -> ExecutionToken:
        """Issue a new execution token.

        Args:
            strategy_id: Strategy requesting the token.
            scope: Token scope (e.g. ``"order:BTC/USDT"``).
            ttl_seconds: Override TTL (defaults to config).
            trace_id: Optional trace_id for audit linkage.

        Returns:
            New :class:`ExecutionToken`.

        Raises:
            ValueError: If max active tokens exceeded.
        """
        # Check active token limit
        active = self.active_count(strategy_id)
        if active >= self._config.max_active_tokens:
            raise ValueError(
                f"Max active tokens ({self._config.max_active_tokens}) "
                f"exceeded for {strategy_id}"
            )

        token = ExecutionToken(
            strategy_id=strategy_id,
            scope=scope,
            ttl_seconds=ttl_seconds or self._config.default_ttl_seconds,
            trace_id=trace_id,
        )
        self._tokens[token.token_id] = token
        logger.debug(
            "Token issued: %s strategy=%s scope=%s ttl=%ds",
            token.token_id[:8],
            strategy_id,
            scope,
            token.ttl_seconds,
        )
        return token

    # ------------------------------------------------------------------
    # Validation & consumption
    # ------------------------------------------------------------------

    def validate(self, token_id: str) -> bool:
        """Check if a token is valid (not expired, not used, not revoked)."""
        token = self._tokens.get(token_id)
        if token is None:
            return False
        return token.is_valid

    def consume(self, token_id: str) -> bool:
        """Mark a token as used. Returns False if invalid.

        A consumed token cannot be used again.
        """
        token = self._tokens.get(token_id)
        if token is None or not token.is_valid:
            return False
        token.used = True
        logger.debug("Token consumed: %s", token_id[:8])
        return True

    def get_token(self, token_id: str) -> ExecutionToken | None:
        """Retrieve a token by ID."""
        return self._tokens.get(token_id)

    # ------------------------------------------------------------------
    # Revocation
    # ------------------------------------------------------------------

    def revoke(self, token_id: str) -> bool:
        """Revoke a specific token. Returns False if not found."""
        token = self._tokens.get(token_id)
        if token is None:
            return False
        token.revoked = True
        logger.info("Token revoked: %s", token_id[:8])
        return True

    def revoke_all(self, strategy_id: str) -> int:
        """Revoke all active tokens for a strategy. Returns count."""
        count = 0
        for token in self._tokens.values():
            if (
                token.strategy_id == strategy_id
                and not token.revoked
                and not token.used
            ):
                token.revoked = True
                count += 1
        if count > 0:
            logger.info(
                "Revoked %d tokens for strategy %s", count, strategy_id
            )
        return count

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """Remove expired and consumed tokens. Returns count removed."""
        to_remove = [
            tid
            for tid, token in self._tokens.items()
            if token.is_expired or token.used
        ]
        for tid in to_remove:
            del self._tokens[tid]
        if to_remove:
            logger.debug("Cleaned up %d tokens", len(to_remove))
        return len(to_remove)

    def active_count(self, strategy_id: str | None = None) -> int:
        """Count active (valid, non-consumed, non-revoked) tokens."""
        return sum(
            1
            for token in self._tokens.values()
            if token.is_valid
            and (strategy_id is None or token.strategy_id == strategy_id)
        )

    def total_count(self) -> int:
        """Total tokens in the store (including expired)."""
        return len(self._tokens)
