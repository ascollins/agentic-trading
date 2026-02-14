"""Tests for governance.tokens — Scoped Execution Tokens."""

import time
from datetime import datetime, timedelta, timezone

import pytest

from agentic_trading.core.config import ExecutionTokenConfig
from agentic_trading.governance.tokens import ExecutionToken, TokenManager


class TestTokenIssuance:
    """Token creation and properties."""

    def test_issue_creates_token(self, token_manager):
        token = token_manager.issue("s1", scope="order:BTC/USDT")
        assert token.strategy_id == "s1"
        assert token.scope == "order:BTC/USDT"
        assert token.is_valid

    def test_token_has_unique_id(self, token_manager):
        t1 = token_manager.issue("s1")
        t2 = token_manager.issue("s1")
        assert t1.token_id != t2.token_id

    def test_custom_ttl(self, token_manager):
        token = token_manager.issue("s1", ttl_seconds=60)
        assert token.ttl_seconds == 60

    def test_default_ttl_from_config(self, token_manager):
        token = token_manager.issue("s1")
        assert token.ttl_seconds == token_manager._config.default_ttl_seconds

    def test_max_active_tokens_enforced(self):
        cfg = ExecutionTokenConfig(max_active_tokens=2)
        mgr = TokenManager(cfg)
        mgr.issue("s1")
        mgr.issue("s1")
        with pytest.raises(ValueError, match="Max active tokens"):
            mgr.issue("s1")

    def test_trace_id_binding(self, token_manager):
        token = token_manager.issue("s1", trace_id="trace-123")
        assert token.trace_id == "trace-123"


class TestTokenValidation:
    """Token validation and consumption."""

    def test_validate_valid_token(self, token_manager):
        token = token_manager.issue("s1")
        assert token_manager.validate(token.token_id) is True

    def test_validate_nonexistent_token(self, token_manager):
        assert token_manager.validate("nonexistent") is False

    def test_consume_valid_token(self, token_manager):
        token = token_manager.issue("s1")
        assert token_manager.consume(token.token_id) is True
        assert token.used is True

    def test_consume_twice_fails(self, token_manager):
        token = token_manager.issue("s1")
        assert token_manager.consume(token.token_id) is True
        assert token_manager.consume(token.token_id) is False

    def test_consumed_token_invalid(self, token_manager):
        token = token_manager.issue("s1")
        token_manager.consume(token.token_id)
        assert token_manager.validate(token.token_id) is False


class TestTokenRevocation:
    """Token revocation."""

    def test_revoke_single_token(self, token_manager):
        token = token_manager.issue("s1")
        assert token_manager.revoke(token.token_id) is True
        assert token.revoked is True
        assert token_manager.validate(token.token_id) is False

    def test_revoke_nonexistent(self, token_manager):
        assert token_manager.revoke("nonexistent") is False

    def test_revoke_all_for_strategy(self, token_manager):
        token_manager.issue("s1")
        token_manager.issue("s1")
        token_manager.issue("s2")
        count = token_manager.revoke_all("s1")
        assert count == 2
        assert token_manager.active_count("s1") == 0
        assert token_manager.active_count("s2") == 1


class TestTokenExpiry:
    """Token expiration."""

    def test_expired_token_is_invalid(self):
        token = ExecutionToken(
            strategy_id="s1",
            ttl_seconds=0,  # Expires immediately
            issued_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        assert token.is_expired is True
        assert token.is_valid is False

    def test_fresh_token_not_expired(self, token_manager):
        token = token_manager.issue("s1", ttl_seconds=3600)
        assert token.is_expired is False

    def test_expires_at_computed(self):
        now = datetime.now(timezone.utc)
        token = ExecutionToken(
            strategy_id="s1",
            ttl_seconds=300,
            issued_at=now,
        )
        expected = now + timedelta(seconds=300)
        assert abs((token.expires_at - expected).total_seconds()) < 1


class TestTokenHousekeeping:
    """Cleanup and counting."""

    def test_cleanup_removes_expired(self):
        cfg = ExecutionTokenConfig(default_ttl_seconds=0)
        mgr = TokenManager(cfg)
        mgr.issue("s1")  # Expires immediately (ttl=0)
        # Need to wait a moment for expiry
        import time
        time.sleep(0.01)
        removed = mgr.cleanup_expired()
        assert removed >= 1

    def test_active_count(self, token_manager):
        token_manager.issue("s1")
        token_manager.issue("s1")
        t3 = token_manager.issue("s2")
        token_manager.consume(t3.token_id)
        assert token_manager.active_count("s1") == 2
        assert token_manager.active_count("s2") == 0
        assert token_manager.active_count() == 2  # All strategies

    def test_total_count(self, token_manager):
        token_manager.issue("s1")
        token_manager.issue("s2")
        assert token_manager.total_count() == 2

    def test_get_token(self, token_manager):
        token = token_manager.issue("s1")
        retrieved = token_manager.get_token(token.token_id)
        assert retrieved is not None
        assert retrieved.strategy_id == "s1"

    def test_get_nonexistent_token(self, token_manager):
        assert token_manager.get_token("nonexistent") is None
