"""Reasoning Message Bus — inter-agent communication for Soteria.

This is **not** the trading ``EventBus`` (Redis Streams / MemoryEventBus).
The ``ReasoningMessageBus`` handles *thinking* messages between agents
during a reasoning conversation. It is synchronous by default (messages
delivered inline during the conversation loop), with an optional async
``post_async()`` for paper/live mode.

Key concepts:
- **subscribe(role, handler)** — an agent subscribes to messages
  addressed to its role (or broadcast messages).
- **post(message)** — sends a message to all matching subscribers.
- **get_conversation(id)** — retrieves the full ``AgentConversation``.
- **get_thread(conversation_id, role)** — messages for a specific agent.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from .agent_conversation import AgentConversation, ConversationOutcome
from .agent_message import AgentMessage, AgentRole, MessageType
from .soteria_trace import SoteriaTrace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

MessageHandler = Callable[[AgentMessage], None]
AsyncMessageHandler = Callable[[AgentMessage], Any]  # coroutine


# ---------------------------------------------------------------------------
# ReasoningMessageBus
# ---------------------------------------------------------------------------


class ReasoningMessageBus:
    """In-process message bus for inter-agent reasoning conversations.

    Thread-safe for single-threaded async usage (no locking needed
    because all access is within the same event loop).

    Parameters
    ----------
    max_conversations:
        Maximum conversations to retain in memory. Oldest evicted
        when exceeded.
    """

    def __init__(self, *, max_conversations: int = 1000) -> None:
        self._max_conversations = max_conversations

        # role -> list of handlers
        self._handlers: dict[AgentRole, list[MessageHandler]] = defaultdict(list)
        self._async_handlers: dict[AgentRole, list[AsyncMessageHandler]] = defaultdict(list)

        # conversation_id -> AgentConversation
        self._conversations: dict[str, AgentConversation] = {}
        self._conversation_order: list[str] = []  # for eviction

        # Stats
        self._messages_posted: int = 0
        self._messages_delivered: int = 0

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    def subscribe(
        self,
        role: AgentRole,
        handler: MessageHandler,
    ) -> None:
        """Subscribe a synchronous handler for messages to a role.

        The handler is also called for broadcast messages (no recipients).
        """
        self._handlers[role].append(handler)
        logger.debug(
            "Subscribed sync handler for role=%s (total=%d)",
            role.value,
            len(self._handlers[role]),
        )

    def subscribe_async(
        self,
        role: AgentRole,
        handler: AsyncMessageHandler,
    ) -> None:
        """Subscribe an async handler for messages to a role."""
        self._async_handlers[role].append(handler)
        logger.debug(
            "Subscribed async handler for role=%s (total=%d)",
            role.value,
            len(self._async_handlers[role]),
        )

    def unsubscribe(self, role: AgentRole, handler: MessageHandler) -> None:
        """Remove a synchronous handler."""
        handlers = self._handlers.get(role, [])
        if handler in handlers:
            handlers.remove(handler)

    def unsubscribe_async(self, role: AgentRole, handler: AsyncMessageHandler) -> None:
        """Remove an async handler."""
        handlers = self._async_handlers.get(role, [])
        if handler in handlers:
            handlers.remove(handler)

    # ------------------------------------------------------------------
    # Post (synchronous)
    # ------------------------------------------------------------------

    def post(self, message: AgentMessage) -> int:
        """Post a message synchronously. Returns number of handlers called.

        Delivers to:
        - All handlers for each recipient role
        - If broadcast (no recipients), delivers to ALL subscribed roles

        Also records the message in the conversation.
        """
        self._messages_posted += 1

        # Record in conversation
        self._record_message(message)

        delivered = 0

        if message.is_broadcast:
            # Deliver to all roles
            for role, handlers in self._handlers.items():
                if role == message.sender:
                    continue  # Don't deliver to self
                for handler in handlers:
                    try:
                        handler(message)
                        delivered += 1
                    except Exception:
                        logger.exception(
                            "Handler error for role=%s msg=%s",
                            role.value,
                            message.message_id[:8],
                        )
        else:
            # Deliver to specific recipients
            for recipient in message.recipients:
                for handler in self._handlers.get(recipient, []):
                    try:
                        handler(message)
                        delivered += 1
                    except Exception:
                        logger.exception(
                            "Handler error for role=%s msg=%s",
                            recipient.value,
                            message.message_id[:8],
                        )

        self._messages_delivered += delivered
        return delivered

    async def post_async(self, message: AgentMessage) -> int:
        """Post a message asynchronously. Returns number of handlers called.

        Same routing as ``post()`` but calls async handlers.
        Sync handlers are also called (inline).
        """
        # Call sync handlers first
        delivered = self.post(message)

        # Then async handlers
        if message.is_broadcast:
            for role, handlers in self._async_handlers.items():
                if role == message.sender:
                    continue
                for handler in handlers:
                    try:
                        await handler(message)
                        delivered += 1
                    except Exception:
                        logger.exception(
                            "Async handler error for role=%s msg=%s",
                            role.value,
                            message.message_id[:8],
                        )
        else:
            for recipient in message.recipients:
                for handler in self._async_handlers.get(recipient, []):
                    try:
                        await handler(message)
                        delivered += 1
                    except Exception:
                        logger.exception(
                            "Async handler error for role=%s msg=%s",
                            recipient.value,
                            message.message_id[:8],
                        )

        return delivered

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    def create_conversation(
        self,
        *,
        symbol: str = "",
        timeframe: str = "",
        trigger_event: str = "",
        strategy_id: str = "",
        context_snapshot: dict[str, Any] | None = None,
    ) -> AgentConversation:
        """Create and register a new conversation."""
        conv = AgentConversation(
            symbol=symbol,
            timeframe=timeframe,
            trigger_event=trigger_event,
            strategy_id=strategy_id,
            context_snapshot=context_snapshot or {},
        )
        self._register_conversation(conv)
        return conv

    def get_conversation(self, conversation_id: str) -> AgentConversation | None:
        """Retrieve a conversation by ID."""
        return self._conversations.get(conversation_id)

    def get_thread(
        self,
        conversation_id: str,
        role: AgentRole | None = None,
    ) -> list[AgentMessage]:
        """Get messages in a conversation, optionally filtered by role.

        Parameters
        ----------
        conversation_id:
            The conversation to query.
        role:
            If provided, returns messages where this role is sender
            or recipient (or broadcast).
        """
        conv = self._conversations.get(conversation_id)
        if conv is None:
            return []

        if role is None:
            return list(conv.messages)

        return [
            m for m in conv.messages
            if m.sender == role
            or role in m.recipients
            or m.is_broadcast
        ]

    def finalize_conversation(
        self,
        conversation_id: str,
        outcome: ConversationOutcome,
        details: dict[str, Any] | None = None,
    ) -> AgentConversation | None:
        """Mark a conversation as complete and return it."""
        conv = self._conversations.get(conversation_id)
        if conv is None:
            return None
        conv.finalize(outcome, details)
        return conv

    def list_conversations(
        self,
        *,
        symbol: str | None = None,
        outcome: ConversationOutcome | None = None,
        limit: int = 50,
    ) -> list[AgentConversation]:
        """List conversations with optional filters."""
        results: list[AgentConversation] = []
        for cid in reversed(self._conversation_order):
            conv = self._conversations.get(cid)
            if conv is None:
                continue
            if symbol and conv.symbol != symbol:
                continue
            if outcome and conv.outcome != outcome:
                continue
            results.append(conv)
            if len(results) >= limit:
                break
        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def messages_posted(self) -> int:
        return self._messages_posted

    @property
    def messages_delivered(self) -> int:
        return self._messages_delivered

    @property
    def conversation_count(self) -> int:
        return len(self._conversations)

    @property
    def subscriber_count(self) -> int:
        """Total number of registered handlers (sync + async)."""
        total = sum(len(h) for h in self._handlers.values())
        total += sum(len(h) for h in self._async_handlers.values())
        return total

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _register_conversation(self, conv: AgentConversation) -> None:
        """Register a conversation and evict oldest if at capacity."""
        self._conversations[conv.conversation_id] = conv
        self._conversation_order.append(conv.conversation_id)

        # Evict oldest if over capacity
        while len(self._conversations) > self._max_conversations:
            oldest_id = self._conversation_order.pop(0)
            self._conversations.pop(oldest_id, None)

    def _record_message(self, message: AgentMessage) -> None:
        """Record a message in its conversation (auto-creates if needed)."""
        cid = message.conversation_id
        if not cid:
            return

        conv = self._conversations.get(cid)
        if conv is None:
            # Auto-create a bare conversation
            conv = AgentConversation(conversation_id=cid)
            self._register_conversation(conv)

        conv.add_message(message)

    def clear(self) -> None:
        """Clear all conversations and stats (for testing)."""
        self._conversations.clear()
        self._conversation_order.clear()
        self._handlers.clear()
        self._async_handlers.clear()
        self._messages_posted = 0
        self._messages_delivered = 0
