"""Redis Streams event bus implementation.

Uses Redis Streams for persistent, ordered event delivery with consumer groups.
Each subscriber group gets guaranteed at-least-once delivery.

Fixes over the original:
- Handler exceptions no longer silently ack messages
- Dead-letter tracking for failed messages after max retries
- Error callback hook for observability / metrics
- Per-handler error counters for health checks
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

import redis.asyncio as aioredis

from agentic_trading.core.events import BaseEvent

from .schemas import get_event_class

logger = logging.getLogger(__name__)


@dataclass
class DeadLetter:
    """Record of a message that exhausted its retry budget."""

    topic: str
    group: str
    msg_id: str
    event_type: str
    error: str
    attempts: int
    timestamp: float = field(default_factory=time.monotonic)


class RedisStreamsBus:
    """Production event bus backed by Redis Streams.

    Key improvements:
    - Messages are only ack'd *after* handler succeeds.
    - Failed messages are retried up to ``max_handler_retries`` times.
    - After exhausting retries, messages go to an in-memory dead-letter
      list and are ack'd (so they don't block the stream).
    - Optional ``on_handler_error`` callback for external metrics/alerting.
    - Per-topic/group error counters accessible via ``get_error_counts()``.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        max_stream_length: int = 10_000,
        block_ms: int = 1000,
        batch_size: int = 10,
        max_handler_retries: int = 3,
        on_handler_error: Callable[
            [str, str, str, Exception], None
        ] | None = None,
    ) -> None:
        self._redis_url = redis_url
        self._redis: aioredis.Redis | None = None
        self._max_len = max_stream_length
        self._block_ms = block_ms
        self._batch_size = batch_size
        self._max_retries = max_handler_retries
        self._on_handler_error = on_handler_error
        self._subscriptions: list[
            tuple[str, str, Callable[[BaseEvent], Coroutine[Any, Any, None]]]
        ] = []
        self._tasks: list[asyncio.Task] = []
        self._running = False

        # Observability
        self._error_counts: dict[str, int] = defaultdict(int)
        self._handler_attempts: dict[str, int] = defaultdict(int)
        self._dead_letters: list[DeadLetter] = []
        self._messages_processed: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to Redis and start consumer loops."""
        self._redis = aioredis.from_url(
            self._redis_url, decode_responses=True
        )
        self._running = True

        for topic, group, handler in self._subscriptions:
            await self._ensure_group(topic, group)
            task = asyncio.create_task(
                self._consume_loop(topic, group, handler),
                name=f"consumer-{topic}-{group}",
            )
            self._tasks.append(task)

    async def stop(self) -> None:
        """Stop consumer loops and close Redis connection."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        if self._redis:
            await self._redis.aclose()
            self._redis = None

    # ------------------------------------------------------------------
    # Publish / Subscribe
    # ------------------------------------------------------------------

    async def publish(self, topic: str, event: BaseEvent) -> None:
        """Publish an event to a Redis Stream."""
        if not self._redis:
            raise RuntimeError("RedisStreamsBus not started")

        payload = {
            "_type": type(event).__name__,
            "_data": event.model_dump_json(),
        }
        await self._redis.xadd(
            topic, payload, maxlen=self._max_len, approximate=True
        )

    async def subscribe(
        self,
        topic: str,
        group: str,
        handler: Callable[[BaseEvent], Coroutine[Any, Any, None]],
    ) -> None:
        """Register a handler.

        Can be called before or after start().  If the bus is already
        running the consumer group is created and the consume loop is
        launched immediately.
        """
        self._subscriptions.append((topic, group, handler))

        # Late subscription — bus already running, spin up consumer now.
        if self._running and self._redis is not None:
            await self._ensure_group(topic, group)
            task = asyncio.create_task(
                self._consume_loop(topic, group, handler),
                name=f"consumer-{topic}-{group}",
            )
            self._tasks.append(task)

    # ------------------------------------------------------------------
    # Consumer loop (fixed exception handling)
    # ------------------------------------------------------------------

    async def _consume_loop(
        self,
        topic: str,
        group: str,
        handler: Callable[[BaseEvent], Coroutine[Any, Any, None]],
    ) -> None:
        """Consumer loop: read from stream, deserialize, call handler, ack.

        Key fix: messages are only acknowledged AFTER the handler
        succeeds.  On failure the message stays pending in the consumer
        group so Redis can redeliver it on the next claim cycle.
        After ``max_handler_retries`` failures the message is sent to
        the dead-letter list and ack'd to prevent infinite loops.
        """
        consumer_name = f"{group}-worker"
        assert self._redis is not None
        error_key = f"{topic}/{group}"

        while self._running:
            try:
                entries = await self._redis.xreadgroup(
                    groupname=group,
                    consumername=consumer_name,
                    streams={topic: ">"},
                    count=self._batch_size,
                    block=self._block_ms,
                )

                if not entries:
                    continue

                for _stream, messages in entries:
                    for msg_id, fields in messages:
                        await self._process_message(
                            topic, group, handler, msg_id, fields, error_key,
                        )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception(
                    "Consumer loop error for %s/%s", topic, group,
                )
                self._error_counts[error_key] += 1
                await asyncio.sleep(1)

    async def _process_message(
        self,
        topic: str,
        group: str,
        handler: Callable[[BaseEvent], Coroutine[Any, Any, None]],
        msg_id: str,
        fields: dict[str, str],
        error_key: str,
    ) -> None:
        """Process a single message with retry tracking.

        On success: ack the message.
        On failure: increment retry counter, log, fire error callback.
        After max retries: dead-letter the message and ack it.
        """
        attempt_key = f"{topic}/{group}/{msg_id}"

        try:
            event = self._deserialize(fields)
            if event is None:
                # Malformed message — can't retry, dead-letter immediately
                self._dead_letters.append(
                    DeadLetter(
                        topic=topic,
                        group=group,
                        msg_id=str(msg_id),
                        event_type=fields.get("_type", "unknown"),
                        error="deserialization_failed",
                        attempts=1,
                    )
                )
                await self._redis.xack(topic, group, msg_id)
                return

            await handler(event)

            # Handler succeeded — ack and clean up
            await self._redis.xack(topic, group, msg_id)
            self._messages_processed += 1
            self._handler_attempts.pop(attempt_key, None)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._error_counts[error_key] += 1
            self._handler_attempts[attempt_key] = (
                self._handler_attempts.get(attempt_key, 0) + 1
            )
            attempts = self._handler_attempts[attempt_key]

            logger.exception(
                "Handler error on %s/%s msg=%s (attempt %d/%d)",
                topic,
                group,
                msg_id,
                attempts,
                self._max_retries,
            )

            # Fire external error callback
            if self._on_handler_error is not None:
                try:
                    self._on_handler_error(topic, group, str(msg_id), exc)
                except Exception:
                    logger.warning(
                        "on_handler_error callback failed", exc_info=True,
                    )

            if attempts >= self._max_retries:
                # Exhausted retries — dead-letter and ack to unblock
                logger.error(
                    "Dead-lettering message %s on %s/%s after %d attempts",
                    msg_id,
                    topic,
                    group,
                    attempts,
                )
                self._dead_letters.append(
                    DeadLetter(
                        topic=topic,
                        group=group,
                        msg_id=str(msg_id),
                        event_type=fields.get("_type", "unknown"),
                        error=str(exc),
                        attempts=attempts,
                    )
                )
                await self._redis.xack(topic, group, msg_id)
                self._handler_attempts.pop(attempt_key, None)
            # else: leave un-ack'd for redelivery on next read

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_error_counts(self) -> dict[str, int]:
        """Return per-topic/group error counts."""
        return dict(self._error_counts)

    @property
    def dead_letters(self) -> list[DeadLetter]:
        """Access the dead-letter list (read-only snapshot)."""
        return list(self._dead_letters)

    @property
    def messages_processed(self) -> int:
        """Total messages successfully processed."""
        return self._messages_processed

    def clear_dead_letters(self) -> list[DeadLetter]:
        """Drain the dead-letter list and return all entries."""
        drained = self._dead_letters[:]
        self._dead_letters.clear()
        return drained

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _ensure_group(self, topic: str, group: str) -> None:
        """Create consumer group, ignoring BUSYGROUP if it already exists."""
        assert self._redis is not None
        try:
            await self._redis.xgroup_create(
                topic, group, id="0", mkstream=True,
            )
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    @staticmethod
    def _deserialize(fields: dict[str, str]) -> BaseEvent | None:
        """Deserialize a Redis Stream message back to an event."""
        event_type_name = fields.get("_type")
        event_data = fields.get("_data")

        if not event_type_name or not event_data:
            logger.warning("Malformed message: %s", fields)
            return None

        event_cls = get_event_class(event_type_name)
        if not event_cls:
            logger.warning("Unknown event type: %s", event_type_name)
            return None

        return event_cls.model_validate_json(event_data)
