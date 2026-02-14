"""Redis Streams event bus implementation.

Uses Redis Streams for persistent, ordered event delivery with consumer groups.
Each subscriber group gets guaranteed at-least-once delivery.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine

import redis.asyncio as aioredis

from agentic_trading.core.events import BaseEvent
from .schemas import get_event_class

logger = logging.getLogger(__name__)


class RedisStreamsBus:
    """Production event bus backed by Redis Streams."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        max_stream_length: int = 10_000,
        block_ms: int = 1000,
        batch_size: int = 10,
    ) -> None:
        self._redis_url = redis_url
        self._redis: aioredis.Redis | None = None
        self._max_len = max_stream_length
        self._block_ms = block_ms
        self._batch_size = batch_size
        self._subscriptions: list[
            tuple[str, str, Callable[[BaseEvent], Coroutine[Any, Any, None]]]
        ] = []
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        """Connect to Redis and start consumer loops."""
        self._redis = aioredis.from_url(
            self._redis_url, decode_responses=True
        )
        self._running = True

        for topic, group, handler in self._subscriptions:
            # Ensure consumer group exists
            try:
                await self._redis.xgroup_create(
                    topic, group, id="0", mkstream=True
                )
            except aioredis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise

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
        """Register a handler. Must be called before start()."""
        self._subscriptions.append((topic, group, handler))

    async def _consume_loop(
        self,
        topic: str,
        group: str,
        handler: Callable[[BaseEvent], Coroutine[Any, Any, None]],
    ) -> None:
        """Consumer loop: read from stream, deserialize, call handler, ack."""
        consumer_name = f"{group}-worker"
        assert self._redis is not None

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
                        try:
                            event = self._deserialize(fields)
                            if event:
                                await handler(event)
                            await self._redis.xack(topic, group, msg_id)
                        except Exception:
                            logger.exception(
                                "Error processing message %s on %s",
                                msg_id,
                                topic,
                            )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Consumer loop error for %s/%s", topic, group)
                await asyncio.sleep(1)

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
