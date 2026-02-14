"""Test MemoryEventBus publish/subscribe, event history, multiple subscribers."""

import pytest

from agentic_trading.core.events import BaseEvent, CandleEvent
from agentic_trading.core.enums import Exchange, Timeframe
from agentic_trading.event_bus.memory_bus import MemoryEventBus


class TestMemoryEventBusPublishSubscribe:
    async def test_publish_invokes_handler(self, memory_bus):
        received = []

        async def handler(event: BaseEvent) -> None:
            received.append(event)

        await memory_bus.subscribe("test.topic", "group1", handler)
        event = BaseEvent(source_module="test")
        await memory_bus.publish("test.topic", event)

        assert len(received) == 1
        assert received[0].event_id == event.event_id

    async def test_no_handler_for_topic(self, memory_bus):
        received = []

        async def handler(event: BaseEvent) -> None:
            received.append(event)

        await memory_bus.subscribe("topic_a", "group1", handler)
        await memory_bus.publish("topic_b", BaseEvent())

        assert len(received) == 0

    async def test_multiple_subscribers(self, memory_bus):
        received_a = []
        received_b = []

        async def handler_a(event: BaseEvent) -> None:
            received_a.append(event)

        async def handler_b(event: BaseEvent) -> None:
            received_b.append(event)

        await memory_bus.subscribe("test.topic", "group_a", handler_a)
        await memory_bus.subscribe("test.topic", "group_b", handler_b)

        event = BaseEvent(source_module="test")
        await memory_bus.publish("test.topic", event)

        assert len(received_a) == 1
        assert len(received_b) == 1
        assert received_a[0].event_id == received_b[0].event_id

    async def test_handler_error_does_not_break_others(self, memory_bus):
        received = []

        async def bad_handler(event: BaseEvent) -> None:
            raise ValueError("boom")

        async def good_handler(event: BaseEvent) -> None:
            received.append(event)

        await memory_bus.subscribe("test.topic", "bad", bad_handler)
        await memory_bus.subscribe("test.topic", "good", good_handler)

        await memory_bus.publish("test.topic", BaseEvent())
        assert len(received) == 1


class TestMemoryEventBusHistory:
    async def test_history_records_events(self, memory_bus):
        await memory_bus.publish("topic_a", BaseEvent(source_module="a"))
        await memory_bus.publish("topic_b", BaseEvent(source_module="b"))
        await memory_bus.publish("topic_a", BaseEvent(source_module="a2"))

        all_history = memory_bus.get_history()
        assert len(all_history) == 3

    async def test_history_filter_by_topic(self, memory_bus):
        await memory_bus.publish("topic_a", BaseEvent(source_module="a"))
        await memory_bus.publish("topic_b", BaseEvent(source_module="b"))

        a_history = memory_bus.get_history(topic="topic_a")
        assert len(a_history) == 1
        assert a_history[0][0] == "topic_a"

    async def test_clear_history(self, memory_bus):
        await memory_bus.publish("topic_a", BaseEvent())
        assert len(memory_bus.get_history()) == 1

        memory_bus.clear_history()
        assert len(memory_bus.get_history()) == 0


class TestMemoryEventBusLifecycle:
    async def test_start_stop(self, memory_bus):
        await memory_bus.start()
        await memory_bus.stop()
        # Should not raise

    async def test_publish_works_without_start(self, memory_bus):
        received = []

        async def handler(event: BaseEvent) -> None:
            received.append(event)

        await memory_bus.subscribe("t", "g", handler)
        await memory_bus.publish("t", BaseEvent())
        assert len(received) == 1
