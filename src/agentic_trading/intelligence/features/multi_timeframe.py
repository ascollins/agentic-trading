"""Multi-timeframe feature alignment.

Aligns feature vectors computed on different timeframes to the fastest
(lowest) timeframe so that strategies can consume a single, unified
feature dict per bar.

Higher-timeframe features are forward-filled to lower-timeframe
timestamps: e.g. the ``4h_ema_12`` value computed at 08:00 is carried
forward across every 5m bar until the next 4h bar closes at 12:00.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

from agentic_trading.core.enums import Timeframe
from agentic_trading.core.events import BaseEvent, FeatureVector
from agentic_trading.core.interfaces import IEventBus

logger = logging.getLogger(__name__)

# Mapping from Timeframe to its duration in minutes, used for ordering.
_TF_MINUTES: dict[Timeframe, int] = {tf: tf.minutes for tf in Timeframe}


class MultiTimeframeAligner:
    """Collects feature vectors from multiple timeframes and produces a
    unified, aligned feature dict keyed with timeframe prefixes.

    Usage::

        aligner = MultiTimeframeAligner(
            symbol="BTC/USDT",
            timeframes=[Timeframe.M5, Timeframe.H1, Timeframe.H4],
            event_bus=bus,
        )
        await aligner.start()

    When feature vectors arrive, the aligner forward-fills higher-TF
    features and publishes a combined ``FeatureVector`` on the
    ``feature.vector.aligned`` topic each time the *fastest* timeframe
    updates.
    """

    def __init__(
        self,
        symbol: str,
        timeframes: list[Timeframe],
        event_bus: IEventBus | None = None,
        publish_topic: str = "feature.vector.aligned",
    ) -> None:
        if not timeframes:
            raise ValueError("At least one timeframe is required")

        self._symbol = symbol
        self._event_bus = event_bus
        self._publish_topic = publish_topic

        # Sort timeframes fastest-first.
        self._timeframes = sorted(timeframes, key=lambda tf: _TF_MINUTES[tf])
        self._base_tf = self._timeframes[0]  # The fastest timeframe.

        # Latest feature snapshot per timeframe.
        # Key: Timeframe -> (timestamp, features_dict)
        self._latest: dict[Timeframe, tuple[datetime, dict[str, float]]] = {}

    # ------------------------------------------------------------------
    # Event bus lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to feature vectors on the event bus."""
        if self._event_bus is None:
            logger.warning(
                "MultiTimeframeAligner started without event bus - "
                "call update() directly"
            )
            return

        await self._event_bus.subscribe(
            topic="feature.vector",
            group=f"mtf_aligner_{self._symbol}",
            handler=self._handle_feature_vector,
        )
        logger.info(
            "MultiTimeframeAligner for %s subscribed to feature.vector "
            "(timeframes=%s)",
            self._symbol,
            [tf.value for tf in self._timeframes],
        )

    async def stop(self) -> None:
        """Clean up (currently a no-op)."""
        logger.info("MultiTimeframeAligner for %s stopped", self._symbol)

    async def _handle_feature_vector(self, event: BaseEvent) -> None:
        """Event-bus callback for incoming feature vectors."""
        if not isinstance(event, FeatureVector):
            return
        if event.symbol != self._symbol:
            return
        if event.timeframe not in self._timeframes:
            return

        combined = self.update(event.timeframe, event.timestamp, event.features)

        # Only publish when the base (fastest) timeframe updates.
        if event.timeframe == self._base_tf and self._event_bus is not None:
            aligned_fv = FeatureVector(
                symbol=self._symbol,
                timeframe=self._base_tf,
                features=combined,
                source_module="features.multi_timeframe",
            )
            await self._event_bus.publish(self._publish_topic, aligned_fv)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        timeframe: Timeframe,
        timestamp: datetime,
        features: dict[str, float],
    ) -> dict[str, float]:
        """Ingest a feature snapshot for *timeframe* and return the
        merged, prefixed feature dict.

        Args:
            timeframe: The timeframe this snapshot belongs to.
            timestamp: Timestamp of the originating bar close.
            features: Raw feature dict (un-prefixed).

        Returns:
            A merged dict containing features from **all** timeframes
            observed so far, with keys prefixed by timeframe value
            (e.g. ``"1h_ema_12"``).
        """
        self._latest[timeframe] = (timestamp, dict(features))
        return self.get_aligned_features()

    def get_aligned_features(self) -> dict[str, float]:
        """Build the merged feature dict from the latest snapshots.

        Features from each timeframe are prefixed with the timeframe
        value, e.g. ``"5m_rsi_14"``, ``"1h_adx_14"``.

        If a higher timeframe has not yet produced any features, it is
        simply omitted (strategies should check for key presence or use
        ``dict.get`` with a default).
        """
        combined: dict[str, float] = {}

        for tf in self._timeframes:
            entry = self._latest.get(tf)
            if entry is None:
                # Higher TF data hasn't arrived yet - skip it.
                continue

            _ts, feats = entry
            prefix = tf.value  # e.g. "1h", "4h", "1d"

            for key, value in feats.items():
                combined[f"{prefix}_{key}"] = value

        return combined

    def has_all_timeframes(self) -> bool:
        """Return ``True`` when at least one snapshot has been received
        for every configured timeframe."""
        return all(tf in self._latest for tf in self._timeframes)

    def get_stale_timeframes(
        self, reference_time: datetime, max_age_bars: int = 2
    ) -> list[Timeframe]:
        """Return timeframes whose last update is older than
        *max_age_bars* bar durations from *reference_time*.

        Useful for detecting data feed issues on higher timeframes.
        """
        stale: list[Timeframe] = []
        for tf in self._timeframes:
            entry = self._latest.get(tf)
            if entry is None:
                stale.append(tf)
                continue
            ts, _ = entry
            age_seconds = (reference_time - ts).total_seconds()
            max_seconds = tf.seconds * max_age_bars
            if age_seconds > max_seconds:
                stale.append(tf)
        return stale

    @property
    def base_timeframe(self) -> Timeframe:
        """The fastest (base) timeframe."""
        return self._base_tf

    @property
    def symbol(self) -> str:
        return self._symbol

    def clear(self) -> None:
        """Reset all stored snapshots."""
        self._latest.clear()
