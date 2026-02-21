"""Data quality checking for OHLCV candles.

``DataQualityChecker`` runs a battery of checks against incoming candle data
and returns a list of ``DataQualityIssue`` objects.  These issues can be used
to trigger circuit breakers, emit risk alerts, or simply log warnings.

Checks implemented
------------------
1. **Gap detection** -- Missing candles within a contiguous series.
2. **Staleness** -- Last candle is older than an acceptable threshold.
3. **Price sanity** -- Price change between consecutive candles exceeds a
   threshold (potential bad tick or exchange glitch).
4. **Volume anomaly** -- Volume is unusually high or low relative to a
   rolling average.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Sequence

from pydantic import BaseModel, Field

from agentic_trading.core.enums import AssetClass, Timeframe
from agentic_trading.core.models import Candle, Instrument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data quality issue model
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """Severity level for a data quality issue."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DataQualityIssue(BaseModel):
    """A single data quality issue detected by the checker."""

    check: str  # Name of the check that produced this issue.
    severity: Severity
    symbol: str
    timeframe: str = ""
    message: str
    details: dict[str, object] = Field(default_factory=dict)
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# DataQualityChecker
# ---------------------------------------------------------------------------


class DataQualityChecker:
    """Stateless checker that produces ``DataQualityIssue`` instances.

    All methods are pure functions (no internal state). They accept candle
    data and return zero or more issues.
    """

    # ------------------------------------------------------------------
    # 1. Gap detection
    # ------------------------------------------------------------------

    @staticmethod
    def check_gaps(
        candles: Sequence[Candle],
        timeframe: Timeframe,
    ) -> list[DataQualityIssue]:
        """Detect missing candles in a contiguous series.

        The candles must be sorted in ascending chronological order.
        A gap is defined as a jump in timestamps that is larger than
        ``timeframe.seconds``.

        Parameters
        ----------
        candles:
            Sorted sequence of candles.
        timeframe:
            Expected timeframe (determines the expected interval).

        Returns
        -------
        list[DataQualityIssue]
            One issue per gap, with details about start/end and count of
            missing candles.
        """
        issues: list[DataQualityIssue] = []
        if len(candles) < 2:
            return issues

        expected_delta_s = timeframe.seconds

        for i in range(1, len(candles)):
            prev = candles[i - 1]
            curr = candles[i]

            delta_s = (curr.timestamp - prev.timestamp).total_seconds()

            # Allow a small tolerance (5 seconds) for timestamp jitter.
            if delta_s > expected_delta_s + 5:
                missing_count = int(delta_s / expected_delta_s) - 1
                severity = (
                    Severity.CRITICAL if missing_count >= 5 else Severity.WARNING
                )
                issues.append(
                    DataQualityIssue(
                        check="gap_detection",
                        severity=severity,
                        symbol=curr.symbol,
                        timeframe=timeframe.value,
                        message=(
                            f"Gap of {missing_count} candle(s) detected "
                            f"between {prev.timestamp.isoformat()} and "
                            f"{curr.timestamp.isoformat()}"
                        ),
                        details={
                            "gap_start": prev.timestamp.isoformat(),
                            "gap_end": curr.timestamp.isoformat(),
                            "missing_count": missing_count,
                            "delta_seconds": delta_s,
                            "expected_delta_seconds": expected_delta_s,
                        },
                    )
                )

        return issues

    # ------------------------------------------------------------------
    # 2. Staleness
    # ------------------------------------------------------------------

    @staticmethod
    def check_staleness(
        last_candle_time: datetime,
        max_age_seconds: float,
        symbol: str = "",
        now: datetime | None = None,
        instrument: Instrument | None = None,
    ) -> DataQualityIssue | None:
        """Check whether the most recent candle is too old.

        Parameters
        ----------
        last_candle_time:
            Timestamp of the most recent candle.
        max_age_seconds:
            Maximum acceptable age in seconds.
        symbol:
            Symbol for the issue report.
        now:
            Current time.  Defaults to ``datetime.now(UTC)``.
        instrument:
            Optional instrument metadata.  When provided for an FX
            instrument with ``weekend_close=True``, staleness is
            suppressed during closed trading sessions.

        Returns
        -------
        DataQualityIssue or None
            An issue if the data is stale, else ``None``.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Make both tz-aware for comparison.
        if last_candle_time.tzinfo is None:
            last_candle_time = last_candle_time.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        # FX session awareness: skip staleness when market is closed
        if instrument is not None and instrument.asset_class == AssetClass.FX:
            from agentic_trading.core.fx_normalizer import is_session_open

            if not is_session_open(
                instrument, now.hour, now.minute, now.isoweekday()
            ):
                return None  # market closed — staleness expected

        age_s = (now - last_candle_time).total_seconds()

        if age_s > max_age_seconds:
            severity = (
                Severity.CRITICAL
                if age_s > max_age_seconds * 3
                else Severity.WARNING
            )
            return DataQualityIssue(
                check="staleness",
                severity=severity,
                symbol=symbol,
                message=(
                    f"Data is stale: last candle at "
                    f"{last_candle_time.isoformat()}, "
                    f"age {age_s:.0f}s exceeds max {max_age_seconds:.0f}s"
                ),
                details={
                    "last_candle_time": last_candle_time.isoformat(),
                    "age_seconds": age_s,
                    "max_age_seconds": max_age_seconds,
                },
            )

        return None

    # ------------------------------------------------------------------
    # 3. Price sanity
    # ------------------------------------------------------------------

    @staticmethod
    def check_price_sanity(
        candle: Candle,
        prev_candle: Candle | None,
        max_change_pct: float = 15.0,
    ) -> list[DataQualityIssue]:
        """Detect unreasonable price movements.

        Checks:
            * Close-to-close change exceeds ``max_change_pct``.
            * Intra-candle range (high - low) exceeds ``max_change_pct`` of
              the open.
            * OHLC ordering: high >= max(open, close) and low <= min(open, close).
            * Non-positive prices.

        Parameters
        ----------
        candle:
            The candle to check.
        prev_candle:
            The preceding candle (for close-to-close comparison).
            Pass ``None`` to skip the close-to-close check.
        max_change_pct:
            Maximum acceptable percentage change.

        Returns
        -------
        list[DataQualityIssue]
        """
        issues: list[DataQualityIssue] = []

        # Non-positive price check.
        for field in ("open", "high", "low", "close"):
            value = getattr(candle, field)
            if value <= 0:
                issues.append(
                    DataQualityIssue(
                        check="price_sanity",
                        severity=Severity.CRITICAL,
                        symbol=candle.symbol,
                        timeframe=candle.timeframe.value,
                        message=(
                            f"Non-positive {field} price: {value} "
                            f"at {candle.timestamp.isoformat()}"
                        ),
                        details={
                            "field": field,
                            "value": value,
                            "timestamp": candle.timestamp.isoformat(),
                        },
                    )
                )

        # OHLC ordering sanity.
        if candle.high < max(candle.open, candle.close):
            issues.append(
                DataQualityIssue(
                    check="price_sanity",
                    severity=Severity.WARNING,
                    symbol=candle.symbol,
                    timeframe=candle.timeframe.value,
                    message=(
                        f"High ({candle.high}) < max(open, close) "
                        f"({max(candle.open, candle.close)}) "
                        f"at {candle.timestamp.isoformat()}"
                    ),
                    details={
                        "high": candle.high,
                        "open": candle.open,
                        "close": candle.close,
                        "timestamp": candle.timestamp.isoformat(),
                    },
                )
            )
        if candle.low > min(candle.open, candle.close):
            issues.append(
                DataQualityIssue(
                    check="price_sanity",
                    severity=Severity.WARNING,
                    symbol=candle.symbol,
                    timeframe=candle.timeframe.value,
                    message=(
                        f"Low ({candle.low}) > min(open, close) "
                        f"({min(candle.open, candle.close)}) "
                        f"at {candle.timestamp.isoformat()}"
                    ),
                    details={
                        "low": candle.low,
                        "open": candle.open,
                        "close": candle.close,
                        "timestamp": candle.timestamp.isoformat(),
                    },
                )
            )

        # Intra-candle range check.
        if candle.open > 0:
            intra_range_pct = (
                (candle.high - candle.low) / candle.open
            ) * 100.0
            if intra_range_pct > max_change_pct:
                issues.append(
                    DataQualityIssue(
                        check="price_sanity",
                        severity=Severity.WARNING,
                        symbol=candle.symbol,
                        timeframe=candle.timeframe.value,
                        message=(
                            f"Large intra-candle range: {intra_range_pct:.2f}% "
                            f"(max {max_change_pct}%) "
                            f"at {candle.timestamp.isoformat()}"
                        ),
                        details={
                            "intra_range_pct": round(intra_range_pct, 4),
                            "max_change_pct": max_change_pct,
                            "high": candle.high,
                            "low": candle.low,
                            "open": candle.open,
                            "timestamp": candle.timestamp.isoformat(),
                        },
                    )
                )

        # Close-to-close change.
        if prev_candle is not None and prev_candle.close > 0:
            change_pct = abs(
                (candle.close - prev_candle.close) / prev_candle.close
            ) * 100.0
            if change_pct > max_change_pct:
                issues.append(
                    DataQualityIssue(
                        check="price_sanity",
                        severity=Severity.CRITICAL,
                        symbol=candle.symbol,
                        timeframe=candle.timeframe.value,
                        message=(
                            f"Extreme close-to-close change: {change_pct:.2f}% "
                            f"(max {max_change_pct}%) from "
                            f"{prev_candle.timestamp.isoformat()} to "
                            f"{candle.timestamp.isoformat()}"
                        ),
                        details={
                            "change_pct": round(change_pct, 4),
                            "max_change_pct": max_change_pct,
                            "prev_close": prev_candle.close,
                            "curr_close": candle.close,
                            "prev_timestamp": prev_candle.timestamp.isoformat(),
                            "curr_timestamp": candle.timestamp.isoformat(),
                        },
                    )
                )

        return issues

    # ------------------------------------------------------------------
    # 4. Volume anomaly
    # ------------------------------------------------------------------

    @staticmethod
    def check_volume_anomaly(
        candle: Candle,
        avg_volume: float,
        threshold: float = 5.0,
    ) -> DataQualityIssue | None:
        """Detect unusual volume relative to a rolling average.

        Parameters
        ----------
        candle:
            The candle to check.
        avg_volume:
            The rolling average volume to compare against.
        threshold:
            Multiplier threshold.  If ``candle.volume > avg_volume * threshold``
            or ``candle.volume < avg_volume / threshold``, an issue is raised.

        Returns
        -------
        DataQualityIssue or None
        """
        if avg_volume <= 0:
            return None

        ratio = candle.volume / avg_volume

        if ratio > threshold:
            return DataQualityIssue(
                check="volume_anomaly",
                severity=Severity.WARNING,
                symbol=candle.symbol,
                timeframe=candle.timeframe.value,
                message=(
                    f"Unusually high volume: {candle.volume:.4f} "
                    f"({ratio:.1f}x average {avg_volume:.4f}) "
                    f"at {candle.timestamp.isoformat()}"
                ),
                details={
                    "volume": candle.volume,
                    "avg_volume": avg_volume,
                    "ratio": round(ratio, 4),
                    "threshold": threshold,
                    "direction": "high",
                    "timestamp": candle.timestamp.isoformat(),
                },
            )

        if threshold > 0 and ratio < (1.0 / threshold):
            return DataQualityIssue(
                check="volume_anomaly",
                severity=Severity.INFO,
                symbol=candle.symbol,
                timeframe=candle.timeframe.value,
                message=(
                    f"Unusually low volume: {candle.volume:.4f} "
                    f"({ratio:.3f}x average {avg_volume:.4f}) "
                    f"at {candle.timestamp.isoformat()}"
                ),
                details={
                    "volume": candle.volume,
                    "avg_volume": avg_volume,
                    "ratio": round(ratio, 4),
                    "threshold": threshold,
                    "direction": "low",
                    "timestamp": candle.timestamp.isoformat(),
                },
            )

        return None

    # ------------------------------------------------------------------
    # Composite check
    # ------------------------------------------------------------------

    def run_all_checks(
        self,
        candles: Sequence[Candle],
        timeframe: Timeframe,
        *,
        max_staleness_seconds: float = 300.0,
        max_price_change_pct: float = 15.0,
        volume_threshold: float = 5.0,
        now: datetime | None = None,
    ) -> list[DataQualityIssue]:
        """Run all quality checks on a sequence of candles.

        This is a convenience method that combines gap detection, staleness,
        price sanity, and volume anomaly checks into a single call.

        Parameters
        ----------
        candles:
            Sorted sequence of candles (ascending by timestamp).
        timeframe:
            Expected timeframe.
        max_staleness_seconds:
            Maximum acceptable age for the most recent candle.
        max_price_change_pct:
            Maximum acceptable price change percentage.
        volume_threshold:
            Volume anomaly threshold multiplier.
        now:
            Current time for staleness check.

        Returns
        -------
        list[DataQualityIssue]
            All issues found, sorted by severity (critical first).
        """
        all_issues: list[DataQualityIssue] = []

        if not candles:
            return all_issues

        # 1. Gap detection.
        all_issues.extend(self.check_gaps(candles, timeframe))

        # 2. Staleness.
        staleness_issue = self.check_staleness(
            candles[-1].timestamp,
            max_staleness_seconds,
            symbol=candles[-1].symbol,
            now=now,
        )
        if staleness_issue is not None:
            all_issues.append(staleness_issue)

        # 3. Price sanity (pairwise).
        prev_candle: Candle | None = None
        for candle in candles:
            all_issues.extend(
                self.check_price_sanity(candle, prev_candle, max_price_change_pct)
            )
            prev_candle = candle

        # 4. Volume anomaly (compute rolling average over the series).
        if len(candles) >= 2:
            total_volume = sum(c.volume for c in candles)
            avg_volume = total_volume / len(candles)
            for candle in candles:
                vol_issue = self.check_volume_anomaly(
                    candle, avg_volume, volume_threshold
                )
                if vol_issue is not None:
                    all_issues.append(vol_issue)

        # Sort by severity: CRITICAL > WARNING > INFO.
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.WARNING: 1,
            Severity.INFO: 2,
        }
        all_issues.sort(key=lambda i: severity_order.get(i.severity, 99))

        return all_issues
