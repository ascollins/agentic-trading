"""External notification dispatcher for risk alerts.

Routes :class:`~agentic_trading.core.events.RiskAlert` events to
external channels: HTTP webhooks, SMTP email, and an extensible
callback interface.

Usage::

    notifier = AlertNotifier()
    notifier.add_webhook("https://hooks.slack.com/...", severity_min="WARNING")
    notifier.add_email(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        from_addr="alerts@example.com",
        to_addrs=["ops@example.com"],
        username="alerts@example.com",
        password="app-password",
    )

    # After AlertEngine produces alerts:
    for alert in fired_alerts:
        await notifier.dispatch(alert)
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
import ssl
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Any, Awaitable, Callable

import aiohttp

from agentic_trading.core.enums import RiskAlertSeverity
from agentic_trading.core.events import RiskAlert

logger = logging.getLogger(__name__)

# Severity ordering for min-severity filtering.
_SEVERITY_ORDER: dict[str, int] = {
    RiskAlertSeverity.WARNING.value: 0,
    RiskAlertSeverity.CRITICAL.value: 1,
    RiskAlertSeverity.EMERGENCY.value: 2,
}


def _severity_gte(actual: str, minimum: str) -> bool:
    return _SEVERITY_ORDER.get(actual, 0) >= _SEVERITY_ORDER.get(minimum, 0)


# ------------------------------------------------------------------
# Channel definitions
# ------------------------------------------------------------------

@dataclass
class WebhookChannel:
    """HTTP POST webhook target."""

    url: str
    severity_min: str = RiskAlertSeverity.WARNING.value
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 10.0

    # Stats
    sent_count: int = field(init=False, default=0)
    error_count: int = field(init=False, default=0)


@dataclass
class EmailChannel:
    """SMTP email target."""

    smtp_host: str
    smtp_port: int = 587
    from_addr: str = ""
    to_addrs: list[str] = field(default_factory=list)
    username: str = ""
    password: str = ""
    use_tls: bool = True
    severity_min: str = RiskAlertSeverity.CRITICAL.value
    subject_prefix: str = "[TRADING ALERT]"

    # Stats
    sent_count: int = field(init=False, default=0)
    error_count: int = field(init=False, default=0)


@dataclass
class CallbackChannel:
    """Arbitrary async callback for custom integrations (Slack, PagerDuty, etc.)."""

    name: str
    callback: Callable[[RiskAlert], Awaitable[bool]]
    severity_min: str = RiskAlertSeverity.WARNING.value

    # Stats
    sent_count: int = field(init=False, default=0)
    error_count: int = field(init=False, default=0)


# ------------------------------------------------------------------
# Notifier
# ------------------------------------------------------------------

class AlertNotifier:
    """Dispatches RiskAlert events to external notification channels."""

    def __init__(self) -> None:
        self._webhooks: list[WebhookChannel] = []
        self._emails: list[EmailChannel] = []
        self._callbacks: list[CallbackChannel] = []
        self._total_dispatched: int = 0
        self._total_errors: int = 0

    # ---- Channel registration ----

    def add_webhook(
        self,
        url: str,
        severity_min: str = RiskAlertSeverity.WARNING.value,
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        """Register an HTTP webhook channel."""
        self._webhooks.append(
            WebhookChannel(
                url=url,
                severity_min=severity_min,
                headers=headers or {},
                timeout_seconds=timeout_seconds,
            )
        )
        logger.info("Registered webhook channel: url=%s min_severity=%s", url, severity_min)

    def add_email(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        from_addr: str = "",
        to_addrs: list[str] | None = None,
        username: str = "",
        password: str = "",
        use_tls: bool = True,
        severity_min: str = RiskAlertSeverity.CRITICAL.value,
        subject_prefix: str = "[TRADING ALERT]",
    ) -> None:
        """Register an SMTP email channel."""
        self._emails.append(
            EmailChannel(
                smtp_host=smtp_host,
                smtp_port=smtp_port,
                from_addr=from_addr,
                to_addrs=to_addrs or [],
                username=username,
                password=password,
                use_tls=use_tls,
                severity_min=severity_min,
                subject_prefix=subject_prefix,
            )
        )
        logger.info(
            "Registered email channel: host=%s to=%s min_severity=%s",
            smtp_host, to_addrs, severity_min,
        )

    def add_callback(
        self,
        name: str,
        callback: Callable[[RiskAlert], Awaitable[bool]],
        severity_min: str = RiskAlertSeverity.WARNING.value,
    ) -> None:
        """Register a custom async callback channel."""
        self._callbacks.append(
            CallbackChannel(name=name, callback=callback, severity_min=severity_min)
        )
        logger.info("Registered callback channel: name=%s min_severity=%s", name, severity_min)

    # ---- Dispatch ----

    async def dispatch(self, alert: RiskAlert) -> int:
        """Send an alert to all matching channels.

        Returns the number of channels that were successfully notified.
        """
        severity = alert.severity.value if hasattr(alert.severity, "value") else str(alert.severity)
        tasks: list[Awaitable[bool]] = []

        for wh in self._webhooks:
            if _severity_gte(severity, wh.severity_min):
                tasks.append(self._send_webhook(wh, alert))

        for em in self._emails:
            if _severity_gte(severity, em.severity_min):
                tasks.append(self._send_email(em, alert))

        for cb in self._callbacks:
            if _severity_gte(severity, cb.severity_min):
                tasks.append(self._send_callback(cb, alert))

        if not tasks:
            return 0

        results = await asyncio.gather(*tasks, return_exceptions=True)
        successes = sum(1 for r in results if r is True)
        failures = len(results) - successes

        self._total_dispatched += successes
        self._total_errors += failures
        return successes

    # ---- Channel senders ----

    async def _send_webhook(self, channel: WebhookChannel, alert: RiskAlert) -> bool:
        """POST alert JSON to a webhook URL."""
        payload = {
            "alert_type": alert.alert_type,
            "severity": alert.severity.value if hasattr(alert.severity, "value") else str(alert.severity),
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "event_id": alert.event_id,
            "details": {
                k: v for k, v in (alert.details or {}).items()
                if isinstance(v, (str, int, float, bool, type(None)))
            },
        }
        try:
            timeout = aiohttp.ClientTimeout(total=channel.timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    channel.url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        **channel.headers,
                    },
                ) as resp:
                    if resp.status < 300:
                        channel.sent_count += 1
                        logger.info(
                            "Webhook sent: url=%s alert=%s status=%d",
                            channel.url, alert.alert_type, resp.status,
                        )
                        return True
                    logger.warning(
                        "Webhook failed: url=%s status=%d body=%s",
                        channel.url, resp.status, await resp.text(),
                    )
                    channel.error_count += 1
                    return False
        except Exception:
            logger.exception("Webhook error: url=%s", channel.url)
            channel.error_count += 1
            return False

    async def _send_email(self, channel: EmailChannel, alert: RiskAlert) -> bool:
        """Send alert via SMTP email (runs in thread to avoid blocking)."""
        severity = alert.severity.value if hasattr(alert.severity, "value") else str(alert.severity)
        subject = f"{channel.subject_prefix} [{severity.upper()}] {alert.alert_type}"
        body = (
            f"Alert: {alert.alert_type}\n"
            f"Severity: {severity}\n"
            f"Time: {alert.timestamp.isoformat()}\n\n"
            f"{alert.message}\n\n"
            f"Event ID: {alert.event_id}\n"
        )

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = channel.from_addr
        msg["To"] = ", ".join(channel.to_addrs)
        msg.set_content(body)

        try:
            await asyncio.to_thread(self._smtp_send, channel, msg)
            channel.sent_count += 1
            logger.info(
                "Email sent: to=%s alert=%s", channel.to_addrs, alert.alert_type,
            )
            return True
        except Exception:
            logger.exception("Email error: host=%s", channel.smtp_host)
            channel.error_count += 1
            return False

    @staticmethod
    def _smtp_send(channel: EmailChannel, msg: EmailMessage) -> None:
        """Blocking SMTP send (called via to_thread)."""
        if channel.use_tls:
            context = ssl.create_default_context()
            with smtplib.SMTP(channel.smtp_host, channel.smtp_port) as server:
                server.starttls(context=context)
                if channel.username:
                    server.login(channel.username, channel.password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(channel.smtp_host, channel.smtp_port) as server:
                if channel.username:
                    server.login(channel.username, channel.password)
                server.send_message(msg)

    async def _send_callback(self, channel: CallbackChannel, alert: RiskAlert) -> bool:
        """Invoke a custom async callback."""
        try:
            result = await channel.callback(alert)
            if result:
                channel.sent_count += 1
            else:
                channel.error_count += 1
            return bool(result)
        except Exception:
            logger.exception("Callback error: name=%s", channel.name)
            channel.error_count += 1
            return False

    # ---- Introspection ----

    @property
    def total_dispatched(self) -> int:
        return self._total_dispatched

    @property
    def total_errors(self) -> int:
        return self._total_errors

    @property
    def channel_count(self) -> int:
        return len(self._webhooks) + len(self._emails) + len(self._callbacks)

    def get_channel_stats(self) -> list[dict[str, Any]]:
        """Return per-channel send/error stats."""
        stats: list[dict[str, Any]] = []
        for wh in self._webhooks:
            stats.append({
                "type": "webhook",
                "target": wh.url,
                "severity_min": wh.severity_min,
                "sent": wh.sent_count,
                "errors": wh.error_count,
            })
        for em in self._emails:
            stats.append({
                "type": "email",
                "target": ", ".join(em.to_addrs),
                "severity_min": em.severity_min,
                "sent": em.sent_count,
                "errors": em.error_count,
            })
        for cb in self._callbacks:
            stats.append({
                "type": "callback",
                "target": cb.name,
                "severity_min": cb.severity_min,
                "sent": cb.sent_count,
                "errors": cb.error_count,
            })
        return stats
