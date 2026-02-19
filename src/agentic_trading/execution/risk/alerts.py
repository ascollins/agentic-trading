"""Smart alert rule engine with hit-rate tracking.

The :class:`AlertEngine` lets callers register arbitrary condition
functions as rules.  On each evaluation cycle the engine runs every
rule, tracks historical hit rates, and returns a list of
:class:`~agentic_trading.core.events.RiskAlert` events for the rules
that fired -- each annotated with *why* it fired.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agentic_trading.core.enums import RiskAlertSeverity
from agentic_trading.core.events import RiskAlert

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Rule definition
# ----------------------------------------------------------------------

@dataclass
class AlertRule:
    """A single alert rule.

    Args:
        name: Unique human-readable identifier.
        condition_fn: A callable that receives a context dict and returns
            a tuple ``(triggered: bool, explanation: str)``.  The
            explanation is included in the alert event when the rule
            fires.
        severity: Alert severity level.
        cooldown_seconds: Minimum time between consecutive firings of
            this rule to avoid alert fatigue.
    """

    name: str
    condition_fn: Callable[[dict[str, Any]], tuple[bool, str]]
    severity: RiskAlertSeverity = RiskAlertSeverity.WARNING
    cooldown_seconds: float = 60.0

    # ---- internal tracking state ----
    eval_count: int = field(init=False, default=0)
    fire_count: int = field(init=False, default=0)
    last_fire_time: float = field(init=False, default=0.0)

    @property
    def hit_rate(self) -> float:
        """Fraction of evaluations that triggered the rule."""
        if self.eval_count == 0:
            return 0.0
        return self.fire_count / self.eval_count


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------

class AlertEngine:
    """Rule engine that evaluates alert conditions and tracks hit rates.

    Usage::

        engine = AlertEngine()

        # Register a rule
        def high_drawdown(ctx: dict) -> tuple[bool, str]:
            dd = ctx.get("drawdown_pct", 0.0)
            if dd > 0.10:
                return True, f"Drawdown at {dd:.1%} exceeds 10% threshold"
            return False, ""

        engine.add_rule(
            name="high_drawdown",
            condition_fn=high_drawdown,
            severity=RiskAlertSeverity.CRITICAL,
            cooldown_seconds=120,
        )

        # On each evaluation cycle
        alerts = engine.evaluate({
            "drawdown_pct": 0.12,
            "leverage": 2.5,
        })
    """

    def __init__(self) -> None:
        self._rules: dict[str, AlertRule] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def add_rule(
        self,
        name: str,
        condition_fn: Callable[[dict[str, Any]], tuple[bool, str]],
        severity: RiskAlertSeverity = RiskAlertSeverity.WARNING,
        cooldown_seconds: float = 60.0,
    ) -> None:
        """Register a new alert rule.

        Args:
            name: Unique rule name.
            condition_fn: ``(context) -> (triggered, explanation)``
            severity: Alert severity when the rule fires.
            cooldown_seconds: Minimum interval between consecutive firings.

        Raises:
            ValueError: If a rule with the same name is already registered.
        """
        if name in self._rules:
            raise ValueError(f"Alert rule '{name}' is already registered")

        self._rules[name] = AlertRule(
            name=name,
            condition_fn=condition_fn,
            severity=severity,
            cooldown_seconds=cooldown_seconds,
        )
        logger.info(
            "Registered alert rule: name=%s severity=%s cooldown=%.0fs",
            name,
            severity.value,
            cooldown_seconds,
        )

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name.  Returns ``True`` if found."""
        removed = self._rules.pop(name, None)
        if removed:
            logger.info("Removed alert rule: %s", name)
        return removed is not None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, context: dict[str, Any]) -> list[RiskAlert]:
        """Evaluate all registered rules against the given context.

        Args:
            context: Arbitrary dict of metrics / state that rules
                inspect.

        Returns:
            List of :class:`RiskAlert` events for rules that fired
            (respecting cooldown).
        """
        now = time.monotonic()
        alerts: list[RiskAlert] = []

        for rule in self._rules.values():
            rule.eval_count += 1

            try:
                triggered, explanation = rule.condition_fn(context)
            except Exception:
                logger.exception(
                    "Alert rule '%s' raised an exception during evaluation",
                    rule.name,
                )
                continue

            if not triggered:
                continue

            # Cooldown check
            if (now - rule.last_fire_time) < rule.cooldown_seconds:
                logger.debug(
                    "Alert rule '%s' triggered but in cooldown "
                    "(%.1fs remaining)",
                    rule.name,
                    rule.cooldown_seconds - (now - rule.last_fire_time),
                )
                continue

            # Fire the alert
            rule.fire_count += 1
            rule.last_fire_time = now

            alert = RiskAlert(
                severity=rule.severity,
                alert_type=rule.name,
                message=explanation,
                details={
                    "rule_name": rule.name,
                    "hit_rate": round(rule.hit_rate, 4),
                    "eval_count": rule.eval_count,
                    "fire_count": rule.fire_count,
                    "context_snapshot": _safe_snapshot(context),
                },
            )
            alerts.append(alert)

            logger.warning(
                "Alert FIRED: [%s] %s  (severity=%s, hit_rate=%.2f%%)",
                rule.name,
                explanation,
                rule.severity.value,
                rule.hit_rate * 100,
            )

        return alerts

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_hit_rates(self) -> dict[str, float]:
        """Return a mapping of rule name -> historical hit rate."""
        return {name: rule.hit_rate for name, rule in self._rules.items()}

    def get_rule_stats(self) -> list[dict[str, Any]]:
        """Return detailed stats for every registered rule."""
        stats: list[dict[str, Any]] = []
        for rule in self._rules.values():
            stats.append({
                "name": rule.name,
                "severity": rule.severity.value,
                "eval_count": rule.eval_count,
                "fire_count": rule.fire_count,
                "hit_rate": round(rule.hit_rate, 4),
                "cooldown_seconds": rule.cooldown_seconds,
            })
        return stats

    def reset_stats(self) -> None:
        """Reset evaluation counters for all rules."""
        for rule in self._rules.values():
            rule.eval_count = 0
            rule.fire_count = 0
            rule.last_fire_time = 0.0
        logger.info("Alert engine stats reset for %d rules", len(self._rules))

    @property
    def rule_count(self) -> int:
        return len(self._rules)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _safe_snapshot(context: dict[str, Any], max_keys: int = 20) -> dict[str, Any]:
    """Create a safe, serialisable snapshot of the context for event details.

    Limits the number of keys and converts non-serialisable values to
    strings to avoid Pydantic validation errors.
    """
    snapshot: dict[str, Any] = {}
    for i, (k, v) in enumerate(context.items()):
        if i >= max_keys:
            snapshot["_truncated"] = True
            break
        try:
            # Keep primitives as-is; convert anything else to str
            if isinstance(v, (int, float, str, bool, type(None))):
                snapshot[k] = v
            else:
                snapshot[k] = str(v)
        except Exception:
            snapshot[k] = "<unserializable>"
    return snapshot
