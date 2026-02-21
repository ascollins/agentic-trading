"""Audit bundle generation (design spec §7.2).

Collects all evidence for a given trace_id into a self-contained JSON
bundle suitable for compliance inquiries and regulatory exports.

A bundle includes:
    - Audit log entries (proposed actions, policy evaluations, approvals)
    - Surveillance cases linked to the trace
    - Model records involved (if any)
    - Timeline of events ordered chronologically

Usage::

    generator = AuditBundleGenerator(
        audit_log=audit_log,
        case_manager=case_manager,
    )
    bundle = generator.generate(trace_id="abc-123")
    json_str = bundle.model_dump_json(indent=2)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agentic_trading.core.ids import new_id, utc_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bundle models
# ---------------------------------------------------------------------------


class BundleEntry(BaseModel):
    """Single entry in an audit bundle."""

    timestamp: datetime
    event_type: str
    actor: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_hash: str = ""


class AuditBundle(BaseModel):
    """Self-contained audit bundle for a trace or correlation ID.

    Fields
    ------
    bundle_id : str
        Unique identifier for this bundle.
    trace_id : str
        The correlation/trace ID this bundle covers.
    generated_at : datetime
        When this bundle was assembled.
    entries : list[BundleEntry]
        Chronologically ordered audit entries.
    surveillance_cases : list[dict]
        Any compliance cases linked to this trace.
    summary : dict
        High-level summary: counts, outcome, actors involved.
    """

    bundle_id: str = Field(default_factory=new_id)
    trace_id: str
    generated_at: datetime = Field(default_factory=utc_now)
    entries: list[BundleEntry] = Field(default_factory=list)
    surveillance_cases: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class AuditBundleGenerator:
    """Assembles audit bundles from system components.

    Parameters
    ----------
    audit_log:
        The control-plane AuditLog for reading entries by correlation ID.
    case_manager:
        Optional CaseManager for retrieving surveillance cases.
    """

    def __init__(
        self,
        audit_log: Any = None,
        case_manager: Any = None,
    ) -> None:
        self._audit_log = audit_log
        self._case_manager = case_manager

    def generate(self, trace_id: str) -> AuditBundle:
        """Generate a complete audit bundle for the given trace_id.

        Collects audit entries and surveillance cases, builds a
        chronological timeline, and produces a summary.
        """
        entries: list[BundleEntry] = []
        cases: list[dict[str, Any]] = []

        # 1. Collect audit log entries
        if self._audit_log is not None:
            try:
                raw_entries = self._audit_log.read(trace_id)
                for entry in raw_entries:
                    entries.append(BundleEntry(
                        timestamp=entry.timestamp,
                        event_type=entry.event_type,
                        actor=entry.actor,
                        payload=entry.payload,
                        payload_hash=entry.payload_hash,
                    ))
            except Exception:
                logger.debug(
                    "Failed to read audit log for trace %s", trace_id,
                    exc_info=True,
                )

        # 2. Collect surveillance cases
        if self._case_manager is not None:
            try:
                all_open = self._case_manager.list_open()
                for case in all_open:
                    # Match cases by checking evidence for trace_id
                    if self._case_matches_trace(case, trace_id):
                        cases.append(case.model_dump())
            except Exception:
                logger.debug(
                    "Failed to read cases for trace %s", trace_id,
                    exc_info=True,
                )

        # 3. Sort entries chronologically
        entries.sort(key=lambda e: e.timestamp)

        # 4. Build summary
        actors = {e.actor for e in entries if e.actor}
        event_types = {e.event_type for e in entries}
        outcome = self._determine_outcome(entries)

        summary = {
            "total_entries": len(entries),
            "total_cases": len(cases),
            "actors": sorted(actors),
            "event_types": sorted(event_types),
            "outcome": outcome,
        }

        bundle = AuditBundle(
            trace_id=trace_id,
            entries=entries,
            surveillance_cases=cases,
            summary=summary,
        )

        logger.info(
            "Audit bundle generated: bundle_id=%s trace_id=%s entries=%d cases=%d",
            bundle.bundle_id, trace_id, len(entries), len(cases),
        )
        return bundle

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _case_matches_trace(case: Any, trace_id: str) -> bool:
        """Check if a compliance case is related to the given trace_id."""
        for evidence_item in getattr(case, "evidence", []):
            if isinstance(evidence_item, dict):
                for val in evidence_item.values():
                    if isinstance(val, str) and trace_id in val:
                        return True
        # Also check case description
        desc = getattr(case, "description", "")
        if trace_id in desc:
            return True
        return False

    @staticmethod
    def _determine_outcome(entries: list[BundleEntry]) -> str:
        """Determine the overall outcome from audit entries."""
        event_types = {e.event_type for e in entries}

        if "information_barrier_blocked" in event_types:
            return "blocked_information_barrier"
        if "policy_blocked" in event_types:
            return "blocked_policy"

        for entry in entries:
            if entry.event_type == "tool_call_recorded":
                success = entry.payload.get("success")
                if success is True:
                    return "executed_success"
                if success is False:
                    return "executed_failure"

        if "approval_decision" in event_types:
            for entry in entries:
                if entry.event_type == "approval_decision":
                    if entry.payload.get("pending_request_id"):
                        return "pending_approval"
                    if not entry.payload.get("approved"):
                        return "approval_denied"

        if entries:
            return "in_progress"
        return "no_data"
