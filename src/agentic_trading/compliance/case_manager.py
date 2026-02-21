"""Compliance case management (design spec §7.2).

Provides the lifecycle model for compliance cases created by the
:class:`SurveillanceAgent` and a persistent store backed by JSONL.

Case lifecycle::

    open → investigating → escalated → closed
                  ↘                ↗
                    → closed (false_positive)

Each case carries an evidence list, timeline entries, and a final
disposition (confirmed, false_positive, inconclusive).
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentic_trading.core.ids import new_id, utc_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TimelineEntry(BaseModel):
    """Single entry in a case timeline."""

    timestamp: datetime = Field(default_factory=utc_now)
    action: str  # "opened", "assigned", "evidence_added", "escalated", "closed"
    actor: str = ""  # Who performed the action (agent_id or operator)
    detail: str = ""


class ComplianceCase(BaseModel):
    """Compliance investigation case with full lifecycle.

    Fields
    ------
    case_id : str
        Unique identifier.
    case_type : str
        Category: ``"wash_trade"``, ``"spoofing"``, ``"self_crossing"``, etc.
    severity : str
        ``"low"``, ``"medium"``, ``"high"``, ``"critical"``.
    status : str
        Current lifecycle state: ``"open"``, ``"investigating"``,
        ``"escalated"``, ``"closed"``.
    disposition : str
        Final outcome (only meaningful when closed):
        ``"confirmed"``, ``"false_positive"``, ``"inconclusive"``, ``""``.
    symbol : str
        Primary symbol involved.
    strategy_id : str
        Strategy that generated the suspicious activity.
    description : str
        Human-readable summary.
    evidence : list[dict]
        Structured evidence items (fills, orders, timestamps, etc.).
    timeline : list[TimelineEntry]
        Audit trail of all actions taken on the case.
    assigned_to : str
        Operator or team assigned to investigate.
    created_at : datetime
        When the case was opened.
    closed_at : datetime | None
        When the case was closed (None if still open).
    """

    case_id: str = Field(default_factory=new_id)
    case_type: str
    severity: str
    status: str = "open"
    disposition: str = ""
    symbol: str = ""
    strategy_id: str = ""
    description: str = ""
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    assigned_to: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    closed_at: datetime | None = None


# Valid status transitions
_VALID_TRANSITIONS: dict[str, set[str]] = {
    "open": {"investigating", "escalated", "closed"},
    "investigating": {"escalated", "closed"},
    "escalated": {"investigating", "closed"},
    "closed": set(),  # Terminal
}


# ---------------------------------------------------------------------------
# Case Manager
# ---------------------------------------------------------------------------


class CaseManager:
    """Manages compliance case lifecycle with optional JSONL persistence.

    Thread-safe.

    Parameters
    ----------
    persistence_path:
        Optional path to a JSONL file for durable storage.
        When provided, cases are appended on write and loaded on init.
    max_memory:
        Maximum number of cases to retain in memory.
    """

    def __init__(
        self,
        persistence_path: str | Path | None = None,
        max_memory: int = 10_000,
    ) -> None:
        self._lock = threading.Lock()
        self._cases: dict[str, ComplianceCase] = {}
        self._order: deque[str] = deque(maxlen=max_memory)
        self._max_memory = max_memory
        self._persistence_path = Path(persistence_path) if persistence_path else None

        if self._persistence_path is not None:
            self._load_from_disk()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def open_case(
        self,
        case_type: str,
        severity: str,
        *,
        case_id: str | None = None,
        symbol: str = "",
        strategy_id: str = "",
        description: str = "",
        evidence: list[dict[str, Any]] | None = None,
    ) -> ComplianceCase:
        """Create and store a new case.

        Returns the newly created :class:`ComplianceCase`.
        """
        case = ComplianceCase(
            case_id=case_id or new_id(),
            case_type=case_type,
            severity=severity,
            symbol=symbol,
            strategy_id=strategy_id,
            description=description,
            evidence=evidence or [],
            timeline=[
                TimelineEntry(
                    action="opened",
                    actor="surveillance",
                    detail=description,
                ),
            ],
        )
        with self._lock:
            self._store(case)

        logger.info(
            "Compliance case opened: case_id=%s type=%s severity=%s symbol=%s",
            case.case_id, case.case_type, case.severity, case.symbol,
        )
        return case

    def get(self, case_id: str) -> ComplianceCase | None:
        """Retrieve a case by ID."""
        with self._lock:
            return self._cases.get(case_id)

    def transition(
        self,
        case_id: str,
        new_status: str,
        *,
        actor: str = "",
        detail: str = "",
        disposition: str = "",
    ) -> ComplianceCase | None:
        """Transition a case to a new status.

        Returns the updated case, or ``None`` if the case doesn't exist
        or the transition is invalid.
        """
        with self._lock:
            case = self._cases.get(case_id)
            if case is None:
                logger.warning("Case not found: %s", case_id)
                return None

            valid = _VALID_TRANSITIONS.get(case.status, set())
            if new_status not in valid:
                logger.warning(
                    "Invalid transition: %s → %s for case %s",
                    case.status, new_status, case_id,
                )
                return None

            case.status = new_status
            case.timeline.append(TimelineEntry(
                action=new_status,
                actor=actor,
                detail=detail,
            ))

            if new_status == "closed":
                case.closed_at = utc_now()
                if disposition:
                    case.disposition = disposition

            self._persist(case)

        logger.info(
            "Case %s transitioned to %s by %s",
            case_id, new_status, actor,
        )
        return case

    def assign(
        self, case_id: str, assignee: str, *, actor: str = "",
    ) -> ComplianceCase | None:
        """Assign a case to an investigator."""
        with self._lock:
            case = self._cases.get(case_id)
            if case is None:
                return None

            case.assigned_to = assignee
            case.timeline.append(TimelineEntry(
                action="assigned",
                actor=actor,
                detail=f"Assigned to {assignee}",
            ))
            self._persist(case)

        return case

    def add_evidence(
        self,
        case_id: str,
        evidence: dict[str, Any],
        *,
        actor: str = "",
    ) -> ComplianceCase | None:
        """Append evidence to an existing case."""
        with self._lock:
            case = self._cases.get(case_id)
            if case is None:
                return None
            if case.status == "closed":
                logger.warning("Cannot add evidence to closed case %s", case_id)
                return None

            case.evidence.append(evidence)
            case.timeline.append(TimelineEntry(
                action="evidence_added",
                actor=actor,
                detail=f"Evidence item added: {evidence.get('type', 'unknown')}",
            ))
            self._persist(case)

        return case

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_open(self) -> list[ComplianceCase]:
        """Return all non-closed cases."""
        with self._lock:
            return [
                c for c in self._cases.values()
                if c.status != "closed"
            ]

    def list_by_status(self, status: str) -> list[ComplianceCase]:
        """Return cases with the given status."""
        with self._lock:
            return [c for c in self._cases.values() if c.status == status]

    def list_by_symbol(self, symbol: str) -> list[ComplianceCase]:
        """Return cases for a specific symbol."""
        with self._lock:
            return [c for c in self._cases.values() if c.symbol == symbol]

    @property
    def total_cases(self) -> int:
        with self._lock:
            return len(self._cases)

    @property
    def open_count(self) -> int:
        with self._lock:
            return sum(1 for c in self._cases.values() if c.status != "closed")

    # ------------------------------------------------------------------
    # Storage internals
    # ------------------------------------------------------------------

    def _store(self, case: ComplianceCase) -> None:
        """Store a case in memory and persist.  Caller holds lock."""
        # Evict oldest if at capacity
        if (
            case.case_id not in self._cases
            and len(self._order) == self._max_memory
        ):
            evicted_id = self._order[0]
            self._cases.pop(evicted_id, None)

        self._cases[case.case_id] = case
        if case.case_id not in self._order:
            self._order.append(case.case_id)

        self._persist(case)

    def _persist(self, case: ComplianceCase) -> None:
        """Append the case to the JSONL file.  Caller holds lock."""
        if self._persistence_path is None:
            return
        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with self._persistence_path.open("a") as f:
                f.write(case.model_dump_json() + "\n")
        except Exception:
            logger.debug("Failed to persist case %s", case.case_id, exc_info=True)

    def _load_from_disk(self) -> None:
        """Load cases from the JSONL file on startup."""
        if self._persistence_path is None or not self._persistence_path.exists():
            return
        try:
            with self._persistence_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    case = ComplianceCase.model_validate_json(line)
                    self._cases[case.case_id] = case
                    self._order.append(case.case_id)
            logger.info(
                "Loaded %d compliance cases from %s",
                len(self._cases), self._persistence_path,
            )
        except Exception:
            logger.warning(
                "Failed to load cases from %s",
                self._persistence_path, exc_info=True,
            )
