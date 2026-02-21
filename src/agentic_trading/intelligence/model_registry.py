"""Model Registry — versioned model lifecycle management (design spec §6).

Tracks model versions with training provenance, performance metrics,
and a staged promotion pipeline:

    RESEARCH → PAPER → LIMITED → PRODUCTION

Each stage transition requires explicit approval and is logged with
the approver's identity and timestamp.

Storage: in-memory with optional JSONL persistence for durability.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentic_trading.core.ids import new_id, utc_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model stage
# ---------------------------------------------------------------------------


class ModelStage(str, Enum):
    """Lifecycle stages for a model version."""

    RESEARCH = "research"
    PAPER = "paper"
    LIMITED = "limited"
    PRODUCTION = "production"
    RETIRED = "retired"


# Valid promotions: key → set of valid next stages
_STAGE_TRANSITIONS: dict[ModelStage, set[ModelStage]] = {
    ModelStage.RESEARCH: {ModelStage.PAPER, ModelStage.RETIRED},
    ModelStage.PAPER: {ModelStage.LIMITED, ModelStage.RESEARCH, ModelStage.RETIRED},
    ModelStage.LIMITED: {ModelStage.PRODUCTION, ModelStage.PAPER, ModelStage.RETIRED},
    ModelStage.PRODUCTION: {ModelStage.LIMITED, ModelStage.RETIRED},
    ModelStage.RETIRED: set(),  # Terminal
}

_STAGE_RANK: dict[ModelStage, int] = {
    ModelStage.RESEARCH: 0,
    ModelStage.PAPER: 1,
    ModelStage.LIMITED: 2,
    ModelStage.PRODUCTION: 3,
    ModelStage.RETIRED: -1,
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class StageTransition(BaseModel):
    """Record of a stage change."""

    from_stage: str
    to_stage: str
    transitioned_at: datetime = Field(default_factory=utc_now)
    approved_by: str = ""
    reason: str = ""


class ModelRecord(BaseModel):
    """Versioned model record with training provenance and metrics.

    Fields
    ------
    model_id : str
        Unique identifier for this model version.
    name : str
        Human-readable model name (e.g., ``"btc_lstm_v3"``).
    version : int
        Monotonically increasing version number per name.
    stage : ModelStage
        Current lifecycle stage.
    training_data_hash : str
        Hash of the training dataset for reproducibility.
    hyperparameters : dict
        Model hyperparameters used during training.
    metrics : dict
        Performance metrics (MSE, Sharpe, hit rate, etc.).
    description : str
        Notes about this model version.
    created_at : datetime
        When this record was created.
    approved_by : str
        Who approved the current stage.
    approved_at : datetime | None
        When the current stage was approved.
    transitions : list[StageTransition]
        History of all stage transitions.
    tags : list[str]
        Free-form tags for filtering/search.
    """

    model_id: str = Field(default_factory=new_id)
    name: str
    version: int = 1
    stage: ModelStage = ModelStage.RESEARCH
    training_data_hash: str = ""
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    description: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    approved_by: str = ""
    approved_at: datetime | None = None
    transitions: list[StageTransition] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Registry for managing model versions and their lifecycle.

    Thread-safe.

    Parameters
    ----------
    persistence_path:
        Optional JSONL file for durable storage.
    """

    def __init__(
        self,
        persistence_path: str | Path | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, ModelRecord] = {}  # model_id → record
        self._by_name: dict[str, list[str]] = defaultdict(list)  # name → [model_ids]
        self._persistence_path = Path(persistence_path) if persistence_path else None

        if self._persistence_path is not None:
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        *,
        training_data_hash: str = "",
        hyperparameters: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        description: str = "",
        tags: list[str] | None = None,
    ) -> ModelRecord:
        """Register a new model version.

        The version number is auto-incremented based on existing records
        with the same name.

        Returns the new :class:`ModelRecord`.
        """
        with self._lock:
            existing = self._by_name.get(name, [])
            max_version = 0
            for mid in existing:
                rec = self._models.get(mid)
                if rec is not None:
                    max_version = max(max_version, rec.version)

            record = ModelRecord(
                name=name,
                version=max_version + 1,
                training_data_hash=training_data_hash,
                hyperparameters=hyperparameters or {},
                metrics=metrics or {},
                description=description,
                tags=tags or [],
            )

            self._models[record.model_id] = record
            self._by_name[name].append(record.model_id)
            self._persist(record)

        logger.info(
            "Model registered: %s v%d (model_id=%s, stage=%s)",
            name, record.version, record.model_id, record.stage.value,
        )
        return record

    # ------------------------------------------------------------------
    # Stage transitions
    # ------------------------------------------------------------------

    def promote(
        self,
        model_id: str,
        to_stage: ModelStage,
        *,
        approved_by: str = "",
        reason: str = "",
    ) -> ModelRecord | None:
        """Transition a model to a new stage.

        Returns the updated record, or ``None`` if the model doesn't
        exist or the transition is invalid.
        """
        with self._lock:
            record = self._models.get(model_id)
            if record is None:
                logger.warning("Model not found: %s", model_id)
                return None

            valid = _STAGE_TRANSITIONS.get(record.stage, set())
            if to_stage not in valid:
                logger.warning(
                    "Invalid stage transition: %s → %s for model %s",
                    record.stage.value, to_stage.value, model_id,
                )
                return None

            old_stage = record.stage
            record.stage = to_stage
            record.approved_by = approved_by
            record.approved_at = utc_now()
            record.transitions.append(StageTransition(
                from_stage=old_stage.value,
                to_stage=to_stage.value,
                approved_by=approved_by,
                reason=reason,
            ))
            self._persist(record)

        logger.info(
            "Model %s (%s v%d) promoted: %s → %s by %s",
            model_id, record.name, record.version,
            old_stage.value, to_stage.value, approved_by,
        )
        return record

    def retire(
        self, model_id: str, *, reason: str = "", actor: str = "",
    ) -> ModelRecord | None:
        """Retire a model (terminal state)."""
        return self.promote(
            model_id, ModelStage.RETIRED,
            approved_by=actor, reason=reason,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, model_id: str) -> ModelRecord | None:
        """Get a model record by ID."""
        with self._lock:
            return self._models.get(model_id)

    def get_latest(self, name: str) -> ModelRecord | None:
        """Get the latest version of a model by name."""
        with self._lock:
            ids = self._by_name.get(name, [])
            if not ids:
                return None
            best = None
            for mid in ids:
                rec = self._models.get(mid)
                if rec is not None:
                    if best is None or rec.version > best.version:
                        best = rec
            return best

    def get_production(self, name: str) -> ModelRecord | None:
        """Get the current production version of a model by name."""
        with self._lock:
            ids = self._by_name.get(name, [])
            for mid in ids:
                rec = self._models.get(mid)
                if rec is not None and rec.stage == ModelStage.PRODUCTION:
                    return rec
            return None

    def list_by_name(self, name: str) -> list[ModelRecord]:
        """List all versions of a model, ordered by version."""
        with self._lock:
            ids = self._by_name.get(name, [])
            records = [self._models[mid] for mid in ids if mid in self._models]
            return sorted(records, key=lambda r: r.version)

    def list_by_stage(self, stage: ModelStage) -> list[ModelRecord]:
        """List all models at a given stage."""
        with self._lock:
            return [r for r in self._models.values() if r.stage == stage]

    def list_all(self) -> list[ModelRecord]:
        """List all model records."""
        with self._lock:
            return list(self._models.values())

    def update_metrics(
        self, model_id: str, metrics: dict[str, float],
    ) -> ModelRecord | None:
        """Update a model's performance metrics."""
        with self._lock:
            record = self._models.get(model_id)
            if record is None:
                return None
            record.metrics.update(metrics)
            self._persist(record)
        return record

    @property
    def total_models(self) -> int:
        with self._lock:
            return len(self._models)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self, record: ModelRecord) -> None:
        """Append record to JSONL file.  Caller holds lock."""
        if self._persistence_path is None:
            return
        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with self._persistence_path.open("a") as f:
                f.write(record.model_dump_json() + "\n")
        except Exception:
            logger.debug(
                "Failed to persist model %s", record.model_id, exc_info=True,
            )

    def _load_from_disk(self) -> None:
        """Load records from JSONL file.  Called once at init (no lock needed)."""
        if self._persistence_path is None or not self._persistence_path.exists():
            return
        try:
            loaded: dict[str, ModelRecord] = {}
            with self._persistence_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = ModelRecord.model_validate_json(line)
                    loaded[record.model_id] = record

            # Deduplicate: latest line for each model_id wins
            for record in loaded.values():
                self._models[record.model_id] = record
                if record.model_id not in self._by_name[record.name]:
                    self._by_name[record.name].append(record.model_id)

            logger.info(
                "Loaded %d model records from %s",
                len(self._models), self._persistence_path,
            )
        except Exception:
            logger.warning(
                "Failed to load model registry from %s",
                self._persistence_path, exc_info=True,
            )
