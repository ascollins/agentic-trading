"""Policy store: versioned policy management with persistence.

Manages multiple :class:`PolicySet` instances with versioning,
rollback, and optional file-based persistence.

Usage::

    store = PolicyStore(persist_dir="data/policies")
    store.save(policy_set)
    store.activate("pre_trade_risk_v1", version=2)
    current = store.get_active("pre_trade_risk_v1")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .policy_models import PolicyMode, PolicySet

logger = logging.getLogger(__name__)


class PolicyStore:
    """Manages versioned policy sets with optional file persistence.

    Features:
    - Multiple versions per policy set ID
    - Activate/rollback specific versions
    - JSON file persistence (one file per version)
    - Shadow/enforced mode switching
    """

    def __init__(self, persist_dir: str | None = None) -> None:
        # versions[set_id] -> {version: PolicySet}
        self._versions: dict[str, dict[int, PolicySet]] = {}
        # active[set_id] -> version number
        self._active: dict[str, int] = {}
        self._persist_dir = Path(persist_dir) if persist_dir else None

        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save / retrieve
    # ------------------------------------------------------------------

    def save(self, policy_set: PolicySet, *, activate: bool = True) -> None:
        """Save a policy set version. Optionally activate it."""
        sid = policy_set.set_id
        ver = policy_set.version

        if sid not in self._versions:
            self._versions[sid] = {}

        self._versions[sid][ver] = policy_set

        if activate:
            self._active[sid] = ver

        # Persist to disk
        if self._persist_dir is not None:
            self._persist_to_file(policy_set)

        logger.info(
            "PolicyStore: saved %s v%d (%s)%s",
            sid,
            ver,
            policy_set.mode.value,
            " [active]" if activate else "",
        )

    def get_active(self, set_id: str) -> PolicySet | None:
        """Get the currently active version of a policy set."""
        ver = self._active.get(set_id)
        if ver is None:
            return None
        return self._versions.get(set_id, {}).get(ver)

    def get_version(self, set_id: str, version: int) -> PolicySet | None:
        """Get a specific version of a policy set."""
        return self._versions.get(set_id, {}).get(version)

    def list_versions(self, set_id: str) -> list[int]:
        """List all saved versions for a policy set."""
        return sorted(self._versions.get(set_id, {}).keys())

    def active_version(self, set_id: str) -> int | None:
        """Return the active version number for a policy set."""
        return self._active.get(set_id)

    @property
    def active_sets(self) -> dict[str, int]:
        """Map of set_id -> active version for all sets."""
        return dict(self._active)

    # ------------------------------------------------------------------
    # Version management
    # ------------------------------------------------------------------

    def activate(self, set_id: str, version: int) -> PolicySet | None:
        """Activate a specific version. Returns the activated set or None."""
        ps = self.get_version(set_id, version)
        if ps is None:
            logger.warning(
                "PolicyStore: cannot activate %s v%d (not found)",
                set_id,
                version,
            )
            return None

        self._active[set_id] = version
        logger.info("PolicyStore: activated %s v%d", set_id, version)
        return ps

    def rollback(self, set_id: str) -> PolicySet | None:
        """Roll back to the previous version. Returns it or None."""
        versions = self.list_versions(set_id)
        current = self._active.get(set_id)

        if current is None or len(versions) < 2:
            logger.warning("PolicyStore: no previous version for %s", set_id)
            return None

        idx = versions.index(current) if current in versions else -1
        if idx <= 0:
            logger.warning("PolicyStore: already at earliest version for %s", set_id)
            return None

        prev = versions[idx - 1]
        return self.activate(set_id, prev)

    def set_mode(self, set_id: str, mode: PolicyMode) -> bool:
        """Switch the active version's mode between shadow/enforced."""
        ps = self.get_active(set_id)
        if ps is None:
            return False

        ps.mode = mode
        ps.updated_at = datetime.now(timezone.utc)

        if self._persist_dir is not None:
            self._persist_to_file(ps)

        logger.info(
            "PolicyStore: %s v%d mode -> %s",
            set_id,
            ps.version,
            mode.value,
        )
        return True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_to_file(self, policy_set: PolicySet) -> Path:
        """Save a policy set to a JSON file."""
        if self._persist_dir is None:
            raise RuntimeError("No persist directory configured")

        filename = f"{policy_set.set_id}_v{policy_set.version}.json"
        filepath = self._persist_dir / filename

        with open(filepath, "w") as f:
            json.dump(policy_set.model_dump(mode="json"), f, indent=2, default=str)

        return filepath

    def load_from_dir(self) -> int:
        """Load all policy sets from the persist directory.

        Returns the number of policy sets loaded.
        """
        if self._persist_dir is None or not self._persist_dir.exists():
            return 0

        loaded = 0
        for filepath in sorted(self._persist_dir.glob("*_v*.json")):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                ps = PolicySet.model_validate(data)
                self.save(ps, activate=True)
                loaded += 1
            except Exception:
                logger.warning(
                    "PolicyStore: failed to load %s", filepath.name,
                    exc_info=True,
                )

        logger.info("PolicyStore: loaded %d policy sets from disk", loaded)
        return loaded

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> list[dict[str, Any]]:
        """Return a summary of all policy sets for logging/UI."""
        result = []
        for set_id, versions in self._versions.items():
            active_ver = self._active.get(set_id)
            for ver, ps in sorted(versions.items()):
                result.append({
                    "set_id": set_id,
                    "version": ver,
                    "name": ps.name,
                    "mode": ps.mode.value,
                    "rules": len(ps.active_rules),
                    "active": ver == active_ver,
                })
        return result
