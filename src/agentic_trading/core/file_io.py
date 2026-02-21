"""Safe file I/O utilities.

Provides atomic-ish append for JSONL files with file locking (``fcntl``)
and ``fsync`` to minimise data loss on crash or concurrent access.
"""

from __future__ import annotations

import fcntl
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def safe_append_line(path: Path, line: str) -> None:
    """Append a single line to a file with locking and fsync.

    * ``fcntl.LOCK_EX`` prevents interleaved writes from concurrent
      processes / async tasks that share the same file.
    * ``os.fsync`` ensures the data hits disk before the lock is
      released, so a crash immediately after return won't lose the line.
    * The caller is responsible for creating parent directories.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
