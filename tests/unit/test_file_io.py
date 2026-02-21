"""Unit tests for core.file_io — safe JSONL append with file locking."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from agentic_trading.core.file_io import safe_append_line


class TestSafeAppendLine:
    """Tests for safe_append_line utility."""

    def test_appends_single_line(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        safe_append_line(path, '{"key": "value"}')

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"key": "value"}

    def test_appends_multiple_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        safe_append_line(path, '{"n": 1}')
        safe_append_line(path, '{"n": 2}')
        safe_append_line(path, '{"n": 3}')

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3
        for i, line in enumerate(lines, 1):
            assert json.loads(line) == {"n": i}

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "c" / "test.jsonl"
        safe_append_line(path, '{"nested": true}')

        assert path.exists()
        assert json.loads(path.read_text().strip()) == {"nested": True}

    def test_concurrent_writes_no_corruption(self, tmp_path: Path) -> None:
        """50 threads writing simultaneously — all lines must be valid JSON."""
        path = tmp_path / "concurrent.jsonl"
        n_threads = 50

        def writer(thread_id: int) -> None:
            for i in range(10):
                line = json.dumps({"thread": thread_id, "seq": i})
                safe_append_line(path, line)

        threads = [
            threading.Thread(target=writer, args=(tid,))
            for tid in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Every single line must be valid JSON — no interleaving
        lines = path.read_text().strip().split("\n")
        assert len(lines) == n_threads * 10

        parsed = []
        for i, line in enumerate(lines):
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"Line {i} is not valid JSON (file corruption): {line!r}"
                ) from exc

        # All 500 entries present (order may vary)
        expected = {
            (d["thread"], d["seq"]) for d in parsed
        }
        assert len(expected) == n_threads * 10

    def test_existing_file_not_overwritten(self, tmp_path: Path) -> None:
        """Appending to an existing file preserves prior content."""
        path = tmp_path / "test.jsonl"
        path.write_text('{"existing": true}\n')

        safe_append_line(path, '{"new": true}')

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"existing": True}
        assert json.loads(lines[1]) == {"new": True}
