"""Deterministic backtest hash verification.

Ensures that running the same backtest with the same inputs
produces identical results every time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .results import BacktestResult


class GoldenTestVerifier:
    """Verifies backtest determinism via hash comparison."""

    def __init__(self, golden_dir: str = "tests/golden/golden_data") -> None:
        self._golden_dir = Path(golden_dir)
        self._golden_dir.mkdir(parents=True, exist_ok=True)

    def save_golden(self, test_name: str, result: BacktestResult) -> str:
        """Save a golden result for future comparison."""
        data = {
            "deterministic_hash": result.deterministic_hash,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "total_trades": result.total_trades,
            "total_fees": result.total_fees,
            "equity_curve_length": len(result.equity_curve),
            "equity_final": result.equity_curve[-1] if result.equity_curve else 0,
        }

        path = self._golden_dir / f"{test_name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return result.deterministic_hash

    def verify(self, test_name: str, result: BacktestResult) -> tuple[bool, str]:
        """Verify a backtest result against its golden hash.

        Returns (passed, message).
        """
        path = self._golden_dir / f"{test_name}.json"
        if not path.exists():
            return False, f"No golden data found for '{test_name}'. Run with --save-golden first."

        with open(path) as f:
            golden = json.load(f)

        expected_hash = golden["deterministic_hash"]
        actual_hash = result.deterministic_hash

        if expected_hash == actual_hash:
            return True, f"Deterministic hash matches: {actual_hash}"

        # Detailed diff
        diffs = []
        if result.total_return != golden.get("total_return"):
            diffs.append(
                f"total_return: expected={golden.get('total_return')}, got={result.total_return}"
            )
        if result.total_trades != golden.get("total_trades"):
            diffs.append(
                f"total_trades: expected={golden.get('total_trades')}, got={result.total_trades}"
            )

        return False, (
            f"Hash mismatch: expected={expected_hash}, got={actual_hash}. "
            f"Diffs: {'; '.join(diffs) if diffs else 'unknown'}"
        )
