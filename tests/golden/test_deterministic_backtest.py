"""Golden test: deterministic backtest verification.

Verifies that running the backtest engine twice with the same inputs
produces byte-identical outputs.  This guards against non-determinism
creeping into the pipeline (e.g. dict ordering, floating point drift,
time-dependent seeds).
"""

import pytest


class TestDeterministicBacktest:
    """Golden test: same inputs -> same output hash."""

    @pytest.mark.skip(reason="Requires historical data fixture")
    def test_deterministic_hash(self):
        """Run backtest twice with same seed, verify identical hash."""
        # TODO: Once historical data fixture is available:
        # 1. Run BacktestEngine with fixed seed
        # 2. Capture result hash
        # 3. Run again with same seed
        # 4. Verify hashes match
        pass

    @pytest.mark.skip(reason="Requires historical data fixture")
    def test_golden_data_matches_reference(self):
        """Compare backtest output against a stored golden reference file.

        Steps:
        1. Load golden reference from tests/golden/golden_data/
        2. Run backtest with the same config and seed
        3. Compare output hash against the stored reference hash
        4. If mismatch, report which fields diverged
        """
        # TODO: Once golden reference data is generated:
        # 1. Load reference hash from golden_data/reference_hash.json
        # 2. Run BacktestEngine with matching config
        # 3. Compare hashes
        # 4. On failure, diff the result structures to identify divergence
        pass
