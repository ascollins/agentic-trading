#!/usr/bin/env python3
"""Run live trading.

REQUIRES:
  - Environment variable: I_UNDERSTAND_LIVE_TRADING=true
  - CLI flag: --live

Example:
  I_UNDERSTAND_LIVE_TRADING=true python scripts/run_live.py --live --config configs/live.toml
"""

import sys
sys.path.insert(0, "src")

from agentic_trading.cli import main

if __name__ == "__main__":
    main(["live"] + sys.argv[1:], standalone_mode=False)
