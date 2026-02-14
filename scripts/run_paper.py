#!/usr/bin/env python3
"""Run paper trading."""

import sys
sys.path.insert(0, "src")

from agentic_trading.cli import main

if __name__ == "__main__":
    main(["paper"] + sys.argv[1:], standalone_mode=False)
