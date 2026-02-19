"""Backward-compat shim — canonical location is ``agentic_trading.bus``.

This package re-exports everything from the new location so that
existing ``from agentic_trading.event_bus.X import Y`` statements
continue to work.  Will be removed in PR 16.
"""
