#!/usr/bin/env python3
"""
Position Sizer for Crypto Trading
Calculates optimal position size based on account size, risk percentage,
entry price, and stop loss level.
"""

import sys
import json


def calculate_position_size(account_size: float, risk_pct: float, entry: float, stop_loss: float) -> dict:
    """
    Calculate position size and related metrics.
    
    Args:
        account_size: Total trading capital in USD
        risk_pct: Risk per trade as percentage (e.g., 1.0 for 1%)
        entry: Planned entry price
        stop_loss: Stop loss price
    
    Returns:
        Dictionary with position sizing details
    """
    risk_amount = account_size * (risk_pct / 100)
    stop_distance = abs(entry - stop_loss)
    stop_distance_pct = (stop_distance / entry) * 100
    
    if stop_distance == 0:
        return {"error": "Entry and stop loss cannot be the same price"}
    
    position_size_units = risk_amount / stop_distance
    position_size_usd = position_size_units * entry
    leverage_implied = position_size_usd / account_size
    
    return {
        "account_size_usd": account_size,
        "risk_per_trade_pct": risk_pct,
        "risk_amount_usd": round(risk_amount, 2),
        "entry_price": entry,
        "stop_loss_price": stop_loss,
        "stop_distance_usd": round(stop_distance, 4),
        "stop_distance_pct": round(stop_distance_pct, 2),
        "position_size_units": round(position_size_units, 6),
        "position_size_usd": round(position_size_usd, 2),
        "implied_leverage": round(leverage_implied, 2),
        "direction": "LONG" if entry > stop_loss else "SHORT",
        "warning": "Implied leverage > 3x — consider reducing size" if leverage_implied > 3 else None
    }


def calculate_scaled_entries(account_size: float, risk_pct: float, entries: list, stop_loss: float) -> dict:
    """
    Calculate position sizing for scaled/laddered entries.
    
    Args:
        account_size: Total trading capital in USD
        risk_pct: Risk per trade as percentage
        entries: List of [price, allocation_pct] pairs e.g., [[95000, 40], [93000, 35], [91000, 25]]
        stop_loss: Common stop loss price
    """
    risk_amount = account_size * (risk_pct / 100)
    
    total_alloc = sum(e[1] for e in entries)
    if abs(total_alloc - 100) > 0.01:
        return {"error": f"Entry allocations must sum to 100%, got {total_alloc}%"}
    
    weighted_entry = sum(e[0] * (e[1] / 100) for e in entries)
    total_stop_distance = abs(weighted_entry - stop_loss)
    
    if total_stop_distance == 0:
        return {"error": "Weighted entry equals stop loss"}
    
    total_units = risk_amount / total_stop_distance
    total_usd = total_units * weighted_entry
    
    entry_details = []
    for price, alloc_pct in entries:
        units = total_units * (alloc_pct / 100)
        usd = units * price
        entry_details.append({
            "entry_price": price,
            "allocation_pct": alloc_pct,
            "units": round(units, 6),
            "usd_value": round(usd, 2)
        })
    
    return {
        "account_size_usd": account_size,
        "risk_per_trade_pct": risk_pct,
        "risk_amount_usd": round(risk_amount, 2),
        "weighted_avg_entry": round(weighted_entry, 2),
        "stop_loss_price": stop_loss,
        "total_position_units": round(total_units, 6),
        "total_position_usd": round(total_usd, 2),
        "implied_leverage": round(total_usd / account_size, 2),
        "entries": entry_details,
        "direction": "LONG" if weighted_entry > stop_loss else "SHORT"
    }


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python position_sizer.py <account_size> <risk_pct> <entry> <stop_loss>")
        print("  Scaled: python position_sizer.py <account_size> <risk_pct> <stop_loss> --scaled '<entries_json>'")
        print("  Example: python position_sizer.py 100000 1.5 95000 92000")
        print("  Scaled:  python position_sizer.py 100000 1.5 92000 --scaled '[[95000,40],[93000,35],[91000,25]]'")
        sys.exit(1)
    
    account = float(sys.argv[1])
    risk = float(sys.argv[2])
    
    if "--scaled" in sys.argv:
        stop = float(sys.argv[3])
        entries_json = sys.argv[sys.argv.index("--scaled") + 1]
        entries = json.loads(entries_json)
        result = calculate_scaled_entries(account, risk, entries, stop)
    else:
        entry = float(sys.argv[3])
        stop = float(sys.argv[4])
        result = calculate_position_size(account, risk, entry, stop)
    
    print(json.dumps(result, indent=2))
