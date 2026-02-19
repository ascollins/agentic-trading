#!/usr/bin/env python3
"""
Risk-Reward Calculator for Multi-Target Crypto Setups
Computes R:R ratios, expected value, and scale-out P&L projections.
"""

import sys
import json


def calculate_rr(entry: float, stop_loss: float, targets: list, scale_out_pcts: list = None) -> dict:
    """
    Calculate risk-reward metrics for a trade setup with multiple targets.
    
    Args:
        entry: Entry price
        stop_loss: Stop loss price
        targets: List of target prices [TP1, TP2, TP3, ...]
        scale_out_pcts: Optional list of scale-out percentages per target (must sum to 100)
    """
    direction = "LONG" if entry > stop_loss else "SHORT"
    risk_per_unit = abs(entry - stop_loss)
    risk_pct = (risk_per_unit / entry) * 100
    
    if risk_per_unit == 0:
        return {"error": "Entry and stop loss cannot be the same"}
    
    if scale_out_pcts is None:
        n = len(targets)
        scale_out_pcts = [round(100 / n, 1)] * n
        scale_out_pcts[-1] = round(100 - sum(scale_out_pcts[:-1]), 1)
    
    if abs(sum(scale_out_pcts) - 100) > 0.01:
        return {"error": f"Scale-out percentages must sum to 100%, got {sum(scale_out_pcts)}%"}
    
    target_details = []
    cumulative_realised_r = 0
    
    for i, (tp, pct) in enumerate(zip(targets, scale_out_pcts)):
        if direction == "LONG":
            reward_per_unit = tp - entry
        else:
            reward_per_unit = entry - tp
        
        rr_ratio = reward_per_unit / risk_per_unit
        reward_pct = (reward_per_unit / entry) * 100
        weighted_r = rr_ratio * (pct / 100)
        cumulative_realised_r += weighted_r
        
        target_details.append({
            "target": i + 1,
            "price": tp,
            "reward_per_unit": round(reward_per_unit, 4),
            "reward_pct": round(reward_pct, 2),
            "rr_ratio": round(rr_ratio, 2),
            "scale_out_pct": pct,
            "weighted_r": round(weighted_r, 2)
        })
    
    # Expected value calculation (simplified: assumes equal probability of hitting each target)
    # In practice, closer targets have higher probability
    prob_weights = []
    for i in range(len(targets)):
        prob_weights.append(1.0 / (1.0 + i * 0.3))  # Decreasing probability for further targets
    total_weight = sum(prob_weights)
    prob_weights = [w / total_weight for w in prob_weights]
    
    expected_r = sum(td["rr_ratio"] * pw for td, pw in zip(target_details, prob_weights))
    
    return {
        "direction": direction,
        "entry": entry,
        "stop_loss": stop_loss,
        "risk_per_unit": round(risk_per_unit, 4),
        "risk_pct": round(risk_pct, 2),
        "targets": target_details,
        "blended_rr_if_all_hit": round(cumulative_realised_r, 2),
        "expected_r_probability_weighted": round(expected_r, 2),
        "breakeven_at_tp1": target_details[0]["rr_ratio"] >= 1.0,
        "meets_minimum_rr": target_details[0]["rr_ratio"] >= 2.0,
        "assessment": _assess_setup(target_details, direction, cumulative_realised_r)
    }


def _assess_setup(targets, direction, blended_rr):
    """Provide a qualitative assessment of the setup."""
    tp1_rr = targets[0]["rr_ratio"]
    
    if tp1_rr < 1.0:
        return "POOR — TP1 does not cover risk. Consider tighter stop or better entry."
    elif tp1_rr < 2.0:
        return "MARGINAL — TP1 R:R below 2:1. Acceptable only with high confluence and trend alignment."
    elif tp1_rr < 3.0:
        if blended_rr >= 3.0:
            return "GOOD — Solid R:R profile. TP1 covers risk, blended R:R is strong."
        return "DECENT — TP1 R:R is adequate. Look for additional confluence to increase conviction."
    else:
        return "EXCELLENT — Strong R:R across all targets. High-quality setup."


def project_pnl(account_size: float, risk_pct: float, rr_result: dict) -> dict:
    """Project P&L for each scenario given account size and risk."""
    risk_amount = account_size * (risk_pct / 100)
    
    scenarios = {
        "full_stop_loss": {
            "pnl_usd": round(-risk_amount, 2),
            "pnl_pct": round(-risk_pct, 2)
        }
    }
    
    running_pnl = 0
    remaining_pct = 100
    
    for t in rr_result["targets"]:
        profit_this_target = risk_amount * t["rr_ratio"] * (t["scale_out_pct"] / 100)
        running_pnl += profit_this_target
        remaining_pct -= t["scale_out_pct"]
        
        scenarios[f"tp{t['target']}_hit"] = {
            "realised_pnl_usd": round(running_pnl, 2),
            "realised_pnl_pct": round((running_pnl / account_size) * 100, 2),
            "remaining_position_pct": remaining_pct
        }
    
    scenarios["all_targets_hit"] = {
        "total_pnl_usd": round(running_pnl, 2),
        "total_pnl_pct": round((running_pnl / account_size) * 100, 2),
        "return_on_risk": round(running_pnl / risk_amount, 2)
    }
    
    return scenarios


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python rr_calculator.py <entry> <stop_loss> <targets_json> [scale_out_json]")
        print("  Example: python rr_calculator.py 95000 92000 '[98000,102000,108000]'")
        print("  With scale: python rr_calculator.py 95000 92000 '[98000,102000,108000]' '[40,35,25]'")
        sys.exit(1)
    
    entry = float(sys.argv[1])
    stop = float(sys.argv[2])
    targets = json.loads(sys.argv[3])
    scale_out = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
    
    result = calculate_rr(entry, stop, targets, scale_out)
    print(json.dumps(result, indent=2))
    
    # If account size provided via env var, also project P&L
    import os
    acct = os.environ.get("ACCOUNT_SIZE")
    risk = os.environ.get("RISK_PCT")
    if acct and risk:
        pnl = project_pnl(float(acct), float(risk), result)
        print("\n--- P&L Projection ---")
        print(json.dumps(pnl, indent=2))
