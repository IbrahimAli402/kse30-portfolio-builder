"""
kse/rebalance.py
=================
Rebalancing guidance engine.

Shows:
    - Portfolio drift from target weights
    - When to rebalance (threshold vs calendar)
    - Cost of rebalancing (transaction costs + tax)
"""

import numpy as np
import pandas as pd


def analyze_drift(
    target_weights: dict,
    current_weights: dict,
    threshold: float = 0.05,
) -> dict:
    """
    Analyze how far a portfolio has drifted from target weights.

    Parameters
    ----------
    target_weights : dict
        Ticker → target weight (0-1).
    current_weights : dict
        Ticker → current weight (0-1).
    threshold : float
        Rebalancing trigger threshold (default 5% absolute drift).

    Returns
    -------
    dict with:
        - drift_table : DataFrame
        - max_drift_ticker : str
        - max_drift : float
        - needs_rebalance : bool
        - n_breached : int
    """
    tickers = sorted(set(list(target_weights.keys()) + list(current_weights.keys())))
    rows = []
    for t in tickers:
        target = target_weights.get(t, 0)
        current = current_weights.get(t, 0)
        drift = current - target
        breached = abs(drift) > threshold
        rows.append({
            "Ticker": t,
            "Target": target,
            "Current": current,
            "Drift": drift,
            "Abs_Drift": abs(drift),
            "Breached": breached,
        })

    df = pd.DataFrame(rows).sort_values("Abs_Drift", ascending=False)
    max_row = df.iloc[0]

    return {
        "drift_table": df,
        "max_drift_ticker": max_row["Ticker"],
        "max_drift": max_row["Drift"],
        "needs_rebalance": bool(df["Breached"].any()),
        "n_breached": int(df["Breached"].sum()),
    }


def rebalancing_cost(
    target_weights: dict,
    current_weights: dict,
    portfolio_value: float,
    tx_cost_pct: float = 0.002,
    cgt_rate: float = 0.15,
    avg_gain_pct: float = 0.20,
) -> dict:
    """
    Estimate the cost of rebalancing.

    Parameters
    ----------
    target_weights : dict
    current_weights : dict
    portfolio_value : float
        Total portfolio value in PKR.
    tx_cost_pct : float
        Transaction cost per trade (default 0.2%).
    cgt_rate : float
        Capital gains tax rate (default 15% for <12mo).
    avg_gain_pct : float
        Average gain on positions being sold (default 20%).

    Returns
    -------
    dict with:
        - trades_needed : list of dicts
        - total_turnover : float
        - tx_cost : float
        - cgt_cost : float
        - total_cost : float
        - total_cost_pct : float
    """
    trades = []
    total_turnover = 0.0

    for ticker, target_w in target_weights.items():
        current_w = current_weights.get(ticker, 0)
        diff = target_w - current_w  # positive = buy, negative = sell
        if abs(diff) < 0.001:
            continue
        trade_value = abs(diff) * portfolio_value
        total_turnover += trade_value
        trades.append({
            "Ticker": ticker,
            "Action": "Buy" if diff > 0 else "Sell",
            "Weight_Change": diff,
            "Trade_Value": trade_value,
        })

    tx_cost = total_turnover * tx_cost_pct
    # CGT only on sells, only on the gain portion
    sell_value = sum(t["Trade_Value"] for t in trades if t["Action"] == "Sell")
    gain_on_sells = sell_value * avg_gain_pct
    cgt_cost = gain_on_sells * cgt_rate
    total_cost = tx_cost + cgt_cost
    total_cost_pct = total_cost / portfolio_value if portfolio_value > 0 else 0

    return {
        "trades_needed": trades,
        "total_turnover": total_turnover,
        "tx_cost": tx_cost,
        "cgt_cost": cgt_cost,
        "total_cost": total_cost,
        "total_cost_pct": total_cost_pct,
    }