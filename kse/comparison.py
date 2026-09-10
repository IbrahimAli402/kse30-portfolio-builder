"""
kse/comparison.py
=================
Portfolio comparison engine.

Compares two portfolios side by side: returns, risk, Sharpe, max drawdown,
sector exposure, and overlapping growth chart.
"""

import numpy as np
import pandas as pd
from kse.risk_metrics import compute_drawdown


def compare_portfolios(
    weights_a: dict,
    weights_b: dict,
    returns: pd.DataFrame,
    labels: tuple = ("Portfolio A", "Portfolio B"),
) -> dict:
    """
    Compare two portfolios.

    Parameters
    ----------
    weights_a : dict
        Ticker → weight for portfolio A.
    weights_b : dict
        Ticker → weight for portfolio B.
    returns : pd.DataFrame
        Stock returns matrix (columns = tickers).
    labels : tuple
        (label_a, label_b).

    Returns
    -------
    dict with:
        - stats_table : pd.DataFrame (metric, A, B, difference)
        - port_a_returns : pd.Series
        - port_b_returns : pd.Series
        - cumulative_a : pd.Series
        - cumulative_b : pd.Series
    """
    # Align weights to available stocks
    available = returns.columns
    w_a = pd.Series(weights_a).reindex(available).fillna(0)
    w_b = pd.Series(weights_b).reindex(available).fillna(0)

    # Normalize
    if w_a.sum() > 0:
        w_a = w_a / w_a.sum()
    if w_b.sum() > 0:
        w_b = w_b / w_b.sum()

    # Compute portfolio returns
    port_a_returns = (returns * w_a).sum(axis=1)
    port_b_returns = (returns * w_b).sum(axis=1)

    # Compute stats
    stats_a = _compute_stats(port_a_returns)
    stats_b = _compute_stats(port_b_returns)

    # Build comparison table
    metrics = ["Annual Return", "Volatility", "Sharpe", "Max Drawdown", "Cumulative Return"]
    rows = []
    for m in metrics:
        val_a = stats_a.get(m, 0)
        val_b = stats_b.get(m, 0)
        diff = val_a - val_b
        rows.append({"Metric": m, labels[0]: val_a, labels[1]: val_b, "Difference": diff})

    stats_table = pd.DataFrame(rows)

    # Cumulative growth (indexed to 100)
    cum_a = (1 + port_a_returns).cumprod() * 100
    cum_b = (1 + port_b_returns).cumprod() * 100

    return {
        "stats_table": stats_table,
        "port_a_returns": port_a_returns,
        "port_b_returns": port_b_returns,
        "cumulative_a": cum_a,
        "cumulative_b": cum_b,
        "stats_a": stats_a,
        "stats_b": stats_b,
    }


def _compute_stats(returns: pd.Series) -> dict:
    """Compute standard risk-return statistics."""
    annual_return = (1 + returns.mean()) ** 12 - 1
    annual_vol = returns.std() * np.sqrt(12)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    dd = compute_drawdown(returns)
    cumulative = (1 + returns).prod() - 1

    return {
        "Annual Return": annual_return,
        "Volatility": annual_vol,
        "Sharpe": sharpe,
        "Max Drawdown": dd["max_drawdown"],
        "Cumulative Return": cumulative,
    }