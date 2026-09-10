"""
kse/liquidity.py
=================
Liquidity-adjusted position sizing.

For small portfolios, the flat 20% max weight is fine. But for larger
portfolios, illiquid stocks can't be exited without slippage. This module
computes a per-stock max weight based on average daily volume.
"""

import numpy as np
import pandas as pd
from typing import Dict


def compute_liquidity_scores(
    volume_data: pd.Series,
    portfolio_size: float,
    max_participation: float = 0.10,
    hard_cap: float = 0.20,
) -> pd.DataFrame:
    """
    Compute liquidity-adjusted max weights and days-to-liquidate.

    Parameters
    ----------
    volume_data : pd.Series
        Average daily volume in PKR, indexed by ticker.
    portfolio_size : float
        Total portfolio value in PKR.
    max_participation : float
        Max % of daily volume you're willing to trade (default 10%).
    hard_cap : float
        Absolute max weight regardless of liquidity (default 20%).

    Returns
    -------
    pd.DataFrame indexed by ticker with columns:
        - Avg_Daily_Volume
        - Max_Weight — liquidity-adjusted max weight
        - Days_To_Liquidate — days to exit a max-weight position
    """
    rows = []
    for ticker, adv in volume_data.items():
        if pd.isna(adv) or adv <= 0:
            rows.append({
                "Ticker": ticker,
                "Avg_Daily_Volume": 0,
                "Max_Weight": 0.05,  # conservative default for missing data
                "Days_To_Liquidate": 999,
            })
            continue

        # Max position size = max_participation * ADV
        max_position_pkr = max_participation * adv
        max_weight = min(hard_cap, max_position_pkr / portfolio_size) if portfolio_size > 0 else hard_cap

        # Days to liquidate a full max_weight position
        position_pkr = max_weight * portfolio_size
        days_to_liquidate = position_pkr / (max_participation * adv) if adv > 0 else 999

        rows.append({
            "Ticker": ticker,
            "Avg_Daily_Volume": adv,
            "Max_Weight": max_weight,
            "Days_To_Liquidate": days_to_liquidate,
        })

    return pd.DataFrame(rows).set_index("Ticker")


def get_liquidity_bounds(
    volume_data: pd.Series,
    portfolio_size: float,
    max_participation: float = 0.10,
    hard_cap: float = 0.20,
) -> Dict[str, tuple]:
    """
    Get per-stock (min, max) weight bounds for the optimizer.

    Returns
    -------
    dict mapping ticker -> (0.0, max_weight)
    """
    scores = compute_liquidity_scores(volume_data, portfolio_size, max_participation, hard_cap)
    return {ticker: (0.0, row["Max_Weight"]) for ticker, row in scores.iterrows()}