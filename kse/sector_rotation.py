"""
kse/sector_rotation.py
=======================
Sector rotation analysis engine.

Computes sector-level returns from the stock return matrix,
then calculates relative strength vs the KSE-100 index
to identify leading and lagging sectors.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Optional


_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _load_sector_map(path: Optional[str] = None) -> Dict[str, str]:
    """Load ticker → sector mapping."""
    if path is None:
        path = _CONFIG_DIR / "sectors.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("sectors", data)


def compute_sector_returns(
    stock_returns: pd.DataFrame,
    sector_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Compute equal-weighted sector returns from individual stock returns.

    Parameters
    ----------
    stock_returns : pd.DataFrame
        Monthly stock returns, columns = tickers.
    sector_map : dict, optional
        Ticker → sector mapping. Loaded from config if not provided.

    Returns
    -------
    pd.DataFrame — monthly returns by sector (columns = sectors)
    """
    if sector_map is None:
        sector_map = _load_sector_map()

    # Group stocks by sector
    sector_stocks: Dict[str, List[str]] = {}
    for ticker, sector in sector_map.items():
        if ticker in stock_returns.columns:
            sector_stocks.setdefault(sector, []).append(ticker)

    # Compute equal-weighted sector returns
    sector_returns = pd.DataFrame(index=stock_returns.index)
    for sector, tickers in sector_stocks.items():
        if len(tickers) > 0:
            sector_returns[sector] = stock_returns[tickers].mean(axis=1)

    return sector_returns


def compute_sector_rotation(
    stock_returns: pd.DataFrame,
    index_returns: pd.Series,
    sector_map: Optional[Dict[str, str]] = None,
    periods: List[int] = [1, 3, 6, 12],
) -> pd.DataFrame:
    """
    Compute sector rotation metrics: returns, relative strength, momentum.

    Parameters
    ----------
    stock_returns : pd.DataFrame
        Monthly stock returns, columns = tickers.
    index_returns : pd.Series
        Monthly KSE-100 returns.
    sector_map : dict, optional
        Ticker → sector mapping.
    periods : list of int
        Return periods in months (default: 1, 3, 6, 12).

    Returns
    -------
    pd.DataFrame indexed by Sector with columns:
        - {p}M_Return  — raw sector return for each period
        - {p}M_RS       — relative strength vs index
        - Volatility   — annualized
        - Sharpe       — annualized (rf=0)
        - Momentum_Score — weighted composite of relative strengths
    """
    sector_returns = compute_sector_returns(stock_returns, sector_map)

    results = []
    for sector in sector_returns.columns:
        rets = sector_returns[sector].dropna()

        row = {"Sector": sector}

        # Returns and relative strength for each period
        for p in periods:
            if len(rets) >= p:
                sector_ret = (1 + rets.tail(p)).prod() - 1
                row[f"{p}M_Return"] = sector_ret

                if len(index_returns) >= p:
                    idx_ret = (1 + index_returns.tail(p)).prod() - 1
                    row[f"{p}M_RS"] = sector_ret / idx_ret if idx_ret != 0 else np.nan
                else:
                    row[f"{p}M_RS"] = np.nan
            else:
                row[f"{p}M_Return"] = np.nan
                row[f"{p}M_RS"] = np.nan

        # Volatility (annualized)
        row["Volatility"] = rets.std() * np.sqrt(12) if len(rets) > 0 else np.nan

        # Sharpe (annualized, rf=0)
        annual_return = (1 + rets.mean()) ** 12 - 1 if len(rets) > 0 else np.nan
        row["Sharpe"] = annual_return / row["Volatility"] if row["Volatility"] and row["Volatility"] > 0 else 0

        results.append(row)

    df = pd.DataFrame(results).set_index("Sector")

    # Composite momentum score: weighted average of relative strengths
    # More weight on short-term (1M) for tactical signal
    weights = {1: 0.40, 3: 0.30, 6: 0.20, 12: 0.10}
    df["Momentum_Score"] = sum(
        df[f"{p}M_RS"] * w for p, w in weights.items() if f"{p}M_RS" in df.columns
    )

    return df.sort_values("Momentum_Score", ascending=False)