"""
kse/mutual_funds.py
====================
Mutual fund comparison engine.

Compares a DIY portfolio against typical Pakistani mutual fund categories,
showing the fee drag and long-term wealth destruction from high expense ratios.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List


_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _load_fund_config() -> Dict:
    """Load mutual fund category data from YAML."""
    path = _CONFIG_DIR / "mutual_funds.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("categories", {})


def compare_to_mutual_funds(
    diy_annual_return: float,
    diy_annual_fee: float,
    monthly_amount: float,
    horizon_years: int,
    tx_cost_pct: float = 0.002,
) -> Dict:
    """
    Compare a DIY portfolio against mutual fund categories.

    Parameters
    ----------
    diy_annual_return : float
        Expected annual return of the DIY portfolio (gross of fees).
    diy_annual_fee : float
        Annual management fee for the DIY portfolio (e.g., 0.015 = 1.5%).
    monthly_amount : float
        Monthly SIP contribution.
    horizon_years : int
        Investment horizon in years.
    tx_cost_pct : float
        Transaction cost per deposit (default 0.2%).

    Returns
    -------
    dict with:
        - comparison_table : pd.DataFrame
        - diy_terminal : float
        - fee_drag_analysis : dict
    """
    categories = _load_fund_config()
    if not categories:
        raise FileNotFoundError("Mutual fund config not found. Create config/mutual_funds.yaml")

    months = horizon_years * 12
    net_inv = monthly_amount * (1 - tx_cost_pct)

    results = []

    # DIY portfolio
    diy_net_return = diy_annual_return - diy_annual_fee
    diy_monthly_rate = (1 + diy_net_return) ** (1 / 12) - 1
    diy_pv = 0.0
    for m in range(1, months + 1):
        diy_pv = diy_pv * (1 + diy_monthly_rate) + net_inv
    diy_invested = monthly_amount * months
    diy_profit = diy_pv - diy_invested

    results.append({
        "Option": "Your DIY Portfolio",
        "Gross_Return": diy_annual_return,
        "Fee": diy_annual_fee,
        "Net_Return": diy_net_return,
        "Terminal_Value": diy_pv,
        "Total_Invested": diy_invested,
        "Profit": diy_profit,
        "Fee_Paid": diy_invested * diy_annual_fee * horizon_years,  # approx
    })

    # Mutual fund categories
    for key, cat in categories.items():
        gross_ret = cat["annual_return"]
        fee = cat["annual_fee"]
        net_ret = gross_ret - fee
        monthly_rate = (1 + net_ret) ** (1 / 12) - 1

        pv = 0.0
        for m in range(1, months + 1):
            pv = pv * (1 + monthly_rate) + net_inv

        invested = monthly_amount * months
        profit = pv - invested
        fee_paid = invested * fee * horizon_years  # approx

        results.append({
            "Option": cat["label"],
            "Gross_Return": gross_ret,
            "Fee": fee,
            "Net_Return": net_ret,
            "Terminal_Value": pv,
            "Total_Invested": invested,
            "Profit": profit,
            "Fee_Paid": fee_paid,
        })

    df = pd.DataFrame(results)

    # Fee drag analysis: DIY vs most comparable fund (equity)
    diy_row = df[df["Option"] == "Your DIY Portfolio"].iloc[0]
    equity_fund = df[df["Option"].str.contains("Equity Fund")].iloc[0] if len(df[df["Option"].str.contains("Equity Fund")]) > 0 else df.iloc[1]

    fee_diff = equity_fund["Fee"] - diy_row["Fee"]
    terminal_diff = diy_row["Terminal_Value"] - equity_fund["Terminal_Value"]
    terminal_diff_pct = (diy_row["Terminal_Value"] / equity_fund["Terminal_Value"] - 1) if equity_fund["Terminal_Value"] > 0 else 0

    return {
        "comparison_table": df,
        "diy_terminal": diy_pv,
        "fee_drag_analysis": {
            "diy_fee": diy_row["Fee"],
            "equity_fund_fee": equity_fund["Fee"],
            "fee_difference": fee_diff,
            "terminal_difference": terminal_diff,
            "terminal_difference_pct": terminal_diff_pct,
        },
    }


def fee_impact_chart(
    monthly_amount: float,
    horizon_years: int,
    gross_return: float,
    fee_range: List[float] = None,
) -> pd.DataFrame:
    """
    Show how different fee levels destroy long-term wealth.

    Parameters
    ----------
    monthly_amount : float
    horizon_years : int
    gross_return : float
        Gross annual return before fees.
    fee_range : list of float
        Fee levels to test (default: 0%, 0.5%, 1.0%, 1.5%, 2.0%, 2.5%, 3.0%).

    Returns
    -------
    pd.DataFrame with Fee, Terminal_Value, Wealth_Destroyed, Destroyed_Pct
    """
    if fee_range is None:
        fee_range = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]

    months = horizon_years * 12
    net_inv = monthly_amount

    rows = []
    # Baseline (no fee)
    net_ret = gross_return
    monthly_rate = (1 + net_ret) ** (1 / 12) - 1
    baseline_pv = 0.0
    for m in range(1, months + 1):
        baseline_pv = baseline_pv * (1 + monthly_rate) + net_inv

    for fee in fee_range:
        net_ret = gross_return - fee
        monthly_rate = (1 + net_ret) ** (1 / 12) - 1
        pv = 0.0
        for m in range(1, months + 1):
            pv = pv * (1 + monthly_rate) + net_inv

        destroyed = baseline_pv - pv
        destroyed_pct = destroyed / baseline_pv if baseline_pv > 0 else 0

        rows.append({
            "Fee": fee,
            "Terminal_Value": pv,
            "Wealth_Destroyed": destroyed,
            "Destroyed_Pct": destroyed_pct,
        })

    return pd.DataFrame(rows)