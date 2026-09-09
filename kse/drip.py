"""
kse/drip.py
============
Dividend Reinvestment (DRIP) analysis.

Compares two strategies:
    1. Cash dividends: dividends taken as cash, portfolio = capital gains
    2. DRIP: dividends reinvested, portfolio = capital gains + reinvested dividends
"""

import numpy as np
import pandas as pd


def drip_comparison(
    monthly_amount: float,
    annual_return: float,
    annual_dividend_yield: float,
    annual_fee: float = 0.015,
    tx_cost_pct: float = 0.002,
    horizon_years: int = 10,
) -> dict:
    """
    Compare cash dividends vs DRIP over a horizon.

    Returns dict with:
        - cash_terminal, drip_terminal
        - cash_dividends_total, drip_dividends_reinvested
        - drip_advantage (absolute and percentage)
        - yearly_comparison DataFrame
    """
    months = horizon_years * 12
    monthly_capital_return = (1 + annual_return) ** (1/12) - 1
    monthly_div = (1 + annual_dividend_yield) ** (1/12) - 1
    monthly_fee = annual_fee / 12
    net_inv = monthly_amount * (1 - tx_cost_pct)

    # Cash dividends: capital growth only, dividends taken as cash
    cash_pv = 0.0
    cash_div_total = 0.0
    yearly = []

    for m in range(1, months + 1):
        # Dividend earned on current portfolio value (before this month's deposit)
        monthly_div_income = cash_pv * monthly_div
        cash_div_total += monthly_div_income
        # Capital growth (total return minus dividend yield)
        capital_growth = monthly_capital_return - monthly_div - monthly_fee
        cash_pv = cash_pv * (1 + capital_growth) + net_inv

        if m % 12 == 0:
            yearly.append({
                "Year": m // 12,
                "Cash_Portfolio": cash_pv,
                "Cash_Dividends_Annual": cash_pv * annual_dividend_yield,
                "DRIP_Portfolio": 0,  # filled below
            })

    # DRIP: dividends reinvested (total return compounds)
    drip_pv = 0.0
    for m in range(1, months + 1):
        net_return = monthly_capital_return - monthly_fee
        drip_pv = drip_pv * (1 + net_return) + net_inv
        if m % 12 == 0:
            year_idx = m // 12 - 1
            yearly[year_idx]["DRIP_Portfolio"] = drip_pv

    df = pd.DataFrame(yearly)
    drip_advantage = drip_pv - cash_pv
    drip_advantage_pct = (drip_pv / cash_pv - 1) if cash_pv > 0 else 0

    return {
        "cash_terminal": cash_pv,
        "drip_terminal": drip_pv,
        "cash_dividends_total": cash_div_total,
        "drip_advantage": drip_advantage,
        "drip_advantage_pct": drip_advantage_pct,
        "yearly_comparison": df,
    }