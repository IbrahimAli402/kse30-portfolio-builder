"""
kse/goals.py
=============
Goal-based investing framework.

Lets users define real financial goals (university, retirement, house)
and works backward to show:
    - Required monthly SIP contribution
    - Probability of reaching the goal (from Monte Carlo)
    - Cost of delay (what happens if you start 5 years later)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# GOAL TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

GOAL_TEMPLATES = {
    "university": {
        "label": "Child's university education",
        "default_target": 15_000_000,   # PKR 1.5 Cr
        "default_horizon": 18,          # 18 years
        "description": "University education in Pakistan or abroad.",
    },
    "retirement": {
        "label": "Retirement corpus",
        "default_target": 50_000_000,  # PKR 5 Cr
        "default_horizon": 25,          # 25 years
        "description": "A corpus to generate passive income after retirement.",
    },
    "house": {
        "label": "Buy a house",
        "default_target": 20_000_000,  # PKR 2 Cr
        "default_horizon": 8,           # 8 years
        "description": "Down payment or full purchase of a home.",
    },
    "custom": {
        "label": "Custom goal",
        "default_target": 10_000_000,
        "default_horizon": 10,
        "description": "Define your own financial goal.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# REQUIRED CONTRIBUTION CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def required_monthly_sip(
    target_amount: float,
    horizon_years: int,
    annual_return: float,
    annual_fee: float = 0.015,
    tx_cost_pct: float = 0.002,
) -> Dict:
    """
    Calculate the monthly SIP needed to reach a target amount.

    Uses the future value of an annuity formula, adjusted for fees and
    transaction costs.

    Parameters
    ----------
    target_amount : float
        Target portfolio value at end of horizon.
    horizon_years : int
        Investment horizon in years.
    annual_return : float
        Expected annual return (e.g., 0.15 for 15%).
    annual_fee : float
        Annual management fee (default 1.5%).
    tx_cost_pct : float
        Transaction cost per deposit (default 0.2%).

    Returns
    -------
    dict with:
        - monthly_sip      : float — required monthly contribution
        - total_invested   : float — total capital invested
        - total_growth     : float — investment growth (target - invested)
        - effective_return : float — annualized return on invested capital
        - monthly_net_return: float — net monthly return used
    """
    months = horizon_years * 12
    net_annual = annual_return - annual_fee
    monthly_rate = (1 + net_annual) ** (1 / 12) - 1

    # Future value of annuity: FV = PMT * [((1+r)^n - 1) / r]
    # Solve for PMT: PMT = FV * r / [((1+r)^n - 1)]
    if monthly_rate == 0:
        monthly_sip_gross = target_amount / months
    else:
        monthly_sip_gross = target_amount * monthly_rate / ((1 + monthly_rate) ** months - 1)

    # Adjust for transaction costs
    monthly_sip = monthly_sip_gross / (1 - tx_cost_pct)

    total_invested = monthly_sip * months
    total_growth = target_amount - total_invested
    effective_return = (target_amount / total_invested - 1) if total_invested > 0 else 0

    return {
        "monthly_sip": monthly_sip,
        "total_invested": total_invested,
        "total_growth": total_growth,
        "effective_return": effective_return,
        "monthly_net_return": monthly_rate,
        "months": months,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GOAL PROBABILITY FROM MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════════

def goal_probability(
    mc_terminal_values: np.ndarray,
    target_amount: float,
    total_invested: float,
) -> Dict:
    """
    Compute probability of reaching a goal from Monte Carlo simulation.

    Parameters
    ----------
    mc_terminal_values : np.ndarray
        Terminal portfolio values from Monte Carlo (shape: n_paths).
    target_amount : float
        Goal target amount.
    total_invested : float
        Total amount that will be invested over the horizon.

    Returns
    -------
    dict with:
        - prob_reach_target    : float — P(terminal >= target)
        - prob_reach_invested  : float — P(terminal >= invested)
        - prob_above_1_5x      : float — P(terminal >= 1.5x invested)
        - prob_above_2x        : float — P(terminal >= 2x invested)
        - median_terminal      : float
        - p10_terminal         : float
        - p90_terminal         : float
        - shortfall_p50        : float — median shortfall if target not met
    """
    n = len(mc_terminal_values)

    prob_reach = (mc_terminal_values >= target_amount).mean()
    prob_invested = (mc_terminal_values >= total_invested).mean()
    prob_1_5x = (mc_terminal_values >= total_invested * 1.5).mean()
    prob_2x = (mc_terminal_values >= total_invested * 2).mean()

    median = np.median(mc_terminal_values)
    p10 = np.percentile(mc_terminal_values, 10)
    p90 = np.percentile(mc_terminal_values, 90)

    # Shortfall: how far below target if you're in the median
    shortfall = max(0, target_amount - median)

    return {
        "prob_reach_target": prob_reach,
        "prob_reach_invested": prob_invested,
        "prob_above_1_5x": prob_1_5x,
        "prob_above_2x": prob_2x,
        "median_terminal": median,
        "p10_terminal": p10,
        "p90_terminal": p90,
        "shortfall_p50": shortfall,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COST OF DELAY
# ═══════════════════════════════════════════════════════════════════════════════

def cost_of_delay(
    target_amount: float,
    horizon_years: int,
    annual_return: float,
    delay_years: int = 5,
    annual_fee: float = 0.015,
    tx_cost_pct: float = 0.002,
) -> Dict:
    """
    Show the cost of delaying investment by N years.

    If you start `delay_years` later, you have fewer months to reach the
    same target, so your required monthly SIP goes up.

    Parameters
    ----------
    target_amount : float
        Goal target.
    horizon_years : int
        Original horizon.
    annual_return : float
        Expected annual return.
    delay_years : int
        Years of delay to simulate (default 5).
    annual_fee : float
    tx_cost_pct : float

    Returns
    -------
    dict with:
        - original_sip       : float
        - delayed_sip        : float
        - sip_increase       : float — absolute increase
        - sip_increase_pct   : float — percentage increase
        - original_invested  : float
        - delayed_invested   : float
        - extra_invested     : float — additional capital needed
    """
    original = required_monthly_sip(target_amount, horizon_years, annual_return, annual_fee, tx_cost_pct)
    delayed_horizon = max(1, horizon_years - delay_years)
    delayed = required_monthly_sip(target_amount, delayed_horizon, annual_return, annual_fee, tx_cost_pct)

    sip_increase = delayed["monthly_sip"] - original["monthly_sip"]
    sip_increase_pct = sip_increase / original["monthly_sip"] if original["monthly_sip"] > 0 else 0

    return {
        "original_sip": original["monthly_sip"],
        "delayed_sip": delayed["monthly_sip"],
        "sip_increase": sip_increase,
        "sip_increase_pct": sip_increase_pct,
        "original_invested": original["total_invested"],
        " delayed_invested": delayed["total_invested"],
        "extra_invested": delayed["total_invested"] - original["total_invested"],
        "delayed_horizon": delayed_horizon,
    }

