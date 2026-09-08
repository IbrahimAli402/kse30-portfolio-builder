"""
KSE 100 Portfolio Builder — SIP Engine

The core Systematic Investment Plan engine.
Used by both Phase 1 (historical backtest) and Phase 2 (Monte Carlo).

All financial constants come from config/scenarios.yaml — nothing is
hardcoded in this file.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_PATH = PROJECT_ROOT / "config" / "scenarios.yaml"

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

TIERS = CONFIG["tiers"]
MODEL = CONFIG["model"]

ANNUAL_FEE = MODEL["annual_fee"]
TX_COST_PCT = MODEL["tx_cost_pct"]
CGT_SHORT = MODEL["cgt_short"]
CGT_MEDIUM = MODEL["cgt_medium"]
CGT_LONG = MODEL["cgt_long"]


# ── Data loaders ───────────────────────────────────────────────────────

def load_total_return_index():
    """Load KSE 100 monthly total return index (2010-2024)."""
    df = pd.read_csv(DATA_DIR / "kse_monthly_total_return.csv",
                     parse_dates=["Date"])
    return df.set_index("Date")


def load_monthly_returns():
    """Load monthly total returns as a numpy array."""
    df = load_total_return_index()
    return df["Total_Return"].values


def load_portfolio_returns():
    """Load pre-blended tier returns (net of fee)."""
    df = pd.read_csv(DATA_DIR / "portfolio_returns.csv", parse_dates=["Date"])
    return df.set_index("Date")


# ── SIP Engine ─────────────────────────────────────────────────────────

def run_sip(monthly_returns, monthly_amount, tx_cost_pct=TX_COST_PCT):
    """
    Run a SIP on a series of monthly returns (already blended, post-fee).

    Parameters:
        monthly_returns: array of monthly returns (blended by tier, net of fee)
        monthly_amount: PKR deposited each month
        tx_cost_pct: transaction cost per purchase (default 0.2%)

    Returns:
        dict with portfolio_values, cumulative_invested, final_value,
        total_invested, total_profit, return_pct, underwater_months,
        max_drawdown, total_transaction_costs
    """
    net_inv = monthly_amount * (1 - tx_cost_pct)

    pv = 0.0
    ci = 0.0
    tc = 0.0
    peak = 0.0
    max_dd = 0.0
    underwater = 0

    portfolio_values = []
    cumulative_invested = []

    for ret in monthly_returns:
        pv = pv * (1 + ret) + net_inv
        ci += monthly_amount
        tc += monthly_amount * tx_cost_pct
        portfolio_values.append(pv)
        cumulative_invested.append(ci)

        if pv < ci:
            underwater += 1

        if pv > peak:
            peak = pv
        dd = (peak - pv) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    final_value = pv
    total_profit = final_value - ci

    return {
        "portfolio_values": np.array(portfolio_values),
        "cumulative_invested": np.array(cumulative_invested),
        "final_value": final_value,
        "total_invested": ci,
        "total_profit": total_profit,
        "return_pct": (total_profit / ci * 100) if ci > 0 else 0,
        "underwater_months": underwater,
        "max_drawdown": max_dd,
        "total_transaction_costs": tc,
    }


def run_sip_vectorized(paths, monthly_amount, tx_cost_pct=TX_COST_PCT):
    """
    Run SIP on many paths simultaneously (for Monte Carlo).

    Parameters:
        paths: numpy array of shape (num_paths, horizon) — monthly returns
        monthly_amount: PKR deposited each month
        tx_cost_pct: transaction cost per purchase

    Returns:
        dict with portfolio_paths, terminal_values, max_drawdowns,
        underwater_months, total_invested
    """
    net_inv = monthly_amount * (1 - tx_cost_pct)
    num_paths, horizon = paths.shape

    portfolio_paths = np.zeros((num_paths, horizon))
    pv = np.zeros(num_paths)

    for m in range(horizon):
        pv = pv * (1 + paths[:, m]) + net_inv
        portfolio_paths[:, m] = pv

    terminal_values = portfolio_paths[:, -1]

    # Max drawdown per path
    peaks = np.maximum.accumulate(portfolio_paths, axis=1)
    drawdowns = (peaks - portfolio_paths) / np.where(peaks > 0, peaks, 1)
    max_drawdowns = np.max(drawdowns, axis=1)

    # Underwater months per path
    cumulative_invested = monthly_amount * np.arange(1, horizon + 1)
    underwater = np.sum(portfolio_paths < cumulative_invested, axis=1)

    return {
        "portfolio_paths": portfolio_paths,
        "terminal_values": terminal_values,
        "max_drawdowns": max_drawdowns,
        "underwater_months": underwater,
        "total_invested": monthly_amount * horizon,
    }


# ── Return blending ────────────────────────────────────────────────────

def blend_tier_returns(equity_returns, income_return_annual, tier,
                       annual_fee=ANNUAL_FEE):
    """
    Blend equity and income returns by risk tier, deduct annual fee.

    Parameters:
        equity_returns: array of monthly equity returns
        income_return_annual: annual income sleeve return (scenario-dependent)
        tier: "Conservative", "Moderate", or "Aggressive"
        annual_fee: annual fund fee (default 1.5%)

    Returns:
        array of blended monthly returns (net of fee)
    """
    tier_config = TIERS[tier]
    income_monthly = (1 + income_return_annual) ** (1 / 12) - 1
    monthly_fee = annual_fee / 12

    blended = (tier_config["equity"] * np.array(equity_returns)
               + tier_config["income"] * income_monthly)
    return blended - monthly_fee


def blend_tier_returns_vectorized(equity_paths, income_return_annual, tier,
                                  annual_fee=ANNUAL_FEE):
    """
    Blend equity paths with income return by tier (vectorized for Monte Carlo).

    Parameters:
        equity_paths: numpy array (num_paths, horizon) of monthly equity returns
        income_return_annual: annual income sleeve return
        tier: risk tier name
        annual_fee: annual fund fee

    Returns:
        numpy array (num_paths, horizon) of blended monthly returns
    """
    tier_config = TIERS[tier]
    income_monthly = (1 + income_return_annual) ** (1 / 12) - 1
    monthly_fee = annual_fee / 12

    blended = (tier_config["equity"] * equity_paths
               + tier_config["income"] * income_monthly)
    return blended - monthly_fee


# ── Risk metrics ───────────────────────────────────────────────────────

def drawdown(values):
    """Drawdown series from a portfolio value series."""
    values = np.array(values)
    peaks = np.maximum.accumulate(values)
    dd = (values - peaks) / np.where(peaks > 0, peaks, 1)
    return dd


def probability_of_loss(returns, holding_periods=None):
    """
    Probability of negative return for each holding period.

    Parameters:
        returns: array of monthly returns
        holding_periods: list of months (default: [1, 3, 6, 12, 24, 36, 48, 60])

    Returns:
        dict mapping holding period (months) to probability of loss
    """
    if holding_periods is None:
        holding_periods = [1, 3, 6, 12, 24, 36, 48, 60]

    returns = np.array(returns)
    n = len(returns)
    result = {}

    for period in holding_periods:
        if period > n:
            result[period] = None
            continue
        rolling = np.array([
            np.prod(1 + returns[i:i + period]) - 1
            for i in range(n - period + 1)
        ])
        result[period] = np.mean(rolling < 0)

    return result


# ── Capital gains tax ──────────────────────────────────────────────────

def blended_cgt_rate(total_months):
    """
    Blended CGT rate for a SIP of total_months duration.

    Each monthly deposit is a separate lot with its own holding period:
      Lot 1 (first deposit):  held for total_months
      Lot N (last deposit):   held for 1 month

    CGT brackets (Pakistan):
      < 12 months:   15%
      12-24 months:  12.5%
      > 24 months:   0%
    """
    if total_months <= 0:
        return 0.0

    long_lots = max(0, total_months - 24)
    medium_lots = min(12, max(0, total_months - 12))
    short_lots = min(12, total_months)

    rate = (long_lots * CGT_LONG
            + medium_lots * CGT_MEDIUM
            + short_lots * CGT_SHORT) / total_months
    return rate


def apply_cgt(profit, total_months):
    """Apply blended CGT to total profit. Returns (cgt_amount, blended_rate)."""
    rate = blended_cgt_rate(total_months)
    return profit * rate, rate