"""
kse/benchmark.py
=================
Benchmark comparison engine.

Compares a custom portfolio against the KSE-100 index:
    - Returns: annualized, cumulative
    - Risk: volatility, max drawdown, Sharpe, Sortino
    - Alpha & beta (CAPM regression)
    - Up/down capture ratios
    - Rolling alpha
"""

import numpy as np
import pandas as pd
from kse.risk_metrics import compute_drawdown


def compare_to_benchmark(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Compare a portfolio against a benchmark.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Monthly portfolio returns.
    benchmark_returns : pd.Series
        Monthly benchmark (KSE-100) returns.
    risk_free_rate : float
        Annual risk-free rate (default 0 for simplicity).

    Returns
    -------
    dict with:
        - portfolio_stats  : dict (annual return, vol, sharpe, max_dd, sortino)
        - benchmark_stats  : dict (same metrics)
        - alpha            : float — annualized CAPM alpha
        - beta             : float — CAPM beta
        - r_squared        : float
        - up_capture       : float — % of benchmark up-moves captured
        - down_capture     : float — % of benchmark down-moves captured
        - excess_return    : float — annualized portfolio - benchmark
        - tracking_error   : float — annualized std of excess returns
        - information_ratio: float — excess return / tracking error
    """
    # Align dates
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    port = portfolio_returns.loc[common]
    bench = benchmark_returns.loc[common]

    if len(common) < 12:
        raise ValueError(
            f"Only {len(common)} overlapping months — need at least 12 "
            "for meaningful benchmark comparison."
        )

    rf_monthly = (1 + risk_free_rate) ** (1/12) - 1

    # ── Portfolio stats ──
    port_stats = _compute_stats(port, rf_monthly)

    # ── Benchmark stats ──
    bench_stats = _compute_stats(bench, rf_monthly)

    # ── CAPM regression: R_port - Rf = alpha + beta * (R_bench - Rf) + epsilon ──
    excess_port = port - rf_monthly
    excess_bench = bench - rf_monthly

    # OLS regression
    X = np.column_stack([np.ones(len(excess_bench)), excess_bench])
    beta_vec, residuals, rank, sv = np.linalg.lstsq(
        X, excess_port, rcond=None
    )
    alpha_monthly = beta_vec[0]
    beta = beta_vec[1]

    # R-squared
    y_pred = X @ beta_vec
    ss_res = np.sum((excess_port - y_pred) ** 2)
    ss_tot = np.sum((excess_port - excess_port.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Annualize alpha
    alpha_annual = (1 + alpha_monthly) ** 12 - 1

    # ── Capture ratios ──
    up_mask = bench > 0
    down_mask = bench < 0

    if up_mask.sum() > 0:
        up_capture = port[up_mask].mean() / bench[up_mask].mean()
    else:
        up_capture = 0.0

    if down_mask.sum() > 0:
        # Down capture: how much of the downside does the portfolio capture?
        # Positive value = moves in same direction; we want this < 1
        down_capture = port[down_mask].mean() / bench[down_mask].mean()
    else:
        down_capture = 0.0

    # ── Excess return & tracking error ──
    excess = port - bench
    excess_annual = (1 + excess.mean()) ** 12 - 1
    tracking_error = excess.std() * np.sqrt(12)
    information_ratio = excess_annual / tracking_error if tracking_error > 0 else 0

    return {
        "portfolio_stats": port_stats,
        "benchmark_stats": bench_stats,
        "alpha": alpha_annual,
        "beta": beta,
        "r_squared": r_squared,
        "up_capture": up_capture,
        "down_capture": down_capture,
        "excess_return": excess_annual,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "n_months": len(common),
    }


def _compute_stats(returns: pd.Series, rf_monthly: float = 0) -> dict:
    """Compute standard risk-return statistics."""
    annual_return = (1 + returns.mean()) ** 12 - 1
    annual_vol = returns.std() * np.sqrt(12)
    sharpe = (annual_return - rf_monthly * 12) / annual_vol if annual_vol > 0 else 0

    # Sortino: uses downside deviation only
    downside = returns[returns < 0]
    downside_dev = downside.std() * np.sqrt(12) if len(downside) > 0 else 0
    sortino = (annual_return - rf_monthly * 12) / downside_dev if downside_dev > 0 else 0

    # Max drawdown
    dd = compute_drawdown(returns)

    # Cumulative return
    cumulative = (1 + returns).prod() - 1

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": dd["max_drawdown"],
        "cumulative_return": cumulative,
        "n_months": len(returns),
    }


def rolling_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 36,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Compute rolling alpha and beta over a window.

    Parameters
    ----------
    portfolio_returns : pd.Series
    benchmark_returns : pd.Series
    window : int
        Rolling window in months (default 36 = 3 years).
    risk_free_rate : float
        Annual risk-free rate.

    Returns
    -------
    pd.DataFrame with columns: alpha, beta, r_squared
    """
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    port = portfolio_returns.loc[common]
    bench = benchmark_returns.loc[common]
    rf_monthly = (1 + risk_free_rate) ** (1/12) - 1

    results = []
    for i in range(window, len(common)):
        port_window = port.iloc[i-window:i]
        bench_window = bench.iloc[i-window:i]

        excess_port = port_window - rf_monthly
        excess_bench = bench_window - rf_monthly

        X = np.column_stack([np.ones(len(excess_bench)), excess_bench])
        beta_vec, _, _, _ = np.linalg.lstsq(X, excess_port, rcond=None)

        y_pred = X @ beta_vec
        ss_res = np.sum((excess_port - y_pred) ** 2)
        ss_tot = np.sum((excess_port - excess_port.mean()) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        alpha_monthly = beta_vec[0]
        alpha_annual = (1 + alpha_monthly) ** 12 - 1

        results.append({
            "date": common[i],
            "alpha": alpha_annual,
            "beta": beta_vec[1],
            "r_squared": r_sq,
        })

    return pd.DataFrame(results).set_index("date")