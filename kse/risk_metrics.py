"""
kse/risk_metrics.py
====================
Advanced risk analytics for the KSE 100 Portfolio Builder.

Provides:
    - compute_drawdown()           — historical drawdown analysis
    - monte_carlo_drawdown()       — drawdown distribution across MC paths
    - compute_var_cvar()           — VaR & CVaR (historical, parametric, MC)
    - stress_test_portfolio()      — crisis scenario replay
    - sector_concentration()       — sector exposure & HHI
    - correlation_regime_analysis() — bull vs bear correlation structure
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy import stats
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG LOADING
# ═══════════════════════════════════════════════════════════════════════════════

_CONFIG_DIR = Path(__file__).parent.parent / 'config'


def _load_crisis_periods(path: Optional[str] = None) -> Dict:
    """Load crisis period definitions from YAML."""
    if path is None:
        path = _CONFIG_DIR / 'crisis_periods.yaml'
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('crises', data)


def _load_sector_mapping(path: Optional[str] = None) -> Dict[str, str]:
    """Load ticker → sector mapping from YAML."""
    if path is None:
        path = _CONFIG_DIR / 'sectors.yaml'
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('sectors', data)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DRAWDOWN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_drawdown(returns: pd.Series) -> Dict:
    """
    Compute comprehensive drawdown statistics from a return series.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns (monthly, daily, etc.), indexed by date.

    Returns
    -------
    dict with:
        - drawdown_series      : pd.Series — drawdown at each point in time
        - max_drawdown         : float — worst drawdown (negative)
        - max_dd_start         : Timestamp — when the drawdown began (peak)
        - max_dd_trough        : Timestamp — the bottom
        - max_dd_recovery      : Timestamp or None — when it recovered
        - max_dd_duration      : int or None — months peak to recovery
        - max_dd_decline_duration : int — months peak to trough
        - drawdown_episodes    : list of dicts — all episodes > 5% depth
        - underwater_curve     : pd.Series — same as drawdown_series, for plotting
    """
    # Wealth index: growth of 1 unit invested at the start
    wealth = (1 + returns).cumprod()

    # Running peak at each point in time
    running_max = wealth.cummax()

    # Drawdown: how far below the peak, as a fraction
    drawdown = (wealth - running_max) / running_max

    # Maximum drawdown and its timing
    max_dd = drawdown.min()
    max_dd_trough = drawdown.idxmin()

    # Find the peak that preceded the trough
    pre_trough = wealth.loc[:max_dd_trough]
    peak_mask = pre_trough == running_max.loc[:max_dd_trough]
    peak_candidates = pre_trough[peak_mask]
    max_dd_start = peak_candidates.idxmax()

    # Find recovery (if any): when wealth returns to the pre-drawdown peak
    peak_value = wealth.loc[max_dd_start]
    post_trough = wealth.loc[max_dd_trough:]
    recovery_mask = post_trough >= peak_value

    if recovery_mask.any():
        max_dd_recovery = post_trough[recovery_mask].index[0]
        max_dd_duration = len(wealth.loc[max_dd_start:max_dd_recovery])
    else:
        max_dd_recovery = None
        max_dd_duration = None

    # Decline duration (peak → trough)
    max_dd_decline_duration = len(wealth.loc[max_dd_start:max_dd_trough])

    # All drawdown episodes deeper than 5%
    episodes = _identify_drawdown_episodes(drawdown, threshold=-0.05)

    return {
        'drawdown_series': drawdown,
        'max_drawdown': max_dd,
        'max_dd_start': max_dd_start,
        'max_dd_trough': max_dd_trough,
        'max_dd_recovery': max_dd_recovery,
        'max_dd_duration': max_dd_duration,
        'max_dd_decline_duration': max_dd_decline_duration,
        'drawdown_episodes': episodes,
        'underwater_curve': drawdown,
    }


def _identify_drawdown_episodes(drawdown: pd.Series,
                                 threshold: float = -0.05) -> List[Dict]:
    """Identify all drawdown episodes deeper than the threshold."""
    episodes = []
    in_episode = False
    start = None
    trough_val = 0.0
    trough_date = None

    for date, dd in drawdown.items():
        if dd < threshold and not in_episode:
            in_episode = True
            start = date
            trough_val = dd
            trough_date = date
        elif dd < threshold and in_episode:
            if dd < trough_val:
                trough_val = dd
                trough_date = date
        elif dd >= threshold and in_episode:
            in_episode = False
            episodes.append({
                'start': start,
                'trough': trough_date,
                'recovery': date,
                'depth': trough_val,
                'duration_months': len(drawdown.loc[start:date]),
            })

    # Handle an ongoing (unrecovered) episode
    if in_episode:
        episodes.append({
            'start': start,
            'trough': trough_date,
            'recovery': None,
            'depth': trough_val,
            'duration_months': len(drawdown.loc[start:]),
        })

    return episodes


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MONTE CARLO DRAWDOWN DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

def monte_carlo_drawdown(path_values: np.ndarray) -> Dict:
    """
    Compute drawdown statistics across Monte Carlo paths.

    Parameters
    ----------
    path_values : np.ndarray
        Shape (n_paths, horizon) — portfolio values over time from MC.

    Returns
    -------
    dict with:
        - max_drawdown_distribution : np.ndarray — max DD per path
        - p50_max_dd : float — median worst-case drawdown
        - p90_max_dd : float — 90th percentile (severe but plausible)
        - p95_max_dd : float — 95th percentile (tail risk)
        - avg_max_dd : float — average max drawdown
        - avg_dd_duration : float — average duration (months)
        - p90_dd_duration : float — 90th percentile duration
    """
    n_paths, horizon = path_values.shape
    max_dds = np.zeros(n_paths)
    dd_durations = np.zeros(n_paths)

    for i in range(n_paths):
        path = path_values[i]
        running_max = np.maximum.accumulate(path)
        drawdown = (path - running_max) / np.where(running_max == 0, 1, running_max)
        max_dds[i] = drawdown.min()

        # Duration: peak → trough
        trough_idx = int(drawdown.argmin())
        peak_idx = int(np.argmax(path[:trough_idx + 1]))
        dd_durations[i] = trough_idx - peak_idx

    return {
        'max_drawdown_distribution': max_dds,
        'p50_max_dd': np.percentile(max_dds, 50),
        'p90_max_dd': np.percentile(max_dds, 90),
        'p95_max_dd': np.percentile(max_dds, 95),
        'avg_max_dd': max_dds.mean(),
        'avg_dd_duration': dd_durations.mean(),
        'p90_dd_duration': np.percentile(dd_durations, 90),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. VALUE AT RISK (VaR) & CONDITIONAL VaR (CVaR)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_var_cvar(
    returns: pd.Series,
    confidence_levels: List[float] = [0.95, 0.99],
    mc_path_values: Optional[np.ndarray] = None,
    mc_total_invested: Optional[float] = None,
) -> Dict:
    """
    Compute VaR and CVaR using historical, parametric, and Monte Carlo methods.

    Parameters
    ----------
    returns : pd.Series
        Historical periodic returns (e.g., monthly KSE-100 returns).
    confidence_levels : list of float
        Confidence levels — [0.95, 0.99] for 95% and 99% VaR.
    mc_path_values : np.ndarray, optional
        Shape (n_paths, horizon) — portfolio values from Monte Carlo.
        When provided, computes MC VaR on total return on invested capital.
    mc_total_invested : float, optional
        Total capital invested over the horizon (monthly_contribution × horizon).
        Required to compute MC return-on-capital. If None, uses raw terminal
        value distribution.

    Returns
    -------
    dict keyed by confidence level (0.95, 0.99), each containing:
        - var_historical  : float — empirical quantile
        - cvar_historical : float — mean of tail beyond VaR
        - var_parametric  : float — normal-distribution VaR
        - cvar_parametric : float — normal-distribution CVaR
        - var_mc          : float or None — Monte Carlo VaR
        - cvar_mc         : float or None — Monte Carlo CVaR
    """
    results = {}
    mu = returns.mean()
    sigma = returns.std()

    # Pre-compute MC total returns if path values provided
    mc_returns = None
    if mc_path_values is not None:
        terminal = mc_path_values[:, -1]
        if mc_total_invested and mc_total_invested > 0:
            mc_returns = (terminal - mc_total_invested) / mc_total_invested
        else:
            # Fall back to terminal value distribution
            mc_returns = terminal

    for cl in confidence_levels:
        alpha = 1 - cl

        # ── Historical ──
        var_hist = returns.quantile(alpha)
        tail = returns[returns <= var_hist]
        cvar_hist = tail.mean() if len(tail) > 0 else var_hist

        # ── Parametric (normal) ──
        z = stats.norm.ppf(alpha)
        var_param = mu + sigma * z
        cvar_param = mu - sigma * stats.norm.pdf(z) / alpha

        result = {
            'var_historical': var_hist,
            'cvar_historical': cvar_hist,
            'var_parametric': var_param,
            'cvar_parametric': cvar_param,
        }

        # ── Monte Carlo ──
        if mc_returns is not None:
            var_mc = float(np.percentile(mc_returns, alpha * 100))
            tail_mc = mc_returns[mc_returns <= var_mc]
            cvar_mc = float(tail_mc.mean()) if len(tail_mc) > 0 else var_mc
            result['var_mc'] = var_mc
            result['cvar_mc'] = cvar_mc

        results[cl] = result

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. STRESS TESTING — CRISIS SCENARIO REPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def stress_test_portfolio(
    stock_returns: pd.DataFrame,
    weights: pd.Series,
    index_returns: Optional[pd.Series] = None,
    crisis_periods: Optional[Dict] = None,
) -> Dict:
    """
    Replay historical crisis scenarios on a portfolio.

    Parameters
    ----------
    stock_returns : pd.DataFrame
        Historical stock returns, columns = tickers, index = dates.
    weights : pd.Series
        Portfolio weights, indexed by ticker.
    index_returns : pd.Series, optional
        KSE-100 index returns for benchmark comparison during crises.
    crisis_periods : dict, optional
        Crisis definitions from config. Defaults to config/crisis_periods.yaml.

    Returns
    -------
    dict keyed by crisis name, each containing:
        - label, description, start, end
        - portfolio_return      : float — cumulative return during crisis
        - portfolio_drawdown    : float — max drawdown during crisis
        - index_return          : float or None
        - index_drawdown        : float or None
        - recovery_months       : int or None — months to recover pre-crisis peak
        - n_months              : int
        - monthly_returns       : pd.Series — portfolio monthly returns
    """
    if crisis_periods is None:
        crisis_periods = _load_crisis_periods()

    # Align weights to available stocks and renormalize
    available = weights.index.intersection(stock_returns.columns)
    w = weights[available].copy()
    w = w / w.sum()

    # Portfolio monthly returns
    portfolio_returns = (stock_returns[available] * w).sum(axis=1)

    results = {}
    for name, period in crisis_periods.items():
        start = pd.Timestamp(period['start'])
        end = pd.Timestamp(period['end'])

        mask = (portfolio_returns.index >= start) & (portfolio_returns.index <= end)
        crisis_port = portfolio_returns[mask]

        if len(crisis_port) == 0:
            results[name] = {
                'label': period.get('label', name),
                'description': period.get('description', ''),
                'error': 'No data available for this period',
            }
            continue

        # Portfolio cumulative return & drawdown during crisis
        cum_return = (1 + crisis_port).prod() - 1
        wealth = (1 + crisis_port).cumprod()
        running_max = wealth.cummax()
        drawdown = (wealth - running_max) / running_max
        max_dd = drawdown.min()

        # Index comparison
        idx_return = None
        idx_dd = None
        if index_returns is not None:
            crisis_idx = index_returns[mask]
            if len(crisis_idx) > 0:
                idx_return = (1 + crisis_idx).prod() - 1
                idx_wealth = (1 + crisis_idx).cumprod()
                idx_running_max = idx_wealth.cummax()
                idx_dd = ((idx_wealth - idx_running_max) / idx_running_max).min()

        # Recovery: how many months after the crisis to reach pre-crisis peak
        pre_crisis_wealth = (1 + portfolio_returns.loc[:start]).prod()
        post_crisis = portfolio_returns.loc[end:]
        recovery_months = None
        if len(post_crisis) > 0:
            post_wealth = (1 + post_crisis).cumprod() * (1 + portfolio_returns.loc[:end]).prod()
            recovery_mask = post_wealth >= pre_crisis_wealth
            if recovery_mask.any():
                recovery_date = post_wealth[recovery_mask].index[0]
                recovery_months = len(portfolio_returns.loc[end:recovery_date])

        results[name] = {
            'label': period.get('label', name),
            'description': period.get('description', ''),
            'start': start,
            'end': end,
            'portfolio_return': cum_return,
            'portfolio_drawdown': max_dd,
            'index_return': idx_return,
            'index_drawdown': idx_dd,
            'recovery_months': recovery_months,
            'n_months': len(crisis_port),
            'monthly_returns': crisis_port,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SECTOR CONCENTRATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def sector_concentration(
    weights: pd.Series,
    sector_map: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Analyze sector concentration of a portfolio.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights, indexed by ticker.
    sector_map : dict, optional
        Ticker → sector mapping. Loaded from config/sectors.yaml if not provided.

    Returns
    -------
    dict with:
        - sector_weights     : pd.Series — weight by sector (sorted desc)
        - max_sector         : str — most concentrated sector
        - max_sector_weight  : float
        - concentration_flag : bool — True if any sector > 40%
        - hhi                : float — Herfindahl-Hirschman Index (0–1)
        - effective_sectors  : float — 1/HHI (effective number of sectors)
    """
    if sector_map is None:
        sector_map = _load_sector_mapping()

    # Aggregate weights by sector
    sector_w = {}
    for ticker, weight in weights.items():
        if weight < 0.001:
            continue
        sector = sector_map.get(ticker, 'Other')
        sector_w[sector] = sector_w.get(sector, 0) + weight

    sector_weights = pd.Series(sector_w).sort_values(ascending=False)

    max_sector = sector_weights.index[0]
    max_sector_weight = sector_weights.iloc[0]
    concentration_flag = max_sector_weight > 0.40

    # Herfindahl-Hirschman Index
    hhi = float((sector_weights ** 2).sum())
    effective_sectors = 1 / hhi if hhi > 0 else 0

    return {
        'sector_weights': sector_weights,
        'max_sector': max_sector,
        'max_sector_weight': max_sector_weight,
        'concentration_flag': concentration_flag,
        'hhi': hhi,
        'effective_sectors': effective_sectors,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CORRELATION REGIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def correlation_regime_analysis(
    stock_returns: pd.DataFrame,
    regime_labels: pd.Series,
) -> Dict:
    """
    Compare correlation structure across regimes (bull vs bear).

    Shows how diversification benefits erode in crises — correlations
    spike toward 1.0 exactly when you need them most.

    Parameters
    ----------
    stock_returns : pd.DataFrame
        Stock returns, columns = tickers.
    regime_labels : pd.Series
        0 = bull, 1 = bear, indexed by date (from kse/regimes.py).

    Returns
    -------
    dict with:
        - bull_correlation      : pd.DataFrame
        - bear_correlation      : pd.DataFrame
        - avg_corr_bull         : float
        - avg_corr_bear         : float
        - correlation_increase  : float — ratio bear/bull
        - diversification_decay : float — 1 - (bull/bear)
    """
    # Align indices
    common = stock_returns.index.intersection(regime_labels.index)
    stock_returns = stock_returns.loc[common]
    regime_labels = regime_labels.loc[common]

    bull_mask = regime_labels == 0
    bear_mask = regime_labels == 1

    bull_corr = stock_returns[bull_mask].corr()
    bear_corr = stock_returns[bear_mask].corr()

    def _avg_pairwise(corr_matrix: pd.DataFrame) -> float:
        """Average of upper-triangle pairwise correlations."""
        n = len(corr_matrix)
        if n < 2:
            return 0.0
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        )
        return float(upper.stack().mean())

    avg_corr_bull = _avg_pairwise(bull_corr)
    avg_corr_bear = _avg_pairwise(bear_corr)
    correlation_increase = avg_corr_bear / avg_corr_bull if avg_corr_bull > 0 else 0
    diversification_decay = 1 - (avg_corr_bull / avg_corr_bear) if avg_corr_bear > 0 else 0

    return {
        'bull_correlation': bull_corr,
        'bear_correlation': bear_corr,
        'avg_corr_bull': avg_corr_bull,
        'avg_corr_bear': avg_corr_bear,
        'correlation_increase': correlation_increase,
        'diversification_decay': diversification_decay,
    }