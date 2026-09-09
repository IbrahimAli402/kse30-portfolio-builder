"""
kse/multi_asset.py
===================
Multi-asset comparison engine.

Compares KSE-100 against gold, USD, and real estate
to contextualize equity allocation decisions.
"""

import numpy as np
import pandas as pd
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# Pakistan real estate: ~12% annualized (conservative for major cities)
REAL_ESTATE_ANNUAL_RETURN = 0.12


def load_gold_data() -> pd.DataFrame:
    path = _DATA_DIR / "gold_monthly.csv"
    if not path.exists():
        raise FileNotFoundError(f"Gold data not found at {path}. Run: python scripts/fetch_multi_asset.py")
    return pd.read_csv(path, parse_dates=["Date"]).set_index("Date")


def compare_assets(
    kse_returns: pd.Series,
    usd_pkr_data: pd.DataFrame,
    gold_data: pd.DataFrame | None = None,
    real_estate_annual: float = REAL_ESTATE_ANNUAL_RETURN,
) -> pd.DataFrame:
    """
    Build a comparison table of KSE-100 vs Gold vs USD vs Real Estate.

    Returns DataFrame with columns: Asset, Annual_Return, Volatility,
    Sharpe, Max_Drawdown, Cumulative_Return
    """
    results = []

    # KSE-100
    kse_annual = (1 + kse_returns.mean()) ** 12 - 1
    kse_vol = kse_returns.std() * np.sqrt(12)
    kse_sharpe = kse_annual / kse_vol if kse_vol > 0 else 0
    kse_cum = (1 + kse_returns).prod() - 1
    kse_dd = _max_drawdown(kse_returns)
    results.append({
        "Asset": "KSE-100 (PKR)", "Annual_Return": kse_annual,
        "Volatility": kse_vol, "Sharpe": kse_sharpe,
        "Max_Drawdown": kse_dd, "Cumulative_Return": kse_cum
    })

    # USD (from PKR perspective: holding USD)
    from kse.currency import convert_returns_to_usd
    usd_returns = convert_returns_to_usd(kse_returns, usd_pkr_data)
    usd_annual = (1 + usd_returns.mean()) ** 12 - 1
    usd_vol = usd_returns.std() * np.sqrt(12)
    usd_sharpe = usd_annual / usd_vol if usd_vol > 0 else 0
    usd_cum = (1 + usd_returns).prod() - 1
    usd_dd = _max_drawdown(usd_returns)
    results.append({
        "Asset": "KSE-100 (USD)", "Annual_Return": usd_annual,
        "Volatility": usd_vol, "Sharpe": usd_sharpe,
        "Max_Drawdown": usd_dd, "Cumulative_Return": usd_cum
    })

    # Gold (PKR perspective: gold in PKR = gold in USD * USD/PKR)
    if gold_data is not None:
        # Align gold with USD/PKR
        common = gold_data.index.intersection(usd_pkr_data.index)
        if len(common) > 12:
            gold_usd_ret = gold_data.loc[common, "Monthly_Return"]
            pkr_ret = usd_pkr_data.loc[common, "Monthly_Return"]
            gold_pkr_ret = (1 + gold_usd_ret) * (1 + pkr_ret) - 1

            g_annual = (1 + gold_pkr_ret.mean()) ** 12 - 1
            g_vol = gold_pkr_ret.std() * np.sqrt(12)
            g_sharpe = g_annual / g_vol if g_vol > 0 else 0
            g_cum = (1 + gold_pkr_ret).prod() - 1
            g_dd = _max_drawdown(gold_pkr_ret)
            results.append({
                "Asset": "Gold (PKR)", "Annual_Return": g_annual,
                "Volatility": g_vol, "Sharpe": g_sharpe,
                "Max_Drawdown": g_dd, "Cumulative_Return": g_cum
            })

    # Real Estate (static assumption)
    re_monthly = (1 + real_estate_annual) ** (1/12) - 1
    re_vol = 0.05  # low observed volatility (illiquid market)
    re_sharpe = real_estate_annual / re_vol if re_vol > 0 else 0
    results.append({
        "Asset": "Real Estate (PKR)", "Annual_Return": real_estate_annual,
        "Volatility": re_vol, "Sharpe": re_sharpe,
        "Max_Drawdown": -0.15, "Cumulative_Return": None
    })

    return pd.DataFrame(results)


def _max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    return float(((cum / cum.cummax()) - 1).min())