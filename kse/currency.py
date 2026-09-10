"""
kse/currency.py
================
Currency-adjusted return calculations.

Converts PKR-denominated returns to USD-denominated returns using
actual historical USD/PKR exchange rates. This is critical for
Pakistani investors — PKR returns overstate real wealth creation
when the currency is depreciating.

Key identity:
    R_usd = (1 + R_pkr) / (1 + R_fx) - 1
where R_fx is the monthly return of the USD/PKR rate.
"""

import pandas as pd
import numpy as np
from pathlib import Path


_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_fx_data() -> pd.DataFrame:
    """
    Load USD/PKR monthly exchange rate data.

    Returns
    -------
    pd.DataFrame with columns: Rate, Monthly_Return
    """
    path = _DATA_DIR / "usd_pkr_monthly.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"USD/PKR data not found at {path}. "
            "Run: python scripts/fetch_fx.py"
        )
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    return df


def convert_returns_to_usd(
    pkr_returns: pd.Series,
    fx_data: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Convert PKR-denominated returns to USD-denominated returns.

    A Pakistani investor converting PKR to USD each month and
    investing abroad would earn the USD return. Conversely,     a foreign investor in KSE-100 earns R_usd = (1+R_pkr)/(1+R_fx) - 1.

    Parameters
    ----------
    pkr_returns : pd.Series
        Monthly returns in PKR (e.g., KSE-100 total return).
    fx_data : pd.DataFrame, optional
        Exchange rate data from load_fx_data(). Loaded if not provided.

    Returns
    -------
    pd.Series — monthly returns in USD terms
    """
    if fx_data is None:
        fx_data = load_fx_data()

        # Align indices — normalize to month-start so month-end and month-start dates match
    pkr_norm = pkr_returns.copy()
    pkr_norm.index = pkr_norm.index.to_period("M").to_timestamp()
    fx_norm = fx_data.copy()
    fx_norm.index = fx_norm.index.to_period("M").to_timestamp()

    common = pkr_norm.index.intersection(fx_norm.index)
    if len(common) == 0:
        raise ValueError(
            "No overlapping dates between returns and FX data. "
            f"Returns: {pkr_returns.index[0]} to {pkr_returns.index[-1]}, "
            f"FX: {fx_data.index[0]} to {fx_data.index[-1]}"
        )

    pkr = pkr_norm.loc[common]
    fx_ret = fx_norm.loc[common, "Monthly_Return"]

    # USD return = (1 + PKR return) / (1 + FX return) - 1
    # divide because pkr depreciation (positive fx_ret) reduces usd value
    usd_returns = (1 + pkr) / (1 + fx_ret) - 1

    return usd_returns


def convert_values_to_usd(
    pkr_values: np.ndarray | pd.Series,
    fx_data: pd.DataFrame | None = None,
    start_date: pd.Timestamp | None = None,
) -> np.ndarray:
    """
    Convert a series of PKR portfolio values to USD.

    Parameters
    ----------
    pkr_values : array-like
        Portfolio values in PKR over time.
    fx_data : pd.DataFrame, optional
        Exchange rate data. Loaded if not provided.
    start_date : pd.Timestamp, optional
        Date corresponding to the first value. If provided,
        uses actual historical rates. If None, uses the latest
        available rate (for forward projections).

    Returns
    -------
    np.ndarray — values in USD
    """
    if fx_data is None:
        fx_data = load_fx_data()

    if start_date is not None:
        # Use historical rates
        rates = fx_data.loc[fx_data.index >= start_date, "Rate"]
        if len(rates) < len(pkr_values):
            # Pad with last available rate
            last_rate = rates.iloc[-1]
            padding = pd.Series(
                [last_rate] * (len(pkr_values) - len(rates)),
                index=pd.date_range(
                    start=rates.index[-1] + pd.DateOffset(months=1),
                    periods=len(pkr_values) - len(rates),
                    freq="MS"
                )
            )
            rates = pd.concat([rates, padding])
        return np.array(pkr_values) / rates.values[:len(pkr_values)]
    else:
        # Use latest rate (for forward projections)
        latest_rate = fx_data["Rate"].iloc[-1]
        return np.array(pkr_values) / latest_rate


def get_currency_stats(
    pkr_returns: pd.Series,
    fx_data: pd.DataFrame | None = None,
) -> dict:
    """
    Compute currency impact statistics.

    Parameters
    ----------
    pkr_returns : pd.Series
        Monthly returns in PKR.
    fx_data : pd.DataFrame, optional
        Exchange rate data. Loaded if not provided.

    Returns
    -------
    dict with:
        - pkr_annual_return   : float
        - usd_annual_return   : float
        - currency_drag       : float — annualized PKR vs USD return gap
        - pkr_volatility      : float
        - usd_volatility      : float
        - pkr_sharpe          : float (rf=0)
        - usd_sharpe          : float (rf=0)
        - total_depreciation   : float — cumulative PKR depreciation over period
    """
    if fx_data is None:
        fx_data = load_fx_data()

    usd_returns = convert_returns_to_usd(pkr_returns, fx_data)

    # Annualized returns
    pkr_annual = (1 + pkr_returns.mean()) ** 12 - 1
    usd_annual = (1 + usd_returns.mean()) ** 12 - 1
    currency_drag = pkr_annual - usd_annual

    # Volatility
    pkr_vol = pkr_returns.std() * np.sqrt(12)
    usd_vol = usd_returns.std() * np.sqrt(12)

    # Sharpe (rf=0)
    pkr_sharpe = pkr_annual / pkr_vol if pkr_vol > 0 else 0
    usd_sharpe = usd_annual / usd_vol if usd_vol > 0 else 0

    # Total depreciation over the period — normalize to month-start
    pkr_norm_idx = pkr_returns.index.to_period("M").to_timestamp()
    fx_norm_idx = fx_data.index.to_period("M").to_timestamp()
    common = pkr_norm_idx.intersection(fx_norm_idx)
    start_rate = fx_data.loc[fx_norm_idx == common[0], "Rate"].iloc[0]
    end_rate = fx_data.loc[fx_norm_idx == common[-1], "Rate"].iloc[0]
    total_depreciation = (end_rate / start_rate - 1)

    return {
        "pkr_annual_return": pkr_annual,
        "usd_annual_return": usd_annual,
        "currency_drag": currency_drag,
        "pkr_volatility": pkr_vol,
        "usd_volatility": usd_vol,
        "pkr_sharpe": pkr_sharpe,
        "usd_sharpe": usd_sharpe,
        "total_depreciation": total_depreciation,
        "start_rate": start_rate,
        "end_rate": end_rate,
    }