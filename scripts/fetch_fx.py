"""
scripts/fetch_fx.py
====================
Fetch USD/PKR exchange rate history from yfinance.

Saves monthly exchange rates to data/processed/usd_pkr_monthly.csv
for use in currency-adjusted return calculations.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yfinance as yf


def fetch_usd_pkr(start: str = "2005-01-01") -> pd.DataFrame:
    """
    Fetch USD/PKR exchange rate history from yfinance.

    Parameters
    ----------
    start : str
        Start date for history (default 2005-01-01).

    Returns
    -------
    pd.DataFrame with columns: Date, Close, Monthly_Return
    """
    print("Fetching USD/PKR exchange rate from yfinance...")

    # yfinance ticker for USD/PKR
    df = yf.download("PKR=X", start=start, interval="1mo",
                      progress=False, auto_adjust=True)

    if df.empty:
        raise RuntimeError("No data returned from yfinance for USD/PKR")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep Close price
    df = df[["Close"]].rename(columns={"Close": "Rate"})
    df.index.name = "Date"

    # Calculate monthly return
    df["Monthly_Return"] = df["Rate"].pct_change()

    # Drop first row (NaN return)
    df = df.dropna()

    print(f"  Fetched {len(df)} monthly observations")
    print(f"  Period: {df.index[0]:%b %Y} to {df.index[-1]:%b %Y}")
    print(f"  Start rate: {df['Rate'].iloc[0]:.2f} PKR/USD")
    print(f"  End rate:   {df['Rate'].iloc[-1]:.2f} PKR/USD")
    print(f"  Total depreciation: {(df['Rate'].iloc[-1] / df['Rate'].iloc[0] - 1) * 100:.1f}%")

    return df


def main():
    """Fetch and save USD/PKR data."""
    output_dir = ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    fx_data = fetch_usd_pkr()

    output_path = output_dir / "usd_pkr_monthly.csv"
    fx_data.to_csv(output_path)
    print(f"\nSaved to {output_path}")
    print(f"Columns: {list(fx_data.columns)}")


if __name__ == "__main__":
    main()