"""
scripts/fetch_multi_asset.py
==============================
Fetch gold price history for multi-asset comparison.

Saves monthly gold prices to data/processed/gold_monthly.csv
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yfinance as yf


def fetch_gold(start: str = "2005-01-01") -> pd.DataFrame:
    """Fetch gold futures (GC=F) monthly close."""
    print("Fetching gold prices from yfinance...")
    df = yf.download("GC=F", start=start, interval="1mo",
                      progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError("No gold data returned from yfinance")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": "Rate"})
    df.index.name = "Date"
    df["Monthly_Return"] = df["Rate"].pct_change()
    df = df.dropna()
    print(f"  Fetched {len(df)} monthly observations")
    print(f"  Period: {df.index[0]:%b %Y} to {df.index[-1]:%b %Y}")
    return df


def main():
    output_dir = ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    gold = fetch_gold()
    path = output_dir / "gold_monthly.csv"
    gold.to_csv(path)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()