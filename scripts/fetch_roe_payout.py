"""
Fetch ROE and payout ratio by extracting Net Income and Total Equity
from yfinance financial statements.

Updates the existing stock_metrics.csv — does NOT re-download prices.

Usage:
    python scripts/fetch_roe_payout.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def fetch_roe_payout(ticker, yf_suffix=".KA"):
    """Try to fetch ROE and payout from yfinance financial statements."""
    import yfinance as yf
    yf_ticker = f"{ticker}{yf_suffix}"

    try:
        stock = yf.Ticker(yf_ticker)

        # Get financial statements
        income = stock.financials
        balance = stock.balance_sheet

        if income.empty or balance.empty:
            return {}

        # Find Net Income (try different possible row names)
        net_income = None
        for name in ["Net Income", "Net Income Common Stockholders",
                     "Net Income From Continuing Ops"]:
            if name in income.index:
                net_income = income.loc[name].iloc[0]  # Most recent year
                break

        # Find Total Equity (try different possible row names)
        total_equity = None
        for name in ["Total Stockholder Equity", "Stockholders Equity",
                     "Total Equity", "Common Stock Equity"]:
            if name in balance.index:
                total_equity = balance.loc[name].iloc[0]
                break

        # Get dividends (most recent 12 months)
        dividends = stock.dividends
        annual_div = None
        if not dividends.empty:
            annual_div = dividends.resample("YE").sum().iloc[-1]

        results = {}

        # Calculate ROE
        if net_income is not None and total_equity is not None and total_equity != 0:
            roe = float(net_income / total_equity)
            if -1 < roe < 2:  # Sanity check (ROE should be between -100% and 200%)
                results["roe"] = roe

        # Calculate payout ratio
        if annual_div is not None and net_income is not None and net_income > 0:
            payout = float(annual_div / net_income)
            if 0 <= payout <= 2:  # Sanity check (0% to 200%)
                results["payout_ratio"] = payout

        return results

    except Exception as e:
        print(f"    Error: {e}")
        return {}


def update_roe_payout():
    """Update stock_metrics.csv with ROE and payout from yfinance."""
    print("=" * 60)
    print("FETCHING ROE AND PAYOUT FROM YFINANCE FINANCIALS")
    print("=" * 60)

    path = DATA_DIR / "stock_metrics.csv"
    df = pd.read_csv(path)

    updated = 0
    still_missing = []

    for idx, row in df.iterrows():
        ticker = row["ticker"]
        print(f"\n[{idx + 1}/{len(df)}] {ticker}")

        # Check if we already have values
        has_roe = pd.notna(row.get("roe")) and row.get("roe") is not None
        has_payout = pd.notna(row.get("payout_ratio")) and row.get("payout_ratio") is not None

        if has_roe and has_payout:
            print("  Already have ROE and payout — skipping")
            continue

        print("  Fetching financial statements...")
        results = fetch_roe_payout(ticker)

        found_any = False
        if results:
            if not has_roe and "roe" in results:
                df.at[idx, "roe"] = results["roe"]
                print(f"    ROE: {results['roe']:.1%}")
                found_any = True
            if not has_payout and "payout_ratio" in results:
                df.at[idx, "payout_ratio"] = results["payout_ratio"]
                print(f"    Payout: {results['payout_ratio']:.1%}")
                found_any = True

        if found_any:
            updated += 1
        else:
            print("    No data available from yfinance")
            still_missing.append(ticker)

        time.sleep(0.5)

    # Save updated CSV
    df.to_csv(path, index=False)
    print(f"\n{'=' * 60}")
    print(f"Updated {updated} stocks with ROE/payout")
    print(f"Saved to {path}")
    if still_missing:
        print(f"\nStill missing ROE/payout for {len(still_missing)} stocks:")
        for t in still_missing:
            print(f"  - {t}")
    else:
        print("\nAll stocks have ROE and payout!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    update_roe_payout()