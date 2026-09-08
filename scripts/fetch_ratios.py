"""
Fetch financial ratios (P/E, P/B, ROE, payout) from yfinance .info.

Updates the existing stock_metrics.csv — does NOT re-download prices.
For PSX stocks, yfinance may not have all ratios. This script reports
which stocks still need manual entry.

Usage:
    python scripts/fetch_ratios.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import time

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def fetch_ratios_from_yfinance(ticker, yf_suffix=".KA"):
    """Try to fetch P/E, P/B, ROE, payout from yfinance .info."""
    import yfinance as yf
    yf_ticker = f"{ticker}{yf_suffix}"

    try:
        stock = yf.Ticker(yf_ticker)
        info = stock.info

        return {
            "pe": info.get("trailingPE"),
            "pb": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "payout_ratio": info.get("payoutRatio"),
        }
    except Exception as e:
        print(f"    Error fetching ratios for {ticker}: {e}")
        return {}


def update_stock_metrics():
    """Update stock_metrics.csv with yfinance ratios."""
    print("=" * 60)
    print("FETCHING FINANCIAL RATIOS FROM YFINANCE")
    print("=" * 60)

    path = DATA_DIR / "stock_metrics.csv"
    df = pd.read_csv(path)

    updated = 0
    still_missing = []

    for idx, row in df.iterrows():
        ticker = row["ticker"]
        print(f"\n[{idx + 1}/{len(df)}] {ticker} ({row['name']})")

        # Check if we already have all values
        has_pe = pd.notna(row.get("pe")) and row.get("pe") is not None
        has_pb = pd.notna(row.get("pb")) and row.get("pb") is not None
        has_roe = pd.notna(row.get("roe")) and row.get("roe") is not None
        has_payout = pd.notna(row.get("payout_ratio")) and row.get("payout_ratio") is not None

        if has_pe and has_pb and has_roe and has_payout:
            print("  All ratios already present — skipping")
            continue

        # Fetch from yfinance
        print("  Fetching from yfinance...")
        ratios = fetch_ratios_from_yfinance(ticker)

        found_any = False
        if ratios:
            if not has_pe and ratios.get("pe") is not None:
                df.at[idx, "pe"] = ratios["pe"]
                print(f"    P/E: {ratios['pe']}")
                found_any = True
            if not has_pb and ratios.get("pb") is not None:
                df.at[idx, "pb"] = ratios["pb"]
                print(f"    P/B: {ratios['pb']}")
                found_any = True
            if not has_roe and ratios.get("roe") is not None:
                df.at[idx, "roe"] = ratios["roe"]
                print(f"    ROE: {ratios['roe']}")
                found_any = True
            if not has_payout and ratios.get("payout_ratio") is not None:
                df.at[idx, "payout_ratio"] = ratios["payout_ratio"]
                print(f"    Payout: {ratios['payout_ratio']}")
                found_any = True

        if found_any:
            updated += 1
        else:
            print("    No ratios available from yfinance")
            still_missing.append(ticker)

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    # Save updated CSV
    df.to_csv(path, index=False)
    print(f"\n{'=' * 60}")
    print(f"Updated {updated} stocks with yfinance ratios")
    print(f"Saved to {path}")
    if still_missing:
        print(f"\nStill missing ratios for {len(still_missing)} stocks:")
        for t in still_missing:
            print(f"  - {t}")
        print("\nThese need manual entry from https://dps.psx.com.pk/")
    else:
        print("\nAll stocks have ratios!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    update_stock_metrics()