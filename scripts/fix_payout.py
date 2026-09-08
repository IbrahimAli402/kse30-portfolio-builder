"""
Fix payout ratios by using dividends_per_share / EPS instead of
total_dividends / net_income.

Usage:
    python scripts/fix_payout.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def fetch_payout(ticker, yf_suffix=".KA"):
    """Fetch payout ratio using dividends_per_share / EPS."""
    import yfinance as yf
    yf_ticker = f"{ticker}{yf_suffix}"

    try:
        stock = yf.Ticker(yf_ticker)
        info = stock.info

        # Get EPS from yfinance info
        eps = info.get("trailingEps")
        if eps is None or eps <= 0:
            return None

        # Get dividends per share from yfinance info
        dps = info.get("trailingAnnualDividendRate")
        if dps is None or dps <= 0:
            # Try calculating from dividends history
            dividends = stock.dividends
            if dividends.empty:
                return None
            # Strip timezone
            if dividends.index.tz is not None:
                dividends.index = dividends.index.tz_localize(None)
            recent = dividends[dividends.index > pd.Timestamp.now() - pd.DateOffset(years=1)]
            dps = recent.sum()
            if dps <= 0:
                return None

        payout = float(dps / eps)

        # Sanity check (0% to 200%)
        if 0 <= payout <= 2:
            return payout
        return None

    except Exception as e:
        return None


def update_payouts():
    """Update stock_metrics.csv with corrected payout ratios."""
    print("=" * 60)
    print("FIXING PAYOUT RATIOS")
    print("=" * 60)

    path = DATA_DIR / "stock_metrics.csv"
    df = pd.read_csv(path)

    updated = 0
    still_missing = []

    for idx, row in df.iterrows():
        ticker = row["ticker"]
        print(f"\n[{idx + 1}/{len(df)}] {ticker}")

        # Check if payout is already reasonable (> 0.01)
        current_payout = row.get("payout_ratio")
        if pd.notna(current_payout) and current_payout > 0.01:
            print(f"  Already have payout: {current_payout:.1%} — skipping")
            continue

        print("  Fetching payout...")
        payout = fetch_payout(ticker)

        if payout is not None:
            df.at[idx, "payout_ratio"] = payout
            print(f"    Payout: {payout:.1%}")
            updated += 1
        else:
            print("    No data available")
            still_missing.append(ticker)

        time.sleep(0.5)

    # Save
    df.to_csv(path, index=False)
    print(f"\n{'=' * 60}")
    print(f"Updated {updated} stocks with payout ratios")
    if still_missing:
        print(f"\nStill missing payout for {len(still_missing)} stocks:")
        for t in still_missing:
            print(f"  - {t}")
    else:
        print("\nAll stocks have payout ratios!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    update_payouts()