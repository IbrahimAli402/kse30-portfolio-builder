"""
Nightly price update script.

Fetches the latest prices and dividends for KSE 30 stocks from yfinance.
Updates the existing stock_metrics.csv WITHOUT overwriting manual financials
(P/E, P/B, ROE, payout ratio).

Designed to run automatically via GitHub Actions.

Usage:
    python scripts/update_prices.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import yaml
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "stocks"
YF_SUFFIX = ".KA"


def load_constituents():
    """Load KSE 30 constituents."""
    path = DATA_DIR / "kse30_constituents.csv"
    return pd.read_csv(path)


def fetch_latest_price(ticker):
    """Fetch latest price and 5y daily history from yfinance."""
    import yfinance as yf
    yf_ticker = f"{ticker}{YF_SUFFIX}"
    try:
        data = yf.download(yf_ticker, period="5y", progress=False)
        if data.empty:
            return None, None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.to_csv(RAW_DIR / f"{ticker}.csv")
        latest_price = float(data["Close"].iloc[-1])
        return latest_price, data
    except Exception as e:
        print(f"    Error fetching {ticker}: {e}")
        return None, None


def calculate_price_metrics(ticker):
    """Calculate price-based metrics from fetched data."""
    path = RAW_DIR / f"{ticker}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]

    if "Close" not in df.columns:
        return None

    close = df["Close"].dropna()
    if len(close) < 252:
        return None

    monthly = close.resample("ME").last()
    monthly_returns = monthly.pct_change().dropna()

    years = len(monthly_returns) / 12
    if years > 0 and len(close) > 1:
        cagr = (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1
    else:
        cagr = None

    vol = monthly_returns.std() * np.sqrt(12)

    peak = close.expanding().max()
    drawdown = (close - peak) / peak
    max_dd = abs(drawdown.min())

    beta = vol / 0.20

    latest_price = close.iloc[-1]

    # Average daily volume (last 30 days, in PKR)
    if "Volume" in df.columns:
        vol_data = df["Volume"].dropna()
        if len(vol_data) >= 30:
            avg_volume = float(vol_data.tail(30).mean() * latest_price)
        elif len(vol_data) > 0:
            avg_volume = float(vol_data.mean() * latest_price)
        else:
            avg_volume = None
    else:
        avg_volume = None

    return {
        "price": float(latest_price),
        "tr_cagr_5y": float(cagr) if cagr else None,
        "volatility": float(vol),
        "max_drawdown": float(max_dd),
        "beta": float(beta),
        "avg_volume": avg_volume,
        "data_points": len(close),
    }


def fetch_dividend_metrics(ticker, latest_price):
    """Fetch and calculate dividend metrics."""
    import yfinance as yf
    yf_ticker = f"{ticker}{YF_SUFFIX}"
    try:
        stock = yf.Ticker(yf_ticker)
        dividends = stock.dividends
        if dividends.empty:
            return {
                "dividend_yield": None,
                "years_without_div_cut": None,
                "div_growth_5y": None,
            }

        if dividends.index.tz is not None:
            dividends.index = dividends.index.tz_localize(None)

        recent = dividends[dividends.index > pd.Timestamp.now() - pd.DateOffset(years=1)]
        annual_div = recent.sum()
        div_yield = annual_div / latest_price if latest_price > 0 else None

        annual = dividends.resample("YE").sum()
        if len(annual) >= 2:
            cuts = 0
            for i in range(1, len(annual)):
                if annual.iloc[i] < annual.iloc[i - 1] * 0.9:
                    cuts += 1
            years_without_cut = len(annual) - cuts
        else:
            years_without_cut = None

        if len(annual) >= 6:
            start = annual.iloc[-6]
            end = annual.iloc[-1]
            div_growth = (end / start) ** (1 / 5) - 1
        else:
            div_growth = None

        return {
            "dividend_yield": float(div_yield) if div_yield else None,
            "years_without_div_cut": years_without_cut,
            "div_growth_5y": float(div_growth) if div_growth else None,
        }
    except:
        return {
            "dividend_yield": None,
            "years_without_div_cut": None,
            "div_growth_5y": None,
        }


def update_prices():
    """Update stock_metrics.csv with latest prices and dividends."""
    print("=" * 60)
    print("NIGHTLY PRICE UPDATE")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    constituents = load_constituents()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing metrics to preserve manual financials
    existing_path = DATA_DIR / "stock_metrics.csv"
    if existing_path.exists():
        existing_df = pd.read_csv(existing_path)
    else:
        existing_df = pd.DataFrame()

    updated_count = 0
    updated_rows = []

    for _, row in constituents.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        sector = row["sector"]

        print(f"\n[{row.name + 1}/{len(constituents)}] {ticker}")

        # Fetch latest price
        latest_price, _ = fetch_latest_price(ticker)
        if latest_price is None:
            print(f"  No price data — skipping")
            if not existing_df.empty and ticker in existing_df["ticker"].values:
                old_row = existing_df[existing_df["ticker"] == ticker].iloc[0]
                updated_rows.append(old_row.to_dict())
            continue

        # Calculate price metrics
        price_metrics = calculate_price_metrics(ticker)
        if price_metrics is None:
            print(f"  Insufficient price data — skipping")
            if not existing_df.empty and ticker in existing_df["ticker"].values:
                old_row = existing_df[existing_df["ticker"] == ticker].iloc[0]
                updated_rows.append(old_row.to_dict())
            continue

        # Fetch dividend metrics
        div_metrics = fetch_dividend_metrics(ticker, latest_price)

        # Get manual financials from existing data (don't overwrite)
        manual_financials = {}
        if not existing_df.empty and ticker in existing_df["ticker"].values:
            old_row = existing_df[existing_df["ticker"] == ticker].iloc[0]
            for col in ["pe", "pb", "roe", "payout_ratio"]:
                if col in old_row.index and pd.notna(old_row[col]):
                    manual_financials[col] = old_row[col]

        # Combine
        metrics = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            **price_metrics,
            **div_metrics,
            **manual_financials,
            "fetch_date": datetime.now().date(),
        }
        updated_rows.append(metrics)
        updated_count += 1
        print(f"  Price: {latest_price:,.0f}")

    # Save
    df = pd.DataFrame(updated_rows)
    df.to_csv(DATA_DIR / "stock_metrics.csv", index=False)
    print(f"\n{'=' * 60}")
    print(f"Updated {updated_count} stocks")
    print(f"Saved to {DATA_DIR / 'stock_metrics.csv'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    update_prices()