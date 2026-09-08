"""
Fetch stock-level data for KSE 30 constituents.

For each stock:
  - Daily prices from yfinance (may be incomplete for some PSX stocks)
  - Dividends from yfinance (may be incomplete)
  - Financial metrics: manual entry from PSX Data Portal (dps.psx.com.pk)

Usage:
    python scripts/fetch_stocks.py

Output:
    data/processed/stock_metrics.csv
    data/raw/stocks/{TICKER}.csv (daily prices)
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
CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_constituents():
    """Load KSE 30 constituents."""
    path = DATA_DIR / "kse30_constituents.csv"
    return pd.read_csv(path)


def fetch_stock_prices(ticker, yf_suffix=".KA"):
    """Fetch daily prices from yfinance."""
    import yfinance as yf
    yf_ticker = f"{ticker}{yf_suffix}"
    try:
        data = yf.download(yf_ticker, period="5y", progress=False)
        if data.empty:
            return None
        # Flatten multi-level columns if present (yfinance 0.2.x+)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.to_csv(RAW_DIR / f"{ticker}.csv")
        return data
    except Exception as e:
        print(f"    Error fetching {ticker}: {e}")
        return None


def calculate_price_metrics(ticker):
    """Calculate price-based metrics from fetched data."""
    path = RAW_DIR / f"{ticker}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, index_col=0)
    # Convert index to datetime, coercing errors (like "Ticker") to NaT
    df.index = pd.to_datetime(df.index, errors='coerce')
    # Drop rows with NaT index (header rows from yfinance)
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

    return {
        "price": float(latest_price),
        "tr_cagr_5y": float(cagr) if cagr else None,
        "volatility": float(vol),
        "max_drawdown": float(max_dd),
        "beta": float(beta),
        "data_points": len(close),
    }


def fetch_dividends(ticker, yf_suffix=".KA"):
    """Fetch dividend history from yfinance."""
    import yfinance as yf
    yf_ticker = f"{ticker}{yf_suffix}"
    try:
        stock = yf.Ticker(yf_ticker)
        dividends = stock.dividends
        if dividends.empty:
            return None
        return dividends
    except:
        return None


def calculate_dividend_metrics(ticker, latest_price):
    """Calculate dividend-based metrics."""
    dividends = fetch_dividends(ticker)
    if dividends is None or latest_price is None:
        return {
            "dividend_yield": None,
            "years_without_div_cut": None,
            "div_growth_5y": None,
        }

    # Strip timezone info to avoid tz-naive vs tz-aware comparison errors
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


def manual_financials(ticker, name):
    """Prompt for manual entry of financial metrics."""
    print(f"\n  --- {ticker} ({name}) ---")
    print(f"  Source: https://dps.psx.com.pk/ (PSX Data Portal)")
    print(f"  Press Enter to skip any field.")

    pe = input(f"  P/E ratio: ").strip()
    pb = input(f"  P/B ratio: ").strip()
    roe = input(f"  ROE (%): ").strip()
    payout = input(f"  Payout ratio (%): ").strip()

    return {
        "pe": float(pe) if pe else None,
        "pb": float(pb) if pb else None,
        "roe": float(roe) / 100 if roe else None,
        "payout_ratio": float(payout) / 100 if payout else None,
    }


def fetch_all_stocks():
    """Fetch data for all KSE 30 constituents."""
    print("=" * 60)
    print("FETCHING STOCK DATA FOR KSE 30 CONSTITUENTS")
    print("=" * 60)

    constituents = load_constituents()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for _, row in constituents.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        sector = row["sector"]

        print(f"\n[{row.name + 1}/{len(constituents)}] {ticker} ({name})")

        print(f"  Fetching prices...")
        fetch_stock_prices(ticker)
        price_metrics = calculate_price_metrics(ticker)

        if price_metrics is None:
            print(f"  No price data — skipping")
            continue

        print(f"  Fetching dividends...")
        div_metrics = calculate_dividend_metrics(ticker, price_metrics["price"])

        print(f"  Manual financials (press Enter to skip):")
        fin = manual_financials(ticker, name)

        metrics = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            **price_metrics,
            **div_metrics,
            **fin,
            "fetch_date": datetime.now().date(),
        }
        all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)
    df.to_csv(DATA_DIR / "stock_metrics.csv", index=False)
    print(f"\n{'=' * 60}")
    print(f"Saved {len(df)} stocks to {DATA_DIR / 'stock_metrics.csv'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    fetch_all_stocks()