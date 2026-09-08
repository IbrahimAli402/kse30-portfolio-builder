"""
Fetch latest macro values from authoritative sources.

For PKR/USD: yfinance (automatic)
For other values: manual entry from SBP/PBS websites

Every value records its source and observation date.

Usage:
    python scripts/fetch_macro.py

Output:
    data/processed/macro_snapshot.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
CONFIG_DIR = Path(__file__).parent.parent / "config"

with open(CONFIG_DIR / "sources.yaml") as f:
    SOURCES = yaml.safe_load(f)


def fetch_pkr_usd():
    """Fetch PKR/USD exchange rate from yfinance."""
    import yfinance as yf
    ticker = yf.Ticker("USDPKR=X")
    data = ticker.history(period="5d")
    if data.empty:
        raise ValueError("Could not fetch PKR/USD from yfinance")
    rate = data["Close"].iloc[-1]
    obs_date = data.index[-1].date()
    return {
        "metric": "pkr_usd",
        "value": float(rate),
        "source": "yfinance (USDPKR=X)",
        "observation_date": pd.Timestamp(obs_date),
        "fetch_date": pd.Timestamp(datetime.now().date()),
    }


def manual_entry(metric, prompt_text, source_url, default_value=None):
    """Prompt user to enter a value manually with source recording."""
    print(f"\n  {prompt_text}")
    print(f"  Source: {source_url}")

    default_str = f" [default: {default_value}]" if default_value is not None else ""
    value_input = input(f"  Enter value:{default_str} ").strip()

    if not value_input and default_value is not None:
        value = float(default_value)
    elif value_input:
        value = float(value_input)
    else:
        print("  Skipping — no value entered")
        return None

    return {
        "metric": metric,
        "value": value,
        "source": source_url,
        "observation_date": pd.Timestamp(datetime.now().date()),
        "fetch_date": pd.Timestamp(datetime.now().date()),
    }


def fetch_all_macro():
    """Fetch all macro values and save to CSV."""
    print("=" * 60)
    print("FETCHING MACRO DATA")
    print("=" * 60)

    records = []

    # 1. PKR/USD (automatic)
    print("\n1. PKR/USD exchange rate...")
    try:
        pkr_usd = fetch_pkr_usd()
        print(f"   Rate: {pkr_usd['value']:.2f} (source: {pkr_usd['source']})")
        records.append(pkr_usd)
    except Exception as e:
        print(f"   ERROR: {e}")
        print("   Falling back to manual entry")
        src = SOURCES["macro_sources"]["pkr_usd"]
        manual = manual_entry("pkr_usd", "Enter PKR/USD rate", src["url"], 280)
        if manual:
            records.append(manual)

    # 2. SBP Policy Rate (manual)
    print("\n2. SBP Policy Rate...")
    src = SOURCES["macro_sources"]["sbp_policy_rate"]
    policy = manual_entry("sbp_policy_rate", "Enter current SBP policy rate (%)",
                          src["url"], 12.0)
    if policy:
        policy["value"] = policy["value"] / 100
        records.append(policy)

    # 3. CPI Inflation (manual)
    print("\n3. CPI Inflation (year-on-year)...")
    src = SOURCES["macro_sources"]["cpi"]
    cpi = manual_entry("cpi", "Enter latest CPI inflation (%)",
                       src["url"], 7.0)
    if cpi:
        cpi["value"] = cpi["value"] / 100
        records.append(cpi)

    # 4. SBP Reserves (manual)
    print("\n4. SBP Foreign Exchange Reserves...")
    src = SOURCES["macro_sources"]["sbp_reserves"]
    reserves = manual_entry("sbp_reserves",
                            "Enter SBP reserves (months of imports)",
                            src["url"], 3.0)
    if reserves:
        records.append(reserves)

    # 5. KSE 100 Trailing P/E (manual)
    print("\n5. KSE 100 Trailing P/E ratio...")
    src = SOURCES["macro_sources"]["kse100_pe"]
    pe = manual_entry("kse100_pe", "Enter KSE 100 trailing P/E",
                      src["url"], 7.5)
    if pe:
        records.append(pe)

    # 6. KSE 100 Dividend Yield (manual)
    print("\n6. KSE 100 Dividend Yield...")
    src = SOURCES["macro_sources"]["kse100_dividend_yield"]
    div = manual_entry("kse100_dividend_yield",
                        "Enter KSE 100 dividend yield (%)",
                        src["url"], 6.5)
    if div:
        div["value"] = div["value"] / 100
        records.append(div)

    # 7. IMF Programme Status (text, not numeric)
    print("\n7. IMF Programme Status...")
    src = SOURCES["macro_sources"]["imf_status"]
    imf = input(f"  Enter IMF programme status [default: EFF underway]: ").strip()
    if not imf:
        imf = "EFF underway"
    records.append({
        "metric": "imf_status",
        "value": imf,
        "source": src["url"],
        "observation_date": pd.Timestamp(datetime.now().date()),
        "fetch_date": pd.Timestamp(datetime.now().date()),
    })

    # Save to CSV
    df = pd.DataFrame(records)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "macro_snapshot.csv", index=False)
    print(f"\n{'=' * 60}")
    print(f"Saved {len(records)} macro values to {DATA_DIR / 'macro_snapshot.csv'}")
    print(f"{'=' * 60}")
    print("\nSummary:")
    for _, row in df.iterrows():
        val = row["value"]
        if isinstance(val, float):
            print(f"  {row['metric']:<25} {val}")
        else:
            print(f"  {row['metric']:<25} {val}")
        print(f"    source: {row['source']}")
        print(f"    observed: {row['observation_date'].date()}")


if __name__ == "__main__":
    fetch_all_macro()