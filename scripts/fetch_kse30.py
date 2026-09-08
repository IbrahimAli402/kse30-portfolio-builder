"""
Fetch the current KSE 30 constituent list from PSX.

Tries to fetch from the PSX website first. If that fails, falls back
to the manually maintained list in config/kse30_constituents.yaml.

Usage:
    python scripts/fetch_kse30.py

Output:
    data/processed/kse30_constituents.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml
import requests
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
CONFIG_DIR = Path(__file__).parent.parent / "config"


def fetch_from_psx():
    """
    Try to fetch KSE 30 constituents from the PSX website.

    The PSX website may not have a clean API, so this uses a best-effort
    HTTP request. If it fails, we fall back to the manual list.
    """
    try:
        url = "https://www.psx.com.pk/api/index/PSX"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "constituents" in data:
                return data["constituents"]
        print("  PSX API returned no constituents, using fallback")
    except Exception as e:
        print(f"  Could not fetch from PSX: {e}")
    return None


def load_fallback_list():
    """Load the manually maintained fallback list."""
    with open(CONFIG_DIR / "kse30_constituents.yaml") as f:
        config = yaml.safe_load(f)
    return config["constituents"]


def fetch_kse30():
    """Fetch KSE 30 constituents and save to CSV."""
    print("=" * 60)
    print("FETCHING KSE 30 CONSTITUENTS")
    print("=" * 60)

    print("\n1. Trying PSX website...")
    constituents = fetch_from_psx()

    source = "PSX website"
    if constituents is None:
        print("2. Using fallback list from config/kse30_constituents.yaml")
        constituents = load_fallback_list()
        source = "config/kse30_constituents.yaml (manual fallback)"

    df = pd.DataFrame(constituents)
    df["fetch_date"] = datetime.now().date()
    df["source"] = source

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "kse30_constituents.csv", index=False)

    print(f"\nSaved {len(df)} constituents to {DATA_DIR / 'kse30_constituents.csv'}")
    print(f"Source: {source}")
    print(f"Fetch date: {datetime.now().date()}")
    print(f"\nSectors:")
    print(df.groupby("sector").size().to_string())

    return df


if __name__ == "__main__":
    fetch_kse30()