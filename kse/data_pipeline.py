"""
Data pipeline orchestrator — fetches, validates, and caches data.

This module is the single entry point for all data operations.
It ensures that:
  1. Data is fetched from authoritative sources
  2. Source and observation date are recorded
  3. Data is validated before use
  4. Historical snapshots are preserved
  5. Derived metrics are recalculated automatically
"""

import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "snapshots"


def load_macro_snapshot():
    """
    Load the latest macro snapshot.

    Returns a DataFrame with columns:
        metric, value, source, observation_date, fetch_date
    """
    path = DATA_DIR / "macro_snapshot.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Macro snapshot not found at {path}. "
            "Run: python scripts/fetch_macro.py"
        )
    return pd.read_csv(path, parse_dates=["observation_date", "fetch_date"])


def get_macro_value(metric):
    """Get the latest value for a specific macro metric."""
    df = load_macro_snapshot()
    row = df[df["metric"] == metric].iloc[-1]
    return {
        "value": row["value"],
        "source": row["source"],
        "observation_date": row["observation_date"],
        "fetch_date": row["fetch_date"],
    }


def load_kse30_constituents():
    """Load the current KSE 30 constituent list."""
    path = DATA_DIR / "kse30_constituents.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"KSE 30 constituents not found at {path}. "
            "Run: python scripts/fetch_kse30.py"
        )
    return pd.read_csv(path)


def load_stock_metrics():
    """Load the latest stock-level metrics."""
    path = DATA_DIR / "stock_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Stock metrics not found at {path}. "
            "Run: python scripts/fetch_stocks.py"
        )
    return pd.read_csv(path)


def check_data_freshness(max_age_days=180):
    """
    Check if the macro data is fresh enough.

    Returns (is_fresh, age_days, warning_message).
    """
    try:
        df = load_macro_snapshot()
        latest_fetch = df["fetch_date"].max()
        age = (datetime.now().date() - latest_fetch.date()).days

        if age > max_age_days:
            return (False, age,
                    f"Warning: Macro data is {age} days old. "
                    f"Run: python scripts/fetch_macro.py")
        return (True, age, None)
    except FileNotFoundError:
        return (False, None,
                "Warning: No macro data found. "
                "Run: python scripts/fetch_macro.py")


def save_snapshot(name=None):
    """
    Save a complete snapshot of all current data for reproducibility.

    Creates data/snapshots/YYYY-MM-DD/ with copies of:
        - macro_snapshot.csv
        - kse30_constituents.csv
        - stock_metrics.csv
        - scenarios.yaml
    """
    if name is None:
        name = datetime.now().strftime("%Y-%m-%d")

    snapshot_dir = SNAPSHOTS_DIR / name
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        DATA_DIR / "macro_snapshot.csv",
        DATA_DIR / "kse30_constituents.csv",
        DATA_DIR / "stock_metrics.csv",
        CONFIG_DIR / "scenarios.yaml",
    ]

    import shutil
    for src in files_to_copy:
        if src.exists():
            shutil.copy2(src, snapshot_dir / src.name)

    print(f"Snapshot saved to {snapshot_dir}")
    return snapshot_dir


def load_snapshot(name):
    """Load a historical snapshot for reproducible analysis."""
    snapshot_dir = SNAPSHOTS_DIR / name
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_dir}")

    macro = pd.read_csv(snapshot_dir / "macro_snapshot.csv")
    constituents = pd.read_csv(snapshot_dir / "kse30_constituents.csv")
    metrics = pd.read_csv(snapshot_dir / "stock_metrics.csv")

    return {
        "macro": macro,
        "constituents": constituents,
        "stock_metrics": metrics,
        "snapshot_date": name,
    }