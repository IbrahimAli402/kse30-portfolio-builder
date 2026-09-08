"""
Save a reproducible snapshot of all current data.

Usage:
    python scripts/snapshot.py
    python scripts/snapshot.py --name pre-publish-2026-09-09

Output:
    data/snapshots/YYYY-MM-DD/
        macro_snapshot.csv
        kse30_constituents.csv
        stock_metrics.csv
        scenarios.yaml
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kse.data_pipeline import save_snapshot

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None,
                        help="Snapshot name (default: today's date)")
    args = parser.parse_args()

    save_snapshot(args.name)