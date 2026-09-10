"""Update stock_metrics.csv with avg_volume from existing daily price CSVs."""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "stocks"

df = pd.read_csv(DATA_DIR / "stock_metrics.csv")

avg_volumes = []
for ticker in df["ticker"]:
    path = RAW_DIR / f"{ticker}.csv"
    if path.exists():
        raw = pd.read_csv(path, index_col=0)
        raw.index = pd.to_datetime(raw.index, errors="coerce")
        raw = raw[raw.index.notna()]
        if "Volume" in raw.columns and "Close" in raw.columns:
            vol = raw["Volume"].dropna()
            close = raw["Close"].dropna()
            if len(vol) >= 30 and len(close) > 0:
                latest_price = close.iloc[-1]
                avg_vol = float(vol.tail(30).mean() * latest_price)
                avg_volumes.append(avg_vol)
            else:
                avg_volumes.append(None)
        else:
            avg_volumes.append(None)
    else:
        avg_volumes.append(None)

df["avg_volume"] = avg_volumes
df.to_csv(DATA_DIR / "stock_metrics.csv", index=False)

print(f"Updated {len(df)} stocks with avg_volume data")
print(f"Stocks with volume data: {sum(1 for v in avg_volumes if v is not None)}")
print(f"Stocks without volume data: {sum(1 for v in avg_volumes if v is None)}")