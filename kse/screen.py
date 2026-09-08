"""
5-pillar stock screen for KSE 30 stocks.

Pillars:
  1. Performance   — 5yr TR CAGR, volatility, max drawdown, beta
  2. Fundamentals  — ROE
  3. Valuation     — P/E, P/B
  4. Dividend quality — Dividend yield, years without cut, growth
  5. Risk          — Beta, drawdown, volatility

Each pillar scored 1-5. Composite = weighted average.

Scores are calculated dynamically from data/processed/stock_metrics.csv.
If a metric is missing, that pillar scores based on available metrics only.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_stock_data():
    """Load stock metrics from the data pipeline."""
    from kse.data_pipeline import load_stock_metrics
    return load_stock_metrics()


def score_performance(row):
    """Score 1-5 based on 5yr TR CAGR and max drawdown."""
    cagr = row.get("tr_cagr_5y")
    max_dd = row.get("max_drawdown")

    if pd.isna(cagr):
        return 3  # Neutral if no data

    # Score by CAGR
    if cagr > 0.20:
        score = 5
    elif cagr > 0.15:
        score = 4
    elif cagr > 0.10:
        score = 3
    elif cagr > 0.05:
        score = 2
    else:
        score = 1

    # Penalty for high drawdown
    if not pd.isna(max_dd) and max_dd > 0.45:
        score = max(1, score - 1)

    return score


def score_fundamentals(row):
    """Score 1-5 based on ROE."""
    roe = row.get("roe")

    if pd.isna(roe):
        return 3  # Neutral if no data

    if roe > 0.30:
        return 5
    elif roe > 0.25:
        return 4
    elif roe > 0.20:
        return 3
    elif roe > 0.15:
        return 2
    else:
        return 1


def score_valuation(row):
    """Score 1-5 based on P/E and P/B (lower = cheaper = higher score)."""
    pe = row.get("pe")
    pb = row.get("pb")

    pe_score = 3  # Default neutral
    pb_score = 3

    if not pd.isna(pe):
        if pe < 4.5:
            pe_score = 5
        elif pe < 5.5:
            pe_score = 4
        elif pe < 7.0:
            pe_score = 3
        elif pe < 9.0:
            pe_score = 2
        else:
            pe_score = 1

    if not pd.isna(pb):
        if pb < 0.8:
            pb_score = 5
        elif pb < 1.2:
            pb_score = 4
        elif pb < 2.0:
            pb_score = 3
        elif pb < 3.0:
            pb_score = 2
        else:
            pb_score = 1

    # Average of P/E and P/B scores
    return int(np.round((pe_score + pb_score) / 2))


def score_dividend_quality(row):
    """Score 1-5 based on dividend yield, years without cut, growth."""
    div_yield = row.get("dividend_yield")
    years = row.get("years_without_div_cut")
    growth = row.get("div_growth_5y")

    if pd.isna(div_yield) and pd.isna(years):
        return 3  # Neutral if no data

    # Start with yield-based score
    if not pd.isna(div_yield):
        if div_yield > 0.08:
            score = 5
        elif div_yield > 0.06:
            score = 4
        elif div_yield > 0.04:
            score = 3
        elif div_yield > 0.02:
            score = 2
        else:
            score = 1
    else:
        score = 3

    # Bonus for consistency
    if not pd.isna(years):
        if years >= 10:
            score = min(5, score + 1)
        elif years >= 5:
            pass  # No change
        elif years < 3:
            score = max(1, score - 1)

    # Bonus for growth
    if not pd.isna(growth) and growth > 0.10:
        score = min(5, score + 1)

    return score


def score_risk(row):
    """Score 1-5 based on drawdown and volatility."""
    max_dd = row.get("max_drawdown")
    vol = row.get("volatility")

    if pd.isna(max_dd) and pd.isna(vol):
        return 3  # Neutral if no data

    # Score by drawdown (lower drawdown = higher score)
    if not pd.isna(max_dd):
        if max_dd < 0.35:
            dd_score = 5
        elif max_dd < 0.40:
            dd_score = 4
        elif max_dd < 0.45:
            dd_score = 3
        elif max_dd < 0.50:
            dd_score = 2
        else:
            dd_score = 1
    else:
        dd_score = 3

    # Adjust by volatility
    if not pd.isna(vol):
        if vol > 0.30:
            dd_score = max(1, dd_score - 1)
        elif vol < 0.15:
            dd_score = min(5, dd_score + 1)

    return dd_score


def score_stock(row):
    """Composite score across all 5 pillars."""
    scores = {
        "Performance": score_performance(row),
        "Fundamentals": score_fundamentals(row),
        "Valuation": score_valuation(row),
        "Dividend Quality": score_dividend_quality(row),
        "Risk": score_risk(row),
    }

    composite = np.mean(list(scores.values()))
    return scores, composite


def screen_all():
    """
    Screen all KSE 30 stocks and return results DataFrame.

    Returns:
        DataFrame with columns: Ticker, Name, Sector, Perf, Fund, Val,
        Div, Risk, Composite (sorted by Composite descending)
    """
    df = load_stock_data()

    results = []
    for _, row in df.iterrows():
        scores, composite = score_stock(row)
        results.append({
            "Ticker": row["ticker"],
            "Name": row["name"],
            "Sector": row["sector"],
            "Perf": scores["Performance"],
            "Fund": scores["Fundamentals"],
            "Val": scores["Valuation"],
            "Div": scores["Dividend Quality"],
            "Risk": scores["Risk"],
            "Composite": composite,
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("Composite", ascending=False)
    return result_df


def print_screen_results():
    """Print the screening results."""
    df = screen_all()

    print("=" * 90)
    print("STOCK SCREEN — 5-Pillar Composite Scores (DYNAMIC)")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)
    print()
    print("Sector averages:")
    print(df.groupby("Sector")["Composite"].mean().sort_values(ascending=False).to_string())
    print()
    print("Top 5 stocks:")
    print(df.head(5)[["Ticker", "Name", "Sector", "Composite"]].to_string(index=False))
    print()
    print("Bottom 5 stocks:")
    print(df.tail(5)[["Ticker", "Name", "Sector", "Composite"]].to_string(index=False))


if __name__ == "__main__":
    print_screen_results()