"""
Tests that the packaged engine reproduces Phase 1 outputs.
Run: python -m pytest tests/test_engine.py -v
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from kse.engine import (
    load_monthly_returns, load_portfolio_returns, run_sip,
    blend_tier_returns, blended_cgt_rate, drawdown, probability_of_loss
)

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def test_load_monthly_returns():
    """Should load 179 monthly observations."""
    returns = load_monthly_returns()
    assert len(returns) == 179, f"Expected 179, got {len(returns)}"
    assert not np.any(np.isnan(returns)), "NaN in returns"


def test_sip_reproduces_aggressive():
    """SIP on Aggressive tier returns should match sip_aggressive.csv."""
    rets_df = load_portfolio_returns()
    returns = rets_df["Aggressive"].values

    result = run_sip(returns, monthly_amount=50000, tx_cost_pct=0.0)

    phase1 = pd.read_csv(DATA_DIR / "sip_aggressive.csv")

    assert abs(result["final_value"] - phase1["Portfolio_Value"].iloc[-1]) < 1, \
        f"Final value mismatch: engine={result['final_value']}, " \
        f"phase1={phase1['Portfolio_Value'].iloc[-1]}"


def test_sip_reproduces_worst_start():
    """SIP from May 2017 should match sip_worst_start.csv."""
    rets_df = load_portfolio_returns()
    start_idx = rets_df.index.get_loc(pd.Timestamp("2017-05-31"))
    returns = rets_df["Aggressive"].iloc[start_idx:].values

    result = run_sip(returns, monthly_amount=50000, tx_cost_pct=0.0)

    phase1 = pd.read_csv(DATA_DIR / "sip_worst_start.csv")

    assert abs(result["final_value"] - phase1["Portfolio_Value"].iloc[-1]) < 1, \
        f"Worst start mismatch: engine={result['final_value']}, " \
        f"phase1={phase1['Portfolio_Value'].iloc[-1]}"


def test_cgt_rate_15yr():
    """Blended CGT for 15-year SIP should be ~1.83%."""
    rate = blended_cgt_rate(180)
    assert abs(rate - 0.0183) < 0.001, f"CGT rate: {rate}"


def test_cgt_rate_1yr():
    """Blended CGT for 1-year SIP should be 15%."""
    rate = blended_cgt_rate(12)
    assert abs(rate - 0.15) < 0.001, f"CGT rate: {rate}"


def test_drawdown():
    """Drawdown of a simple series."""
    values = [100, 110, 90, 95, 105]
    dd = drawdown(values)
    assert abs(dd[2] - (-0.1818)) < 0.001


def test_probability_of_loss():
    """Probability of loss should decrease with holding period."""
    returns = load_monthly_returns()
    probs = probability_of_loss(returns)
    assert probs[1] > probs[12], "1-month loss prob should exceed 12-month"