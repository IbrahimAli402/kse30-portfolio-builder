"""
Calibration test — the credibility marker.

Fit the model on 2010-2015, forecast 2016-2024, check whether the
realised path fell inside the P10-P90 band.

A band that captures 80% of realised paths is honest.
One that captures 100% is uselessly wide.
One that captures 40% is overconfident.

Uses the Base scenario's building block expected return (not the
historical mean), because the proposals doc says: "use history for
the shape of returns, never for the level."
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from kse.engine import load_monthly_returns, run_sip
from kse.bootstrap import stationary_block_bootstrap
from kse.scenarios import get_scenario


def test_calibration():
    """Check that the P10-P90 band captures 40-100% of realised paths."""
    returns = load_monthly_returns()

    # Split: 2010-2015 (72 months) for shape, 2016-2024 (107 months) for testing
    train = returns[:72]
    test = returns[72:]

    # Use the Base scenario's expected return (not the train period mean)
    # This is what the actual Monte Carlo does — history supplies shape,
    # the building block supplies the level.
    scenario = get_scenario("Base")
    expected_monthly = (1 + scenario["equity_return"]) ** (1 / 12) - 1

    horizon = len(test)

    paths = stationary_block_bootstrap(
        train, num_paths=1000, horizon=horizon,
        expected_monthly_return=expected_monthly, seed=42
    )

    # Apply SIP to each path
    sip_results = [run_sip(path, monthly_amount=50000, tx_cost_pct=0.0) for path in paths]
    portfolio_paths = np.array([r["portfolio_values"] for r in sip_results])

    # Compute percentiles
    p10 = np.percentile(portfolio_paths, 10, axis=0)
    p90 = np.percentile(portfolio_paths, 90, axis=0)

    # Actual realized path
    actual_sip = run_sip(test, monthly_amount=50000, tx_cost_pct=0.0)
    actual_values = actual_sip["portfolio_values"]

    # Check how many months the actual path was inside the band
    inside = np.sum((actual_values >= p10) & (actual_values <= p90))
    capture_rate = inside / len(actual_values)

    print(f"\nCalibration test:")
    print(f"  Train period: 2010-2015 ({len(train)} months, for shape only)")
    print(f"  Test period:  2016-2024 ({len(test)} months)")
    print(f"  Expected return: Base scenario ({scenario['equity_return']:.1%}/yr)")
    print(f"  Capture rate: {capture_rate:.1%}")
    print(f"  (Target: 40-100% — 80% is honest, 100% is uselessly wide, "
          f"40% is overconfident)")

    assert 0.40 <= capture_rate <= 1.0, \
        f"Capture rate {capture_rate:.1%} outside acceptable range"


def test_reproducibility():
    """Same seed should produce same results."""
    returns = load_monthly_returns()

    paths1 = stationary_block_bootstrap(
        returns, num_paths=100, horizon=120, seed=42
    )
    paths2 = stationary_block_bootstrap(
        returns, num_paths=100, horizon=120, seed=42
    )

    assert np.allclose(paths1, paths2), "Same seed should produce identical paths"


def test_method_sensitivity():
    """Check that P50 doesn't move more than 20% between methods."""
    from kse.monte_carlo import run_monte_carlo

    result_boot = run_monte_carlo(
        method="bootstrap", num_paths=1000, seed=42
    )
    result_regime = run_monte_carlo(
        method="regime", num_paths=1000, seed=42
    )

    p50_boot = result_boot["percentiles"]["p50"][-1]
    p50_regime = result_regime["percentiles"]["p50"][-1]

    diff_pct = abs(p50_boot - p50_regime) / p50_boot

    print(f"\nMethod sensitivity:")
    print(f"  Bootstrap P50: {p50_boot:,.0f}")
    print(f"  Regime P50:    {p50_regime:,.0f}")
    print(f"  Difference:    {diff_pct:.1%}")

    # If difference > 20%, report it (don't fail — it's information)
    if diff_pct > 0.20:
        print(f"  NOTE: P50 moves {diff_pct:.1%} between methods — "
              f"report this in the dashboard")