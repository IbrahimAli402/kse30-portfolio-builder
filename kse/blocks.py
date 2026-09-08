"""
Building block expected returns for KSE 100 scenarios.

Identity: Expected return = dividend_yield + real_earnings_growth
                                + inflation + valuation_change

For Bull and Base: standard identity works.
For Bear: nominal_earnings_growth used directly (identity breaks in crisis).

Observed values (dividend_yield, current P/E) are read dynamically from
data/processed/macro_snapshot.csv.
Forecast assumptions (real_earnings_growth, inflation, ending P/E) come from
config/scenarios.yaml.
"""

import yaml
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "scenarios.yaml"

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

BUILDING_BLOCKS = CONFIG["building_blocks"]
SCENARIOS_CFG = CONFIG["scenarios"]


def get_macro_values():
    """
    Load observed macro values from the data pipeline.

    Returns a dict with keys: dividend_yield, kse100_pe
    """
    from kse.data_pipeline import load_macro_snapshot

    df = load_macro_snapshot()

    def get_value(metric):
        row = df[df["metric"] == metric].iloc[-1]
        return float(row["value"])

    return {
        "dividend_yield": get_value("kse100_dividend_yield"),
        "kse100_pe": get_value("kse100_pe"),
    }


def valuation_change_annualized(starting_pe, ending_pe, horizon_years):
    """
    Annualized valuation change from P/E re-rating or de-rating.

    Example: P/E goes from 7 to 11 over 10 years
             (11/7)^(1/10) - 1 = 4.6%/year
    """
    if starting_pe <= 0 or horizon_years <= 0:
        return 0.0

    if ending_pe == "flat":
        return 0.0

    return (float(ending_pe) / starting_pe) ** (1 / horizon_years) - 1


def expected_return(scenario, macro_values=None):
    """
    Calculate expected nominal return from building blocks for a scenario.

    Parameters:
        scenario: "Bull", "Base", or "Bear"
        macro_values: dict from get_macro_values() (loaded if None)

    Returns:
        (expected_return, valuation_change, components_dict)
    """
    if macro_values is None:
        macro_values = get_macro_values()

    bb = BUILDING_BLOCKS[scenario]

    # Get observed values from macro data
    dividend_yield = macro_values["dividend_yield"]
    pe_start = macro_values["kse100_pe"]
    pe_end = bb["pe_end"]
    horizon = bb["horizon_years"]

    # Get forecast inflation from scenario config
    inflation = SCENARIOS_CFG[scenario]["inflation"]

    # Calculate valuation change
    val_change = valuation_change_annualized(pe_start, pe_end, horizon)

    # Calculate expected return
    if bb.get("use_nominal_growth", False):
        # Bear case: nominal earnings growth directly
        nominal_growth = bb["nominal_earnings_growth"]
        ret = dividend_yield + nominal_growth + val_change
    else:
        # Standard identity
        real_growth = bb["real_earnings_growth"]
        ret = dividend_yield + real_growth + inflation + val_change

    # Build components dict for display
    components = {
        "dividend_yield": dividend_yield,
        "inflation": inflation,
        "pe_start": pe_start,
        "pe_end": pe_end,
        "valuation_change": val_change,
        "horizon_years": horizon,
    }

    if bb.get("use_nominal_growth", False):
        components["nominal_earnings_growth"] = bb["nominal_earnings_growth"]
    else:
        components["real_earnings_growth"] = bb["real_earnings_growth"]

    return ret, val_change, components


def calculate_all_expected_returns():
    """
    Calculate expected returns for all three scenarios.

    Returns:
        dict: {
            "Bull": {"expected_return": float, "valuation_change": float, "components": dict},
            "Base": {...},
            "Bear": {...}
        }
    """
    macro_values = get_macro_values()

    results = {}
    for scenario in ["Bull", "Base", "Bear"]:
        ret, val_change, components = expected_return(scenario, macro_values)
        results[scenario] = {
            "expected_return": ret,
            "valuation_change": val_change,
            "components": components,
        }

    return results


def print_building_block_table():
    """Print the building block decomposition for all scenarios."""
    results = calculate_all_expected_returns()
    macro = get_macro_values()

    print("=" * 80)
    print("BUILDING BLOCK EXPECTED RETURNS (DYNAMIC)")
    print(f"  Macro data as of: {macro['dividend_yield']:.1%} div yield, "
          f"{macro['kse100_pe']:.1f}x P/E")
    print("=" * 80)
    print(f"{'Component':<25} {'Bull':>10} {'Base':>10} {'Bear':>10}")
    print("-" * 80)

    # Dividend yield (same for all, from macro)
    div_y = macro["dividend_yield"]
    print(f"{'dividend_yield':<25} {div_y:>10.1%} {div_y:>10.1%} {div_y:>10.1%}")

    # Earnings growth
    bull_g = BUILDING_BLOCKS["Bull"].get("real_earnings_growth", 0)
    base_g = BUILDING_BLOCKS["Base"].get("real_earnings_growth", 0)
    bear_g = BUILDING_BLOCKS["Bear"].get("nominal_earnings_growth", 0)
    print(f"{'earnings_growth':<25} {bull_g:>10.1%} {base_g:>10.1%} {bear_g:>10.1%}")

    # Inflation
    bull_i = SCENARIOS_CFG["Bull"]["inflation"]
    base_i = SCENARIOS_CFG["Base"]["inflation"]
    bear_i = SCENARIOS_CFG["Bear"]["inflation"]
    print(f"{'inflation':<25} {bull_i:>10.1%} {base_i:>10.1%} {'—':>10}")

    # Valuation change
    bull_vc = results["Bull"]["valuation_change"]
    base_vc = results["Base"]["valuation_change"]
    bear_vc = results["Bear"]["valuation_change"]
    print(f"{'valuation_change':<25} {bull_vc:>10.1%} {base_vc:>10.1%} {bear_vc:>10.1%}")

    # Expected return
    print("-" * 80)
    bull_ret = results["Bull"]["expected_return"]
    base_ret = results["Base"]["expected_return"]
    bear_ret = results["Bear"]["expected_return"]
    print(f"{'EXPECTED RETURN':<25} {bull_ret:>10.1%} {base_ret:>10.1%} {bear_ret:>10.1%}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    print_building_block_table()