"""
Scenario definitions and mixture distribution.

Scenarios are loaded from config/scenarios.yaml.
Expected equity returns are calculated dynamically by kse/blocks.py.
Income sleeve returns are calculated from the policy rate in each scenario
(income_return = policy_rate - 0.01, approximating money market fund returns).
"""

import yaml
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "scenarios.yaml"

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

SCENARIOS = CONFIG["scenarios"]
DEFAULT_WEIGHTS = {s: SCENARIOS[s]["weight"] for s in SCENARIOS}


def get_scenario(name, custom_weight=None):
    """
    Get full scenario parameters including dynamically calculated returns.

    Parameters:
        name: "Bull", "Base", or "Bear"
        custom_weight: optional override of the weight

    Returns:
        dict with equity_return, income_return, inflation, depreciation, etc.
    """
    from kse.blocks import expected_return

    scenario = SCENARIOS[name].copy()

    # Calculate expected equity return dynamically
    ret, val_change, components = expected_return(name)
    scenario["equity_return"] = ret
    scenario["valuation_change"] = val_change

    # Calculate income return from policy rate
    # Money market funds typically earn policy_rate - 1%
    scenario["income_return"] = max(0, scenario["policy_rate"] - 0.01)

    if custom_weight is not None:
        scenario["weight"] = custom_weight

    return scenario


def get_all_scenarios():
    """
    Get all three scenarios with dynamically calculated returns.

    Returns:
        dict: {"Bull": {...}, "Base": {...}, "Bear": {...}}
    """
    return {s: get_scenario(s) for s in ["Bull", "Base", "Bear"]}


def blend_scenarios(weights=None):
    """
    Blend scenarios by weight to get expected values.

    Returns dict with blended equity_return, income_return, inflation,
    depreciation, etc.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    total = sum(weights.values())
    blended = {}

    all_scenarios = get_all_scenarios()

    for key in ["equity_return", "income_return", "inflation",
                "depreciation", "gdp_growth", "policy_rate"]:
        blended[key] = sum(
            all_scenarios[s][key] * weights[s] for s in all_scenarios
        ) / total

    return blended


def print_scenario_table():
    """Print the full scenario table with dynamically calculated values."""
    all_scenarios = get_all_scenarios()

    print("=" * 100)
    print("SCENARIO TABLE — KSE 100 Outlook (DYNAMIC)")
    print("=" * 100)
    print(f"{'Parameter':<30} {'Bull (25%)':>15} {'Base (50%)':>15} {'Bear (25%)':>15}")
    print("-" * 100)

    rows = [
        ("equity_return", "Equity return", "%"),
        ("income_return", "Income sleeve", "%"),
        ("policy_rate", "SBP policy rate", "%"),
        ("inflation", "CPI inflation", "%"),
        ("depreciation", "PKR/USD deprec.", "%"),
        ("gdp_growth", "Real GDP growth", "%"),
        ("valuation_change", "Valuation change", "%"),
    ]

    for key, label, fmt in rows:
        values = []
        for s in ["Bull", "Base", "Bear"]:
            val = all_scenarios[s][key]
            values.append(f"{val:>14.1%}")
        print(f"{label:<30} {values[0]} {values[1]} {values[2]}")

    print("=" * 100)

    # Print blended values
    blended = blend_scenarios()
    print(f"\nBlended (default weights 25/50/25):")
    print(f"  Equity return:  {blended['equity_return']:.1%}")
    print(f"  Income return:   {blended['income_return']:.1%}")
    print(f"  Inflation:       {blended['inflation']:.1%}")
    print(f"  Depreciation:    {blended['depreciation']:.1%}")


if __name__ == "__main__":
    print_scenario_table()