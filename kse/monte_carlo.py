"""
Monte Carlo pipeline: generate paths, blend scenarios, compute statistics.

This is the core of Phase 2. It replaces Phase 1's three straight lines
(22/15/5%) with a distribution of 5,000 paths, blended by scenario weight.
"""

import numpy as np
from kse.engine import (
    load_monthly_returns, blend_tier_returns_vectorized,
    run_sip_vectorized, TIERS, ANNUAL_FEE, TX_COST_PCT
)
from kse.scenarios import get_all_scenarios, DEFAULT_WEIGHTS
from kse.bootstrap import generate_scenario_paths


def run_monte_carlo(weights=None, tier="Aggressive", monthly_amount=50000,
                    horizon=120, method="bootstrap", seed=42, num_paths=5000):
    """
    Run full Monte Carlo simulation.

    Parameters:
        weights: dict {"Bull": 0.25, "Base": 0.50, "Bear": 0.25}
        tier: "Conservative", "Moderate", or "Aggressive"
        monthly_amount: PKR deposited each month
        horizon: investment horizon in months
        method: "bootstrap", "regime", or "garch"
        seed: random seed for reproducibility
        num_paths: total number of paths across all scenarios

    Returns:
        dict with:
          - portfolio_paths: (N, horizon) array of portfolio values over time
          - terminal_values: (N,) array of final portfolio values
          - max_drawdowns: (N,) array of max drawdown per path
          - underwater_months: (N,) array of underwater months per path
          - total_invested: total amount deposited
          - percentiles: dict with p10, p25, p50, p75, p90 arrays
          - probability_table: dict with outcome probabilities
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Generate blended return paths for each scenario
    all_returns = []
    returns = load_monthly_returns()
    total_weight = sum(weights.values())
    all_scenarios = get_all_scenarios()

    for scenario_name in all_scenarios:
        n_paths = int(num_paths * weights[scenario_name] / total_weight)
        if n_paths == 0:
            continue

        scenario = all_scenarios[scenario_name]
        expected_monthly = (1 + scenario["equity_return"]) ** (1 / 12) - 1

        # Generate equity return paths
        if method == "bootstrap":
            from kse.bootstrap import stationary_block_bootstrap
            equity_paths = stationary_block_bootstrap(
                returns, n_paths, horizon,
                expected_monthly_return=expected_monthly,
                seed=seed
            )
        elif method == "regime":
            from kse.regimes import generate_regime_paths
            equity_paths = generate_regime_paths(
                returns, expected_monthly, n_paths, horizon, seed
            )
        elif method == "garch":
            from kse.garch import generate_garch_paths
            equity_paths = generate_garch_paths(
                returns, expected_monthly, n_paths, horizon, seed
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Blend with income sleeve by tier
        blended = blend_tier_returns_vectorized(
            equity_paths, scenario["income_return"], tier
        )
        all_returns.append(blended)

    # Stack all paths
    all_returns = np.vstack(all_returns)

    # Apply SIP engine (vectorized)
    sip_result = run_sip_vectorized(all_returns, monthly_amount)

    # Compute percentiles at each month
    portfolio_paths = sip_result["portfolio_paths"]
    percentiles = compute_percentiles(portfolio_paths)

    # Compute probability table
    total_invested = monthly_amount * horizon
    prob_table = compute_probability_table(
        sip_result["terminal_values"],
        sip_result["max_drawdowns"],
        total_invested
    )

    return {
        "portfolio_paths": portfolio_paths,
        "terminal_values": sip_result["terminal_values"],
        "max_drawdowns": sip_result["max_drawdowns"],
        "underwater_months": sip_result["underwater_months"],
        "total_invested": total_invested,
        "percentiles": percentiles,
        "probability_table": prob_table,
        "weights": weights,
        "tier": tier,
        "monthly_amount": monthly_amount,
        "horizon": horizon,
        "method": method,
    }


def compute_percentiles(portfolio_paths):
    """Compute P10, P25, P50, P75, P90 at each month."""
    return {
        "p10": np.percentile(portfolio_paths, 10, axis=0),
        "p25": np.percentile(portfolio_paths, 25, axis=0),
        "p50": np.percentile(portfolio_paths, 50, axis=0),
        "p75": np.percentile(portfolio_paths, 75, axis=0),
        "p90": np.percentile(portfolio_paths, 90, axis=0),
    }


def compute_probability_table(terminal_values, max_drawdowns, total_invested):
    """
    Compute probability of various outcomes.

    This is the deliverable: honest probabilities, not point forecasts.
    """
    return {
        "above_1.5x_deposits": np.mean(terminal_values > 1.5 * total_invested),
        "above_2x_deposits": np.mean(terminal_values > 2.0 * total_invested),
        "above_deposits": np.mean(terminal_values > total_invested),
        "below_deposits": np.mean(terminal_values < total_invested),
        "drawdown_30pct": np.mean(max_drawdowns > 0.30),
        "drawdown_50pct": np.mean(max_drawdowns > 0.50),
    }


def print_monte_carlo_summary(result):
    """Print a summary of the Monte Carlo results."""
    p = result["percentiles"]
    pt = result["probability_table"]
    ti = result["total_invested"]

    print("=" * 70)
    print("MONTE CARLO RESULTS")
    print(f"  {result['monthly_amount']:,.0f}/month · {result['tier']} tier · "
          f"{result['horizon']} months · {result['method']}")
    print(f"  Weights: Bull={result['weights'].get('Bull', 0):.0%} "
          f"Base={result['weights'].get('Base', 0):.0%} "
          f"Bear={result['weights'].get('Bear', 0):.0%}")
    print("=" * 70)
    print(f"Total invested:          PKR {ti:,.0f}")
    print()
    print("Terminal wealth percentiles:")
    print(f"  P10 (worst case):      PKR {p['p10'][-1]:,.0f}")
    print(f"  P25:                   PKR {p['p25'][-1]:,.0f}")
    print(f"  P50 (median):          PKR {p['p50'][-1]:,.0f}")
    print(f"  P75:                   PKR {p['p75'][-1]:,.0f}")
    print(f"  P90 (best case):       PKR {p['p90'][-1]:,.0f}")
    print()
    print("Probabilities:")
    print(f"  Finish above 1.5× deposits:  {pt['above_1.5x_deposits']:.1%}")
    print(f"  Finish above 2× deposits:    {pt['above_2x_deposits']:.1%}")
    print(f"  Finish above deposits:       {pt['above_deposits']:.1%}")
    print(f"  Finish below deposits:       {pt['below_deposits']:.1%}")
    print(f"  30%+ drawdown along the way: {pt['drawdown_30pct']:.1%}")
    print(f"  50%+ drawdown along the way: {pt['drawdown_50pct']:.1%}")
    print("=" * 70)


if __name__ == "__main__":
    # Quick test: 5,000 paths, Aggressive tier, PKR 50k/month, 10 years
    result = run_monte_carlo(
        weights=DEFAULT_WEIGHTS,
        tier="Aggressive",
        monthly_amount=50000,
        horizon=120,
        method="bootstrap",
        seed=42,
        num_paths=5000
    )
    print_monte_carlo_summary(result)