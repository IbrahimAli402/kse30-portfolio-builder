"""
Stationary block bootstrap for KSE 100 return paths.

Method:
  1. Demean the historical returns (subtract sample mean)
  2. For each path: sample blocks of random length (geometric, mean=block_length)
     until we have `horizon` months
  3. Add back the scenario's expected monthly return

The bootstrap supplies SHAPE (fat tails, clustering, drawdowns).
The scenario expected return supplies LOCATION (the mean).
"""

import numpy as np
from kse.engine import load_monthly_returns


def stationary_block_bootstrap(returns, num_paths=10000, horizon=120,
                               block_length=8, expected_monthly_return=None,
                               seed=42):
    """
    Stationary block bootstrap with demean-and-shift.

    Parameters:
        returns: historical monthly returns array (179 observations)
        num_paths: number of paths to generate
        horizon: length of each path in months
        block_length: mean block length (geometric distribution)
        expected_monthly_return: scenario's expected monthly return (location)
        seed: random seed for reproducibility

    Returns:
        numpy array of shape (num_paths, horizon)
    """
    rng = np.random.default_rng(seed)
    n = len(returns)

    # Demean: remove sample mean, keep shape
    sample_mean = returns.mean()
    demeaned = returns - sample_mean

    # If no expected return provided, use sample mean (pure bootstrap)
    if expected_monthly_return is None:
        expected_monthly_return = sample_mean

    paths = np.zeros((num_paths, horizon))

    for p in range(num_paths):
        t = 0
        while t < horizon:
            # Random block length from geometric distribution
            block_len = rng.geometric(1 / block_length)
            # Random starting point (wrap around)
            start = rng.integers(0, n)

            for j in range(min(block_len, horizon - t)):
                paths[p, t] = demeaned[(start + j) % n] + expected_monthly_return
                t += 1
                if t >= horizon:
                    break

    return paths


def generate_scenario_paths(scenario_name, num_paths=10000, horizon=120,
                            method="bootstrap", seed=42):
    """
    Generate simulated equity return paths for a single scenario.

    Parameters:
        scenario_name: "Bull", "Base", or "Bear"
        num_paths: number of paths
        horizon: path length in months
        method: "bootstrap", "regime", or "garch"
        seed: random seed

    Returns:
        numpy array (num_paths, horizon) of monthly equity returns
    """
    from kse.scenarios import get_scenario

    returns = load_monthly_returns()
    scenario = get_scenario(scenario_name)
    expected_monthly = (1 + scenario["equity_return"]) ** (1 / 12) - 1

    if method == "bootstrap":
        paths = stationary_block_bootstrap(
            returns, num_paths, horizon,
            block_length=8,
            expected_monthly_return=expected_monthly,
            seed=seed
        )
    elif method == "regime":
        from kse.regimes import generate_regime_paths
        paths = generate_regime_paths(
            returns, expected_monthly, num_paths, horizon, seed
        )
    elif method == "garch":
        from kse.garch import generate_garch_paths
        paths = generate_garch_paths(
            returns, expected_monthly, num_paths, horizon, seed
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return paths


def validate_bootstrap(returns, num_paths=1000, horizon=120, seed=42):
    """
    Validate that bootstrap paths reproduce historical statistics.

    Checks:
      - Path mean approx equals expected return
      - Path volatility approx equals historical volatility
      - Max drawdown distribution spans 15-45%
    """
    expected_monthly = returns.mean()
    paths = stationary_block_bootstrap(
        returns, num_paths, horizon,
        expected_monthly_return=expected_monthly,
        seed=seed
    )

    # Path statistics
    path_means = paths.mean(axis=1)
    path_stds = paths.std(axis=1)

    # Historical statistics
    hist_mean = returns.mean()
    hist_std = returns.std()

    print("Bootstrap Validation")
    print("=" * 50)
    print(f"Historical monthly mean:     {hist_mean:.4f} ({hist_mean*12:.1%}/yr)")
    print(f"Bootstrap path mean:         {path_means.mean():.4f} ({path_means.mean()*12:.1%}/yr)")
    print(f"Historical monthly std:      {hist_std:.4f} ({hist_std*np.sqrt(12):.1%}/yr)")
    print(f"Bootstrap path std:          {path_stds.mean():.4f} ({path_stds.mean()*np.sqrt(12):.1%}/yr)")
    print(f"Path mean range:             [{path_means.min():.4f}, {path_means.max():.4f}]")
    print(f"Path std range:              [{path_stds.min():.4f}, {path_stds.max():.4f}]")

    # Drawdown check
    from kse.engine import drawdown
    max_dds = []
    for p in range(min(100, num_paths)):
        # Convert returns to cumulative index
        cum = np.cumprod(1 + paths[p])
        dd = drawdown(cum)
        max_dds.append(abs(dd.min()))
    max_dds = np.array(max_dds)

    print(f"Max drawdown (100 paths):    mean={max_dds.mean():.1%}, "
          f"range=[{max_dds.min():.1%}, {max_dds.max():.1%}]")
    print("=" * 50)


if __name__ == "__main__":
    returns = load_monthly_returns()
    validate_bootstrap(returns)