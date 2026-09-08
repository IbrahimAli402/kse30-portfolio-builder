"""
Two-state Markov regime switching model for KSE 100 returns.

Pakistan's index behaves like a regime market:
  - Long calm bull states (2-4 years)
  - Sharp bear states (1-2 years, 30-45% drawdowns)

The fitted transition matrix gives expected regime durations, which is
a more honest way to generate "a crash every N years" than a fixed assumption.
"""

import numpy as np
import warnings


def fit_regimes(returns):
    """
    Fit two-state Markov switching model on monthly returns.

    Returns:
        dict with transition_matrix, regime_means, regime_vars,
        smoothed_probabilities, result
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    import pandas as pd

    # Convert to pandas Series if needed
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    # Fit the model with regime-specific variances
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MarkovRegression(returns, k_regimes=2, trend='c',
                                 switching_variance=True)
        result = model.fit(maxiter=200, disp=False)

    # Extract parameters
    params = result.params

    # Get regime means
    regime_means = np.array([params[f'const[{i}]'] for i in range(2)])

    # Get regime variances — try regime-specific first, then single variance
    try:
        regime_vars = np.array([params[f'sigma2[{i}]'] for i in range(2)])
    except KeyError:
        # Single variance model — use same variance for both regimes
        single_var = params['sigma2']
        regime_vars = np.array([single_var, single_var])

    # Get transition probabilities
    p00 = params['p[0->0]']
    p10 = params['p[1->0]']

    # Build transition matrix: transition[i,j] = P(regime j | regime i)
    transition = np.array([[p00, 1 - p00], [p10, 1 - p10]])

    # Sort so regime 0 = calm (lower volatility)
    if regime_vars[0] > regime_vars[1]:
        regime_means = regime_means[::-1]
        regime_vars = regime_vars[::-1]
        transition = transition[::-1, ::-1]

    return {
        'transition_matrix': transition,
        'regime_means': regime_means,
        'regime_vars': regime_vars,
        'regime_stds': np.sqrt(np.maximum(regime_vars, 1e-10)),
        'smoothed_probabilities': result.smoothed_marginal_probabilities,
        'result': result,
    }


def simulate_regimes(fit_result, num_paths=10000, horizon=120,
                     expected_monthly_return=None, seed=42):
    """
    Simulate paths using the fitted transition matrix.

    Shifts regime means so the blended expected return matches the scenario.
    Vectorized across paths for speed.
    """
    rng = np.random.default_rng(seed)

    trans = fit_result['transition_matrix']
    means = fit_result['regime_means'].copy()
    stds = fit_result['regime_stds']

    # Shift means so blended expected return matches scenario
    if expected_monthly_return is not None:
        # Calculate stationary distribution
        p01 = trans[0, 1]  # Prob of going from bull to bear
        p10 = trans[1, 0]  # Prob of going from bear to bull

        if p01 + p10 > 0:
            pi_0 = p10 / (p01 + p10)  # Stationary prob of bull
            pi_1 = p01 / (p01 + p10)  # Stationary prob of bear
        else:
            pi_0, pi_1 = 0.5, 0.5

        current_blended = pi_0 * means[0] + pi_1 * means[1]
        shift = expected_monthly_return - current_blended
        means = means + shift

    # Vectorized simulation across all paths
    paths = np.zeros((num_paths, horizon))
    states = np.zeros(num_paths, dtype=int)  # All start in bull (regime 0)

    for t in range(horizon):
        # Generate returns for all paths at once
        path_means = means[states]
        path_stds = stds[states]
        paths[:, t] = rng.normal(path_means, path_stds)

        # Transition for all paths at once
        stay_probs = trans[states, states]
        transitions = rng.random(num_paths) > stay_probs
        states[transitions] = 1 - states[transitions]

    return paths


def generate_regime_paths(returns, expected_monthly_return, num_paths=10000,
                          horizon=120, seed=42):
    """Fit and simulate in one call. Falls back to bootstrap if fit fails."""
    try:
        fit = fit_regimes(returns)
        return simulate_regimes(fit, num_paths, horizon,
                                expected_monthly_return, seed)
    except Exception as e:
        print(f"Warning: Regime model failed ({e}), falling back to bootstrap")
        from kse.bootstrap import stationary_block_bootstrap
        return stationary_block_bootstrap(
            returns, num_paths, horizon,
            expected_monthly_return=expected_monthly_return,
            seed=seed
        )


def print_regime_summary(fit_result):
    """Print regime model summary."""
    trans = fit_result['transition_matrix']
    means = fit_result['regime_means']
    stds = fit_result['regime_stds']

    p00 = trans[0, 0]  # Bull -> Bull
    p11 = trans[1, 1]  # Bear -> Bear

    # Expected duration = 1 / (1 - p_ii)
    bull_duration = 1 / (1 - p00) if p00 < 1 else float('inf')
    bear_duration = 1 / (1 - p11) if p11 < 1 else float('inf')

    print("=" * 60)
    print("MARKOV REGIME SWITCHING MODEL")
    print("=" * 60)
    print(f"Transition matrix:")
    print(f"  Bull -> Bull: {p00:.3f}   Bull -> Bear: {1-p00:.3f}")
    print(f"  Bear -> Bull: {1-p11:.3f}   Bear -> Bear: {p11:.3f}")
    print()
    print(f"Regime 0 (Bull):")
    print(f"  Mean return:     {means[0]:.4f}/mo ({means[0]*12:.1%}/yr)")
    print(f"  Volatility:      {stds[0]:.4f}/mo ({stds[0]*np.sqrt(12):.1%}/yr)")
    print(f"  Expected duration: {bull_duration:.0f} months ({bull_duration/12:.1f} years)")
    print()
    print(f"Regime 1 (Bear):")
    print(f"  Mean return:     {means[1]:.4f}/mo ({means[1]*12:.1%}/yr)")
    print(f"  Volatility:      {stds[1]:.4f}/mo ({stds[1]*np.sqrt(12):.1%}/yr)")
    print(f"  Expected duration: {bear_duration:.0f} months ({bear_duration/12:.1f} years)")
    print("=" * 60)


if __name__ == "__main__":
    from kse.engine import load_monthly_returns
    returns = load_monthly_returns()
    fit = fit_regimes(returns)
    print_regime_summary(fit)