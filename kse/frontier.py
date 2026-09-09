"""
Markowitz efficient frontier and diversification curve.

Computes:
1. Diversification curve — portfolio volatility vs number of stocks
2. Efficient frontier — minimum variance portfolios at each return level

Uses historical monthly returns from data/raw/stocks/{TICKER}.csv.

CAVEAT: The efficient frontier is based on historical covariance and is
subject to estimation error. The "optimal" portfolio may not be optimal
out-of-sample. This is a well-known limitation of Markowitz optimization.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "stocks"


def load_stock_return_matrix():
    """
    Load and align monthly returns for all stocks with sufficient data.

    Returns:
        (returns_df, tickers) where returns_df is a DataFrame with
        dates as index and tickers as columns
    """
    from kse.data_pipeline import load_stock_metrics

    stock_df = load_stock_metrics()
    all_returns = {}

    for _, row in stock_df.iterrows():
        ticker = row["ticker"]
        path = RAW_DIR / f"{ticker}.csv"

        if not path.exists():
            continue

        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()]

        if "Close" not in df.columns:
            continue

        close = df["Close"].dropna()
        if len(close) < 252:
            continue

        monthly = close.resample("ME").last()
        monthly_returns = monthly.pct_change().dropna()

        if len(monthly_returns) < 24:
            continue

        all_returns[ticker] = monthly_returns

    if not all_returns:
        return pd.DataFrame(), []

    returns_df = pd.DataFrame(all_returns)
    returns_df = returns_df.dropna()

    if len(returns_df) < 12:
        return pd.DataFrame(), []

    return returns_df, list(returns_df.columns)


def compute_diversification_curve(returns_df, max_trials=100):
    """
    Compute portfolio volatility as a function of number of stocks.

    For each N from 1 to max_stocks:
      - Run max_trials random samples of N stocks
      - Compute equal-weight portfolio volatility for each sample
      - Average across trials

    Returns:
        DataFrame with columns: n_stocks, avg_volatility, min_volatility,
        max_volatility, p10_volatility, p90_volatility
    """
    rng = np.random.default_rng(42)
    tickers = returns_df.columns
    max_stocks = len(tickers)

    results = []

    for n in range(1, max_stocks + 1):
        vols = []
        for _ in range(max_trials):
            if n == max_stocks:
                selected = tickers
            else:
                selected = rng.choice(tickers, size=n, replace=False)

            port_returns = returns_df[selected].mean(axis=1)
            vol = port_returns.std() * np.sqrt(12)
            vols.append(vol)

        results.append({
            "n_stocks": n,
            "avg_volatility": np.mean(vols),
            "min_volatility": np.min(vols),
            "max_volatility": np.max(vols),
            "p10_volatility": np.percentile(vols, 10),
            "p90_volatility": np.percentile(vols, 90),
        })

    return pd.DataFrame(results)


def compute_efficient_frontier(returns_df, n_points=25):
    """
    Compute the Markowitz efficient frontier.

    Uses scipy.optimize to find minimum-variance portfolios at each return level.
    Long-only constraint (weights >= 0).

    Returns:
        dict with:
          - frontier: DataFrame with columns: return, volatility, weights
          - stock_stats: DataFrame with individual stock return and volatility
          - equal_weight: dict with return and volatility of equal-weight portfolio
          - min_var: dict with return and volatility of minimum variance portfolio
    """
    from scipy.optimize import minimize

    tickers = returns_df.columns
    n = len(tickers)

    # Annualized mean returns and covariance
    mean_returns = returns_df.mean() * 12
    cov_matrix = returns_df.cov() * 12

    # Individual stock stats
    stock_stats = pd.DataFrame({
        "ticker": tickers,
        "return": mean_returns.values,
        "volatility": np.sqrt(np.diag(cov_matrix)),
    })

    # Equal-weight portfolio
    ew_weights = np.ones(n) / n
    ew_return = np.dot(ew_weights, mean_returns)
    ew_vol = np.sqrt(ew_weights @ cov_matrix @ ew_weights)
    equal_weight = {"return": ew_return, "volatility": ew_vol, "weights": ew_weights, "tickers": list(tickers)}

    # Optimization helpers
    def portfolio_variance(w):
        return w @ cov_matrix @ w

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    ]
    bounds = [(0.01, 1)] * n  # force a minimum 1% weight per stock

    # Sweep target returns
    min_ret = mean_returns.min()
    max_ret = mean_returns.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier = []
    for target in target_returns:
        constraints_with_return = constraints + [
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_returns) - t}
        ]

        result = minimize(
            portfolio_variance,
            x0=ew_weights,
            method="SLSQP",
            constraints=constraints_with_return,
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-10}
        )

        if result.success:
            w = result.x
            ret = np.dot(w, mean_returns)
            vol = np.sqrt(w @ cov_matrix @ w)
            frontier.append({
                "return": ret,
                "volatility": vol,
                "weights": w,
            })

    frontier_df = pd.DataFrame(frontier)
    frontier_df = frontier_df.sort_values("volatility").reset_index(drop=True)

    # Minimum variance portfolio (no return constraint)
    result_min_var = minimize(
        portfolio_variance,
        x0=ew_weights,
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-10}
    )

    if result_min_var.success:
        mv_weights = result_min_var.x
        mv_return = np.dot(mv_weights, mean_returns)
        mv_vol = np.sqrt(mv_weights @ cov_matrix @ mv_weights)
        min_var = {"return": mv_return, "volatility": mv_vol, "weights": mv_weights, "tickers": list(tickers)}
    else:
        min_var = {"return": ew_return, "volatility": ew_vol, "weights": ew_weights, "tickers": list(tickers)}

    # maximum return portfolio (top of the efficient frontier)
    max_ret_idx = frontier_df["return"].idxmax()
    max_ret_weights = frontier_df.loc[max_ret_idx, "weights"]
    max_ret_return = frontier_df.loc[max_ret_idx, "return"]
    max_ret_vol = frontier_df.loc[max_ret_idx, "volatility"]
    max_ret = {"return": max_ret_return, "volatility": max_ret_vol, "weights": max_ret_weights, "tickers": list(tickers)}

    return {
        "frontier": frontier_df,
        "stock_stats": stock_stats,
        "equal_weight": equal_weight,
        "min_var": min_var,
        "max_ret": max_ret,
    }


def compute_portfolio_stats(returns_df, weights_dict):
    """
    Compute return and volatility for a portfolio with given weights.

    Parameters:
        returns_df: DataFrame of monthly returns (from load_stock_return_matrix)
        weights_dict: {ticker: weight}

    Returns:
        dict with return and volatility (annualized)
    """
    tickers = [t for t in weights_dict if t in returns_df.columns]
    if not tickers:
        return None

    weights = np.array([weights_dict[t] for t in tickers])
    weights = weights / weights.sum()  # Normalize

    mean_returns = returns_df[tickers].mean() * 12
    cov_matrix = returns_df[tickers].cov() * 12

    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(weights @ cov_matrix @ weights)

    return {"return": port_return, "volatility": port_vol}
def get_recommended_portfolio(returns_df, screen_df, n_stocks=8):
    """
    Generate a recommended portfolio:
    1. Select top N stocks by composite screen score.
    2. Weight them using the minimum variance portfolio.
    
    Returns dict with tickers, weights, return, volatility.
    """
    from scipy.optimize import minimize
    
    # 1. Select top N stocks by screen score
    top_stocks = screen_df.head(n_stocks)["Ticker"].tolist()
    
    # Filter to only include stocks that have return data
    available_stocks = [t for t in top_stocks if t in returns_df.columns]
    
    if not available_stocks:
        return None
        
    ret_df = returns_df[available_stocks]
    n = len(available_stocks)
    
    # 2. Calculate minimum variance weights
    mean_returns = ret_df.mean() * 12
    cov_matrix = ret_df.cov() * 12
    
    def portfolio_variance(w):
        return w @ cov_matrix @ w
    
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 1)] * n  # force a minimum 1% weight per stock
    x0 = np.ones(n) / n
    
    result = minimize(
        portfolio_variance,
        x0=x0,
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-10}
    )
    
    if not result.success:
        weights = np.ones(n) / n
    else:
        weights = result.x
        
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(weights @ cov_matrix @ weights)
    
    return {
        "tickers": available_stocks,
        "weights": weights,
        "return": port_return,
        "volatility": port_vol
    }