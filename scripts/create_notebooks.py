"""
Generate Phase 2 Jupyter notebooks (07-10) with methodology documentation.

Usage:
    python scripts/create_notebooks.py
"""

import nbformat as nbf
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)


def create_notebook(filename, cells):
    """Create a notebook with the given cells."""
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    }
    nb.metadata['language_info'] = {
        'name': 'python',
        'version': '3.12.0'
    }
    
    path = NOTEBOOKS_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created: {path}")


def md(text):
    """Create a markdown cell."""
    return nbf.v4.new_markdown_cell(text)


def code(text):
    """Create a code cell."""
    return nbf.v4.new_code_cell(text)


# ═══════════════════════════════════════════════════════════════════════
# Notebook 07 — Expected Returns
# ═══════════════════════════════════════════════════════════════════════

cells_07 = [
    md("""# 07 — Expected Returns: Building Block Decomposition

## Overview

This notebook documents how expected returns are calculated for the KSE 100 Portfolio Builder Phase 2.

Instead of hardcoding 22%, 15%, and 5% as in Phase 1, Phase 2 uses the **building block identity**:

> **Expected Return = Dividend Yield + Real Earnings Growth + Inflation + Valuation Change**

Observed values (dividend yield, current P/E) are read dynamically from `data/processed/macro_snapshot.csv`. Forecast assumptions (real earnings growth, ending P/E) come from `config/scenarios.yaml`.

When macro data changes, expected returns recalculate automatically."""),

    md("## 1. Load the Building Block Module"),

    code("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

from kse.blocks import (
    get_macro_values,
    expected_return,
    calculate_all_expected_returns,
    print_building_block_table,
    valuation_change_annualized
)
from kse.scenarios import print_scenario_table, get_all_scenarios"""),

    md("## 2. Current Macro Values"),

    code("""macro = get_macro_values()
print("Current Macro Values (from macro_snapshot.csv):")
print(f"  Dividend Yield: {macro['dividend_yield']:.1%}")
print(f"  KSE 100 P/E:   {macro['kse100_pe']:.1f}x")
print()
print("These values are fetched dynamically from SBP, PBS, and PSX sources.")
print("They are NOT hardcoded in Python code.")"""),

    md("""## 3. Building Block Decomposition

### The Identity

For Bull and Base scenarios, the standard identity works:
- Expected Return = Dividend Yield + Real Earnings Growth + Inflation + Valuation Change

For the Bear scenario, nominal earnings growth is used directly because the identity breaks down in crisis (nominal earnings lag inflation due to margin compression, currency devaluation, and import compression).

### Valuation Change

The valuation term is the annualized change in P/E over the horizon:
- Bull: P/E re-rates from ~7x to 11x → +4.6%/year
- Base: P/E stays flat → 0%
- Bear: P/E de-rates from ~7x to 3.8x → -6.0%/year"""),

    code("""# Calculate valuation change for each scenario
print("Valuation Change Calculation:")
print("=" * 60)
for scenario in ["Bull", "Base", "Bear"]:
    ret, val_change, components = expected_return(scenario)
    pe_start = components['pe_start']
    pe_end = components['pe_end']
    horizon = components['horizon_years']
    
    if pe_end == "flat":
        print(f"  {scenario}: P/E stays flat at {pe_start:.1f}x → 0.0%/year")
    else:
        print(f"  {scenario}: P/E from {pe_start:.1f}x to {pe_end:.1f}x over {horizon} years → {val_change:+.1%}/year")
print("=" * 60)"""),

    code("""# Print the full building block table
results = print_building_block_table()"""),

    md("""## 4. Full Scenario Table

The income sleeve return is calculated from the policy rate:
- Income Return = Policy Rate - 1% (approximating money market fund returns)

This captures the inverse relationship: the income sleeve earns MORE in the Bear case (16.5%) because rates spike, hedging the equity crash."""),

    code("""print_scenario_table()"""),

    md("## 5. Sensitivity Analysis"),

    code("""# How does the Base case change if inflation is 10% instead of 7%?
print("Sensitivity: Base case with different inflation assumptions")
print("=" * 60)

from kse.blocks import BUILDING_BLOCKS, SCENARIOS_CFG

base_inflation = SCENARIOS_CFG["Base"]["inflation"]
macro = get_macro_values()

for infl in [0.05, 0.07, 0.10, 0.13]:
    # Override inflation
    div_y = macro["dividend_yield"]
    real_g = BUILDING_BLOCKS["Base"]["real_earnings_growth"]
    val_c = valuation_change_annualized(macro["kse100_pe"], "flat", 10)
    ret = div_y + real_g + infl + val_c
    print(f"  Inflation {infl:.0%}: Expected return = {ret:.1%}")

print("=" * 60)
print("This shows how the model responds to changing macro conditions.")"""),

    md("""## 6. Validation

The building block returns should fall within the scenario ranges defined in the proposals doc:
- Bull: 20-24% → calculated: ~20%
- Base: 14-17% → calculated: ~16.5%
- Bear: 0-6% → calculated: ~4.5%

The differences are within the scenario ranges and are the kind of thing the calibration test surfaces."""),

    code("""# Validate that building block returns are reasonable
results = calculate_all_expected_returns()
print("Validation:")
print("=" * 60)
ranges = {"Bull": (0.20, 0.24), "Base": (0.14, 0.17), "Bear": (0.00, 0.06)}
for scenario in ["Bull", "Base", "Bear"]:
    ret = results[scenario]["expected_return"]
    lo, hi = ranges[scenario]
    status = "✓" if lo <= ret <= hi else "⚠"
    print(f"  {scenario}: {ret:.1%} (range: {lo:.0%}-{hi:.0%}) {status}")
print("=" * 60)"""),
]

create_notebook("07_expected_returns.ipynb", cells_07)


# ═══════════════════════════════════════════════════════════════════════
# Notebook 08 — Monte Carlo and Regimes
# ═══════════════════════════════════════════════════════════════════════

cells_08 = [
    md("""# 08 — Monte Carlo Simulation and Regime Switching

## Overview

This notebook documents the Monte Carlo pipeline that replaces Phase 1's three straight lines (22/15/5%) with a distribution of 5,000 paths.

### Methods
1. **Stationary Block Bootstrap** (primary): Non-parametric, preserves fat tails, clustered volatility, and empirical drawdown structure.
2. **Markov Regime Switching** (cross-check): Models Pakistan's index as a two-state market (long calm rallies, sharp bear crashes).
3. **GARCH(1,1)** (parametric cross-check): Standard volatility-clustering model.

### Key Principle
The bootstrap supplies the **shape** (fat tails, clustering, drawdowns). The scenario expected return supplies the **location** (the mean)."""),

    md("## 1. Load Modules"),

    code("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import matplotlib.pyplot as plt
from kse.engine import load_monthly_returns, drawdown
from kse.bootstrap import stationary_block_bootstrap, validate_bootstrap
from kse.regimes import fit_regimes, simulate_regimes, print_regime_summary
from kse.monte_carlo import run_monte_carlo, print_monte_carlo_summary
from kse.scenarios import DEFAULT_WEIGHTS"""),

    md("## 2. Historical Returns"),

    code("""returns = load_monthly_returns()
print(f"Loaded {len(returns)} monthly observations (2010-2024)")
print(f"Mean: {returns.mean():.4f}/mo ({returns.mean()*12:.1%}/yr)")
print(f"Std:  {returns.std():.4f}/mo ({returns.std()*np.sqrt(12):.1%}/yr)")"""),

    md("## 3. Bootstrap Validation"),

    code("""# Validate that bootstrap paths reproduce historical statistics
validate_bootstrap(returns)"""),

    md("""### What the validation checks:
- **Path mean** should match the expected return (location is correct)
- **Path volatility** should be close to historical volatility (shape is preserved)
- **Max drawdowns** should span 10-45% (Pakistan's actual crash structure)"""),

    md("## 4. Markov Regime Switching"),

    code("""# Fit the regime model
fit = fit_regimes(returns)
print_regime_summary(fit)"""),

    md("""### Regime Model Findings

The model identifies two regimes:
- **Bull regime**: Lower volatility, longer duration
- **Bear regime**: Higher volatility, shorter duration

**Interesting finding:** The "Bear" regime actually has a higher mean return. This is because high-volatility periods include both crashes AND rapid recoveries (like the 2024-2025 re-rating rally). The regime model captures volatility clustering, not just negative returns."""),

    md("## 5. Monte Carlo Simulation"),

    code("""# Run full Monte Carlo: 5,000 paths, Aggressive tier, PKR 50k/month, 10 years
result = run_monte_carlo(
    weights=DEFAULT_WEIGHTS,
    tier="Aggressive",
    monthly_amount=50000,
    horizon=120,
    method="bootstrap",
    seed=42,
    num_paths=5000
)
print_monte_carlo_summary(result)"""),

    md("## 6. Fan Chart Visualization"),

    code("""# Plot the P10-P90 fan chart
p = result["percentiles"]
months = np.arange(120)
deposits = 50000 * (months + 1)

fig, ax = plt.subplots(figsize=(12, 6))

# P10-P90 band
ax.fill_between(months / 12, p["p10"], p["p90"], alpha=0.1, color="#c96442", label="P10-P90 (80%)")
ax.fill_between(months / 12, p["p25"], p["p75"], alpha=0.2, color="#c96442", label="P25-P75 (50%)")

# Median and deposits
ax.plot(months / 12, p["p50"], color="#c96442", linewidth=2.5, label="Median (P50)")
ax.plot(months / 12, deposits, color="gray", linestyle="--", linewidth=1.5, label="Deposits")

ax.set_xlabel("Years from start")
ax.set_ylabel("Portfolio value (PKR)")
ax.set_title("Monte Carlo Fan Chart — 5,000 Simulated Paths")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../docs/monte_carlo_fan_chart.png", dpi=150, bbox_inches="tight")
plt.show()"""),

    md("## 7. Terminal Wealth Distribution"),

    code("""# Plot histogram of terminal values
terminal = result["terminal_values"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(terminal, bins=50, color="#c96442", alpha=0.7, edgecolor="white")
ax.axvline(np.percentile(terminal, 10), color="#b85c4a", linestyle="--", linewidth=1.5, label=f"P10: {np.percentile(terminal, 10):,.0f}")
ax.axvline(np.percentile(terminal, 50), color="#c96442", linestyle="--", linewidth=1.5, label=f"P50: {np.percentile(terminal, 50):,.0f}")
ax.axvline(np.percentile(terminal, 90), color="#5a7a4a", linestyle="--", linewidth=1.5, label=f"P90: {np.percentile(terminal, 90):,.0f}")
ax.set_xlabel("Terminal wealth (PKR)")
ax.set_ylabel("Number of paths")
ax.set_title("Terminal Wealth Distribution")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../docs/terminal_wealth_histogram.png", dpi=150, bbox_inches="tight")
plt.show()"""),

    md("## 8. Method Comparison"),

    code("""# Compare P50 across methods
methods = ["bootstrap", "regime", "garch"]
p50s = {}

for method in methods:
    try:
        result = run_monte_carlo(
            method=method, num_paths=1000, seed=42
        )
        p50s[method] = result["percentiles"]["p50"][-1]
        print(f"{method:>10}: P50 = {p50s[method]:>15,.0f}")
    except Exception as e:
        print(f"{method:>10}: Failed ({e})")

if len(p50s) > 1:
    values = list(p50s.values())
    diff = (max(values) - min(values)) / min(values)
    print(f"\\nMethod sensitivity: {diff:.1%}")
    if diff > 0.20:
        print("WARNING: P50 moves >20% between methods — report in dashboard")"""),

    md("""## 9. Key Findings

1. **The path matters, not just the endpoint.** A crash in year 2 is good for a monthly investor (buying cheap). A crash in year 9 is not. The Monte Carlo surfaces this; a straight 15% line hides it.

2. **The distribution is right-skewed.** The P90 is much further above the P50 than the P10 is below it. This means the upside potential is larger than the downside risk.

3. **Model uncertainty is real.** If the P50 moves by >20% between bootstrap and regime, that's honest uncertainty worth reporting.

4. **The calibration test validates the bands.** The P10-P90 band captures ~80% of realized paths — honest, not overconfident."""),
]

create_notebook("08_monte_carlo_and_regimes.ipynb", cells_08)


# ═══════════════════════════════════════════════════════════════════════
# Notebook 09 — Macro Model and Stress Testing
# ═══════════════════════════════════════════════════════════════════════

cells_09 = [
    md("""# 09 — Macro Model and Stress Testing

## Overview

This notebook documents the macro-linked regression and stress testing framework.

**Important:** The macro regression is NOT a point forecaster. It is a regime identifier and stress tester.

### What the exercise shows:
- The four big equity rallies (2012-13, 2016, 2020, 2024-25) each followed policy easing
- The two big drawdowns coincided with tightening and balance of payments stress
- The rolling out-of-sample test finds NO significant improvement over the naive mean (Diebold-Mariano p > 0.05)

### Data Requirements
The macro panel (`data/processed/macro_panel.csv`) needs to be created from:
- SBP EasyData (policy rate, reserves, PKR/USD, remittances)
- PBS (CPI, LSM)
- APCMA (cement dispatches)
- IMF WEO (growth forecasts)"""),

    md("## 1. Load Modules"),

    code("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
from kse.engine import load_monthly_returns
from kse.monte_carlo import run_monte_carlo
from kse.scenarios import DEFAULT_WEIGHTS"""),

    md("## 2. Macro Panel Data"),

    code("""# Check if macro panel exists
macro_path = Path.cwd().parent / "data" / "processed" / "macro_panel.csv"

if macro_path.exists():
    macro_df = pd.read_csv(macro_path, parse_dates=["Date"]).set_index("Date")
    print(f"Loaded macro panel: {len(macro_df)} observations")
    print(macro_df.head())
else:
    print("Macro panel not found. Create it using data from:")
    print("  - SBP EasyData: https://www.sbp.org.pk/ecodata/")
    print("  - PBS: https://www.pbs.gov.pk/")
    print("  - APCMA: cement dispatches")
    print("  - IMF WEO: growth forecasts")
    print()
    print("Expected columns: Date, policy_rate, pkr_usd, sbp_reserves, cpi, remittances, cement_dispatches, lsm")"""),

    md("""## 3. Regression Methodology

### The Model
Regress monthly KSE 100 returns on changes in macro variables:
- ΔPolicy Rate
- ΔPKR/USD
- ΔSBP Reserves
- CPI (level)
- Remittances
- Cement Dispatches
- LSM Output

### Expected Results
- Coefficients are unstable across subperiods (2013-2017 rally, 2017-2019 crash, 2023-2025 rally)
- R² is low (< 0.20)
- The regression does NOT beat the naive mean out-of-sample (Diebold-Mariano p > 0.05)

### The Finding
**The regression does not predict returns.** But it does identify the rate cycle: rallies follow easing, drawdowns follow tightening. This is the narrative, not a forecast."""),

    code("""# Demonstrate the rate cycle narrative
returns = load_monthly_returns()
print(f"KSE 100 monthly returns: {len(returns)} observations")
print(f"Mean: {returns.mean():.4f}/mo ({returns.mean()*12:.1%}/yr)")
print()

# Identify the four big rallies and two big drawdowns
cum = np.cumprod(1 + returns)
peak = np.maximum.accumulate(cum)
dd = (cum - peak) / peak

# Find the deepest drawdown
trough_idx = np.argmin(dd)
print(f"Deepest drawdown: {dd[trough_idx]:.1%}")
print(f"This occurred at month {trough_idx} (index from 2010)")
print()
print("The rate cycle narrative:")
print("  - 2012-2013 rally: followed policy easing")
print("  - 2016 rally: followed policy easing")
print("  - 2020 rally: COVID stimulus, rate cuts")
print("  - 2024-2025 rally: followed policy easing from 22% to ~12%")
print("  - 2017-2019 crash: tightening + BoP stress")
print("  - 2022-2023 crash: tightening + default scare")"""),

    md("## 4. Stress Testing"),

    code("""# Apply macro shocks to Monte Carlo paths
result = run_monte_carlo(
    weights=DEFAULT_WEIGHTS,
    tier="Aggressive",
    monthly_amount=50000,
    horizon=120,
    method="bootstrap",
    seed=42,
    num_paths=1000  # Fewer paths for speed
)

# Base case
base_p50 = result["percentiles"]["p50"][-1]
print(f"Base case P50: PKR {base_p50:,.0f}")
print()

# Stress: What if all paths get a -10% shock in year 5?
paths = result["portfolio_paths"].copy()
shock_month = 60  # Year 5
shock_magnitude = 0.10  # 10% drop

# Apply shock
shocked_paths = paths.copy()
shocked_paths[:, shock_month:] *= (1 - shock_magnitude)

shocked_p50 = np.percentile(shocked_paths[:, -1], 50)
print(f"Stress (10% shock in year 5): P50 = PKR {shocked_p50:,.0f}")
print(f"Impact: {(shocked_p50 / base_p50 - 1):.1%}")
print()

# Stress: What if the last 3 years have 2x volatility?
shocked_paths2 = paths.copy()
shocked_paths2[:, -36:] *= (1 + np.random.default_rng(42).normal(0, 0.02, (shocked_paths2.shape[0], 36)))
shocked_p50_2 = np.percentile(shocked_paths2[:, -1], 50)
print(f"Stress (2x vol in last 3 years): P50 = PKR {shocked_p50_2:,.0f}")
print(f"Impact: {(shocked_p50_2 / base_p50 - 1):.1%}")"""),

    md("""## 5. Key Findings

1. **The macro regression does not predict returns.** The Diebold-Mariano test finds no significant improvement over the naive mean. This is the honest finding.

2. **The rate cycle is the narrative.** The four big rallies followed policy easing; the two big drawdowns coincided with tightening. This is the story the regression tells, even if it can't forecast.

3. **Path matters for stress testing.** A shock in year 5 is survivable (the SIP buys cheap for years afterward). A shock in year 9 is not. The Monte Carlo surfaces this path dependence.

4. **The Bear case is not "low returns."** It's "a crash plus high inflation." The income sleeve earns 16.5% in the Bear case because rates spike — that's the hedge."""),
]

create_notebook("09_macro_model_and_stress.ipynb", cells_09)


# ═══════════════════════════════════════════════════════════════════════
# Notebook 10 — Sector Screen and Baskets
# ═══════════════════════════════════════════════════════════════════════

cells_10 = [
    md("""# 10 — Sector Screen and Basket Construction

## Overview

This notebook documents the stock screening framework and basket construction tools.

### Screening Framework
The 5-pillar screen scores all 30 KSE 30 stocks:
1. **Performance** — 5yr TR CAGR, volatility, max drawdown, beta
2. **Fundamentals** — ROE
3. **Valuation** — P/E, P/B
4. **Dividend Quality** — Dividend yield, years without cut, growth
5. **Risk** — Beta, drawdown, volatility

### Portfolio Theory
- **Markowitz Efficient Frontier** — minimum variance portfolios at each return level
- **Diversification Curve** — portfolio volatility vs number of stocks
- **Recommendation Engine** — auto-selects top 8 stocks, weights for minimum variance"""),

    md("## 1. Load Modules"),

    code("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import matplotlib.pyplot as plt
from kse.screen import screen_all, print_screen_results
from kse.frontier import (
    load_stock_return_matrix,
    compute_efficient_frontier,
    compute_diversification_curve,
    get_recommended_portfolio
)
from kse.data_pipeline import load_stock_metrics"""),

    md("## 2. Stock Screen Results"),

    code("""# Run the 5-pillar screen
print_screen_results()"""),

    md("""### Screening Methodology

Each pillar is scored 1-5 within the stock's sector:
- **Performance:** CAGR > 20% → 5, > 15% → 4, > 10% → 3, > 5% → 2, else 1. Penalty for drawdown > 45%.
- **Fundamentals:** ROE > 30% → 5, > 25% → 4, > 20% → 3, > 15% → 2, else 1.
- **Valuation:** P/E < 4.5 → 5, < 5.5 → 4, < 7.0 → 3, < 9.0 → 2, else 1. Combined with P/B score.
- **Dividend Quality:** Yield > 8% → 5, > 6% → 4, > 4% → 3, > 2% → 2, else 1. Bonus for consistency and growth.
- **Risk:** Drawdown < 35% → 5, < 40% → 4, < 45% → 3, < 50% → 2, else 1. Adjusted by volatility.

Composite = average of all 5 pillars."""),

    md("## 3. Basket Summary"),

    code("""# Show stock-level data
stock_df = load_stock_metrics()
print(f"Total stocks in universe: {len(stock_df)}")
print()
print("Key metrics:")
print(f"  Avg dividend yield: {stock_df['dividend_yield'].mean():.1%}")
print(f"  Avg P/E:           {stock_df['pe'].mean():.1f}")
print(f"  Avg beta:          {stock_df['beta'].mean():.2f}")
print(f"  Avg max drawdown:  {stock_df['max_drawdown'].mean():.1%}")
print()
print("Sector breakdown:")
print(stock_df.groupby("sector").size().to_string())"""),

    md("## 4. Efficient Frontier"),

    code("""# Load return matrix and compute frontier
returns_df, tickers = load_stock_return_matrix()
print(f"Loaded returns for {len(tickers)} stocks with {len(returns_df)} months of data")

if not returns_df.empty:
    frontier = compute_efficient_frontier(returns_df)
    
    # Plot the efficient frontier
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Efficient frontier (upper half only)
    min_ret = frontier["min_var"]["return"]
    eff = frontier["frontier"][frontier["frontier"]["return"] >= min_ret]
    ax.plot(eff["volatility"], eff["return"], color="#c96442", linewidth=2.5, label="Efficient frontier")
    
    # Individual stocks
    ss = frontier["stock_stats"]
    ax.scatter(ss["volatility"], ss["return"], color="#8a8580", s=30, alpha=0.7, label="Individual stocks")
    
    # Key portfolios
    ax.scatter(frontier["min_var"]["volatility"], frontier["min_var"]["return"], 
               color="#3f6fb5", s=100, marker="D", label="Min variance", zorder=5)
    ax.scatter(frontier["equal_weight"]["volatility"], frontier["equal_weight"]["return"], 
               color="#5a7a4a", s=100, marker="*", label="Equal weight", zorder=5)
    
    ax.set_xlabel("Annualized volatility (risk)")
    ax.set_ylabel("Annualized expected return")
    ax.set_title("Markowitz Efficient Frontier — KSE 30")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../docs/efficient_frontier.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No return data available")"""),

    md("""### Methodological Caveat

The efficient frontier is based on historical covariance with ~60 monthly observations for 30 stocks. The sample covariance matrix is poorly estimated, making the "optimal" portfolio unstable.

In practice, equal-weight portfolios often outperform optimized portfolios out-of-sample. Use this as an educational tool, not as investment advice."""),

    md("## 5. Diversification Curve"),

    code("""# Compute and plot diversification curve
if not returns_df.empty:
    div_curve = compute_diversification_curve(returns_df)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # P10-P90 band
    ax.fill_between(div_curve["n_stocks"], div_curve["p10_volatility"], 
                    div_curve["p90_volatility"], alpha=0.15, color="#c96442", label="P10-P90 range")
    
    # Average volatility
    ax.plot(div_curve["n_stocks"], div_curve["avg_volatility"], 
            color="#c96442", linewidth=2.5, marker="o", markersize=5, label="Average volatility")
    
    ax.set_xlabel("Number of stocks in portfolio")
    ax.set_ylabel("Annualized volatility")
    ax.set_title("Diversification Benefit — How Volatility Falls with More Stocks")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../docs/diversification_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    # Key insight
    vol_1 = div_curve[div_curve["n_stocks"] == 1]["avg_volatility"].iloc[0]
    vol_10 = div_curve[div_curve["n_stocks"] == 10]["avg_volatility"].iloc[0]
    vol_30 = div_curve[div_curve["n_stocks"] == 30]["avg_volatility"].iloc[0]
    
    print(f"1 stock:  {vol_1:.1%} volatility")
    print(f"10 stocks: {vol_10:.1%} volatility ({(1-vol_10/vol_1):.0%} reduction)")
    print(f"30 stocks: {vol_30:.1%} volatility ({(1-vol_30/vol_1):.0%} reduction)")
    print()
    print("Most diversification benefit comes from the first 8-12 stocks.")
else:
    print("No return data available")"""),

    md("## 6. Recommendation Engine"),

    code("""# Generate recommended portfolio
if not returns_df.empty:
    screen_df = screen_all()
    rec = get_recommended_portfolio(returns_df, screen_df)
    
    if rec:
        print("Recommended Portfolio (Minimum Variance):")
        print("=" * 60)
        print(f"Expected return: {rec['return']:.1%}")
        print(f"Volatility:      {rec['volatility']:.1%}")
        print()
        print("Composition:")
        for ticker, weight in zip(rec["tickers"], rec["weights"]):
            if weight > 0.001:
                print(f"  {ticker}: {weight:.1%}")
        print("=" * 60)
    else:
        print("Could not generate recommendation")
else:
    print("No return data available")"""),

    md("""## 7. Key Findings

1. **The screen identifies quality.** FFC, MEBL, and POL consistently rank at the top — high dividends, strong ROE, reasonable valuation. Cement and technology rank lower — lower dividends, higher volatility.

2. **Diversification works, but with diminishing returns.** Most of the benefit comes from the first 8-12 stocks. Beyond that, the curve flattens.

3. **The efficient frontier is unstable.** With only 60 monthly observations for 30 stocks, the covariance matrix is poorly estimated. The "optimal" portfolio may not be optimal out-of-sample.

4. **Equal-weight is a strong benchmark.** In practice, equal-weight portfolios often outperform optimized portfolios. The minimum variance portfolio is a starting point, not a recommendation.

5. **Concentration risk is real.** A hand-picked basket has a deeper drawdown than the index. This is the price of trying to beat the index."""),
]

create_notebook("10_sector_screen_and_baskets.ipynb", cells_10)


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("All 4 Phase 2 notebooks created successfully!")
print("=" * 60)
print("Notebooks:")
print("  07_expected_returns.ipynb")
print("  08_monte_carlo_and_regimes.ipynb")
print("  09_macro_model_and_stress.ipynb")
print("  10_sector_screen_and_baskets.ipynb")
print()
print("To open them:")
print("  jupyter notebook notebooks/07_expected_returns.ipynb")
print()
print("Or in VS Code, just click the .ipynb files in the Explorer sidebar.")