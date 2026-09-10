# KSE 100 Portfolio Builder

A comprehensive, institution-grade investment analysis platform for the Pakistan Stock Exchange. Built for young Pakistani earners who want to make data-driven SIP (Systematic Investment Plan) investment decisions.

**Live App:** https://kse30-portfolio-builder.streamlit.app

---

## What It Does

The app helps you:
- **Plan** — Define real financial goals (university, retirement, house) and calculate the required monthly investment
- **Project** — See forward-looking portfolio growth with Monte Carlo probability distributions
- **Assess risk** — View drawdowns, VaR, crisis scenarios, and probability of loss
- **Build portfolios** — Screen KSE-30 stocks, construct efficient frontiers, recommend minimum-variance portfolios
- **Compare** — Benchmark against the KSE-100 index (alpha, beta, capture ratios) and mutual funds (fee drag)
- **Contextualize** — See returns in PKR and USD, compare equities to gold and real estate
- **Guide** — Get rebalancing guidance with cost estimation and tax considerations
- **Export** — Generate PDF reports and CSV files for sharing and broker execution

---

## Dashboard Tabs (8)

| Tab | Features |
|-----|----------|
| **Growth** | SIP projection, scenario comparison, historical backtest, market event overlay |
| **Dividends** | Dividend milestones, DRIP (reinvestment) vs cash comparison |
| **Risk** | Probability of loss, drawdown analysis, VaR/CVaR, stress testing (4 crises), sector concentration, correlation regime analysis, currency-adjusted returns (PKR to USD), multi-asset comparison (gold, USD, real estate) |
| **Worst Case** | Worst-time-to-start SIP backtest |
| **Outlook** | Monte Carlo fan chart (5,000 paths), terminal wealth distribution, probability table, what-if sliders, building block decomposition |
| **Goals** | Goal-based investing (university, retirement, house, custom), required SIP, probability of success, cost of delay |
| **Market Context** | Political/IMF event timeline, sector rotation dashboard (relative strength, momentum scores, quadrant scatter) |
| **Basket** | KSE-30 stock picker, 5-pillar screen, Markowitz efficient frontier, minimum-variance recommendation, benchmark comparison (alpha/beta/capture), rebalancing guidance, mutual fund comparison (fee drag), liquidity profile, portfolio comparison mode |

---

## Tech Stack

- **Language:** Python 3.12
- **App Framework:** Streamlit
- **Charts:** Plotly
- **Data:** Pandas, NumPy, yfinance
- **Modeling:** SciPy (optimization), statsmodels (regime switching), arch (GARCH — local only)
- **Reporting:** reportlab (PDF generation)
- **Automation:** GitHub Actions (nightly price, dividend, and volume updates)
- **Deployment:** Streamlit Cloud

---

## Engine Modules (`kse/`)

| Module | Purpose |
|--------|---------|
| `engine.py` | SIP engine, CGT calculations, return blending |
| `blocks.py` | Building block expected returns (div yield + growth + inflation + val change) |
| `scenarios.py` | Dynamic scenario definitions (Bull/Base/Bear) |
| `bootstrap.py` | Stationary block bootstrap path generator |
| `regimes.py` | Markov regime switching model |
| `garch.py` | GARCH(1,1) cross-check (local only) |
| `monte_carlo.py` | 5,000-path pipeline, percentiles, probability table |
| `screen.py` | 5-pillar stock screen for KSE-30 |
| `frontier.py` | Markowitz efficient frontier, diversification curve, recommendation engine |
| `data_pipeline.py` | Data loading/saving with provenance |
| `risk_metrics.py` | Drawdown, VaR/CVaR, stress testing, sector concentration, correlation regime |
| `currency.py` | PKR to USD return conversion |
| `benchmark.py` | CAPM alpha/beta, capture ratios, rolling alpha |
| `compliance.py` | Disclaimers, data provenance, suitability assessment |
| `goals.py` | Goal templates, required SIP, goal probability, cost of delay |
| `reporting.py` | PDF report generation, CSV export |
| `multi_asset.py` | Multi-asset comparison (KSE, gold, USD, real estate) |
| `drip.py` | Dividend reinvestment vs cash analysis |
| `rebalance.py` | Drift detection, rebalancing cost estimation |
| `mutual_funds.py` | Mutual fund comparison, fee impact analysis |
| `liquidity.py` | Liquidity-adjusted max weights, days to liquidate |
| `comparison.py` | Side-by-side portfolio comparison |
| `events.py` | Market event timeline overlay |
| `sector_rotation.py` | Sector returns, relative strength, momentum scores |

---

## Configuration (`config/`)

All financial assumptions are in YAML — no hardcoded values in the codebase.

| File | Purpose |
|------|---------|
| `scenarios.yaml` | Bull/Base/Bear scenario definitions |
| `sectors.yaml` | KSE-30 ticker to sector mapping |
| `crisis_periods.yaml` | 4 Pakistan-specific crisis periods for stress testing |
| `disclaimers.yaml` | Disclaimer text, data source metadata, suitability questions |
| `mutual_funds.yaml` | 4 mutual fund categories with returns, fees, volatility |
| `market_events.yaml` | 14 major Pakistani market events (2008-2024) |

---

## Automation

A GitHub Action (`.github/workflows/update-prices.yml`) runs nightly at 2:00 AM UTC (7:00 AM PKT) to:
- Fetch latest prices and dividends for all KSE-30 stocks from yfinance
- Calculate price metrics (CAGR, volatility, max drawdown, beta)
- Calculate average daily volume for liquidity analysis
- Update `data/processed/stock_metrics.csv` without overwriting manual financials
- Commit and push to GitHub, triggering a Streamlit Cloud redeploy

---

## Running Locally

    # Clone the repo
    git clone https://github.com/IbrahimAli402/kse30-portfolio-builder.git
    cd kse30-portfolio-builder

    # Create virtual environment
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # or: venv\Scripts\activate  # Windows

    # Install dependencies
    pip install -r requirements.txt

    # Run the app
    streamlit run app/dashboard.py

---

## Disclaimer

This application is an analytical and educational tool, not investment advice. It is not registered with the Securities and Exchange Commission of Pakistan (SECP) as an investment adviser. Past performance does not guarantee future results. Consult a licensed financial adviser before making investment decisions.

---

## License

MIT