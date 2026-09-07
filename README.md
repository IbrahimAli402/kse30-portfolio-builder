📈 KSE 100 Portfolio Builder for Young Pakistani Professionals
A data-driven tool that helps young earners in Pakistan (PKR 200k–500k/month) construct and evaluate a passive KSE 100 investment portfolio using rupee-cost averaging (monthly SIP).

🔗 Live Dashboard: [Insert your Streamlit URL here] 📂 Methodology: See docs/methodology.md

🎯 The Problem
Most young professionals in Pakistan want to invest but don't know where to start. They hear "the market is risky" and hesitate. This project answers the questions they actually have:

If I invest PKR 50,000 every month, how much will I have in 10 years?
What if I start right before a market crash?
How much will I pay in fees and taxes?
How likely am I to lose money if I hold for 5 years?
🚀 Key Features
SIP/DCA Backtest Engine: Simulates monthly investing over 14 years of KSE 100 history
Three Risk Tiers: Conservative (60/40), Moderate (80/20), Aggressive (100% equity)
Bull/Base/Bear Scenarios: Forward projections under three market conditions
Pakistan-Specific Costs: Transaction fees (0.2%), Capital Gains Tax (15%/12.5%/0%)
Risk Metrics: Volatility, Max Drawdown, Sortino Ratio, Probability of Loss
Worst-Time-to-Start Analysis: Shows how DCA rescues bad market timing
Interactive Dashboard: Live web app with sliders for monthly amount, horizon, and risk tier
📊 Key Insights
Compounding works: PKR 100k/month for 15 years (Base scenario) grows to ~PKR 40M+ — a 170%+ return.
Time destroys risk: The probability of losing money drops from 42% (1-month hold) to under 5% (5-year hold) and effectively 0% (10-year hold).
DCA rescues bad timing: Even starting at the May 2017 peak (before a 38% crash), a disciplined SIP recovered and profited within ~3 years.
Costs matter: Over 15 years, transaction costs and taxes eat ~5-8% of your final value — which is why low-turnover passive investing beats active trading.
🛠️ Tech Stack
Python (pandas, numpy, matplotlib, scipy)
Jupyter Notebooks (analysis and modeling)
Streamlit (interactive dashboard)
Git & GitHub (version control)
📂 Project Structure
text
Copy
kse30-portfolio-builder/
├── data/
│   ├── raw/                    # Raw KSE 100 data from Investing.com
│   └── processed/              # Cleaned monthly total return data
├── notebooks/
│   ├── 01_data_pipeline.ipynb          # Data download, cleaning, TR index
│   ├── 02_portfolio_construction.ipynb  # Risk tiers, allocation, blended returns
│   ├── 03_sip_dca_backtest.ipynb        # Monthly SIP simulation
│   ├── 04_dividends_costs_taxes.ipynb   # Dividends, transaction costs, CGT
│   ├── 05_risk_metrics.ipynb            # Volatility, drawdown, Sortino, prob of loss
│   └── 06_scenario_analysis.ipynb       # Bull/Base/Bear projections
├── app/
│   └── dashboard.py              # Streamlit interactive dashboard
├── docs/
│   ├── methodology.md            # Detailed methodology writeup
│   └── *.png                     # Charts saved from notebooks
├── requirements.txt
└── README.md
🔧 Setup & Usage
Prerequisites
Python 3.11 or 3.12
Git
Installation
Clone the repo:

bash
Copy
git clone https://github.com/IbrahimAli402/kse30-portfolio-builder.git
cd kse30-portfolio-builder
Create and activate a virtual environment:

bash
Copy
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
Install dependencies:

bash
Copy
pip install -r requirements.txt
Running the Dashboard Locally
bash
Copy
streamlit run app/dashboard.py
Running the Notebooks
Open any notebook in the notebooks/ folder with Jupyter:

bash
Copy
jupyter notebook
📈 Methodology Summary
Data: KSE 100 daily prices from Investing.com (2010–2024), resampled to monthly.
Total Return Index: Price return + 7% annual dividend yield (reinvested).
Portfolio Tiers: Blended returns of KSE 100 (equity) and 10% money market (income), net of 1.5% annual fee.
SIP Engine: Monthly deposit, portfolio grows by blended return, cumulative tracking.
Costs: 0.2% per transaction (broker + CDC + PSX + SECP + FED).
Taxes: CGT at 15% (<12mo), 12.5% (12–24mo), 0% (>24mo). Blended rate applied on exit.
Risk: Annualized volatility, max drawdown (peak-to-trough), Sortino ratio, rolling holding-period probability of loss.
See docs/methodology.md for the full writeup.

📜 License
MIT

⚠️ Disclaimer
This project is for educational purposes only. It is not investment advice. Past performance does not guarantee future results. Always consult a licensed financial advisor before making investment decisions.