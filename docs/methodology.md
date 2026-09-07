Methodology: KSE 100 Portfolio Builder
1. Data Source and Pipeline
Source
Index: KSE 100 (Karachi Stock Exchange 100 Index)
Provider: Investing.com (free historical data)
Frequency: Daily closing prices
Date Range: January 2010 to December 2024 (14 years)
Pipeline
Downloaded daily KSE 100 data as CSV from Investing.com.
Cleaned data: converted dates (DD/MM/YYYY format), removed commas from prices, sorted oldest-first.
Resampled to monthly frequency by taking the last trading day's close of each month.
Calculated monthly price returns: (Close_t / Close_t-1) - 1.
Estimated monthly dividend yield: 7% annual / 12 = 0.583%.
Built Total Return Index: TR = (1 + Price Return) × (1 + Dividend Yield) - 1, rebased to 100.
Why Total Return?
The KSE 100 index only tracks price changes. But investors also earn dividends. The Total Return Index captures both, giving a more accurate picture of actual investor returns. Over 14 years, dividends contribute roughly 40% of total returns.

2. Portfolio Construction
Risk Tiers
Three risk profiles, each a blend of equity (KSE 100) and income (money market fund):

Tier	Equity %	Income %	Description
Conservative	60%	40%	Lower volatility, softer drawdowns
Moderate	80%	20%	Balanced risk/return
Aggressive	100%	0%	Maximum long-term return, full volatility
Income Sleeve
Modeled as a money market fund returning 10% annually (approximate long-term average for Pakistani money market funds, tracking the SBP policy rate).
Converted to monthly: (1 + 0.10)^(1/12) - 1 = 0.797%.
Fees
Annual management fee of 1.5% deducted monthly (0.125%/month).
Applied to the blended portfolio return.
3. SIP/DCA Backtest Engine
Methodology
Simulates a Systematic Investment Plan (SIP): a fixed PKR amount invested at the end of every month.
Each month:
Existing portfolio grows by the blended monthly return (net of fees).
Investor deposits the monthly amount.
Portfolio value updated.
Tracks: monthly investment, cumulative invested, portfolio value, profit/loss, return on invested capital.
Worst-Time-to-Start Analysis
Identified the worst peak-to-trough drawdown in KSE 100 history (May 2017 peak, -38% drawdown to August 2019).
Ran a SIP starting from that peak.
Tracked "underwater" months (portfolio value < cumulative invested).
Demonstrates how rupee-cost averaging smooths out bad market entry timing.
4. Dividends, Transaction Costs, and Taxes
Dividends
KSE 100 average dividend yield: ~7% annually.
Projected annual dividend income at portfolio milestones (Years 1, 3, 5, 10, 15).
Two scenarios: reinvest (compounds into total return) vs. take as cash (income projection).
Transaction Costs (Pakistan-specific)
Every stock purchase on the PSX incurs:

Cost	Rate
Broker Commission	0.15%
CDC Fee	0.02%
PSX Transaction Charge	0.02%
SECP Regulatory Fee	0.01%
FED Levy	0.01%
Total	0.20%
Applied to every monthly purchase in the SIP.

Capital Gains Tax (CGT)
Pakistan's CGT on securities based on holding period:

Holding Period	Rate
< 12 months	15%
12–24 months	12.5%
> 24 months	0% (exempt)
In a SIP, different units have different holding periods. The model calculates a blended CGT rate based on the fraction of units in each bracket at exit.

5. Risk Metrics
Volatility
Monthly standard deviation of returns.
Annualized: monthly_vol × √12.
Max Drawdown
Largest peak-to-trough decline in the portfolio's cumulative value.
Formula: (Current Value - Running Maximum) / Running Maximum.
Tracks peak date, trough date, and recovery date.
Sortino Ratio
Like Sharpe, but only penalizes downside volatility (negative returns).
Formula: (Annual Return - Risk-Free Rate) / Annual Downside Deviation.
Risk-free rate: 10% (Pakistan 1-year T-Bill).
Sortino > 1 = good, > 2 = excellent.
Probability of Loss
Calculated across all possible N-month holding periods in the dataset.
For each holding period length (1, 3, 6, 12, 24, 36, 48, 60 months), counts the fraction of periods that resulted in a negative return.
Key finding: probability drops sharply with holding period length.
6. Scenario Analysis
Forward Projections
Instead of relying solely on historical data, the model projects forward under three scenarios:

Scenario	Equity Annual Return	Income Annual Return	Description
Bull	22%	10%	Strong growth, stability, PKR stabilizes
Base	15%	10%	Historical average performance
Bear	5%	10%	Stagnant market, economic slowdown
Sensitivity Analysis
Monthly Amount: Tested PKR 25k, 50k, 100k, 150k, 200k.
Time Horizon: Tested 3, 5, 10, 15, 20 years.
Shows how final value scales with inputs.
7. Limitations and Assumptions
Dividend yield is assumed constant at 7%. In reality, it varies year to year.
Money market return is assumed constant at 10%. In reality, it tracks the policy rate.
Transaction costs are applied to the full monthly amount. In reality, broker commissions can be negotiated down for larger trades.
CGT rates are based on current Pakistan tax law. These may change.
Forward projections use fixed annual returns. Real markets have volatility clustering, fat tails, and regime changes.
No rebalancing is modeled. In practice, the equity/income drift would require periodic rebalancing.
Inflation is not explicitly modeled. Real returns would be lower than the nominal returns shown.
8. Key Findings
Compounding is powerful: A PKR 100k/month SIP over 15 years (Base scenario) grows to ~PKR 40M+.
Time in the market beats timing: Probability of loss drops from 42% (1-month) to <5% (5-year) to ~0% (10-year).
DCA rescues bad timing: Even starting at the worst peak (May 2017), a SIP recovered within ~3 years.
Costs are manageable: Over 15 years, total costs and taxes consume ~5-8% of final value.
Aggressive tier wins long-term: For young investors with 10+ year horizons, 100% equity maximizes returns despite higher volatility.