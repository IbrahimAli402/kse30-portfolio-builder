"""In-app search: glossary + FAQ, local and instant.

Two tiers, no network, no API key:

1. GLOSSARY — finance terms used anywhere in the dashboard.
2. FAQ      — questions about the app itself ("what is this for?",
              "is the data real?", "is this advice?").

``answer(query)`` returns the best match as a dict so the dialog can render
it consistently. Matching is exact -> alias -> substring -> fuzzy (difflib), then
keyword overlap for FAQ entries.

Upgrade path (deliberately not built): ``llm_fallback`` is a stub. If the
static tiers keep missing real questions, wire an LLM there, scoped by a
system prompt to this app and the values in ``st.session_state``, and always
append DISCLAIMER. Prove the free tier is insufficient first.
"""

from __future__ import annotations

import difflib
import re

DISCLAIMER = "Educational definitions only. Not investment advice."

# ---------------------------------------------------------------------------
# Tier 1 — glossary
# ---------------------------------------------------------------------------
# key: (definition, aliases, related keys)
GLOSSARY: dict[str, tuple[str, list[str], list[str]]] = {
    # ── Beginner: the building blocks ──────────────────────────────────
    "stock": (
        "A share of ownership in a company. If you own one share of ENGRO, you own a "
        "tiny fraction of ENGRO's business and are entitled to a share of its profits.",
        ["share", "equity share", "common stock"], ["dividend", "market cap"],
    ),
    "bond": (
        "A loan you make to a company or government. It pays fixed interest and returns "
        "your principal at maturity. Bonds are generally less risky than stocks but offer "
        "lower returns.",
        ["fixed income security", "debt security"], ["fixed income", "risk"],
    ),
    "mutual fund": (
        "A pooled investment where a professional manager buys stocks or bonds for all "
        "investors. You pay an annual fee (expense ratio) for this. This app's DIY vs "
        "Mutual Fund section shows how that fee compounds against you over time.",
        ["fund", "managed fund", "unit trust"], ["expense ratio", "nav"],
    ),
    "dividend": (
        "Cash that a company pays out from its profits to shareholders. The KSE-100 has "
        "historically paid around 7% of its value every year in dividends.",
        ["cash dividend", "dividend payment"], ["dividend yield", "drip"],
    ),
    "portfolio": (
        "The collection of all investments you hold. A portfolio with 10 stocks is more "
        "diversified than one with 2, but only if those stocks don't all move together.",
        ["holdings", "investments"], ["diversification", "asset allocation"],
    ),
    "compound interest": (
        "Earning returns on your returns. If you invest PKR 10,000 at 15% for 15 years, "
        "you don't earn 225% — you earn about 714%, because each year's gain becomes part "
        "of next year's base. This is the single most powerful force in long-term investing.",
        ["compounding", "compound return", "compound growth"], ["sip", "return"],
    ),
    "return": (
        "How much your investment grew (or shrank), expressed as a percentage. A PKR 100 "
        "investment worth PKR 115 has a 15% return. Total return includes dividends; price "
        "return does not.",
        ["investment return", "gain", "profit"], ["total return", "compound interest"],
    ),
    "risk": (
        "The chance that your investment loses money or grows less than expected. In this "
        "app, risk is measured by volatility (how much returns swing), max drawdown (how "
        "far you fall from peak), and probability of loss.",
        ["investment risk", "downside"], ["volatility", "max drawdown"],
    ),
    "inflation": (
        "The general rise in prices over time, which reduces what your money can buy. If "
        "your portfolio returns 15% but inflation is 12%, your real return is only about 3%.",
        ["cpi", "price inflation"], ["real return", "usd return"],
    ),
    "bull market": (
        "A period of rising prices, usually driven by economic growth and investor optimism. "
        "The Bull scenario in this app assumes 22% annual equity returns.",
        ["bull", "bullish"], ["bear market", "scenario"],
    ),
    "bear market": (
        "A period of falling prices, usually 20% or more from the peak, driven by economic "
        "contraction or crisis. The Bear scenario assumes only 5% annual equity returns.",
        ["bear", "bearish"], ["bull market", "scenario", "max drawdown"],
    ),
    "market cap": (
        "Market capitalisation: share price multiplied by total shares. A company with 100 "
        "million shares at PKR 50 each has a market cap of PKR 5 billion. Large-cap stocks "
        "tend to be more stable; small-cap stocks more volatile.",
        ["market capitalisation", "cap", "large cap", "small cap"], ["stock", "p/e ratio"],
    ),
    "equity": (
        "Ownership in a company, as opposed to lending to it (bonds). The Aggressive risk "
        "tier is 100% equity; Conservative is 60% equity, 40% income fund.",
        ["equity investment"], ["stock", "fixed income", "risk tier"],
    ),
    "fixed income": (
        "Investments that pay a fixed return: bonds, term deposits, money market funds. "
        "Lower risk and lower return than equity. The Conservative tier uses 40% income fund.",
        ["income fund", "bonds", "debt"], ["bond", "risk tier"],
    ),
    "rebalancing": (
        "Selling assets that grew too large and buying ones that shrank, to return to your "
        "target weights. This app's Basket tab shows how far your basket has drifted and "
        "what rebalancing would cost in fees and tax.",
        ["rebalance", "drift"], ["asset allocation", "portfolio"],
    ),
    "asset allocation": (
        "How you split your money across asset classes: equity, fixed income, gold, cash. "
        "Studies show allocation explains 90% of long-term return differences between portfolios.",
        ["allocation"], ["diversification", "risk tier", "rebalancing"],
    ),
    "time horizon": (
        "How long you plan to hold your investment before needing the money. Longer horizons "
        "can tolerate more equity (and more volatility) because there's time to recover from "
        "drawdowns. This app supports 5 to 20 year horizons.",
        ["horizon", "investment period"], ["risk tier", "risk tolerance"],
    ),
    "risk tolerance": (
        "Your psychological capacity to watch your portfolio fall without panic-selling. "
        "The suitability quiz at the start of this app estimates your tolerance and suggests "
        "a risk tier.",
        ["risk appetite", "risk profile"], ["risk tier", "time horizon"],
    ),
    "capital gain": (
        "Profit from selling an investment above what you paid. In Pakistan, capital gains "
        "on shares are taxed at 15% if held under 12 months, 12.5% for 12 to 24 months, and "
        "0% beyond 24 months.",
        ["capital gains", "profit on sale"], ["cgt", "capital loss"],
    ),
    "capital loss": (
        "Loss from selling an investment below what you paid. The Risk tab shows how often "
        "different holding periods ended in a loss based on KSE-100 history.",
        ["loss"], ["capital gain", "max drawdown"],
    ),
    "broker": (
        "A licensed intermediary who executes buy and sell orders on the stock exchange for "
        "a commission. In Pakistan, you need a CDC sub-account through a broker to trade PSX.",
        ["brokerage", "stockbroker"], ["exchange", "transaction cost"],
    ),
    "exchange": (
        "A marketplace where stocks are bought and sold. The Pakistan Stock Exchange (PSX) "
        "is where all KSE-100 and KSE-30 companies trade.",
        ["psx", "stock exchange"], ["kse-100", "broker"],
    ),
    "nav": (
        "Net Asset Value: the per-unit value of a mutual fund, calculated daily as total "
        "assets minus liabilities divided by units outstanding. You buy and sell funds at NAV.",
        ["net asset value"], ["mutual fund", "expense ratio"],
    ),
    "expense ratio": (
        "The annual fee a mutual fund charges, expressed as a percentage of your investment. "
        "Pakistani equity funds typically charge 1.5% to 2.5%. This app's DIY portfolio "
        "assumes 1.5%, and shows how even 1% more compounds into a massive wealth gap.",
        ["fund fee", "management fee", "mer"], ["mutual fund", "fees"],
    ),
    "eps": (
        "Earnings Per Share: a company's total profit divided by its number of shares. If a "
        "company earns PKR 10 billion and has 1 billion shares, EPS is PKR 10.",
        ["earnings per share"], ["p/e ratio", "roe"],
    ),
    "price to book": (
        "Price to Book ratio: share price divided by book value (assets minus liabilities) "
        "per share. Below 1 means the stock trades for less than its accounting worth.",
        ["p/b", "pb", "book value"], ["p/e ratio", "market cap"],
    ),
    "dividend payout ratio": (
        "The share of earnings a company pays out as dividends. A company earning PKR 10 per "
        "share and paying PKR 7 has a 70% payout ratio. High payout means less money reinvested "
        "in growth.",
        ["payout ratio", "payout"], ["dividend", "dividend yield"],
    ),
    "market order": (
        "An instruction to buy or sell immediately at the best available price. Fast execution "
        "but no price guarantee — on illiquid stocks you may get a worse price than expected.",
        [], ["limit order", "adv"],
    ),
    "limit order": (
        "An instruction to buy or sell only at a specified price or better. Gives you price "
        "control but may not execute if the market doesn't reach your limit.",
        [], ["market order"],
    ),
    "stop loss": (
        "An order to sell automatically if the price falls to a set level, limiting your "
        "downside. Not commonly used in SIP investing, where you buy more on dips instead.",
        ["stop loss order"], ["max drawdown", "risk"],
    ),
    "asset class": (
        "A category of investments with similar characteristics: equity (stocks), fixed income "
        "(bonds), commodities (gold, oil), real estate, cash. Each class has its own risk-return "
        "profile. The Multi-Asset Comparison section shows how KSE-100, gold and USD compare.",
        ["asset classes"], ["asset allocation", "diversification"],
    ),
    "gold": (
        "A traditional store of value in Pakistan. Gold tends to hold its worth during currency "
        "depreciation and equity crashes, making it a diversifier. The Multi-Asset Comparison "
        "section shows its risk-return profile alongside KSE-100.",
        ["precious metal", "soney"], ["asset class", "usd return"],
    ),
    "real estate": (
        "Physical property as an investment. Illiquid (hard to sell quickly) but can provide "
        "rental income and capital appreciation. Not directly modelled in this app, but discussed "
        "in the Multi-Asset Comparison as context.",
        ["property"], ["asset class", "liquidity"],
    ),
    "liquidity": (
        "How easily you can convert an investment to cash without affecting its price. Stocks "
        "with high daily volume are liquid; real estate is not. The Basket tab's Liquidity "
        "Profile shows how many days it would take to exit each position.",
        ["marketability"], ["adv", "days to liquidate"],
    ),
    "ipo": (
        "Initial Public Offering: when a company first sells shares to the public on a stock "
        "exchange. IPOs can be volatile and are not part of this app's KSE-30 universe.",
        ["initial public offering", "listing"], ["exchange", "stock"],
    ),
    "real return": (
        "Your return after subtracting inflation. If your portfolio returns 15% and inflation "
        "is 12%, your real return is about 3%. The Forecast tab has a 'Real' toggle to show "
        "inflation-adjusted values.",
        ["inflation adjusted return", "real return"], ["inflation", "return"],
    ),
    "emergency fund": (
        "3 to 6 months of expenses saved in a safe, accessible account before you start investing. "
        "Without one, a market downturn can force you to sell at a loss. Build this first.",
        ["emergency savings", "rainy day fund"], ["risk tolerance", "time horizon"],
    ),

    # ── Intermediate: ratios and analysis ──────────────────────────────
    "sharpe ratio": (
        "Risk-adjusted return: how much return you earned per unit of volatility taken on. "
        "Above 1 is generally considered good; the KSE-100 has historically sat below that.",
        ["sharpe", "sharp ratio", "sharp"], ["sortino ratio", "volatility"],
    ),
    "sortino ratio": (
        "Like the Sharpe ratio, but it only penalises downside swings. A portfolio that jumps up a lot "
        "and rarely falls scores better on Sortino than on Sharpe.",
        ["sortino"], ["sharpe ratio", "max drawdown"],
    ),
    "max drawdown": (
        "The largest percentage fall from a portfolio's peak to its lowest point before it recovered. "
        "It answers: how bad did it get, at the worst moment?",
        ["drawdown", "maximum drawdown", "peak to trough"], ["volatility", "worst case"],
    ),
    "volatility": (
        "How much monthly returns swing around their average, annualised. Higher volatility means a "
        "bumpier ride, not necessarily a worse outcome.",
        ["vol", "annual volatility", "standard deviation"], ["sharpe ratio", "max drawdown"],
    ),
    "var": (
        "Value at Risk: the most you would expect to lose over a period at a given confidence level. "
        "A 95% one-month VaR of 8% means one month in twenty is expected to lose more than 8%.",
        ["value at risk", "value-at-risk"], ["cvar", "stress test"],
    ),
    "cvar": (
        "Conditional VaR (expected shortfall): the average loss in the months that breach VaR. "
        "It describes how bad the bad months are, so it is always worse than VaR.",
        ["conditional var", "expected shortfall", "es"], ["var"],
    ),
    "beta": (
        "How much a stock or basket moves relative to the index. Beta 1.2 means it tends to move 20% "
        "more than the KSE-100, in both directions.",
        [], ["alpha", "up capture", "capm"],
    ),
    "alpha": (
        "Return earned above what beta alone would predict. Positive alpha means the basket did better "
        "than its market exposure explains.",
        ["jensen alpha", "jensen's alpha"], ["beta", "capm"],
    ),
    "up capture": (
        "When the index rises, the share of that gain your basket captured. 110% means it rose more "
        "than the market in up months.",
        ["upside capture"], ["down capture", "beta"],
    ),
    "down capture": (
        "When the index falls, the share of that loss your basket took on. Below 100% is what you want.",
        ["downside capture"], ["up capture", "beta"],
    ),
    "hhi": (
        "Herfindahl-Hirschman Index: a concentration score built from squared weights. Higher means "
        "your money is bunched into fewer stocks or sectors.",
        ["herfindahl", "concentration"], ["sector concentration"],
    ),
    "sector concentration": (
        "How much of the portfolio sits in one sector. The KSE-30 is structurally heavy in banks, "
        "energy and cement, so an 'index' basket can be less diversified than it looks.",
        ["sector exposure"], ["hhi", "sector rotation"],
    ),
    "p/e ratio": (
        "Price to earnings: how much you pay for one rupee of a company's annual earnings. Lower can "
        "mean cheaper, or that the market expects trouble.",
        ["pe", "p/e", "price to earnings", "pe ratio"], ["roe", "eps"],
    ),
    "roe": (
        "Return on equity: how efficiently a company turns shareholders' capital into profit.",
        ["return on equity"], ["p/e ratio", "eps"],
    ),
    "cgt": (
        "Capital gains tax on shares in Pakistan, as modelled here: 15% if held under 12 months, "
        "12.5% for 12 to 24 months, 0% beyond 24 months. Check current FBR rules before acting.",
        ["capital gains tax", "tax"], ["capital gain", "sip"],
    ),
    "sip": (
        "Systematic Investment Plan: investing a fixed amount on a fixed schedule (here, monthly) "
        "regardless of price. Also called rupee-cost averaging.",
        ["systematic investment plan", "rupee cost averaging", "dca", "dollar cost averaging"],
        ["drip", "worst case", "compound interest"],
    ),
    "drip": (
        "Dividend Reinvestment Plan: using cash dividends to buy more shares instead of spending them. "
        "The Dividends tab shows what that compounding is worth.",
        ["dividend reinvestment", "reinvest dividends"], ["sip", "dividend yield"],
    ),
    "dividend yield": (
        "Annual dividends as a share of price. When you select stocks in the Basket tab, the yield "
        "updates to reflect your specific portfolio's weighted average. Otherwise it defaults to 7%.",
        ["yield"], ["drip", "dividend"],
    ),
    "correlation": (
        "How much two stocks move together, from -1 (opposite) to +1 (identical). Low correlation "
        "between stocks is what makes diversification work. The Correlation Regime Analysis shows "
        "how correlations spike in bear markets.",
        ["correlation coefficient"], ["diversification", "correlation regime"],
    ),
    "r-squared": (
        "How well a basket's returns are explained by the index's movements. R-squared of 0.85 means "
        "85% of the basket's ups and downs track the KSE-100. Higher means less idiosyncratic risk.",
        ["r squared", "r2", "coefficient of determination"], ["beta", "alpha", "tracking error"],
    ),
    "tracking error": (
        "How much a basket's returns deviate from the index, measured as the standard deviation of "
        "the difference. A passive index fund targets near-zero tracking error; an active basket "
        "accepts higher tracking error in pursuit of alpha.",
        ["te"], ["alpha", "information ratio", "r-squared"],
    ),
    "information ratio": (
        "Alpha divided by tracking error. How much excess return you earned per unit of deviation "
        "from the index. Above 0.5 is generally considered good active management.",
        ["ir"], ["alpha", "tracking error"],
    ),
    "benchmark": (
        "A reference point for comparison. The KSE-100 is the benchmark for this app: your basket's "
        "return, risk, alpha and beta are all measured against it.",
        ["benchmark index"], ["kse-100", "alpha", "beta"],
    ),
    "excess return": (
        "Return above the benchmark or risk-free rate. If the KSE-100 returns 15% and your basket "
        "returns 18%, your excess return is 3%.",
        ["outperformance", "surplus return"], ["alpha", "benchmark"],
    ),
    "risk-free rate": (
        "The return on an investment with zero default risk, usually a government Treasury bill. In "
        "Pakistan, the 12-month T-bill yield is the typical risk-free rate. The Sharpe ratio subtracts "
        "it from returns before dividing by volatility.",
        ["risk free rate", "t-bill", "treasury bill"], ["sharpe ratio", "excess return"],
    ),
    "systematic risk": (
        "Market-wide risk that cannot be diversified away: recessions, wars, currency crises. Beta "
        "measures exposure to systematic risk. Even a perfectly diversified portfolio still faces it.",
        ["market risk", "undiversifiable risk"], ["unsystematic risk", "beta"],
    ),
    "unsystematic risk": (
        "Risk specific to a single company or sector: management changes, sector downturns, regulatory "
        "action. Diversification reduces this; holding 12+ stocks eliminates most of it.",
        ["specific risk", "idiosyncratic risk", "diversifiable risk"], ["systematic risk", "diversification"],
    ),
    "recovery": (
        "The time it takes for a portfolio to climb back to its previous peak after a drawdown. The "
        "Drawdown chart annotates the recovery date if one occurred within the data.",
        ["recovery period"], ["max drawdown", "underwater"],
    ),
    "underwater": (
        "When your portfolio value is below the total amount you've invested. The Growth and Worst Case "
        "tabs shade these periods. Being underwater doesn't mean you've lost money — you only lose when "
        "you sell.",
        ["below cost", "below water"], ["max drawdown", "recovery"],
    ),
    "rolling return": (
        "Return over a fixed window (e.g. 3 years) calculated at every point in time, not just start "
        "to end. Rolling returns show the range of outcomes an investor could have experienced, not "
        "just the one lucky endpoint.",
        ["rolling period return"], ["return", "cagr"],
    ),
    "cagr": (
        "Compound Annual Growth Rate: the constant annual rate that would grow your investment from "
        "start to end value. PKR 100 growing to PKR 400 over 10 years has a CAGR of 14.9%.",
        ["compound annual growth rate", "annualised return"], ["return", "compound interest"],
    ),
    "irr": (
        "Internal Rate of Return: the annualised discount rate that makes the present value of all "
        "cash flows (deposits and final value) equal zero. For a SIP, it's the money-weighted return "
        "that accounts for the timing of each deposit.",
        ["internal rate of return", "money weighted return", "xirr"], ["cagr", "sip"],
    ),
    "turnover ratio": (
        "How much of a portfolio is bought and sold per year, as a fraction of total value. High "
        "turnover means more transaction costs and more short-term capital gains tax. The Rebalancing "
        "Guidance section shows your basket's turnover.",
        ["portfolio turnover"], ["rebalancing", "transaction cost"],
    ),
    "free cash flow": (
        "Cash a company generates after paying for its operations and capital expenditures. Positive "
        "and growing FCF means the company can fund dividends, buybacks, or debt reduction without "
        "borrowing.",
        ["fcf"], ["dividend", "eps"],
    ),
    "calmar ratio": (
        "Annualised return divided by max drawdown. A Calmar of 1 means you earn 1% for every 1% of "
        "worst-case drawdown. Higher is better; it rewards smooth, steady growth.",
        [], ["sharpe ratio", "max drawdown"],
    ),
    "treynor ratio": (
        "Excess return per unit of beta (systematic risk), instead of per unit of volatility (total "
        "risk) like the Sharpe ratio. Useful for comparing portfolios within a diversified whole.",
        [], ["sharpe ratio", "beta"],
    ),
    "transaction cost": (
        "The fee paid on each trade: brokerage commission, SECP fee, CDC fee, and spread. This app "
        "assumes 0.2% per deposit, which is conservative for PSX retail trading.",
        ["brokerage", "trading cost", "commission"], ["expense ratio", "turnover ratio"],
    ),

    # ── Expert: models and methods ─────────────────────────────────────
    "monte carlo": (
        "A simulation run thousands of times with randomised monthly returns, producing a range of "
        "outcomes instead of one number. The Forecast tab runs 5,000 paths.",
        ["simulation", "monte carlo simulation"], ["p10 p50 p90", "fan chart", "bootstrap"],
    ),
    "p10 p50 p90": (
        "Percentiles of the simulation. P50 is the median (half the paths did better, half worse); "
        "P10 is a pessimistic case; P90 an optimistic one. Plan around P10 to P50, not P90.",
        ["percentile", "p10", "p50", "p90", "median"], ["monte carlo", "fan chart"],
    ),
    "fan chart": (
        "The shaded cone on the Forecast tab. The band widens over time because uncertainty compounds.",
        [], ["monte carlo", "p10 p50 p90"],
    ),
    "efficient frontier": (
        "The set of portfolios that delivered the best historical return for each level of risk. "
        "Anything below the curve took on risk it was not paid for.",
        ["frontier", "markowitz frontier"], ["minimum variance", "diversification", "modern portfolio theory"],
    ),
    "minimum variance": (
        "The portfolio on the frontier with the lowest volatility. Used as the 'recommended' basket "
        "because it is the most defensive mix of the stocks you picked.",
        ["min variance", "min-variance", "recommended portfolio"], ["efficient frontier"],
    ),
    "diversification": (
        "Spreading money across stocks that do not move together. The diversification curve shows "
        "how quickly the benefit flattens out, usually after 10 to 15 names.",
        ["diversification curve"], ["efficient frontier", "correlation regime", "unsystematic risk"],
    ),
    "correlation regime": (
        "How much stocks move together shifts with the market. In bear markets correlations rise, so "
        "diversification helps least exactly when you need it most.",
        ["correlation", "regime"], ["diversification", "stress test"],
    ),
    "stress test": (
        "Replaying a historical crisis (2008 crash, 2020 COVID, 2022 floods) against today's portfolio "
        "to estimate the hit.",
        ["stress testing", "crisis replay"], ["max drawdown", "var"],
    ),
    "adv": (
        "Average Daily Volume: the typical PKR value of a stock traded per day. Low ADV means you "
        "cannot buy or sell a large position without moving the price.",
        ["average daily volume", "liquidity", "volume"], ["days to liquidate"],
    ),
    "days to liquidate": (
        "How many trading days it would take to sell a position at 10% of daily volume without "
        "moving the price. Over 5 days is flagged.",
        ["liquidate", "exit"], ["adv"],
    ),
    "relative strength": (
        "A sector's return divided by the KSE-100's over the same period. Above 1 means it led the market.",
        ["rs"], ["momentum score", "sector rotation"],
    ),
    "momentum score": (
        "A composite of 1, 3, 6 and 12-month relative strength (weighted 40/30/20/10). Ranks sectors "
        "by recent leadership.",
        ["momentum"], ["relative strength", "sector rotation"],
    ),
    "sector rotation": (
        "Different sectors lead at different points in the cycle: banks in rate-hiking phases, cement "
        "in construction booms. The Market Context tab shows who is leading now.",
        ["rotation"], ["momentum score", "relative strength"],
    ),
    "risk tier": (
        "The equity/income split of your plan. Conservative 60/40, Moderate 80/20, Aggressive 100/0. "
        "More equity means higher expected growth and deeper drawdowns.",
        ["tier", "conservative", "moderate", "aggressive"], ["scenario", "max drawdown", "asset allocation"],
    ),
    "scenario": (
        "The macro assumption behind the projection. Bull: IMF programme succeeds, reforms proceed. "
        "Base: muddle through. Bear: balance-of-payments shock and high inflation.",
        ["bull", "base", "bear", "bull case", "bear case"], ["risk tier", "monte carlo"],
    ),
    "worst case": (
        "The Worst Case tab starts a monthly plan at the market's peak, right before the biggest crash "
        "in the data, to test whether steady investing survives bad timing.",
        ["worst time to start"], ["sip", "max drawdown"],
    ),
    "total return": (
        "Price change plus dividends reinvested. All history in this app uses total return, which is "
        "why it looks better than the headline KSE-100 index.",
        ["tr", "total return index"], ["dividend yield", "return"],
    ),
    "usd return": (
        "Your return measured in dollars. When the rupee weakens, the USD return is lower than the PKR "
        "return: (1 + PKR return) / (1 + rupee depreciation) - 1.",
        ["currency adjusted", "dollar return", "pkr usd", "fx"], ["total return", "inflation"],
    ),
    "kse-30": (
        "The 30 largest, most liquid PSX companies. The Basket tab lets you build a portfolio from them.",
        ["kse30", "kse 30"], ["kse-100"],
    ),
    "kse-100": (
        "The main Pakistan Stock Exchange index of 100 companies. The benchmark for everything here.",
        ["kse100", "kse 100", "psx", "index"], ["kse-30", "total return", "benchmark"],
    ),
    "bootstrap": (
        "A Monte Carlo method that resamples actual historical monthly returns with replacement, "
        "preserving their real distribution. The default method in the Forecast tab. Simpler than GARCH "
        "but assumes the future will look like the past.",
        ["bootstrap simulation", "historical bootstrap"], ["monte carlo", "garch"],
    ),
    "garch": (
        "Generalised Autoregressive Conditional Heteroskedasticity: a model where volatility clusters "
        "(calm periods follow calm, turbulent follows turbulent). Available as a Forecast method, it "
        "captures the fact that big moves tend to cluster.",
        ["generalised autoregressive conditional heteroskedasticity"], ["bootstrap", "monte carlo", "volatility"],
    ),
    "regime switching": (
        "A model that alternates between distinct market states (e.g. bull and bear) with different "
        "return and volatility characteristics. Available as a Forecast method. More realistic than "
        "a single distribution but harder to calibrate.",
        ["markov regime", "regime model", "regime"], ["monte carlo", "correlation regime", "bootstrap"],
    ),
    "modern portfolio theory": (
        "Harry Markowitz's 1952 framework: for any level of risk, there is an optimal mix of assets "
        "that maximises expected return. The Efficient Frontier is its visual output. The core insight "
        "is that a portfolio's risk depends on how assets interact, not just on their individual risks.",
        ["mpt", "markowitz theory", "portfolio theory"], ["efficient frontier", "diversification", "capm"],
    ),
    "capm": (
        "Capital Asset Pricing Model: expected return = risk-free rate + beta times market premium. "
        "It says you're only compensated for systematic (market) risk, not for risk you could diversify "
        "away. Alpha is what you earn above what CAPM predicts.",
        ["capital asset pricing model"], ["beta", "alpha", "risk-free rate"],
    ),
    "tail risk": (
        "The risk of extreme losses in the far tails of the return distribution — events that are rare "
        "but devastating. VaR and CVaR measure it; standard deviation understates it because returns "
        "are not perfectly normal.",
        ["tail risk", "fat tails", "black swan"], ["var", "cvar", "kurtosis"],
    ),
    "kurtosis": (
        "How 'fat-tailed' a return distribution is. High kurtosis means more extreme outcomes (both "
        "gains and losses) than a normal distribution would predict. KSE-100 returns exhibit excess "
        "kurtosis, meaning crashes and spikes happen more often than models assume.",
        ["fat tails", "leptokurtic"], ["tail risk", "var", "skewness"],
    ),
    "skewness": (
        "Asymmetry in the return distribution. Negative skew means more extreme losses than gains; "
        "positive skew means the opposite. Stock markets typically have negative skew — slow gains, "
        "sudden drops.",
        ["skew", "negative skew"], ["kurtosis", "tail risk"],
    ),
    "parametric var": (
        "VaR calculated assuming returns follow a normal distribution, using only the mean and standard "
        "deviation. Faster to compute than historical VaR but underestimates tail risk if returns have "
        "fat tails (which they do).",
        ["analytical var", "variance covariance"], ["var", "cvar", "tail risk"],
    ),
    "historical simulation": (
        "VaR calculated directly from past returns without assuming any distribution: the 95th percentile "
        "worst month in history is your 95% VaR. More accurate than parametric for fat-tailed data, but "
        "assumes the past represents the future.",
        ["historical var"], ["var", "parametric var", "bootstrap"],
    ),
    "value investing": (
        "Buying stocks that trade below their intrinsic value (low P/E, low P/B, high dividend yield). "
        "The philosophy of Benjamin Graham and Warren Buffett. The stock screen in this app includes "
        "valuation as one of its five pillars.",
        ["value stocks", "value factor"], ["p/e ratio", "price to book", "growth investing"],
    ),
    "growth investing": (
        "Buying stocks with rapidly growing earnings or revenue, even at high valuations. The bet is "
        "that growth will continue and justify the price. Higher risk than value investing but higher "
        "potential upside.",
        ["growth stocks", "growth factor"], ["value investing", "p/e ratio"],
    ),
    "factor investing": (
        "Selecting stocks based on quantifiable characteristics (factors) that explain returns: value, "
        "momentum, quality, low volatility, size. The stock screen's five pillars are a simplified "
        "factor approach.",
        ["smart beta", "factor model", "style investing"], ["value investing", "momentum score"],
    ),
    "risk parity": (
        "An allocation method that equalises risk contribution across assets rather than equalising "
        "rupee amounts. Bonds get more weight (because they're less risky per rupee), leveraged up to "
        "match equity risk. Not used in this app but referenced for context.",
        ["equal risk contribution"], ["asset allocation", "volatility"],
    ),
    "black litterman": (
        "A portfolio optimisation model that blends market equilibrium returns with the investor's own "
        "views, producing more stable weights than pure Markowitz optimisation. Not implemented here, "
        "but it addresses the estimation error problem noted in the Efficient Frontier caveat.",
        ["black-litterman"], ["efficient frontier", "modern portfolio theory"],
    ),
    "withholding tax": (
        "Tax deducted at source on dividends and interest. In Pakistan, companies "
        "deduct tax before paying dividends to your account. Not modelled in this app's "
        "projections, which focus on capital gains tax on exit.",
        ["wht"], ["cgt", "dividend"],
    ),
    "cds": (
        "Central Depository System: the electronic system that holds your shares. You "
        "need a CDC sub-account (opened through your broker) to trade on PSX.",
        ["cdc", "central depository"], ["broker", "exchange"],
    ),
    "kmi": (
        "KSE-MSCI Index: a broader index sometimes referenced for Pakistan equity "
        "performance. This app uses the KSE-100 as its benchmark, not KMI.",
        ["msci", "kse msci"], ["kse-100", "benchmark"],
    ),
    "lot size": (
        "The minimum number of shares you can buy or sell in a single transaction on "
        "PSX. Varies by stock. This app's basket calculator rounds share counts down "
        "to whole numbers, not lot sizes.",
        ["board lot", "minimum lot"], ["market order", "broker"],
    ),
    "spread": (
        "The difference between the buy (ask) and sell (bid) price. Wider spreads on "
        "illiquid stocks mean you effectively pay a hidden cost on every trade. The 0.2% "
        "transaction cost assumed in this app is separate from spread.",
        ["bid ask spread", "bid ask"], ["transaction cost", "liquidity"],
    ),
}

# ---------------------------------------------------------------------------
# Tier 2 — FAQ about the app
# ---------------------------------------------------------------------------
FAQ: list[dict] = [
    dict(
        id="purpose",
        title="What is this app for?",
        keywords="what is this app purpose use for me about summary overview point why do explain application end user",
        body=(
            "It answers one question for a Pakistani earner: if I invest a fixed amount every month in "
            "the stock market, what could happen? It projects your monthly plan forward, tests it against "
            "2010 to 2024 history, shows the risks honestly (drawdowns, probability of loss, crises), and "
            "lets you build your own basket from the KSE-30. Start on Forecast, then Goals."
        ),
        related=["how to use", "is this advice"],
    ),
    dict(
        id="how to use",
        title="How do I use it?",
        keywords="how use start begin steps guide navigate where tabs workflow",
        body=(
            "Set your monthly amount, risk tier, horizon and scenario in the sidebar. Every tab then "
            "reads from those. Forecast shows the range of outcomes. Goals works backward from a target. "
            "Basket lets you pick stocks. Growth, Risk, Worst case and Dividends are the evidence. "
            "Market context is reference material."
        ),
        related=["purpose", "risk tier"],
    ),
    dict(
        id="is this advice",
        title="Is this investment advice?",
        keywords="advice recommend should i invest buy sell financial advisor legal",
        body=(
            "No. It is an analytical and educational tool. It does not know your full finances, does "
            "not recommend specific trades, and its projections are model assumptions, not predictions. "
            "Talk to a licensed advisor before acting."
        ),
        related=["purpose", "data"],
    ),
    dict(
        id="data",
        title="Where does the data come from, and is it real?",
        keywords="data source real accurate fresh updated where from provenance nightly live",
        body=(
            "Yes, it is real. KSE-100 total return, KSE-30 stock prices, USD/PKR and gold are fetched "
            "from public sources and refreshed nightly by an automated job. Historical charts recompute "
            "from those files every time the page loads. There is no live intraday feed."
        ),
        related=["is this advice"],
    ),
    dict(
        id="forecast",
        title="How is the forecast made?",
        keywords="forecast projection predict future how calculated model assumptions",
        body=(
            "Two ways. The fixed projection compounds your monthly deposit at the scenario's assumed "
            "return (Bull 22%, Base 15%, Bear 5% equity), minus fees, costs and tax. The Monte Carlo "
            "forecast draws 5,000 random return paths from history and shows the spread."
        ),
        related=["monte carlo", "scenario"],
    ),
    dict(
        id="lac cr",
        title="What do Lac and Cr mean?",
        keywords="lac lakh crore cr units currency format number",
        body="1 Lac = 100,000 PKR. 1 Cr (crore) = 10,000,000 PKR = 100 Lac.",
        related=[],
    ),
    dict(
        id="fees",
        title="What fees and taxes are assumed?",
        keywords="fees cost charges tax expense ratio brokerage",
        body=(
            "1.5% annual fee, 0.2% transaction cost on every deposit, and capital gains tax on exit "
            "blended across holding periods (15% under 12 months, 12.5% for 12 to 24, 0% after)."
        ),
        related=["cgt"],
    ),
    dict(
        id="methodology",
        title="What is the methodology?",
        keywords="methodology method approach how calculated formula building blocks identity",
        body=(
            "Expected returns use a building-block identity: dividend yield + earnings growth + inflation "
            "+ valuation change. Historical analysis uses KSE-100 total return data from 2010 to 2024. "
            "Monte Carlo uses bootstrap resampling of historical monthly returns. All details are in "
            "docs/methodology.md."
        ),
        related=["forecast", "data"],
    ),
    dict(
        id="basket vs index",
        title="How is the basket different from the index?",
        keywords="basket vs index difference custom portfolio kse 30 kse 100",
        body=(
            "The index (KSE-100) is the benchmark everything is measured against. The basket is your own "
            "selection of up to 30 KSE-30 stocks with custom weights. The Basket tab compares your basket's "
            "return, risk, alpha and beta against the index, and shows how diversified it is."
        ),
        related=["kse-100", "kse-30", "efficient frontier"],
    ),
    dict(
        id="can i lose money",
        title="Can I lose money?",
        keywords="lose money loss risk downside crash go to zero",
        body=(
            "Yes. The Risk tab shows the probability of losing money for each holding period. A single "
            "month lost money about 40% of the time historically. Even over 5 years, there's a small "
            "chance of being below your deposits. Over 10+ years, the probability approaches zero — but "
            "it never reaches it."
        ),
        related=["risk", "max drawdown", "worst case"],
    ),
    dict(
        id="crash",
        title="What happens if the market crashes?",
        keywords="crash crash 2008 covid 2020 crisis fall drop panic",
        body=(
            "The Worst Case tab shows what happened to a SIP started right at the 2017 market peak, before "
            "a significant fall. The Stress Test section replays historical crises (2008, 2020 COVID, 2022 "
            "floods) against the portfolio. The key lesson: rupee-cost averaging means you buy more units "
            "at lower prices, which speeds recovery."
        ),
        related=["worst case", "stress test", "max drawdown"],
    ),
    dict(
        id="pkr vs usd",
        title="Why are PKR and USD returns different?",
        keywords="pkr usd difference currency depreciation dollar rupee exchange rate",
        body=(
            "The PKR has depreciated significantly against the USD over the past 15 years. When you measure "
            "returns in dollars, that depreciation is subtracted. If the KSE-100 returns 15% in PKR but the "
            "rupee falls 8% against the dollar, your USD return is about 7%. For investors with USD expenses "
            "(education, imports, travel), the USD number is the real return."
        ),
        related=["usd return", "inflation", "real return"],
    ),
]

_SUGGESTED = [
    "Sharpe ratio", "Max drawdown", "VaR", "Compound interest",
    "What is this app for?", "Is this advice?", "SIP", "Monte Carlo",
    "Diversification", "P/E ratio",
]


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s/\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_ALIAS_INDEX: dict[str, str] = {}
for _k, (_d, _aliases, _r) in GLOSSARY.items():
    _ALIAS_INDEX[_k] = _k
    for _a in _aliases:
        _ALIAS_INDEX[_norm(_a)] = _k

_STOP = {"what", "is", "the", "a", "an", "of", "does", "mean", "do", "i", "this", "app", "me", "for",
         "how", "to", "in", "it", "and", "or", "my", "can", "you", "tell", "about", "give", "please",
         "whats", "what's", "explain", "define", "definition"}

_DISPLAY = {
    "var": "VaR", "cvar": "CVaR", "hhi": "HHI", "p/e ratio": "P/E ratio", "roe": "ROE", "cgt": "CGT",
    "sip": "SIP", "drip": "DRIP", "adv": "ADV", "p10 p50 p90": "P10 / P50 / P90",
    "kse-30": "KSE-30", "kse-100": "KSE-100", "usd return": "USD return", "monte carlo": "Monte Carlo",
    "nav": "NAV", "eps": "EPS", "ipo": "IPO", "irr": "IRR", "cagr": "CAGR", "fcf": "FCF",
    "capm": "CAPM", "garch": "GARCH", "mpt": "MPT", "r-squared": "R-squared", "p/b": "P/B",
    "price to book": "Price to Book", "real return": "Real return", "bull market": "Bull market",
    "bear market": "Bear market", "market cap": "Market cap", "market order": "Market order",
    "limit order": "Limit order", "stop loss": "Stop loss", "asset class": "Asset class",
    "asset allocation": "Asset allocation", "time horizon": "Time horizon",
    "risk tolerance": "Risk tolerance", "capital gain": "Capital gain", "capital loss": "Capital loss",
    "risk-free rate": "Risk-free rate", "systematic risk": "Systematic risk",
    "unsystematic risk": "Unsystematic risk", "rolling return": "Rolling return",
    "turnover ratio": "Turnover ratio", "free cash flow": "Free cash flow",
    "calmar ratio": "Calmar ratio", "treynor ratio": "Treynor ratio",
    "transaction cost": "Transaction cost", "tail risk": "Tail risk",
    "parametric var": "Parametric VaR", "historical simulation": "Historical simulation",
    "value investing": "Value investing", "growth investing": "Growth investing",
    "factor investing": "Factor investing", "risk parity": "Risk parity",
    "black litterman": "Black-Litterman", "modern portfolio theory": "Modern Portfolio Theory",
    "regime switching": "Regime switching", "bootstrap": "Bootstrap",
    "compound interest": "Compound interest", "dividend yield": "Dividend yield",
    "dividend payout ratio": "Dividend payout ratio", "emergency fund": "Emergency fund",
    "mutual fund": "Mutual fund", "expense ratio": "Expense ratio",
    "efficient frontier": "Efficient frontier", "minimum variance": "Minimum variance",
    "diversification": "Diversification", "correlation regime": "Correlation regime",
    "stress test": "Stress test", "days to liquidate": "Days to liquidate",
    "relative strength": "Relative strength", "momentum score": "Momentum score",
    "sector rotation": "Sector rotation", "risk tier": "Risk tier",
    "scenario": "Scenario", "worst case": "Worst case", "total return": "Total return",
    "fan chart": "Fan chart", "sector concentration": "Sector concentration",
    "correlation": "Correlation", "tracking error": "Tracking error",
    "information ratio": "Information ratio", "benchmark": "Benchmark",
    "excess return": "Excess return", "recovery": "Recovery", "underwater": "Underwater",
    "real estate": "Real estate", "liquidity": "Liquidity",
    "broker": "Broker", "exchange": "Exchange", "stock": "Stock",
    "bond": "Bond", "dividend": "Dividend", "portfolio": "Portfolio",
    "return": "Return", "risk": "Risk", "inflation": "Inflation",
    "equity": "Equity", "fixed income": "Fixed income", "rebalancing": "Rebalancing",
    "alpha": "Alpha", "beta": "Beta", "up capture": "Up capture", "down capture": "Down capture",
    "sharpe ratio": "Sharpe ratio", "sortino ratio": "Sortino ratio",
    "volatility": "Volatility", "max drawdown": "Max drawdown",
    "kurtosis": "Kurtosis", "skewness": "Skewness", "gold": "Gold",
    "withholding tax": "Withholding tax", "cds": "CDS", "kmi": "KMI",
    "lot size": "Lot size", "spread": "Spread",
}


def display_name(key: str) -> str:
    return _DISPLAY.get(key, key[:1].upper() + key[1:])


def _glossary_hit(key: str, score: float) -> dict:
    d, _aliases, related = GLOSSARY[key]
    return dict(kind="term", title=display_name(key), body=d,
                related=[display_name(r) for r in related], score=score)


def _faq_hit(entry: dict, score: float) -> dict:
    return dict(kind="faq", title=entry["title"], body=entry["body"], related=entry["related"], score=score)


def answer(query: str) -> dict | None:
    """Best single answer for a free-text query, or None."""
    q = _norm(query)
    if not q:
        return None

    # 1. exact / alias
    if q in _ALIAS_INDEX:
        return _glossary_hit(_ALIAS_INDEX[q], 1.0)

    # 2. query contains a glossary key or alias (longest wins)
    contained = [k for k in _ALIAS_INDEX if len(k) >= 3 and re.search(rf"\b{re.escape(k)}\b", q)]
    if contained:
        best = max(contained, key=len)
        return _glossary_hit(_ALIAS_INDEX[best], 0.9)

    # 3. FAQ keyword overlap
    words = {w for w in q.split() if w not in _STOP}
    best_faq, best_overlap = None, 0
    for e in FAQ:
        kw = set(e["keywords"].split())
        overlap = len(words & kw)
        if overlap > best_overlap:
            best_faq, best_overlap = e, overlap
    if best_faq and best_overlap >= 1 and (best_overlap >= 2 or len(words) <= 2):
        return _faq_hit(best_faq, 0.6 + 0.1 * best_overlap)

    # 4. fuzzy on glossary keys and aliases (typos: "sharp ratio", "sortinno")
    close = difflib.get_close_matches(q, list(_ALIAS_INDEX), n=1, cutoff=0.72)
    if close:
        return _glossary_hit(_ALIAS_INDEX[close[0]], 0.7)
    for w in words:
        close = difflib.get_close_matches(w, list(_ALIAS_INDEX), n=1, cutoff=0.8)
        if close:
            return _glossary_hit(_ALIAS_INDEX[close[0]], 0.65)

    # 5. weak FAQ fallback: any overlap at all
    if best_faq and best_overlap >= 1:
        return _faq_hit(best_faq, 0.5)
    return None


def suggestions() -> list[str]:
    return list(_SUGGESTED)


def all_terms() -> list[str]:
    return sorted(GLOSSARY)


def llm_fallback(query: str, context: dict | None = None) -> str | None:  # pragma: no cover
    """Upgrade hook. Returns None until an LLM is wired in.

    When enabled: build a system prompt from FAQ + GLOSSARY + ``context``
    (the user's current amount, tier, horizon, scenario), forbid advice, and
    always append DISCLAIMER. Keep the static tiers first; call this last.
    """
    return None