"""KSE 100 Portfolio Builder — local finance glossary.

Instant, zero-cost lookup for beginner finance jargon used across the app.
Educational definitions only — not investment advice.
"""

GLOSSARY = {
    "sharpe ratio": "Risk-adjusted return: how much return you got per unit "
        "of volatility taken on. Higher is better; above 1 is generally good.",
    "sortino ratio": "Like Sharpe, but only penalizes downside volatility.",
    "max drawdown": "The largest percentage drop from a portfolio's peak to "
        "its lowest point before recovering.",
    "var": "Value at Risk — the most you'd expect to lose in a bad period, "
        "at a given confidence level.",
    "cvar": "Conditional VaR — the average loss when you're already in that "
        "worst-case tail. Always worse than VaR.",
    "beta": "How much a stock or basket moves relative to the index.",
    "alpha": "Return earned above what beta alone would predict.",
    "up capture": "When the index rises, the share of that gain your basket captured.",
    "down capture": "When the index falls, the share of that loss your basket took on.",
    "hhi": "Herfindahl-Hirschman Index — a concentration score.",
    "p/e ratio": "Price-to-Earnings — how much you pay for PKR 1 of earnings.",
    "roe": "Return on Equity — how efficiently a company turns capital into profit.",
    "cgt": "Capital Gains Tax. Pakistan: 15% under 12 months, 12.5% for 12–24, 0% beyond 24.",
    "sip": "Systematic Investment Plan — a fixed amount on a fixed schedule.",
    "drip": "Dividend Reinvestment Plan — using cash dividends to buy more shares.",
    "monte carlo": "A simulation run thousands of times under randomized conditions.",
    "p10 p50 p90": "Monte Carlo percentiles — P50 median, P10 pessimistic, P90 optimistic.",
    "efficient frontier": "The best possible return for each level of risk, historically.",
    "correlation regime": "How much stocks move together shifts in bull vs bear markets.",
    "volatility": "How much returns swing month to month.",
    "adv": "Average Daily Volume — typical PKR value of a stock traded per day.",
    "days to liquidate": "How many trading days to sell a position without moving its price.",
    "relative strength": "How a sector performed vs the KSE-100 over a period.",
    "momentum score": "A composite of 1/3/6/12-month relative strength.",
    "sector rotation": "Different sectors leading the market at different points in the cycle.",
}


def lookup(query: str) -> str | None:
    q = query.strip().lower()
    if q in GLOSSARY:
        return GLOSSARY[q]
    for key, definition in GLOSSARY.items():
        if q in key or key in q:
            return definition
    return None