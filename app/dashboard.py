"""KSE 100 Portfolio Builder — interactive dashboard.

Every historical figure on this page (drawdowns, probability of loss, the
worst-time-to-start backtest, the 2010–2024 SIP) is computed at runtime from
``data/processed/*.csv`` — the same files the notebooks wrote. Forward
projections use the fixed scenario assumptions documented in
``docs/methodology.md``.

Styling comes from ``.streamlit/config.toml`` (see DESIGN.md); this file adds
no CSS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
# ── Phase 2 imports ────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kse.monte_carlo import run_monte_carlo, compute_percentiles, compute_probability_table
from kse.scenarios import get_all_scenarios, DEFAULT_WEIGHTS, blend_scenarios
from kse.blocks import expected_return, calculate_all_expected_returns
from kse.screen import screen_all
from kse.data_pipeline import load_stock_metrics, load_kse30_constituents, check_data_freshness

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="KSE 100 Portfolio Builder",
    page_icon=":material/show_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Model assumptions (docs/methodology.md)
# ---------------------------------------------------------------------------
TIERS = {
    "Conservative": {"equity": 0.60, "income": 0.40},
    "Moderate": {"equity": 0.80, "income": 0.20},
    "Aggressive": {"equity": 1.00, "income": 0.00},
}
SCENARIOS = {
    "Bull": {"equity_return": 0.22, "income_return": 0.10},
    "Base": {"equity_return": 0.15, "income_return": 0.10},
    "Bear": {"equity_return": 0.05, "income_return": 0.10},
}
ANNUAL_DIVIDEND_YIELD = 0.07
ANNUAL_FEE = 0.015
TX_COST_PCT = 0.002
CGT_BRACKETS = {"short": 0.15, "medium": 0.125, "long": 0.0}  # <12 mo, 12–24 mo, >24 mo

AMOUNT_MIN, AMOUNT_MAX, AMOUNT_STEP, AMOUNT_DEFAULT = 25_000, 200_000, 5_000, 100_000
HORIZON_MIN, HORIZON_MAX, HORIZON_DEFAULT = 5, 20, 15
HOLDING_PERIODS = [
    (1, "1 mo"), (3, "3 mo"), (6, "6 mo"), (12, "1 yr"), (24, "2 yr"),
    (36, "3 yr"), (48, "4 yr"), (60, "5 yr"), (84, "7 yr"), (120, "10 yr"),
]
# ── Phase 2 constants ──────────────────────────────────────────────────
WEIGHT_PRESETS = {
    "Default (25/50/25)": {"Bull": 0.25, "Base": 0.50, "Bear": 0.25},
    "Equal (33/33/33)": {"Bull": 0.333, "Base": 0.333, "Bear": 0.333},
    "Optimistic (40/40/20)": {"Bull": 0.40, "Base": 0.40, "Bear": 0.20},
    "Pessimistic (15/45/40)": {"Bull": 0.15, "Base": 0.45, "Bear": 0.40},
}

PKR_USD = 280  # Starting exchange rate for USD toggle

# ---------------------------------------------------------------------------
# Design tokens (DESIGN.md) — chart-side mirror of config.toml
# ---------------------------------------------------------------------------
TOKENS = {
    "light": dict(
        bg="#faf9f5", surface="#f5f3ed", text="#2d2a26", muted="#6b665e",
        border="#e3dfd7", accent="#c96442", positive="#2f8f6a", negative="#b85c4a",
        bear="#3f6fb5", invested="#8a8580",
    ),
    "dark": dict(
        bg="#1c1a17", surface="#252320", text="#e8e6e1", muted="#a8a298",
        border="#3a3733", accent="#d97757", positive="#4fae88", negative="#d98a72",
        bear="#7a9bd6", invested="#8a8580",
    ),
}
FONT_STACK = "Instrument Sans, Aptos, Segoe UI, sans-serif"


def theme_tokens() -> dict:
    kind = st.context.theme.type or "light"
    return TOKENS["dark" if kind == "dark" else "light"]


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Formatting — PKR in Lac / Cr
# ---------------------------------------------------------------------------
def fmt_pkr(v: float, decimals: int = 2) -> str:
    v = float(v)
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e7:
        return f"{sign}PKR {_trim(a / 1e7, decimals)} Cr"
    if a >= 1e5:
        return f"{sign}PKR {_trim(a / 1e5, decimals)} Lac"
    return f"{sign}PKR {a:,.0f}"


def _trim(x: float, decimals: int) -> str:
    """1.80 -> 1.8, 92.00 -> 92, 5.26 -> 5.26."""
    return f"{x:.{decimals}f}".rstrip("0").rstrip(".")


def fmt_axis(v: float) -> str:
    a = abs(v)
    if a >= 1e7:
        s = f"{a / 1e7:g} Cr"
    elif a >= 1e5:
        s = f"{a / 1e5:g} L"
    elif a >= 1e3:
        s = f"{a / 1e3:g}k"
    else:
        s = f"{a:g}"
    return ("-" if v < 0 else "") + s


def nice_ticks(vmax: float, n: int = 5) -> list[float]:
    """Round tick values (0 … ≥ vmax) so the axis reads 0 / 2 Cr / 4 Cr, not 3.7 Cr."""
    if vmax <= 0:
        return [0]
    raw = vmax / n
    mag = 10 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    return list(np.arange(0, vmax + step, step))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    tr = pd.read_csv(DATA / "kse_monthly_total_return.csv", parse_dates=["Date"]).set_index("Date")
    rets = pd.read_csv(DATA / "portfolio_returns.csv", parse_dates=["Date"]).set_index("Date")
    risk = pd.read_csv(DATA / "risk_metrics_summary.csv").set_index("Tier")
    return tr, rets, risk


TR, RETS, RISK = load_data()
@st.cache_data(show_spinner=False, ttl=300)
def load_frontier_data():
    """Load stock return matrix and compute frontier (cached)."""
    from kse.frontier import load_stock_return_matrix, compute_efficient_frontier, compute_diversification_curve
    returns_df, tickers = load_stock_return_matrix()
    if returns_df.empty:
        return None
    frontier = compute_efficient_frontier(returns_df)
    div_curve = compute_diversification_curve(returns_df)
    return {"returns": returns_df, "tickers": tickers, "frontier": frontier, "div_curve": div_curve}
@st.cache_data(show_spinner=False, ttl=300)
def run_monte_carlo_cached(weights_tuple, tier, monthly_amount, horizon_months, method, seed=42):
    """Run Monte Carlo simulation (cached for performance)."""
    weights = {"Bull": weights_tuple[0], "Base": weights_tuple[1], "Bear": weights_tuple[2]}
    return run_monte_carlo(
        weights=weights, tier=tier, monthly_amount=monthly_amount,
        horizon=horizon_months, method=method, seed=seed, num_paths=5000
    )
DATA_START, DATA_END = RETS.index[0], RETS.index[-1]


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------
def run_sip(returns: pd.Series, monthly_amount: float, tx_cost_pct: float = 0.0) -> pd.DataFrame:
    """Systematic investment plan: grow last month's value, then deposit.

    Mirrors ``run_sip_backtest`` from notebook 03 (deposit is applied after
    the month's return, so the first month carries no return).
    """
    net = monthly_amount * (1 - tx_cost_pct)
    pv, rows = 0.0, []
    for date, r in returns.items():
        pv = pv * (1 + r) + net
        rows.append((date, pv))
    df = pd.DataFrame(rows, columns=["Date", "Portfolio_Value"]).set_index("Date")
    df["Cumulative_Invested"] = monthly_amount * np.arange(1, len(df) + 1)
    df["Profit"] = df["Portfolio_Value"] - df["Cumulative_Invested"]
    df["Underwater"] = df["Portfolio_Value"] < df["Cumulative_Invested"]
    return df


def project_sip(monthly_amount: float, tier: str, scenario: str, years: int) -> tuple[pd.DataFrame, dict]:
    """Forward projection with fixed annual returns, fees, transaction costs and CGT."""
    months = years * 12
    t, s = TIERS[tier], SCENARIOS[scenario]
    eq_m = (1 + s["equity_return"]) ** (1 / 12) - 1
    inc_m = (1 + s["income_return"]) ** (1 / 12) - 1
    net_r = t["equity"] * eq_m + t["income"] * inc_m - ANNUAL_FEE / 12
    net_inv = monthly_amount * (1 - TX_COST_PCT)

    pv, rows = 0.0, []
    for m in range(1, months + 1):
        pv = pv * (1 + net_r) + net_inv
        if m % 12 == 0:
            rows.append(dict(Year=m // 12, Portfolio_Value=pv, Cumulative_Invested=monthly_amount * m))
    df = pd.DataFrame(rows)
    df["Label"] = "Year " + df["Year"].astype(str)
    df["Profit"] = df["Portfolio_Value"] - df["Cumulative_Invested"]
    df["Annual_Dividend"] = df["Portfolio_Value"] * ANNUAL_DIVIDEND_YIELD
    df["Monthly_Dividend"] = df["Annual_Dividend"] / 12
    df["Transaction_Costs"] = monthly_amount * TX_COST_PCT * df["Year"] * 12

    invested = monthly_amount * months
    profit = pv - invested
    # Blended CGT on exit: share of months held in each bracket (methodology §5).
    short = min(12, months) / months
    medium = min(12, max(0, months - 12)) / months
    long = max(0, months - 24) / months
    blended_cgt = short * CGT_BRACKETS["short"] + medium * CGT_BRACKETS["medium"] + long * CGT_BRACKETS["long"]
    cgt = max(profit, 0) * blended_cgt
    after_tax = pv - cgt
    stats = dict(
        final_value=pv, total_invested=invested, profit=profit,
        return_pct=profit / invested * 100,
        cgt=cgt, blended_cgt_pct=blended_cgt * 100,
        after_tax=after_tax, after_tax_return_pct=(after_tax - invested) / invested * 100,
        tx_costs=monthly_amount * TX_COST_PCT * months,
    )
    return df, stats


def drawdown_series(returns: pd.Series) -> pd.Series:
    cum = (1 + returns).cumprod()
    return cum / cum.cummax() - 1


def drawdown_episode(returns: pd.Series) -> dict:
    """Peak, trough, recovery and depth of the deepest drawdown."""
    dd = drawdown_series(returns)
    trough = dd.idxmin()
    cum = (1 + returns).cumprod()
    peak = cum[:trough].idxmax()
    after = dd[trough:]
    recovered = after[after >= -1e-9]
    recovery = recovered.index[0] if len(recovered) else None
    return dict(peak=peak, trough=trough, recovery=recovery, depth_pct=dd.min() * 100, series=dd)


def sip_irr(df: pd.DataFrame, monthly_amount: float) -> float:
    """Annualised money-weighted return of a SIP (monthly deposits, terminal value)."""
    n = len(df)
    final = df["Portfolio_Value"].iloc[-1]

    def npv(r: float) -> float:  # deposits at t=0..n-1, terminal value at t=n-1
        t = np.arange(n)
        return final - monthly_amount * ((1 + r) ** (n - 1 - t)).sum()

    lo, hi = -0.5, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if npv(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (1 + (lo + hi) / 2) ** 12 - 1


def probability_of_loss(returns: pd.Series) -> pd.DataFrame:
    rows = []
    for n, label in HOLDING_PERIODS:
        roll = (1 + returns).rolling(n).apply(np.prod, raw=True).dropna() - 1
        rows.append(dict(months=n, period=label, prob=(roll < 0).mean() * 100, windows=len(roll)))
    return pd.DataFrame(rows)


def dividend_milestones(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    years = sorted({y for y in (1, 3, 5, 10, horizon) if y <= horizon})
    m = df[df["Year"].isin(years)].copy()
    m["Label"] = "Year " + m["Year"].astype(str)
    return m[["Label", "Year", "Portfolio_Value", "Annual_Dividend", "Monthly_Dividend"]]


# ---------------------------------------------------------------------------
# Charts (Plotly; specs in DESIGN.md)
# ---------------------------------------------------------------------------
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def base_layout(tk: dict, *, height: int = 340, legend: bool = False, hovermode: str = "x unified") -> dict:
    return dict(
        height=height,
        margin=dict(l=8, r=16, t=28 if legend else 12, b=8, pad=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, size=12, color=tk["muted"]),
        hoverlabel=dict(bgcolor=tk["surface"], bordercolor=tk["border"],
                        font=dict(family=FONT_STACK, size=12, color=tk["text"])),
        hovermode=hovermode,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, traceorder="normal",
                    font=dict(size=12, color=tk["text"]), itemwidth=30),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=tk["border"], linewidth=1,
                   tickfont=dict(color=tk["muted"]), ticks="", fixedrange=True, automargin=True),
        yaxis=dict(gridcolor=tk["border"], gridwidth=1, zeroline=False, showline=False,
                   tickfont=dict(color=tk["muted"]), ticks="", fixedrange=True, automargin=True),
    )


def money_axis(fig: go.Figure, vmax: float) -> None:
    ticks = nice_ticks(vmax * 1.05)
    fig.update_yaxes(tickvals=ticks, ticktext=[fmt_axis(t) for t in ticks], range=[0, ticks[-1]])


def end_label(fig: go.Figure, x, y, text: str, tk: dict) -> None:
    """Direct label at a line's end, kept inside the plot so it survives narrow screens."""
    fig.add_annotation(x=x, y=y, text=text, showarrow=False, xanchor="right", yanchor="bottom", yshift=6,
                       font=dict(family=FONT_STACK, size=12, color=tk["text"]), bgcolor=rgba(tk["bg"], 0.7))


def growth_chart(df: pd.DataFrame, tk: dict, *, x_col: str, hover_x: str = "%{x}",
                 show_underwater: bool = False) -> go.Figure:
    fig = go.Figure()
    x = df[x_col]
    # Draw order: portfolio wash, invested line, underwater band, portfolio line (on top).
    fig.add_trace(go.Scatter(
        x=x, y=df["Portfolio_Value"], mode="lines", line=dict(width=0),
        fill="tozeroy", fillcolor=rgba(tk["accent"], 0.10), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df["Cumulative_Invested"], name="Invested", mode="lines", legendrank=2,
        line=dict(color=tk["invested"], width=1.5, dash="4px,4px"),
        hovertemplate="Invested %{customdata}<extra></extra>",
        customdata=[fmt_pkr(v) for v in df["Cumulative_Invested"]],
    ))
    if show_underwater:
        under = df["Portfolio_Value"].where(df["Underwater"])
        fig.add_trace(go.Scatter(
            x=x, y=under, name="Underwater", mode="lines", legendrank=3,
            line=dict(color=tk["negative"], width=0),
            fill="tonexty", fillcolor=rgba(tk["negative"], 0.18), connectgaps=False,
            hoverinfo="skip", showlegend=True,
        ))
    fig.add_trace(go.Scatter(
        x=x, y=df["Portfolio_Value"], name="Portfolio", mode="lines", legendrank=1,
        line=dict(color=tk["accent"], width=2.5),
        hovertemplate="Portfolio %{customdata}<extra></extra>",
        customdata=[fmt_pkr(v) for v in df["Portfolio_Value"]],
    ))
    fig.update_layout(**base_layout(tk, legend=True))
    fig.update_xaxes(title=None, hoverformat=hover_x)
    if x_col == "Label":
        year_ticks(fig, list(x))
    else:
        fig.update_xaxes(range=[x.iloc[0], x.iloc[-1]], dtick="M24", tickformat="%Y")
    money_axis(fig, max(df["Portfolio_Value"].max(), df["Cumulative_Invested"].max()))
    end_label(fig, x.iloc[-1], df["Portfolio_Value"].iloc[-1], fmt_pkr(df["Portfolio_Value"].iloc[-1], 1), tk)
    return fig


def year_ticks(fig: go.Figure, labels: list[str]) -> None:
    keep = [v for i, v in enumerate(labels) if i % 2 == 0 or i == len(labels) - 1]
    fig.update_xaxes(type="category", tickvals=keep, ticktext=[v.replace("Year ", "") + "y" for v in keep])


def compare_chart(runs: dict[str, pd.DataFrame], tk: dict) -> go.Figure:
    colors = {"Bull": tk["positive"], "Base": tk["accent"], "Bear": tk["bear"]}
    fig = go.Figure()
    bull, bear = runs["Bull"], runs["Bear"]
    fig.add_trace(go.Scatter(
        x=pd.concat([bull["Label"], bear["Label"][::-1]]),
        y=pd.concat([bull["Portfolio_Value"], bear["Portfolio_Value"][::-1]]),
        fill="toself", fillcolor=rgba(tk["muted"], 0.07), line=dict(width=0),
        hoverinfo="skip", showlegend=False,
    ))
    for name in ("Bear", "Base", "Bull"):
        d = runs[name]
        fig.add_trace(go.Scatter(
            x=d["Label"], y=d["Portfolio_Value"], name=name, mode="lines",
            legendrank={"Bull": 1, "Base": 2, "Bear": 3}[name],
            line=dict(color=colors[name], width=2.5 if name == "Base" else 2),
            hovertemplate=f"{name} %{{customdata}}<extra></extra>",
            customdata=[fmt_pkr(v) for v in d["Portfolio_Value"]],
        ))
        end_label(fig, d["Label"].iloc[-1], d["Portfolio_Value"].iloc[-1], fmt_pkr(d["Portfolio_Value"].iloc[-1], 1), tk)
    fig.update_layout(**base_layout(tk, legend=True))
    year_ticks(fig, list(bull["Label"]))
    money_axis(fig, bull["Portfolio_Value"].max())
    return fig


def dividend_chart(m: pd.DataFrame, tk: dict) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=m["Label"], y=m["Annual_Dividend"], marker=dict(color=tk["accent"], cornerradius=4),
        width=0.22, text=[fmt_pkr(v, 1) for v in m["Annual_Dividend"]], textposition="outside",
        textfont=dict(family=FONT_STACK, size=12, color=tk["text"]), cliponaxis=False,
        hovertemplate="%{x}<br>%{customdata[0]} per year<br>%{customdata[1]} per month<extra></extra>",
        customdata=np.column_stack([[fmt_pkr(v) for v in m["Annual_Dividend"]],
                                    [fmt_pkr(v) for v in m["Monthly_Dividend"]]]),
    ))
    fig.update_layout(**base_layout(tk, hovermode="closest"), bargap=0.5)
    money_axis(fig, m["Annual_Dividend"].max() * 1.12)
    return fig


def prob_loss_chart(p: pd.DataFrame, tk: dict) -> go.Figure:
    labelled = {"1 mo", "1 yr", "3 yr", "5 yr", "10 yr"}
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=p["period"], y=p["prob"], mode="lines+markers+text",
        line=dict(color=tk["negative"], width=2),
        marker=dict(size=9, color=tk["negative"], line=dict(color=tk["bg"], width=2)),
        fill="tozeroy", fillcolor=rgba(tk["negative"], 0.08),
        text=[f"{v:.0f}%" if lab in labelled else "" for v, lab in zip(p["prob"], p["period"])],
        textposition="top center", textfont=dict(family=FONT_STACK, size=12, color=tk["text"]),
        cliponaxis=False,
        hovertemplate="Hold %{x}: %{y:.0f}% of windows lost money<br>(%{customdata} rolling windows)<extra></extra>",
        customdata=p["windows"],
    ))
    fig.update_layout(**base_layout(tk, height=300, hovermode="closest"))
    fig.update_yaxes(range=[0, max(50, p["prob"].max() + 14)], ticksuffix="%", dtick=10)
    return fig


def drawdown_chart(ep: dict, tk: dict) -> go.Figure:
    dd = ep["series"] * 100
    fig = go.Figure(go.Scatter(
        x=dd.index, y=dd.values, mode="lines", line=dict(color=tk["negative"], width=1.75),
        fill="tozeroy", fillcolor=rgba(tk["negative"], 0.10),
        hovertemplate="%{x|%b %Y}: %{y:.1f}% from peak<extra></extra>",
    ))
    fig.update_layout(**base_layout(tk, height=300, hovermode="x"))
    floor = min(-40, np.floor(dd.min() / 10) * 10 - 5)
    fig.update_yaxes(range=[floor, 3], ticksuffix="%", dtick=10)
    fig.update_xaxes(dtick="M24", tickformat="%Y")
    # Annotate the episode from the data, not from memory.
    fig.add_annotation(x=ep["peak"], y=0, text=f"Peak {ep['peak']:%b %Y}", showarrow=False,
                       yshift=12, font=dict(size=11, color=tk["text"]))
    fig.add_annotation(x=ep["trough"], y=ep["depth_pct"], text=f"{ep['depth_pct']:.1f}% · {ep['trough']:%b %Y}",
                       showarrow=True, arrowhead=0, arrowcolor=tk["muted"], ax=48, ay=18,
                       font=dict(size=11, color=tk["text"]))
    if ep["recovery"] is not None:
        fig.add_annotation(x=ep["recovery"], y=0, text=f"Recovered {ep['recovery']:%b %Y}", showarrow=False,
                           yshift=12, xanchor="left", font=dict(size=11, color=tk["text"]))
    return fig


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------
def insight(text: str) -> None:
    with st.container(border=True):
        st.markdown(f"**Key insight** · {text}")


def table_view(df: pd.DataFrame, label: str = "Table view") -> None:
    with st.expander(label):
        st.dataframe(df, hide_index=True, width="stretch")


def chart(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key, width="stretch")


# ---------------------------------------------------------------------------
# Page body
# ---------------------------------------------------------------------------
tk = theme_tokens()
outer = st.container(horizontal=True, horizontal_alignment="center")
page = outer.container(width=960)

with page:
    st.title("KSE 100 Portfolio Builder")
    st.markdown(
        "A systematic-investment-plan backtest for young Pakistani earners. "
        f"Historical figures are computed from KSE 100 total returns, {DATA_START:%b %Y} – {DATA_END:%b %Y}; "
        "projections use fixed scenario assumptions."
    )

    # ---- Controls ----------------------------------------------------------
    st.space("small")
    monthly_amount = st.select_slider(
        "Monthly investment", options=list(range(AMOUNT_MIN, AMOUNT_MAX + 1, AMOUNT_STEP)),
        value=AMOUNT_DEFAULT, format_func=fmt_pkr, help="Deposited on the last trading day of every month.",
    )
    c1, c2, c3 = st.columns([1.45, 1, 1], gap="large")
    with c1:
        tier = st.segmented_control("Risk tier", list(TIERS), default="Aggressive", required=True, width="stretch")
        st.caption(f"{TIERS[tier]['equity']:.0%} equity · {TIERS[tier]['income']:.0%} income fund")
    with c2:
        horizon = st.slider("Time horizon (years)", HORIZON_MIN, HORIZON_MAX, HORIZON_DEFAULT)
    with c3:
        scenario = st.segmented_control("Scenario", list(SCENARIOS), default="Base", required=True, width="stretch")
        st.caption(f"{SCENARIOS[scenario]['equity_return']:.0%} equity · {SCENARIOS[scenario]['income_return']:.0%} income, per year")

    # ---- Model runs --------------------------------------------------------
    proj, stats = project_sip(monthly_amount, tier, scenario, horizon)
    runs = {s: project_sip(monthly_amount, tier, s, horizon) for s in SCENARIOS}
    tier_rets = RETS[tier]
    risk = RISK.loc[tier]
    episode = drawdown_episode(tier_rets)
    tr_episode = drawdown_episode(TR["Total_Return"])
    worst = run_sip(tier_rets[tr_episode["peak"]:], monthly_amount)  # gross of tx costs, as in notebook 03
    history = run_sip(tier_rets, monthly_amount, TX_COST_PCT)  # net of tx costs, as in notebook 04
    milestones = dividend_milestones(proj, horizon)

    # ---- KPI row -----------------------------------------------------------
    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total invested", fmt_pkr(stats["total_invested"]),
              f"{fmt_pkr(monthly_amount)} × {horizon * 12} months", delta_color="off", delta_arrow="off")
    k2.metric("Portfolio value", fmt_pkr(stats["final_value"]), f"{stats['return_pct']:+.0f}% on invested")
    k3.metric("Annual dividend", fmt_pkr(proj["Annual_Dividend"].iloc[-1]),
              f"at year {horizon} · {ANNUAL_DIVIDEND_YIELD:.0%} yield", delta_color="off", delta_arrow="off")
    k4.metric("Max drawdown", f"{risk['Max_Drawdown_%']:.1f}%",
              f"{risk['Annual_Volatility_%']:.1f}% volatility · Sortino {risk['Sortino_Ratio']:.2f}",
              delta_color="off", delta_arrow="off",
              help=f"Deepest peak-to-trough fall of the {tier} tier, {DATA_START:%Y}–{DATA_END:%Y}.")
    st.divider()

    # ---- Tabs --------------------------------------------------------------
    tab_growth, tab_div, tab_risk, tab_worst, tab_outlook, tab_basket = st.tabs(
        ["Growth", "Dividends", "Risk", "Worst case", "Outlook", "Basket"]
    )

    with tab_growth:
        VIEWS = {"Projection": "Projection", "Scenarios": "Compare scenarios", "Backtest": f"{DATA_START:%Y}–{DATA_END:%Y} backtest"}
        head, ctl = st.columns([3, 2], vertical_alignment="bottom")
        with ctl:
            view = st.segmented_control(
                "View", list(VIEWS), default="Projection", required=True,
                label_visibility="collapsed", width="stretch", help=" · ".join(VIEWS.values()),
            )
        with head:
            if view == "Projection":
                st.subheader("Portfolio growth over time")
                st.caption(f"Projected value versus cumulative invested capital · {tier} tier · {scenario} scenario.")
            elif view == "Scenarios":
                st.subheader("Bull, Base and Bear side by side")
                st.caption("Same deposits, three return assumptions. The shaded band is the range of outcomes.")
            else:
                st.subheader("What actually happened")
                st.caption(f"{fmt_pkr(monthly_amount)} a month into the {tier} tier from {DATA_START:%b %Y}, "
                           "net of 0.2% transaction costs. Real KSE 100 total returns, not a projection.")

        if view == "Projection":
            chart(growth_chart(proj, tk, x_col="Label"), "growth")
            c1, c2, c3 = st.columns(3)
            c1.metric("Transaction costs", fmt_pkr(stats["tx_costs"]), f"{TX_COST_PCT:.1%} per purchase",
                      delta_color="off", delta_arrow="off")
            c2.metric("Capital gains tax", fmt_pkr(stats["cgt"]), f"blended rate {stats['blended_cgt_pct']:.2f}%",
                      delta_color="off", delta_arrow="off")
            c3.metric("After-tax value", fmt_pkr(stats["after_tax"]), f"{stats['after_tax_return_pct']:+.0f}% after tax")
            insight(
                f"Over {horizon} years you invest {fmt_pkr(stats['total_invested'])} and the portfolio grows to "
                f"{fmt_pkr(stats['final_value'])}, a {stats['return_pct']:.0f}% return. After transaction costs "
                f"({fmt_pkr(stats['tx_costs'])}) and capital gains tax ({fmt_pkr(stats['cgt'])}) you keep "
                f"{fmt_pkr(stats['after_tax'])}."
            )
        elif view == "Scenarios":
            chart(compare_chart({s: r[0] for s, r in runs.items()}, tk), "compare")
            c1, c2, c3 = st.columns(3)
            for col, s in zip((c1, c2, c3), ("Bull", "Base", "Bear")):
                col.metric(f"{s} · {SCENARIOS[s]['equity_return']:.0%} equity", fmt_pkr(runs[s][1]["final_value"]),
                           f"{runs[s][1]['return_pct']:+.0f}%")
            insight(
                f"The gap between Bear ({fmt_pkr(runs['Bear'][1]['final_value'])}) and Bull "
                f"({fmt_pkr(runs['Bull'][1]['final_value'])}) is the uncertainty you sign up for. Even the Bear case "
                f"returns {runs['Bear'][1]['return_pct']:+.0f}% on invested capital over {horizon} years."
            )
        else:
            h_final, h_inv = history["Portfolio_Value"].iloc[-1], history["Cumulative_Invested"].iloc[-1]
            h_irr = sip_irr(history, monthly_amount * (1 - TX_COST_PCT)) * 100
            hx = history.reset_index()
            chart(growth_chart(hx, tk, x_col="Date", hover_x="%b %Y", show_underwater=True), "history")
            c1, c2, c3 = st.columns(3)
            c1.metric("Invested", fmt_pkr(h_inv), f"{len(history)} months", delta_color="off", delta_arrow="off")
            c2.metric("Final value", fmt_pkr(h_final), f"{(h_final - h_inv) / h_inv * 100:+.0f}% on invested")
            c3.metric("Annualised return (IRR)", f"{h_irr:.1f}%",
                      f"{int(history['Underwater'].sum())} of {len(history)} months underwater",
                      delta_color="off", delta_arrow="off",
                      help="Money-weighted: the constant annual rate at which the monthly deposits would have to grow to reach the final value.")
            insight(
                f"Deposits made every month since {DATA_START:%b %Y} would have totalled {fmt_pkr(h_inv)} and be worth "
                f"{fmt_pkr(h_final)} by {DATA_END:%b %Y}. History is one path — the projection tabs show the range."
            )

        yby = proj.drop(columns="Label").copy()
        yby["Year"] = yby["Year"].astype(int)
        for c in ("Portfolio_Value", "Cumulative_Invested", "Profit", "Annual_Dividend", "Monthly_Dividend", "Transaction_Costs"):
            yby[c] = yby[c].map(fmt_pkr)
        yby.columns = ["Year", "Portfolio value", "Invested", "Profit", "Annual dividend", "Monthly dividend", "Transaction costs"]
        table_view(yby, "Year-by-year projection")

    with tab_div:
        st.subheader("Dividend income at milestones")
        st.caption(f"Annual cash dividends if not reinvested, at a {ANNUAL_DIVIDEND_YIELD:.0%} yield on portfolio value.")
        chart(dividend_chart(milestones, tk), "dividends")
        y10 = milestones[milestones["Year"] == 10]
        if not y10.empty:
            md = y10["Monthly_Dividend"].iloc[0]
            ratio = md / monthly_amount
            insight(
                f"By year 10 the portfolio pays {fmt_pkr(md)} a month in dividends — "
                f"{ratio:.0%} of what you put in each month. "
                + ("Your portfolio has started paying you." if ratio >= 1 else "Compounding does the rest of the work.")
            )
        else:
            last = milestones.iloc[-1]
            insight(f"By year {int(last['Year'])} the portfolio pays {fmt_pkr(last['Monthly_Dividend'])} a month in dividends.")
        mt = milestones.drop(columns="Year").copy()
        for c in ("Portfolio_Value", "Annual_Dividend", "Monthly_Dividend"):
            mt[c] = mt[c].map(fmt_pkr)
        mt.columns = ["Milestone", "Portfolio value", "Annual dividend", "Monthly dividend"]
        table_view(mt)

    with tab_risk:
        st.subheader("Probability of losing money by holding period")
        st.caption(f"Share of rolling windows with a negative total return · {tier} tier · "
                   f"{DATA_START:%b %Y} – {DATA_END:%b %Y}.")
        p = probability_of_loss(tier_rets)
        chart(prob_loss_chart(p, tk), "prob_loss")
        first_zero = p[p["prob"] == 0]["period"].iloc[0] if (p["prob"] == 0).any() else None
        under5 = p[p["prob"] < 5]["period"].iloc[0] if (p["prob"] < 5).any() else None
        if first_zero and first_zero == under5:
            odds = f"No {first_zero} window in the dataset ended in a loss."
        elif under5:
            odds = f"Hold for {under5} and that drops below 5%" + (f"; no {first_zero} window ended in a loss." if first_zero else ".")
        else:
            odds = "The odds fall with every year held."
        insight(
            f"A single month lost money {p['prob'].iloc[0]:.0f}% of the time. {odds} "
            "Time in the market, not timing."
        )
        pt = p.rename(columns={"period": "Holding period", "prob": "Probability of loss (%)", "windows": "Rolling windows"})
        pt["Probability of loss (%)"] = pt["Probability of loss (%)"].round(1)
        table_view(pt.drop(columns="months"))

        st.space("medium")
        st.subheader("Drawdown from peak")
        st.caption(f"{tier} tier, net of fees. How far and for how long the portfolio sat below its previous high.")
        chart(drawdown_chart(episode, tk), "drawdown")
        rec_txt = (f"and did not regain that peak until {episode['recovery']:%B %Y} — "
                   f"{(episode['recovery'].year - episode['peak'].year) * 12 + episode['recovery'].month - episode['peak'].month} months later"
                   if episode["recovery"] is not None else "and had not recovered by the end of the data")
        insight(
            f"The {tier} tier's deepest fall was {episode['depth_pct']:.1f}%, from {episode['peak']:%B %Y} to "
            f"{episode['trough']:%B %Y}, {rec_txt}. This is the discomfort a long-term equity investor signs up for."
        )
        dt = pd.DataFrame({"Month": episode["series"].index.strftime("%b %Y"),
                           "Drawdown (%)": (episode["series"] * 100).round(1).values})
        table_view(dt)

    with tab_worst:
        st.subheader("The worst time to start")
        st.caption(f"A SIP begun at the {tr_episode['peak']:%B %Y} market peak — right before the "
                   f"{abs(tr_episode['depth_pct']):.0f}% fall — in the {tier} tier, held to {DATA_END:%b %Y}.")
        wx = worst.reset_index()
        chart(growth_chart(wx, tk, x_col="Date", hover_x="%b %Y", show_underwater=True), "worst")
        w_final, w_inv = worst["Portfolio_Value"].iloc[-1], worst["Cumulative_Invested"].iloc[-1]
        w_under = int(worst["Underwater"].sum())
        last_under = worst.index[worst["Underwater"]].max() if w_under else None
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Months underwater", w_under, f"of {len(worst)}", delta_color="off", delta_arrow="off")
        c2.metric("Invested", fmt_pkr(w_inv), f"over {len(worst)} months", delta_color="off", delta_arrow="off")
        c3.metric("Final value", fmt_pkr(w_final), "despite the worst start", delta_color="off", delta_arrow="off")
        c4.metric("Return", f"{(w_final - w_inv) / w_inv * 100:+.0f}%", f"{sip_irr(worst, monthly_amount) * 100:.1f}% a year (IRR)",
                  delta_color="off", delta_arrow="off")
        insight(
            f"Starting at the {tr_episode['peak']:%B %Y} peak, the portfolio was below cost for {w_under} of "
            f"{len(worst)} months"
            + (f" (last time in {last_under:%B %Y})" if last_under is not None else "")
            + f" and still finished at {fmt_pkr(w_final)} on {fmt_pkr(w_inv)} invested. "
            "Rupee-cost averaging turns a crash into cheaper units."
        )
        wt = wx[["Date", "Portfolio_Value", "Cumulative_Invested", "Profit", "Underwater"]].copy()
        wt["Date"] = wt["Date"].dt.strftime("%b %Y")
        for c in ("Portfolio_Value", "Cumulative_Invested", "Profit"):
            wt[c] = wt[c].map(fmt_pkr)
        wt.columns = ["Month", "Portfolio value", "Invested", "Profit", "Underwater"]
        table_view(wt)

    with tab_outlook:
        st.subheader("Outlook 2026–2035")
        st.caption("A probabilistic ten-year projection. 5,000 simulated paths, "
                   "blended by scenario weight. Not a point forecast — a distribution.")

        # ── Controls ──────────────────────────────────────────────────────
        oc1, oc2, oc3, oc4 = st.columns([2, 1, 1, 1], gap="medium")

        with oc1:
            preset = st.select_slider(
                "Scenario weights", list(WEIGHT_PRESETS),
                value="Default (25/50/25)",
                help="Bull/Base/Bear probability weights"
            )
            weights = WEIGHT_PRESETS[preset]
            st.caption(f"Bull {weights['Bull']:.0%} · Base {weights['Base']:.0%} · Bear {weights['Bear']:.0%}")

        with oc2:
            view_mode = st.segmented_control(
                "View", ["Nominal", "Real"], default="Nominal", required=True
            )

        with oc3:
            currency = st.segmented_control(
                "Currency", ["PKR", "USD"], default="PKR", required=True
            )

        with oc4:
            method = st.segmented_control(
                "Method", ["Bootstrap", "Regime", "GARCH"],
                default="Bootstrap", required=True
            )

        horizon_outlook = st.slider(
            "Outlook horizon (years)", 5, 20, 10,
            help="Longer horizons have wider uncertainty bands"
        )
        horizon_months = horizon_outlook * 12

        # ── Run Monte Carlo ───────────────────────────────────────────────
        weights_tuple = (weights["Bull"], weights["Base"], weights["Bear"])
        mc_result = run_monte_carlo_cached(
            weights_tuple, tier, monthly_amount, horizon_months,
            method.lower(), seed=42
        )

        # ── Adjust for real/USD ───────────────────────────────────────────
        blended = blend_scenarios(weights)
        inflation = blended["inflation"]
        depreciation = blended["depreciation"]

        def fmt_money(v, decimals=2):
            """Format value with the selected currency label."""
            prefix = "USD" if currency == "USD" else "PKR"
            v = float(v)
            sign = "-" if v < 0 else ""
            a = abs(v)

            if currency == "USD":
                # Western conventions: K for thousands, M for millions
                if a >= 1e6:
                    return f"{sign}{prefix} {_trim(a / 1e6, decimals)}M"
                if a >= 1e3:
                    return f"{sign}{prefix} {_trim(a / 1e3, decimals)}K"
                return f"{sign}{prefix} {a:,.0f}"
            else:
                # South Asian conventions: Lac, Cr
                if a >= 1e7:
                    return f"{sign}{prefix} {_trim(a / 1e7, decimals)} Cr"
                if a >= 1e5:
                    return f"{sign}{prefix} {_trim(a / 1e5, decimals)} Lac"
                return f"{sign}{prefix} {a:,.0f}"

        def adjust_value(val, month):
            """Adjust value for real/USD view."""
            year = month / 12
            adjusted = val
            if view_mode == "Real":
                adjusted = val / (1 + inflation) ** year
            if currency == "USD":
                adjusted = adjusted / PKR_USD
            return adjusted

        # ── KPI Row ───────────────────────────────────────────────────────
        p = mc_result["percentiles"]
        pt = mc_result["probability_table"]
        total_invested = mc_result["total_invested"]

        kpi_p50 = adjust_value(p["p50"][-1], horizon_months - 1)
        kpi_p10 = adjust_value(p["p10"][-1], horizon_months - 1)
        kpi_p90 = adjust_value(p["p90"][-1], horizon_months - 1)
        kpi_invested = adjust_value(total_invested, horizon_months - 1)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Median (P50)", fmt_money(kpi_p50), help="Most likely outcome")
        with k2:
            st.metric("Worst case (P10)", fmt_money(kpi_p10),
                      delta=f"{(kpi_p10/kpi_invested - 1)*100:.0f}% vs deposits",
                      delta_color="inverse")
        with k3:
            st.metric("Best case (P90)", fmt_money(kpi_p90),
                      delta=f"{(kpi_p90/kpi_invested - 1)*100:.0f}% vs deposits")
        with k4:
            loss_color = "inverse" if pt["below_deposits"] > 0.2 else "off"
            st.metric("Chance of loss", f"{pt['below_deposits']:.0%}",
                      help="Probability of finishing below total deposits")

        st.divider()

        # ── Fan Chart ─────────────────────────────────────────────────────
        st.markdown("**Portfolio value distribution over time**")
        st.caption(f"{mc_result['portfolio_paths'].shape[0]:,} simulated paths · "
                   f"{method} method · {view_mode} {currency}")

        months = np.arange(horizon_months)
        deposits = monthly_amount * (months + 1)

        adj_p10 = np.array([adjust_value(p["p10"][m], m) for m in months])
        adj_p25 = np.array([adjust_value(p["p25"][m], m) for m in months])
        adj_p50 = np.array([adjust_value(p["p50"][m], m) for m in months])
        adj_p75 = np.array([adjust_value(p["p75"][m], m) for m in months])
        adj_p90 = np.array([adjust_value(p["p90"][m], m) for m in months])
        adj_dep = np.array([adjust_value(deposits[m], m) for m in months])

        fig_fan = go.Figure()

        fig_fan.add_trace(go.Scatter(
            x=months / 12, y=adj_p90, mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig_fan.add_trace(go.Scatter(
            x=months / 12, y=adj_p10, mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(201, 100, 66, 0.1)",
            name="P10–P90 (80% of outcomes)"
        ))

        fig_fan.add_trace(go.Scatter(
            x=months / 12, y=adj_p75, mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig_fan.add_trace(go.Scatter(
            x=months / 12, y=adj_p25, mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(201, 100, 66, 0.2)",
            name="P25–P75 (50% of outcomes)"
        ))

        fig_fan.add_trace(go.Scatter(
            x=months / 12, y=adj_p50, mode="lines",
            line=dict(color="#c96442", width=2.5),
            name="Median (P50)"
        ))

        fig_fan.add_trace(go.Scatter(
            x=months / 12, y=adj_dep, mode="lines",
            line=dict(color="gray", width=1.5, dash="dash"),
            name="Cumulative deposits"
        ))

        y_title = "Portfolio value" if currency == "PKR" else "Portfolio value (USD)"
        fig_fan.update_layout(
            xaxis_title="Years from start",
            yaxis_title=y_title,
            hovermode="x unified",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_fan, use_container_width=True)

        st.info(
            f"**Key insight:** Over {horizon_outlook} years, the median outcome is "
            f"{fmt_money(kpi_p50)} on {fmt_money(kpi_invested)} invested. But the range "
            f"is wide: there's a 10% chance of finishing below {fmt_money(kpi_p10)} "
            f"and a 10% chance of exceeding {fmt_money(kpi_p90)}. "
            + ("These are real (inflation-adjusted) values — the nominal headline is higher."
               if view_mode == "Real" else
               "Switch to **Real** view to see inflation-adjusted purchasing power.")
        )

        st.divider()

        # ── Terminal Wealth Histogram ─────────────────────────────────────
        st.markdown("**Terminal wealth distribution**")
        st.caption("Final portfolio value across all simulated paths. "
                   "Vertical lines mark P10, P50 and P90.")

        terminal_adjusted = np.array([
            adjust_value(v, horizon_months - 1)
            for v in mc_result["terminal_values"]
        ])

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=terminal_adjusted, nbinsx=40,
            marker_color="#c96442", opacity=0.7,
            name="Terminal wealth"
        ))

        for i, (label, val, color) in enumerate([
            ("P10", kpi_p10, "#b85c4a"),
            ("P50", kpi_p50, "#c96442"),
            ("P90", kpi_p90, "#5a7a4a"),
        ]):
            # alternate top/bottom so labels don't overlap when percentiles are close
            pos = "top" if i % 2 == 0 else "bottom"
            fig_hist.add_vline(
                x=val, line_dash="dash", line_color=color,
                annotation_text=f"{label}: {fmt_money(val)}",
                annotation_position=pos
            )

        fig_hist.update_layout(
            xaxis_title=y_title, yaxis_title="Number of paths",
            height=300, showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()

        # ── Probability Table ─────────────────────────────────────────────
        st.markdown("**Probability table**")
        st.caption("Honest probabilities, not point forecasts. "
                   "These update live with your inputs.")

        prob_data = [
            {"Outcome": "Finish above 1.5× deposits",
             "Probability": f"{pt['above_1.5x_deposits']:.0%}"},
            {"Outcome": "Finish above 2× deposits",
             "Probability": f"{pt['above_2x_deposits']:.0%}"},
            {"Outcome": "Finish above deposits (break even)",
             "Probability": f"{pt['above_deposits']:.0%}"},
            {"Outcome": "Finish below deposits (loss)",
             "Probability": f"{pt['below_deposits']:.0%}"},
            {"Outcome": "30%+ drawdown along the way",
             "Probability": f"{pt['drawdown_30pct']:.0%}"},
            {"Outcome": "50%+ drawdown along the way",
             "Probability": f"{pt['drawdown_50pct']:.0%}"},
        ]
        st.dataframe(pd.DataFrame(prob_data), use_container_width=True, hide_index=True)

        st.info(
            f"**What this means:** A {pt['above_deposits']:.0%} chance of breaking even "
            f"is not a guarantee — it means that across {mc_result['portfolio_paths'].shape[0]:,} "
            f"plausible futures, that fraction ended above deposits. The "
            f"{pt['drawdown_30pct']:.0%} chance of a 30%+ drawdown is the number to "
            f"prepare for psychologically: it doesn't mean you'll lose money, but it "
            f"means you'll see your portfolio fall by a third and have to keep buying."
        )

        # ── Building Block Table (expandable) ─────────────────────────────
        with st.expander("Where do the expected returns come from?"):
            st.caption("Building block identity: expected return = dividend yield + "
                       "real earnings growth + inflation + valuation change")

            bb_data = []
            from kse.blocks import calculate_all_expected_returns
            results = calculate_all_expected_returns()
            for scenario_name in ["Bull", "Base", "Bear"]:
                ret = results[scenario_name]["expected_return"]
                val_change = results[scenario_name]["valuation_change"]
                components = results[scenario_name]["components"]

                if "nominal_earnings_growth" in components:
                    growth_label = f"{components['nominal_earnings_growth']:.1%} (nominal)"
                else:
                    growth_label = f"{components['real_earnings_growth']:.1%} (real)"

                bb_data.append({
                    "Scenario": scenario_name,
                    "Div yield": f"{components['dividend_yield']:.1%}",
                    "Earnings growth": growth_label,
                    "Inflation": f"{components['inflation']:.1%}",
                    "Valuation Δ": f"{val_change:+.1%}",
                    "Expected return": f"{ret:.1%}",
                })

            st.dataframe(pd.DataFrame(bb_data), use_container_width=True, hide_index=True)
            st.caption("Bear case uses nominal earnings growth directly because the "
                       "standard identity (real growth + inflation) breaks down in crisis.")

    with tab_basket:
        st.subheader("Build a basket")
        st.caption("Replace the index with a basket you chose. See what that does to "
                   "return, drawdown and dividend income. Analysis tool, not a "
                   "recommendation engine.")

        try:
            # ── Load data ─────────────────────────────────────────────────
            stock_df = load_stock_metrics()
            all_tickers = stock_df["ticker"].tolist()

            # Get top 8 from screen, with fallback to first 8 in the list
            try:
                top_8 = screen_all()["Ticker"].head(8).tolist()
            except Exception:
                top_8 = all_tickers[:8]

            # ── Stock picker ──────────────────────────────────────────────
            selected = st.multiselect(
                "Select stocks", all_tickers,
                default=top_8,
                help="Pick from the screened KSE 30 universe"
            )

            if len(selected) < 2:
                st.warning("Select at least 2 stocks for a basket.")
            else:
                # ── Weights ───────────────────────────────────────────────
                bc1, bc2 = st.columns([1, 2])

                with bc1:
                    rebalance = st.segmented_control(
                        "Rebalancing", ["None", "Annual", "Quarterly"],
                        default="Annual", required=True
                    )

                    weight_mode = st.segmented_control(
                        "Weights", ["Equal", "Custom"],
                        default="Equal", required=True
                    )

                with bc2:
                    if weight_mode == "Equal":
                        weights_basket = {t: 1.0 / len(selected) for t in selected}
                    else:
                        weights_basket = {}
                        cols = st.columns(min(len(selected), 4))
                        for i, ticker in enumerate(selected):
                            with cols[i % len(cols)]:
                                # Use 0-100 range for the slider so it displays as percentage
                                w = st.slider(
                                    f"{ticker}", 0.0, 100.0,
                                    100.0 / len(selected), 5.0,
                                    format="%.0f%%"
                                )
                                # Convert back to decimal for calculations
                                weights_basket[ticker] = w / 100.0

                        total_w = sum(weights_basket.values())
                        if total_w > 0:
                            weights_basket = {t: w / total_w for t, w in weights_basket.items()}

                # ── Basket summary ─────────────────────────────────────────
                st.divider()
                st.markdown("**Basket composition**")

                basket_data = []
                for ticker in selected:
                    row = stock_df[stock_df["ticker"] == ticker].iloc[0]
                    basket_data.append({
                        "Ticker": ticker,
                        "Name": row["name"],
                        "Sector": row["sector"],
                        "Weight": f"{weights_basket[ticker]:.1%}",
                        "Div yield": f"{row.get('dividend_yield', 0):.1%}" if pd.notna(row.get("dividend_yield")) else "—",
                        "P/E": f"{row.get('pe', 0):.1f}" if pd.notna(row.get("pe")) else "—",
                        "Beta": f"{row.get('beta', 0):.2f}" if pd.notna(row.get("beta")) else "—",
                        "ROE": f"{row.get('roe', 0):.0%}" if pd.notna(row.get("roe")) else "—",
                    })

                st.dataframe(pd.DataFrame(basket_data), use_container_width=True,
                             hide_index=True)

                # ── Basket statistics ──────────────────────────────────────
                # Calculate weighted averages, handling missing values
                valid_stocks = []
                for t in selected:
                    if t in stock_df["ticker"].values:
                        row = stock_df[stock_df["ticker"] == t].iloc[0]
                        if pd.notna(row.get("dividend_yield")) and pd.notna(row.get("pe")) \
                           and pd.notna(row.get("beta")) and pd.notna(row.get("max_drawdown")):
                            valid_stocks.append(t)

                if valid_stocks:
                    avg_div = np.average(
                        [stock_df[stock_df["ticker"] == t]["dividend_yield"].iloc[0] for t in valid_stocks],
                        weights=[weights_basket[t] for t in valid_stocks]
                    )
                    avg_pe = np.average(
                        [stock_df[stock_df["ticker"] == t]["pe"].iloc[0] for t in valid_stocks],
                        weights=[weights_basket[t] for t in valid_stocks]
                    )
                    avg_beta = np.average(
                        [stock_df[stock_df["ticker"] == t]["beta"].iloc[0] for t in valid_stocks],
                        weights=[weights_basket[t] for t in valid_stocks]
                    )
                    avg_dd = np.average(
                        [stock_df[stock_df["ticker"] == t]["max_drawdown"].iloc[0] for t in valid_stocks],
                        weights=[weights_basket[t] for t in valid_stocks]
                    )
                else:
                    avg_div = avg_pe = avg_beta = avg_dd = 0

                sector_weights = {}
                for ticker in selected:
                    row = stock_df[stock_df["ticker"] == ticker].iloc[0]
                    sector = row["sector"]
                    sector_weights[sector] = sector_weights.get(sector, 0) + weights_basket[ticker]

                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1:
                    st.metric("Avg dividend yield", f"{avg_div:.1%}")
                with sc2:
                    st.metric("Avg P/E", f"{avg_pe:.1f}")
                with sc3:
                    st.metric("Avg beta", f"{avg_beta:.2f}")
                with sc4:
                    st.metric("Avg max drawdown", f"{avg_dd:.0%}")

                st.caption("**Sector concentration:** " + " · ".join(
                    f"{s} {w:.0%}" for s, w in sorted(sector_weights.items(),
                                                       key=lambda x: -x[1])
                ))

                # ── Screen scores ──────────────────────────────────────────
                with st.expander("Stock screen scores (5-pillar)"):
                    try:
                        screen_df = screen_all()
                        screen_df = screen_df[screen_df["Ticker"].isin(selected)]
                        st.dataframe(screen_df, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.warning(f"Screen scores unavailable: {e}")

                # ── Comparison: Basket vs Index ─────────────────────────────
                st.divider()
                st.markdown("**Basket vs KSE 100 comparison**")
                st.caption("Illustrative — actual comparison requires running the SIP "
                           "engine on the basket's total return series from PSX data.")

                try:
                    from kse.blocks import expected_return
                    base_ret, _, _ = expected_return("Base")

                    months_proj = horizon * 12
                    monthly_ret = (1 + base_ret) ** (1/12) - 1
                    monthly_fee = ANNUAL_FEE / 12

                    idx_values = []
                    pv_idx = 0
                    for m in range(months_proj):
                        pv_idx = pv_idx * (1 + monthly_ret - monthly_fee) + monthly_amount * (1 - TX_COST_PCT)
                        idx_values.append(pv_idx)

                    basket_ret_monthly = (1 + base_ret * avg_beta) ** (1/12) - 1
                    basket_values = []
                    pv_basket = 0
                    for m in range(months_proj):
                        pv_basket = pv_basket * (1 + basket_ret_monthly - monthly_fee) + monthly_amount * (1 - TX_COST_PCT)
                        basket_values.append(pv_basket)

                    fig_cmp = go.Figure()
                    fig_cmp.add_trace(go.Scatter(
                        x=list(range(1, months_proj + 1)),
                        y=basket_values, mode="lines",
                        name=f"Basket ({avg_beta:.2f} β)",
                        line=dict(color="#c96442", width=2.5)
                    ))
                    fig_cmp.add_trace(go.Scatter(
                        x=list(range(1, months_proj + 1)),
                        y=idx_values, mode="lines",
                        name="KSE 100 (β=1.0)",
                        line=dict(color="gray", width=2, dash="dash")
                    ))

                    fig_cmp.update_layout(
                        xaxis_title="Months", yaxis_title="Portfolio value",
                        height=350, hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)

                    cmp1, cmp2, cmp3, cmp4 = st.columns(4)
                    with cmp1:
                        st.metric("Basket terminal", fmt_pkr(basket_values[-1]))
                    with cmp2:
                        st.metric("Index terminal", fmt_pkr(idx_values[-1]))
                    with cmp3:
                        diff_pct = (basket_values[-1] / idx_values[-1] - 1) * 100
                        st.metric("Difference", f"{diff_pct:+.0f}%",
                                  delta_color="inverse" if diff_pct < 0 else "normal")
                    with cmp4:
                        st.metric("Basket drawdown", f"{avg_dd:.0%}",
                                  delta=f"{avg_dd - 0.33:+.0%} vs index",
                                  delta_color="inverse")

                    st.info(
                        f"**Key insight:** The basket has a {'higher' if avg_div > 0.065 else 'lower'} "
                        f"dividend yield ({avg_div:.1%}) and {'lower' if avg_pe < 7.5 else 'higher'} "
                        f"P/E ({avg_pe:.1f}) than the index, but a "
                        f"{'deeper' if avg_dd > 0.33 else 'shallower'} drawdown ({avg_dd:.0%} vs ~33%). "
                        f"This is the concentration risk tradeoff — you get more income and cheaper "
                        f"valuation, but you take more drawdown risk."
                    )
                except Exception as e:
                    st.warning(f"Comparison chart unavailable: {e}")
                                    # ── Diversification Curve ───────────────────────────────
                st.divider()
                st.markdown("**Diversification benefit**")
                st.caption("How portfolio volatility falls as you add more stocks. "
                           "Most of the benefit comes from the first 8–12 stocks.")

                try:
                    fd = load_frontier_data()
                    if fd is not None:
                        dc = fd["div_curve"]

                        fig_div = go.Figure()

                        # P10-P90 band
                        fig_div.add_trace(go.Scatter(
                            x=dc["n_stocks"], y=dc["p90_volatility"],
                            mode="lines", line=dict(width=0), showlegend=False,
                            hoverinfo="skip"
                        ))
                        fig_div.add_trace(go.Scatter(
                            x=dc["n_stocks"], y=dc["p10_volatility"],
                            mode="lines", line=dict(width=0), fill="tonexty",
                            fillcolor="rgba(201, 100, 66, 0.12)",
                            name="P10–P90 range", hoverinfo="skip"
                        ))

                        # Average volatility
                        fig_div.add_trace(go.Scatter(
                            x=dc["n_stocks"], y=dc["avg_volatility"],
                            mode="lines+markers",
                            line=dict(color="#c96442", width=2.5),
                            marker=dict(size=6),
                            name="Average volatility",
                            hovertemplate="%{x} stocks: %{y:.1%} volatility<extra></extra>"
                        ))

                        fig_div.update_layout(
                            xaxis_title="Number of stocks in portfolio",
                            yaxis_title="Annualized volatility",
                            height=350, hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            margin=dict(l=10, r=10, t=10, b=10),
                        )
                        st.plotly_chart(fig_div, use_container_width=True)

                        st.info(
                            f"**Key insight:** A single stock has an average volatility of "
                            f"{dc['avg_volatility'].iloc[0]:.1%}. With 10 stocks, that drops "
                            f"to {dc[dc['n_stocks']==10]['avg_volatility'].iloc[0]:.1%}. "
                            f"Adding more stocks beyond 12–15 provides diminishing returns — "
                            f"the curve flattens."
                        )
                except Exception as e:
                    st.warning(f"Diversification curve unavailable: {e}")

                # ── Efficient Frontier ───────────────────────────────────
                st.divider()
                st.markdown("**Efficient frontier (Markowitz)**")
                st.caption("The theoretical risk-return tradeoff. Each point on the curve "
                           "is the minimum-variance portfolio for that expected return. "
                           "**Caveat:** based on historical covariance — subject to estimation error.")

                try:
                    if fd is not None:
                        fr = fd["frontier"]
                        ss = fr["stock_stats"]
                        ew = fr["equal_weight"]
                        mv = fr["min_var"]

                        fig_front = go.Figure()

                        # Frontier curve — only the efficient part (above min variance)
                        frontier_df = fr["frontier"]
                        min_var_return = fr["min_var"]["return"]
                        efficient = frontier_df[frontier_df["return"] >= min_var_return].sort_values("volatility")

                        fig_front.add_trace(go.Scatter(
                            x=efficient["volatility"],
                            y=efficient["return"],
                            mode="lines",
                            line=dict(color="#c96442", width=2.5),
                            name="Efficient frontier",
                            hovertemplate="Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>"
                        ))

                        # Individual stocks
                        fig_front.add_trace(go.Scatter(
                            x=ss["volatility"], y=ss["return"],
                            mode="markers",
                            marker=dict(size=8, color="#8a8580", opacity=0.7),
                            name="Individual stocks",
                            text=ss["ticker"],
                            hovertemplate="%{text}<br>Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>"
                        ))

                        # Equal-weight portfolio
                        fig_front.add_trace(go.Scatter(
                            x=[ew["volatility"]], y=[ew["return"]],
                            mode="markers",
                            marker=dict(size=14, color="#5a7a4a", symbol="star",
                                        line=dict(color="white", width=1)),
                            name="Equal-weight (all stocks)",
                            hovertemplate="Equal-weight<br>Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>"
                        ))

                        # Minimum variance portfolio
                        fig_front.add_trace(go.Scatter(
                            x=[mv["volatility"]], y=[mv["return"]],
                            mode="markers",
                            marker=dict(size=12, color="#3f6fb5", symbol="diamond",
                                        line=dict(color="white", width=1)),
                            name="Minimum variance",
                            hovertemplate="Min variance<br>Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>"
                        ))

                        # User's selected basket
                        from kse.frontier import compute_portfolio_stats
                        basket_stats = compute_portfolio_stats(fd["returns"], weights_basket)
                        if basket_stats:
                            fig_front.add_trace(go.Scatter(
                                x=[basket_stats["volatility"]],
                                y=[basket_stats["return"]],
                                mode="markers",
                                marker=dict(size=14, color="#c96442", symbol="circle",
                                            line=dict(color="white", width=2)),
                                name="Your basket",
                                hovertemplate="Your basket<br>Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>"
                            ))

                        fig_front.update_layout(
                            xaxis_title="Annualized volatility (risk)",
                            yaxis_title="Annualized expected return",
                            height=400, hovermode="closest",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            margin=dict(l=10, r=10, t=10, b=10),
                        )
                        st.plotly_chart(fig_front, use_container_width=True)

                        st.warning(
                            "**Methodological caveat:** This frontier is based on historical "
                            "covariance with ~60 monthly observations for 30 stocks. The sample "
                            "covariance matrix is poorly estimated, making the 'optimal' portfolio "
                            "unstable. In practice, equal-weight portfolios often outform "
                            "optimized portfolios out-of-sample. Use this as an educational tool, "
                            "not as investment advice."
                        )
                except Exception as e:
                    st.warning(f"Efficient frontier unavailable: {e}")

        except Exception as e:
            st.error(f"Basket tab error: {e}")
            st.write("This usually means `stock_metrics.csv` is missing or has an issue.")
            st.write("Run: `python scripts/fetch_stocks.py` to regenerate the data.")


st.space("large")
st.caption(
        f"KSE 100 Portfolio Builder · historical data {DATA_START:%b %Y} – {DATA_END:%b %Y} (Investing.com), "
        f"{ANNUAL_DIVIDEND_YIELD:.0%} dividend yield, {ANNUAL_FEE:.1%} fee · {tier} tier · {scenario} scenario · "
        "Not investment advice."
    )
