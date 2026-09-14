"""KSE 100 Portfolio Builder — design system ("Quiet Ledger").

One place for every visual decision the app makes outside config.toml:

* ``TOKENS``      — the light/dark palettes (mirrored in .streamlit/config.toml)
* ``inject_css``  — dual-theme CSS layer, injected once per run
* primitives      — ``page_header``, ``section``, ``eyebrow``, ``insight``,
                    ``note``, ``band``, ``hairline``, ``footer``

Design lineage (see docs/DESIGN_SYSTEM.md):
    Cosmos  → linen canvas, ink text, one signature radius, flat surfaces,
              eyebrow "stamps", chrome recedes so data leads
    Apple   → rhythm from alternating surfaces (never borders), pill buttons,
              monochrome UI with one accent reserved for interaction
    Dala    → hierarchy from scale not weight, uppercase tracked micro-labels,
              a warm near-black void for dark mode, no cards
    SeatGeek→ dialog craft: one title, one primary action, tinted banners
              with a left rule, labels above inputs

Rule of thumb: if a component needs a border to be understood, the spacing
is wrong. Fix the spacing first.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Tokens — MUST stay in sync with .streamlit/config.toml
# ---------------------------------------------------------------------------
TOKENS: dict[str, dict[str, str]] = {
    "light": dict(
        bg="#faf9f5", surface="#f5f3ed", surface2="#efece4", text="#2d2a26", muted="#6b665e",
        faint="#736d63", border="#e3dfd7", accent="#b0532f", positive="#27775a", negative="#a4412f",
        bear="#3f6fb5", invested="#8a8580",
    ),
    "dark": dict(
        bg="#1c1a17", surface="#252320", surface2="#2d2a26", text="#e8e6e1", muted="#a8a298",
        faint="#928c82", border="#3a3733", accent="#d97757", positive="#4fae88", negative="#d98a72",
        bear="#7a9bd6", invested="#8a8580",
    ),
}

FONT_SANS = "Instrument Sans, Aptos, Segoe UI, sans-serif"
FONT_SERIF = "Newsreader, Georgia, serif"

# One radius family. 16px is the signature (Cosmos); pills for buttons (Apple);
# 10px only for things that live inside a 16px container (chips, thumbnails).
RADIUS = "16px"
RADIUS_INNER = "10px"
RADIUS_PILL = "999px"

# Spacing scale on a 4px base. Section gap is deliberately smaller than the
# marketing-page references (they use 80-120px); a dense analytical tool
# breathes at 40-48px.
SPACE = dict(xs="4px", sm="8px", md="16px", lg="24px", xl="40px", xxl="64px")


def theme_tokens() -> dict[str, str]:
    """Always return dark theme tokens. Single-theme app."""
    return TOKENS["dark"]


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# The CSS layer — split into variables + component rules
# ---------------------------------------------------------------------------
def build_vars(tk: dict[str, str]) -> str:
    """CSS variable declarations for one theme (no selector wrapper)."""
    a = tk["accent"]
    return f"""  --kse-bg: {tk['bg']}; --kse-surface: {tk['surface']}; --kse-surface2: {tk['surface2']};
  --kse-text: {tk['text']}; --kse-muted: {tk['muted']}; --kse-faint: {tk['faint']};
  --kse-border: {tk['border']}; --kse-accent: {a};
  --kse-positive: {tk['positive']}; --kse-negative: {tk['negative']};
  --kse-accent-tint: {rgba(a, 0.10)}; --kse-accent-wash: {rgba(a, 0.06)};
  --kse-radius: {RADIUS}; --kse-radius-inner: {RADIUS_INNER}; --kse-pill: {RADIUS_PILL};
  --kse-sans: {FONT_SANS}; --kse-serif: {FONT_SERIF};"""


def build_component_css() -> str:
    """Theme-agnostic component CSS — every color is a var() reference.

    This string is identical for both themes. Only the variable values
    (from build_vars) change when the theme switches.
    """
    s = SPACE
    return f"""
/* ── Chrome recedes ─────────────────────────────────────────────────── */
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="stAppDeployButton"] {{ display: none; }}
[data-testid="stMainBlockContainer"] {{ padding-top: {s['xl']}; padding-bottom: {s['xxl']}; }}

/* Sidebar shares the page canvas; a hairline is the only separation (Cosmos) */
[data-testid="stSidebar"] {{ background: var(--kse-surface); }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{ font-size: 12px; }}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ font-size: 12px; }}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding-top: {s['sm']}; }}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ padding-top: 0 !important; }}
[data-testid="stSidebar"] hr {{ margin: {s['md']} 0; }}

/* Nav radio styled as premium sidebar nav */
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {{
  gap: 2px;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
  font-family: var(--kse-sans); font-size: 14px; font-weight: 500;
  color: var(--kse-muted); background: transparent;
  margin: 1px 0; padding: 8px 14px; border-radius: 10px;
  cursor: pointer; transition: all 0.2s ease-in-out; border: 0;
  display: flex; align-items: center;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
  background: var(--kse-accent-wash); color: var(--kse-accent);
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
  background: var(--kse-accent-tint); color: var(--kse-accent); font-weight: 600;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label svg {{
  display: none;
}}

/* ── Typography: scale carries hierarchy, weight stays put (Dala) ─────── */
h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}
[data-testid="stHeading"] h1 {{ font-size: 2.25rem; line-height: 1.1; margin-bottom: 0.25rem; color: var(--kse-text); }}
[data-testid="stHeading"] h2 {{ color: var(--kse-text); }}
[data-testid="stHeading"] h3 {{ font-size: 1.375rem; line-height: 1.25; margin-top: 0; padding-top: 0; }}
[data-testid="stMarkdownContainer"] p {{ line-height: 1.6; overflow-wrap: break-word; }}
[data-testid="stCaptionContainer"] p, .kse-lede {{ color: var(--kse-muted); line-height: 1.5; overflow-wrap: break-word; }}

.kse-eyebrow {{
  font-family: var(--kse-sans); font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--kse-muted); margin: 0 0 6px 0;
}}
.kse-eyebrow.accent {{ color: var(--kse-accent); }}
.kse-lede {{ font-size: 15px; line-height: 1.55; max-width: 62ch; margin: 0 0 {s['lg']} 0; }}
.kse-section {{ margin-top: {s['xl']}; }}
.kse-section h3 {{
  font-family: var(--kse-serif); font-weight: 500; font-size: 1.375rem; line-height: 1.25;
  color: var(--kse-text); margin: 0 0 4px 0;
}}
.kse-section .kse-lede {{ margin-bottom: {s['md']}; }}

/* ── Metrics: editorial numerals, quiet delta (no pill) ────────────── */
[data-testid="stMetric"] {{ padding: 0; }}
[data-testid="stMetricLabel"] p {{
  font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--kse-muted);
}}
[data-testid="stMetricValue"] {{
  font-family: var(--kse-serif); font-weight: 500; font-size: 1.75rem; letter-spacing: -0.01em;
  color: var(--kse-text); font-variant-numeric: tabular-nums;
}}
[data-testid="stMetricDelta"] {{
  background: transparent !important; padding: 0 !important; font-size: 13px; color: var(--kse-muted);
}}
[data-testid="stMetricDelta"] p {{ font-size: 13px; }}

/* ── Insight: eyebrow + sentence on a surface band; never a bordered card ── */
.kse-insight {{
  background: var(--kse-surface); border-radius: var(--kse-radius);
  padding: {s['md']} {s['lg']}; margin: {s['md']} 0 {s['lg']} 0;
  overflow: visible; word-break: break-word; overflow-wrap: break-word;
}}
.kse-insight p {{ margin: 0; font-size: 15px; line-height: 1.6; color: var(--kse-text); overflow-wrap: break-word; }}
.kse-insight p.kse-eyebrow {{ margin: 0 0 4px 0; font-size: 11px; line-height: 1.4; color: var(--kse-muted); }}

/* ── Banners (st.info/warning/success/error): SeatGeek tinted rule ──── */
[data-testid="stAlertContainer"] {{
  border-radius: var(--kse-radius); border: 0; padding: 20px 16px 20px 20px;
  background: var(--kse-surface); position: relative;
  overflow: visible; word-break: break-word; overflow-wrap: break-word;
  width: 100%; min-height: auto !important; height: auto !important;
  display: flex; align-items: center; justify-content: flex-start; min-height: 56px;
}}
[data-testid="stAlertContainer"]::before {{
  content: ""; position: absolute; left: 0; top: 10px; bottom: 10px; width: 3px;
  border-radius: 3px; background: var(--kse-accent);
}}
[data-testid="stAlertContainer"] p {{ font-size: 14px; line-height: 1.6; color: var(--kse-text); margin: 0; overflow-wrap: break-word; text-align: left; width: 100%; }}
[data-testid="stAlertContainer"] [data-testid="stIconMaterial"],
[data-testid="stAlertContainer"] [data-testid="stAlertDynamicIcon"] {{ display: none; }}

/* ── band(): bordered containers become surface bands ─────────────── */
[class*="st-key-band-"] {{
  border: 0 !important; background: var(--kse-surface); border-radius: var(--kse-radius);
  padding: {s['lg']} {s['lg']} {s['md']} {s['lg']};
}}
[class*="st-key-band-"] [data-testid="stMetric"] {{ background: transparent; }}

/* ── Expanders: hairline, no box ──────────────────────────────────── */
[data-testid="stExpander"] details {{
  border: 0; border-top: 1px solid var(--kse-border); border-radius: 0; background: transparent;
}}
[data-testid="stExpander"] summary {{ padding-left: 0; padding-right: 0; }}
[data-testid="stExpander"] summary p {{ font-size: 14px; font-weight: 500; color: var(--kse-text); }}
[data-testid="stExpander"] summary:hover {{ color: var(--kse-accent); }}
[data-testid="stExpanderDetails"] {{ padding-left: 0; padding-right: 0; }}

/* ── Dataframes: hairline border, surface header ──────────────────── */
[data-testid="stDataFrame"] {{ border-radius: var(--kse-radius-inner); overflow: hidden; }}

/* ── Buttons: pill primary, ghost secondary, identical size (Apple) ── */
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-secondary"],
[data-testid="stDownloadButton"] button {{
  border-radius: var(--kse-pill); min-height: 40px; padding: 0 20px; font-weight: 500;
}}
[data-testid="stBaseButton-secondary"] {{ border: 1px solid var(--kse-border); background: transparent; }}
[data-testid="stBaseButton-secondary"]:hover {{ border-color: var(--kse-accent); color: var(--kse-accent); background: var(--kse-accent-wash); }}

/* ── Inputs: labels above, quiet chrome ───────────────────────────── */
[data-testid="stWidgetLabel"] p {{ font-size: 13px; font-weight: 500; color: var(--kse-text); }}
[data-testid="stSliderThumbValue"] {{ color: var(--kse-accent); font-weight: 500; }}
[data-testid="stTextInput"] input {{ border-radius: var(--kse-radius-inner); }}

/* ── Search pill in the sidebar (Cosmos nav-pill search) ─────────────── */
.kse-search-hint {{
  font-size: 12px; color: var(--kse-faint); margin-top: -8px; margin-bottom: {s['md']};
}}
.kse-kbd {{
  font-family: var(--kse-sans); font-size: 11px; color: var(--kse-muted);
  border: 1px solid var(--kse-border); border-radius: 6px; padding: 1px 6px; margin-left: 4px;
}}

/* ── Dialog: SeatGeek clarity ─────────────────────────────────────── */
[data-testid="stDialog"] section {{ border-radius: var(--kse-radius); border: 1px solid var(--kse-border); }}
[data-testid="stDialog"] [data-testid="stHeading"] h2 {{ font-family: var(--kse-serif); font-weight: 500; font-size: 1.5rem; }}

/* ── Hairline & footer ─────────────────────────────────────────────── */
hr {{ border-color: var(--kse-border); opacity: 1; margin: {s['md']} 0 0 0; }}
.kse-hairline {{ border-top: 1px solid var(--kse-border); margin: {s['xl']} 0 {s['lg']} 0; }}
.kse-footer {{ color: var(--kse-faint); font-size: 12px; line-height: 1.6; margin-top: {s['xxl']}; }}
.kse-footer a {{ color: var(--kse-muted); }}

/* ── Chips (glossary suggestions) ─────────────────────────────────── */
.kse-chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 {s['md']} 0; }}
.kse-chip {{
  font-size: 12px; color: var(--kse-muted); border: 1px solid var(--kse-border);
  border-radius: var(--kse-pill); padding: 3px 10px; background: transparent;
}}
.kse-answer {{
  background: var(--kse-surface); border-radius: var(--kse-radius); padding: {s['md']} {s['lg']};
}}
.kse-answer h4 {{ font-family: var(--kse-serif); font-weight: 500; font-size: 1.125rem; margin: 0 0 6px 0; }}
.kse-answer p {{ margin: 0 0 8px 0; font-size: 15px; line-height: 1.55; }}
.kse-answer p.kse-eyebrow {{ font-size: 11px; color: var(--kse-muted); margin-bottom: 4px; }}
.kse-answer .kse-related {{ font-size: 12px; color: var(--kse-muted); margin-top: 8px; }}

/* Plotly charts sit flush; the section header does the labelling */
[data-testid="stPlotlyChart"] {{ margin-top: 4px; }}

/* Mobile */
@media (max-width: 760px) {{
  [data-testid="stHeading"] h1 {{ font-size: 1.75rem; }}
  [data-testid="stMetricValue"] {{ font-size: 1.375rem; }}
  .kse-band, .kse-insight {{ padding: {s['md']}; }}
}}
"""


def build_css(tk: dict[str, str]) -> str:
    """Return the full stylesheet for one theme. Backward-compat wrapper."""
    return f"<style>:root {{\n{build_vars(tk)}\n}}\n{build_component_css()}\n</style>"

# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------
def inject_css(tk: dict[str, str] | None = None) -> None:
    tk = tk or theme_tokens()
    st.markdown(build_css(tk), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def eyebrow(text: str, *, accent: bool = False) -> None:
    cls = "kse-eyebrow accent" if accent else "kse-eyebrow"
    st.markdown(f'<p class="{cls}">{text}</p>', unsafe_allow_html=True)


def page_header(app_name: str, title: str, lede: str | None = None) -> None:
    """App wordmark as eyebrow, tab name as the H1, one-sentence lede."""
    eyebrow(app_name)
    st.title(title)
    if lede:
        st.markdown(f'<p class="kse-lede">{lede}</p>', unsafe_allow_html=True)


def section(title: str, lede: str | None = None, label: str | None = None) -> None:
    """Section header: optional eyebrow, serif title, muted lede. Replaces st.subheader + st.caption pairs."""
    parts = ['<div class="kse-section">']
    if label:
        parts.append(f'<p class="kse-eyebrow">{label}</p>')
    parts.append(f"<h3>{title}</h3>")
    if lede:
        parts.append(f'<p class="kse-lede">{lede}</p>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def insight(text: str, label: str = "Key insight") -> None:
    """The one sentence the reader should leave with. Surface band, no border."""
    st.markdown(
        f'<div class="kse-insight"><p class="kse-eyebrow">{label}</p><p>{text}</p></div>',
        unsafe_allow_html=True,
    )


def note(text: str, kind: str = "info") -> None:
    """Thin wrapper so call sites read as intent, not widget. kind: info|warning|success|error."""
    {"info": st.info, "warning": st.warning, "success": st.success, "error": st.error}[kind](text)


def hairline() -> None:
    st.markdown('<div class="kse-hairline"></div>', unsafe_allow_html=True)


def band(key: str):
    """A surface band that can hold widgets: ``with band("kpi"): st.metric(...)``.

    Streamlit's bordered container is restyled by the CSS layer into a flat
    surface block (no border, one radius). Use it for the KPI strip and any
    group that needs to read as one unit without drawing a box around it.
    ``key`` must be unique per page; it becomes the CSS hook ``.st-key-band-<key>``.
    """
    return st.container(border=True, key=f"band-{key}")


def footer(lines: list[str]) -> None:
    body = "<br>".join(lines)
    st.markdown(f'<div class="kse-footer">{body}</div>', unsafe_allow_html=True)


def kbd_hint(text: str, keys: str) -> None:
    st.markdown(
        f'<p class="kse-search-hint">{text}<span class="kse-kbd">{keys}</span></p>',
        unsafe_allow_html=True,
    )