"""
kse/events.py
==============
Market event timeline overlay.

Loads historical political, IMF, and monetary events from config
and provides a function to add them as vertical lines on any Plotly chart.
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import Optional


_CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_events(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load market events from YAML config.

    Returns
    -------
    pd.DataFrame with columns: date, label, category, description
    """
    if path is None:
        path = _CONFIG_DIR / "market_events.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    events = data.get("events", [])
    df = pd.DataFrame(events)
    if len(df) > 0:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def add_event_lines(fig, events_df: pd.DataFrame, tk: dict, max_events: int = 15):
    """
    Add vertical lines and annotations for market events to a Plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to add events to.
    events_df : pd.DataFrame
        Events from load_events().
    tk : dict
        Theme tokens.
    max_events : int
        Maximum number of events to show (most recent first).

    Returns
    -------
    fig (modified in place)
    """
    if len(events_df) == 0:
        return fig

    if len(events_df) > max_events:
        events_df = events_df.tail(max_events)

    for i, (_, event) in enumerate(events_df.iterrows()):
        # Alternate y position to reduce label overlap
        y_pos = 0.95 if i % 2 == 0 else 0.05
        y_anchor = "top" if i % 2 == 0 else "bottom"

        fig.add_vline(
            x=event["date"],
            line_dash="dot",
            line_color=tk["muted"],
            opacity=0.4,
            line_width=1,
        )
        fig.add_annotation(
            x=event["date"],
            y=y_pos,
            yref="paper",
            text=event["label"],
            showarrow=False,
            font=dict(size=9, color=tk["muted"]),
            textangle=-90,
            yanchor=y_anchor,
        )

    return fig