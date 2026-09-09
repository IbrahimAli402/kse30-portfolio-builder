"""
kse/reporting.py
=================
Report generation for the KSE 100 Portfolio Builder.

Provides:
    - generate_pdf_report()   — branded PDF summary of current analysis
    - export_portfolio_csv()  — basket composition as CSV for brokers
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import io


def export_portfolio_csv(
    tickers: list,
    weights: dict,
    prices: dict,
    monthly_amount: float,
    names: Optional[dict] = None,
    sectors: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build a CSV-ready DataFrame of portfolio composition.

    Parameters
    ----------
    tickers : list
        Selected tickers.
    weights : dict
        Ticker → weight (0-1).
    prices : dict
        Ticker → current price.
    monthly_amount : float
        Total monthly investment in PKR.
    names : dict, optional
        Ticker → company name.
    sectors : dict, optional
        Ticker → sector.

    Returns
    -------
    pd.DataFrame with columns: Ticker, Name, Sector, Weight, PKR_Amount, Shares, Price
    """
    rows = []
    for t in tickers:
        w = weights.get(t, 0)
        if w < 0.001:
            continue
        pkr = w * monthly_amount
        price = prices.get(t, 0)
        shares = int(pkr / price) if price > 0 else 0
        rows.append({
            "Ticker": t,
            "Name": (names or {}).get(t, ""),
            "Sector": (sectors or {}).get(t, ""),
            "Weight": f"{w:.1%}",
            "PKR_Amount": round(pkr, 0),
            "Shares": shares,
            "Price": round(price, 2) if price > 0 else "",
        })
    return pd.DataFrame(rows)


def generate_pdf_report(
    monthly_amount: float,
    tier: str,
    scenario: str,
    horizon: int,
    proj_stats: dict,
    mc_result: Optional[dict] = None,
    goal_data: Optional[dict] = None,
    basket_data: Optional[dict] = None,
) -> bytes:
    """
    Generate a branded PDF report as bytes.

    Uses reportlab if available, otherwise falls back to a simple
    HTML-to-PDF via a text buffer. Returns PDF bytes that can be
    passed to st.download_button.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        )
        from reportlab.lib.enums import TA_CENTER

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                               topMargin=20*mm, bottomMargin=20*mm,
                               leftMargin=15*mm, rightMargin=15*mm)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Title'],
            fontSize=20, spaceAfter=5, alignment=TA_CENTER,
            textColor=colors.HexColor('#c96442')
        )
        subtitle_style = ParagraphStyle(
            'CustomSubtitle', parent=styles['Normal'],
            fontSize=10, spaceAfter=15, alignment=TA_CENTER,
            textColor=colors.HexColor('#6b665e')
        )
        heading_style = ParagraphStyle(
            'CustomHeading', parent=styles['Heading2'],
            fontSize=13, spaceBefore=12, spaceAfter=6,
            textColor=colors.HexColor('#2d2a26')
        )
        body_style = styles['Normal']

        elements = []

        # ── Header ──
        elements.append(Paragraph("KSE 100 Portfolio Builder", title_style))
        elements.append(Paragraph(
            f"Analytical Report · {tier} tier · {scenario} scenario · "
            f"{horizon} years · PKR {monthly_amount:,.0f}/month",
            subtitle_style
        ))
        elements.append(Spacer(1, 5*mm))

        # ── Projection Summary ──
        elements.append(Paragraph("Projection Summary", heading_style))
        proj_data = [
            ["Metric", "Value"],
            ["Monthly investment", f"PKR {monthly_amount:,.0f}"],
            ["Time horizon", f"{horizon} years"],
            ["Risk tier", tier],
            ["Scenario", scenario],
            ["Total invested", f"PKR {proj_stats['total_invested']:,.0f}"],
            ["Portfolio value", f"PKR {proj_stats['final_value']:,.0f}"],
            ["Return on invested", f"{proj_stats['return_pct']:+.0f}%"],
            ["Capital gains tax", f"PKR {proj_stats['cgt']:,.0f}"],
            ["After-tax value", f"PKR {proj_stats['after_tax']:,.0f}"],
            ["After-tax return", f"{proj_stats['after_tax_return_pct']:+.0f}%"],
        ]
        t = Table(proj_data, colWidths=[80*mm, 80*mm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#c96442')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e3dfd7')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f3ed')]),
        ]))
        elements.append(t)

        # ── Monte Carlo Summary ──
        if mc_result is not None:
            elements.append(Paragraph("Monte Carlo Forecast", heading_style))
            p = mc_result["percentiles"]
            mc_data = [
                ["Metric", "Value"],
                ["Median (P50)", f"PKR {p['p50'][-1]:,.0f}"],
                ["Worst case (P10)", f"PKR {p['p10'][-1]:,.0f}"],
                ["Best case (P90)", f"PKR {p['p90'][-1]:,.0f}"],
                ["Probability of loss", f"{mc_result['probability_table']['below_deposits']:.0%}"],
                ["Probability of 2x", f"{mc_result['probability_table']['above_2x_deposits']:.0%}"],
            ]
            t2 = Table(mc_data, colWidths=[80*mm, 80*mm])
            t2.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#c96442')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e3dfd7')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f3ed')]),
            ]))
            elements.append(t2)

        # ── Goal Summary ──
        if goal_data is not None:
            elements.append(Paragraph("Goal Analysis", heading_style))
            goal_tbl = [
                ["Metric", "Value"],
                ["Goal target", f"PKR {goal_data['target']:,.0f}"],
                ["Required monthly SIP", f"PKR {goal_data['required_sip']:,.0f}"],
                ["Probability of success", f"{goal_data['prob_success']:.0%}"],
                ["Cost of 5-year delay", f"PKR {goal_data['delay_cost']:,.0f}/month extra"],
            ]
            t3 = Table(goal_tbl, colWidths=[80*mm, 80*mm])
            t3.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#c96442')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e3dfd7')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f3ed')]),
            ]))
            elements.append(t3)

        # ── Basket Composition ──
        if basket_data is not None and len(basket_data) > 0:
            elements.append(Paragraph("Portfolio Composition", heading_style))
            basket_rows = [["Ticker", "Name", "Weight", "PKR Amount", "Shares"]]
            for _, row in basket_data.iterrows():
                basket_rows.append([
                    row["Ticker"], row.get("Name", ""),
                    row["Weight"], row["PKR Amount"], row["Shares"]
                ])
            t4 = Table(basket_rows, colWidths=[25*mm, 50*mm, 25*mm, 30*mm, 20*mm])
            t4.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#c96442')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e3dfd7')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f3ed')]),
            ]))
            elements.append(t4)

        # ── Disclaimer ──
        elements.append(Spacer(1, 10*mm))
        disclaimer = (
            "<b>Disclaimer:</b> This report is generated by the KSE 100 Portfolio Builder, "
            "an analytical and educational tool. It is not investment advice. Past performance "
            "does not guarantee future results. Consult a licensed financial adviser."
        )
        elements.append(Paragraph(disclaimer, ParagraphStyle(
            'Disclaimer', parent=body_style, fontSize=8,
            textColor=colors.HexColor('#6b665e')
        )))

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()

    except ImportError:
        # Fallback: return empty bytes if reportlab not installed
        return b""


def has_pdf_support() -> bool:
    """Check if reportlab is available for PDF generation."""
    try:
        import reportlab
        return True
    except ImportError:
        return False