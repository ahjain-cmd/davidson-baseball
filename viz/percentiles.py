"""Savant-style percentile bar rendering."""
import streamlit as st
import pandas as pd
import numpy as np


def savant_color(pct, higher_is_better=True):
    """Statcast-style gradient: blue(poor) -> gray(avg) -> red(great)"""
    if pd.isna(pct):
        return "#aaa"
    p = pct if higher_is_better else (100 - pct)
    if p >= 95: return "#be0000"
    if p >= 90: return "#d22d49"
    if p >= 80: return "#e65730"
    if p >= 70: return "#ee7e1e"
    if p >= 60: return "#d4a017"
    if p >= 40: return "#9e9e9e"
    if p >= 30: return "#6a9bc3"
    if p >= 20: return "#3d7dab"
    if p >= 10: return "#1f5f8b"
    return "#14365d"


def _pctile_text_color(bg_color):
    """Return white or dark text depending on background brightness."""
    # Dark backgrounds need white text; light/mid backgrounds need dark text
    dark_bgs = {"#14365d", "#1f5f8b", "#3d7dab", "#be0000", "#d22d49", "#e65730"}
    return "#ffffff" if bg_color in dark_bgs else "#1a1a2e"


def render_savant_percentile_section(metrics_data, title=None, legend=None):
    """Render Baseball Savant style percentile ranking section.
    metrics_data: list of (label, value, percentile, fmt, higher_is_better)
    legend: optional tuple of (left_label, center_label, right_label) to override POOR/AVERAGE/GREAT
    """
    if title:
        st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

    # Legend
    l_left, l_center, l_right = legend or ("POOR", "AVERAGE", "GREAT")
    st.markdown(
        '<div style="display:flex;justify-content:space-between;margin-bottom:8px;padding:0 4px;">'
        f'<span style="font-size:10px;font-weight:700;color:#14365d !important;letter-spacing:0.5px;">{l_left}</span>'
        f'<span style="font-size:10px;font-weight:700;color:#9e9e9e !important;letter-spacing:0.5px;">{l_center}</span>'
        f'<span style="font-size:10px;font-weight:700;color:#be0000 !important;letter-spacing:0.5px;">{l_right}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    for label, val, pct, fmt, hib in metrics_data:
        color = savant_color(pct, hib)
        txt_color = _pctile_text_color(color)
        # For "lower is better" metrics, invert position & displayed number
        effective_pct = pct if hib else (100 - pct) if not pd.isna(pct) else pct
        display_pct = int(round(effective_pct)) if not pd.isna(pct) else "-"
        display_val = f"{val:{fmt}}" if not pd.isna(val) else "-"
        bar_left = max(min(effective_pct, 100), 0) if not pd.isna(pct) else 50

        st.markdown(
            f'<div style="display:flex;align-items:center;margin:4px 0;height:30px;background:white;'
            f'border-radius:4px;padding:0 8px;">'
            # Label
            f'<div style="min-width:110px;font-size:11px;font-weight:600;color:#1a1a2e !important;'
            f'white-space:nowrap;text-transform:uppercase;letter-spacing:0.3px;">{label}</div>'
            # Percentile circle on gradient bar
            f'<div style="flex:1;position:relative;height:6px;'
            f'background:linear-gradient(to right, #14365d 0%, #3d7dab 25%, #9e9e9e 50%, #ee7e1e 75%, #be0000 100%);'
            f'border-radius:3px;margin:0 12px;">'
            f'<div style="position:absolute;left:{bar_left}%;top:50%;transform:translate(-50%,-50%);'
            f'width:28px;height:28px;border-radius:50%;background:{color};border:3px solid white;'
            f'box-shadow:0 1px 4px rgba(0,0,0,0.3);display:flex;align-items:center;justify-content:center;'
            f'font-size:10px;font-weight:800;color:{txt_color} !important;'
            f'text-shadow:0 0 2px rgba(0,0,0,0.3);">{display_pct}</div>'
            f'</div>'
            # Value
            f'<div style="min-width:50px;text-align:right;font-size:12px;font-weight:700;color:#1a1a2e !important;">'
            f'{display_val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
