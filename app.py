import os

import streamlit as st
import plotly.io as pio

from config import _APP_DIR, ROSTER_2026
from data.loader import load_davidson_data, get_sidebar_stats
from viz.layout import GLOBAL_CSS


st.set_page_config(page_title="Davidson Baseball Analytics", layout="wide")
pio.templates.default = "plotly_white"

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def _sidebar_brand():
    logo_real = os.path.join(_APP_DIR, "logo_real.png")
    logo_fallback = os.path.join(_APP_DIR, "logo.png")
    logo = logo_real if os.path.exists(logo_real) else logo_fallback
    if os.path.exists(logo):
        _lcol1, _lcol2, _lcol3 = st.sidebar.columns([1, 2, 1])
        with _lcol2:
            st.image(logo, use_container_width=True)
    st.sidebar.markdown(
        '<div style="text-align:center;padding:2px 0 5px 0;">'
        '<span class="sidebar-brand-title" style="display:block;font-size:20px;'
        'font-weight:800;font-family:Inter,sans-serif;letter-spacing:1px;">W.I.L.D.C.A.T.S.</span>'
        '<span class="sidebar-brand-sub" style="display:block;font-size:10px;letter-spacing:1px;'
        'text-transform:uppercase;font-family:Inter,sans-serif;">Davidson Baseball Analytics</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")


def main():
    _sidebar_brand()

    _nav = [
        "Team Overview",
        "Hitting",
        "Pitching",
        "Catcher Analytics",
        "Player Development",
        "Defensive Positioning",
        "Opponent Scouting",
        "Postgame Report",
        "Series Report",
        "Data Quality",
    ]

    page = st.sidebar.radio("Navigation", _nav, label_visibility="collapsed")

    data = load_davidson_data()
    if data is None or data.empty:
        st.error("No data loaded.")
        return

    st.sidebar.markdown("---")
    _sb = get_sidebar_stats()
    st.sidebar.markdown(
        f'<div style="font-size:12px;color:#888 !important;padding:0 10px;">'
        f'<b style="color:#cc0000 !important;">{_sb["total_pitches"]:,}</b> pitches<br>'
        f'<b style="color:#cc0000 !important;">{_sb["n_seasons"]}</b> seasons '
        f'({_sb["min_season"]}-{_sb["max_season"]})<br>'
        f'<b style="color:#cc0000 !important;">{_sb["n_dav_games"]}</b> Davidson games<br>'
        f'<b style="color:#cc0000 !important;">{len(ROSTER_2026)}</b> rostered players<br>'
        f'<b style="color:#cc0000 !important;">{_sb["n_pitchers"]:,}</b> pitchers in DB<br>'
        f'<b style="color:#cc0000 !important;">{_sb["n_batters"]:,}</b> hitters in DB'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Lazy imports — only load the selected page module to speed up startup
    if page == "Team Overview":
        from _pages.team import page_team
        page_team(data)
    elif page == "Hitting":
        from _pages.hitting import page_hitting
        page_hitting(data)
    elif page == "Pitching":
        from _pages.pitching import page_pitching
        page_pitching(data)
    elif page == "Catcher Analytics":
        from _pages.catcher import page_catcher
        page_catcher(data)
    elif page == "Player Development":
        from _pages.development import page_development
        page_development(data)
    elif page == "Defensive Positioning":
        from _pages.defense import page_defensive_positioning
        page_defensive_positioning(data)
    elif page == "Opponent Scouting":
        from _pages.scouting import page_scouting
        page_scouting(data)
    elif page == "Bryant Scouting":
        from _pages.bryant import page_bryant
        page_bryant(data)
    elif page == "In-Game Decision Engine":
        try:
            from _pages.decision_engine import page_decision_engine
            page_decision_engine(data)
        except ImportError:
            st.error("Decision Engine not available.")
    elif page == "Postgame Report":
        from _pages.postgame import page_postgame
        page_postgame(data)
    elif page == "Series Report":
        from _pages.series import page_series
        page_series(data)
    elif page == "Data Quality":
        from _pages.data_quality import page_data_quality
        page_data_quality()


if __name__ == "__main__":
    main()
