"""Layout utilities — CSS, chart defaults, section headers."""
import streamlit as st

GLOBAL_CSS = """<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Force light backgrounds everywhere */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    .stApp { background-color: #f8f9fa !important; }

    /* Sidebar - black background, ALL red text */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > div > div,
    [data-testid="stSidebar"] section,
    [data-testid="stSidebar"] section > div { background-color: #000000 !important; }

    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] a,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] button,
    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] label p,
    section[data-testid="stSidebar"] label span,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #cc0000 !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stSidebar"] label:hover,
    [data-testid="stSidebar"] label:hover p,
    [data-testid="stSidebar"] label:hover span { color: #ff3333 !important; }
    [data-testid="stSidebar"] hr { border-color: #cc0000 !important; opacity: 0.3 !important; }
    /* Sidebar brand header — force red with high specificity */
    .sidebar-brand-title,
    [data-testid="stSidebar"] .sidebar-brand-title,
    section[data-testid="stSidebar"] .sidebar-brand-title,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] .sidebar-brand-title { color: #cc0000 !important; }
    .sidebar-brand-sub,
    [data-testid="stSidebar"] .sidebar-brand-sub,
    section[data-testid="stSidebar"] .sidebar-brand-sub,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] .sidebar-brand-sub { color: #cc0000 !important; opacity: 0.6; }
    /* Radio buttons and icons */
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] button svg,
    [data-testid="stSidebar"] svg,
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapseButton"] svg {
        color: #cc0000 !important;
        fill: #cc0000 !important;
    }

    /* Expander fix */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        margin-top: 8px !important;
    }
    [data-testid="stExpander"] summary {
        color: #1a1a2e !important;
        font-weight: 600 !important;
    }
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p {
        color: #1a1a2e !important;
    }

    /* Force dark text everywhere in main content */
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] h5,
    [data-testid="stAppViewContainer"] h6,
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] span,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] div,
    [data-testid="stAppViewContainer"] .stMarkdown,
    [data-testid="stAppViewContainer"] .stCaption,
    [data-testid="stAppViewContainer"] [data-testid="stMetricValue"],
    [data-testid="stAppViewContainer"] [data-testid="stMetricLabel"],
    [data-testid="stAppViewContainer"] .stSelectbox label,
    [data-testid="stAppViewContainer"] .stMultiSelect label,
    [data-testid="stAppViewContainer"] .stRadio label,
    [data-testid="stAppViewContainer"] .stTabs [data-baseweb="tab"] {
        color: #1a1a2e !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Player header on dark background — override main content dark text */
    .player-header-dark .ph-name { color: #cc0000 !important; }
    .player-header-dark .ph-jersey { color: #cc0000 !important; }
    .player-header-dark .ph-detail { color: #ff6666 !important; }
    .player-header-dark .ph-stat { color: #cc9999 !important; }

    /* Dataframes */
    [data-testid="stAppViewContainer"] .stDataFrame,
    [data-testid="stAppViewContainer"] [data-testid="stDataFrame"] * {
        color: #1a1a2e !important;
    }

    /* Make selectboxes/multiselects white instead of grey */
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #d0d0d0 !important;
        color: #1a1a2e !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stMultiSelect"] div[data-baseweb="select"] span {
        color: #1a1a2e !important;
    }

    /* Chart containers */
    [data-testid="stAppViewContainer"] .stPlotlyChart {
        background-color: #ffffff !important;
        border-radius: 8px;
    }
    /* Force ALL plotly text elements dark */
    .stPlotlyChart text,
    .stPlotlyChart .xtick text,
    .stPlotlyChart .ytick text,
    .stPlotlyChart .gtitle,
    .stPlotlyChart .g-xtitle text,
    .stPlotlyChart .g-ytitle text,
    .stPlotlyChart .cbtitle text,
    .stPlotlyChart .cbaxis text,
    .stPlotlyChart .annotation-text {
        fill: #000000 !important;
        color: #000000 !important;
    }

    .block-container { padding-top: 1rem; max-width: 1400px; }

    /* Section headers */
    .section-header {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #1a1a2e !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 2px solid #cc0000;
        padding-bottom: 6px;
        margin-bottom: 12px !important;
        margin-top: 8px !important;
    }
</style>"""

CHART_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(size=11, color="#000000", family="Inter, Arial, sans-serif"),
    margin=dict(l=45, r=10, t=30, b=40),
    xaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
    yaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
    coloraxis=dict(colorbar=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))),
)


def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def force_dark_fonts(fig):
    """Force all axis tick/title fonts and colorbar fonts to black on a Plotly figure."""
    fig.update_layout(
        font=dict(color="#000000"),
        xaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
        yaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
    )
    if hasattr(fig, 'layout') and hasattr(fig.layout, 'coloraxis'):
        fig.update_layout(
            coloraxis=dict(colorbar=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000")))
        )
    return fig
