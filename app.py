"""
app.py – BioWatch: Lantana camara Habitat Suitability Dashboard

Streamlit application providing:
  A. Home (species profile and key statistics)
  B. Zone Map of India (Folium) – suitability zones and hotspot zones
  C. Model Comparison (ROC curves, metrics, confusion matrices)
  D. Feature Importance
  E. Monitoring Priority

Run:
    streamlit run app.py
"""

import base64
import json
import logging
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from branca.element import MacroElement, Template
from folium.plugins import Fullscreen, HeatMap
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from src.zones import (
    build_class_zones,
    build_hotspot_zones,
    india_mask_geojson,
    load_india_boundary,
)

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent
OUTPUTS      = ROOT / "outputs"
MODELS       = ROOT / "models"
ASSETS       = ROOT / "assets"
DATA_FINAL   = ROOT / "data" / "final" / "invasive_species_lantana_india.csv"
DATA_PRES    = ROOT / "data" / "processed" / "presence_points.csv"
SUITABILITY  = OUTPUTS / "suitability_grid.csv"
METRICS_JSON = OUTPUTS / "model_metrics.json"
STATS_JSON   = OUTPUTS / "suitability_stats.json"


LEAF_DARK = "#1f3d2b"
LEAF      = "#2f6b3f"
CREAM     = "#fffaf2"
SAND      = "#f6efe2"
YELLOW    = "#f9c22e"
ORANGE    = "#f28c28"
PINK      = "#e2558f"
MAGENTA   = "#a4134a"
MUTED     = "#6b7b6e"


CLASS_COLORS = {
    "Very Low":  "#b9d3ae",
    "Low":       "#f9d65c",
    "Moderate":  "#f5a030",
    "High":      "#e2558f",
    "Very High": "#a4134a",
    "No Data":   "#d9d4c7",
}

CLASS_ORDER = ["Very Low", "Low", "Moderate", "High", "Very High"]
MODEL_COLORS = [PINK, LEAF, ORANGE]

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioWatch by maansi| Lantana camara in India",
    page_icon=str(ASSETS / "lantana.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{
    background:
        radial-gradient(circle at 100% 0%, rgba(249,194,46,.12), transparent 35%),
        radial-gradient(circle at 0% 100%, rgba(226,85,143,.08), transparent 40%),
        {CREAM};
    color: {LEAF_DARK};
}}

/* ── Sidebar: deep leaf green ─────────────────────── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {LEAF_DARK} 0%, #234a33 100%);
    border-right: 4px solid;
    border-image: linear-gradient(180deg, {YELLOW}, {ORANGE}, {PINK}, {MAGENTA}) 1;
}}
[data-testid="stSidebar"] * {{ color: #f3efe4 !important; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    padding: 8px 12px; border-radius: 10px; margin-bottom: 4px;
    transition: background .15s;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: rgba(255,255,255,.08); }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
    background: rgba(249,194,46,.18);
    box-shadow: inset 3px 0 0 {YELLOW};
}}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.15); }}

.brand {{ display:flex; align-items:center; gap:10px; margin: 4px 0 2px; }}
.brand img {{ width: 44px; height: 44px; }}
.brand-name {{ font-family:'Fraunces', serif; font-size: 1.7rem; font-weight: 700; letter-spacing: .3px; }}
.brand-name span {{ color: {YELLOW} !important; }}
.brand-tag {{ font-size: .78rem; opacity: .75; margin-bottom: 6px; }}

/* ── Headings ─────────────────────────────────────── */
h1 {{ font-family:'Fraunces', serif; color: {LEAF_DARK} !important; font-weight: 700; }}
h2 {{
    font-family:'Fraunces', serif; color: {LEAF_DARK} !important; font-weight: 600;
    font-size: 1.35rem !important; padding-bottom: 6px;
    border-bottom: 2px solid; border-image: linear-gradient(90deg, {YELLOW}, {ORANGE}, {PINK}, transparent) 1;
}}
h3 {{ color: {LEAF_DARK} !important; font-weight: 600; }}

/* ── Metric cards ─────────────────────────────────── */
[data-testid="stMetric"] {{
    background: #fff;
    border: 1px solid #eadfca;
    border-top: 4px solid {PINK};
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 2px 10px rgba(31,61,43,.06);
}}
[data-testid="column"]:nth-child(2) [data-testid="stMetric"] {{ border-top-color: {ORANGE}; }}
[data-testid="column"]:nth-child(3) [data-testid="stMetric"] {{ border-top-color: {YELLOW}; }}
[data-testid="column"]:nth-child(4) [data-testid="stMetric"] {{ border-top-color: {LEAF}; }}
[data-testid="column"]:nth-child(5) [data-testid="stMetric"] {{ border-top-color: {MAGENTA}; }}
[data-testid="stMetricLabel"] {{ color: {MUTED} !important; font-size: .78rem; }}
[data-testid="stMetricValue"] {{ color: {LEAF_DARK} !important; font-size: 1.55rem; font-weight: 700; }}

/* ── Hero ─────────────────────────────────────────── */
.hero {{
    display: grid; grid-template-columns: 1.35fr 1fr; gap: 28px; align-items: center;
    background: linear-gradient(120deg, {LEAF_DARK} 0%, {LEAF} 100%);
    border-radius: 22px; padding: 34px 38px; color: #fdf8ee; overflow: hidden;
    position: relative; margin-bottom: 18px;
}}
.hero::after {{
    content:""; position:absolute; right:-80px; top:-80px; width:280px; height:280px; border-radius:50%;
    background: radial-gradient(circle, rgba(249,194,46,.35), transparent 70%);
}}
.hero h1 {{ color: #fff !important; font-size: 2.6rem; margin: 0 0 6px; line-height: 1.1; }}
.hero .kicker {{ text-transform: uppercase; letter-spacing: 2px; font-size: .75rem; color: {YELLOW}; font-weight: 700; }}
.hero p {{ color: #e8efe5; font-size: 1rem; line-height: 1.6; max-width: 560px; }}
.hero .sci {{ font-style: italic; color: {YELLOW}; }}
.hero-img {{
    width: 100%; aspect-ratio: 4/3; border-radius: 18px;
    background-color: #f3e5c4; background-size: cover; background-position: center 55%;
    box-shadow: 0 10px 30px rgba(0,0,0,.25); position: relative; z-index: 1;
}}
.hero-img.illustration {{ background-size: contain; background-repeat: no-repeat; background-origin: content-box; padding: 16px; }}
.chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-top: 14px; }}
.chip {{ padding: 4px 12px; border-radius: 999px; font-size: .78rem; font-weight: 600; }}
.chip.y {{ background:{YELLOW}; color:{LEAF_DARK}; }}
.chip.o {{ background:{ORANGE}; color:#fff; }}
.chip.p {{ background:{PINK}; color:#fff; }}
.chip.m {{ background:{MAGENTA}; color:#fff; }}
.photo-credit {{ font-size: .7rem; color: #cfdccb; margin-top: 6px; text-align: right; position: relative; z-index: 1; }}

/* ── Cards ────────────────────────────────────────── */
.card {{
    background: #fff; border: 1px solid #eadfca; border-radius: 16px;
    padding: 18px 20px; box-shadow: 0 2px 10px rgba(31,61,43,.05); height: 100%;
}}
.card h4 {{ margin: 0 0 6px; color: {LEAF_DARK}; font-size: 1rem; }}
.card p {{ margin: 0; color: #44564a; font-size: .9rem; line-height: 1.55; }}
.card .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; }}

/* ── Legend ───────────────────────────────────────── */
.legend-box {{
    background: #fff; border: 1px solid #eadfca; border-radius: 14px;
    padding: 14px 16px; font-size: .86rem; margin-bottom: 12px;
}}
.legend-swatch {{
    display: inline-block; width: 16px; height: 16px; border-radius: 4px;
    margin-right: 8px; vertical-align: middle; border: 1px solid rgba(0,0,0,.08);
}}

/* ── Zone detail ──────────────────────────────────── */
.zone-card {{
    background: linear-gradient(135deg, #fff 0%, #fff4f8 100%);
    border: 1px solid #f1c6d8; border-left: 5px solid {MAGENTA};
    border-radius: 14px; padding: 14px 16px; font-size: .88rem; line-height: 1.7;
}}
.zone-card b.title {{ font-family:'Fraunces', serif; font-size: 1.15rem; color: {MAGENTA}; }}

/* ── Tabs / widgets ───────────────────────────────── */
[data-testid="stTabs"] button[aria-selected="true"] {{ color: {MAGENTA}; }}
.stAlert {{ border-radius: 12px; }}
[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}
hr {{ border-color: #eadfca; }}

/* ── Status pill ──────────────────────────────────── */
.status-pill {{ display:inline-block; padding:2px 10px; border-radius:12px; font-size:.78rem; font-weight:600; }}
.status-ok   {{ background: #dcefd8; color: {LEAF}; }}
.status-warn {{ background: #fde9cc; color: #a35a00; }}

.footer {{ text-align:center; color:{MUTED}; font-size:.78rem; margin-top: 30px; }}

@media (max-width: 900px) {{
    .hero {{ grid-template-columns: 1fr; padding: 24px; }}
    .hero h1 {{ font-size: 2rem; }}
}}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: images and chart styling
# ──────────────────────────────────────────────────────────────────────────────

def lantana_image():
    """Return (data URI, is_photo). A photo in assets/ wins over the illustration."""
    for name in ("lantana.jpg", "lantana.jpeg", "lantana.png", "lantana.webp"):
        p = ASSETS / name
        if p.exists():
            mime = "jpeg" if p.suffix in (".jpg", ".jpeg") else p.suffix[1:]
            return f"data:image/{mime};base64,{base64.b64encode(p.read_bytes()).decode()}", True
    svg = (ASSETS / "lantana.svg").read_bytes()
    return f"data:image/svg+xml;base64,{base64.b64encode(svg).decode()}", False


def style_fig(fig, height=360):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color=LEAF_DARK),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(gridcolor="#f0e8d8", zerolinecolor="#eadfca")
    fig.update_yaxes(gridcolor="#f0e8d8", zerolinecolor="#eadfca")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Data loaders (cached)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_occurrence_data():
    if DATA_FINAL.exists():
        return pd.read_csv(DATA_FINAL)
    return None

@st.cache_data(show_spinner=False)
def load_presence_data():
    if DATA_PRES.exists():
        return pd.read_csv(DATA_PRES)
    return None

@st.cache_data(show_spinner=False)
def load_suitability_grid():
    if SUITABILITY.exists():
        return pd.read_csv(SUITABILITY)
    return None

@st.cache_data(show_spinner=False)
def load_metrics():
    if METRICS_JSON.exists():
        with open(METRICS_JSON) as f:
            return json.load(f)
    return None

@st.cache_data(show_spinner=False)
def load_stats():
    if STATS_JSON.exists():
        with open(STATS_JSON) as f:
            return json.load(f)
    return None

@st.cache_resource(show_spinner=False)
def load_india():
    return load_india_boundary()

@st.cache_data(show_spinner="Building suitability zones…")
def load_class_zones(_df_suit):
    return build_class_zones(_df_suit, load_india())

@st.cache_data(show_spinner="Finding hotspot zones…")
def load_hotspot_zones(_df_suit, _df_pres, threshold):
    return build_hotspot_zones(_df_suit, load_india(), _df_pres, threshold=threshold)


# ──────────────────────────────────────────────────────────────────────────────
# Map building blocks
# ──────────────────────────────────────────────────────────────────────────────

INDIA_BOUNDS = [[6.0, 66.5], [37.5, 98.5]]


def base_india_map():
    """A light basemap locked to India, with everything outside India dimmed."""
    m = folium.Map(
        location=[22.5, 80.0],
        zoom_start=5,
        min_zoom=4,
        max_bounds=True,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Basemap",
        max_zoom=16,
        control=False,
    ).add_to(m)
    m.fit_bounds(INDIA_BOUNDS)
    india = load_india()
    folium.GeoJson(
        india_mask_geojson(india),
        name="Outside India",
        style_function=lambda _: {"fillColor": CREAM, "fillOpacity": 0.82, "weight": 0},
        interactive=False,
        control=False,
    ).add_to(m)
    return m, india


def add_india_outline(m, india):
    folium.GeoJson(
        mapping(india),
        name="India boundary",
        style_function=lambda _: {"color": LEAF_DARK, "weight": 2, "fillOpacity": 0},
        interactive=False,
        control=False,
    ).add_to(m)


def add_map_legend(m, title, items):
    rows = "".join(
        f'<div style="display:flex;align-items:center;margin:3px 0">'
        f'<span style="width:14px;height:14px;border-radius:3px;background:{c};'
        f'border:{b};margin-right:8px;display:inline-block"></span>{label}</div>'
        for label, c, b in items
    )
    tpl = Template(f"""
    {{% macro html(this, kwargs) %}}
    <div style="position:absolute;bottom:24px;left:12px;z-index:9999;background:rgba(255,255,255,.95);
        border-radius:12px;padding:10px 14px;font:12px Inter,sans-serif;color:{LEAF_DARK};
        box-shadow:0 2px 12px rgba(0,0,0,.15);border-top:4px solid {PINK}">
      <div style="font-weight:700;margin-bottom:4px">{title}</div>{rows}
    </div>
    {{% endmacro %}}
    """)
    el = MacroElement()
    el._template = tpl
    m.get_root().add_child(el)


def render_map(m, height):
    st.components.v1.html(m._repr_html_(), height=height, scrolling=False)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

logo_uri = "data:image/svg+xml;base64," + base64.b64encode((ASSETS / "lantana.svg").read_bytes()).decode()

with st.sidebar:
    st.markdown(f"""
<div class="brand"><img src="{logo_uri}"/><div class="brand-name">Bio<span>Watch</span></div></div>
<div class="brand-tag">Tracking <i>Lantana camara</i> across India</div>
""", unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigate to",
        ["Home", "Zone Map", "Model Comparison",
         "Feature Importance", "Monitoring Priority"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("""
<div style='font-size:0.75rem;line-height:1.6;opacity:.85'>
<b>Data source:</b> GBIF occurrence records<br>
<b>Environment:</b> WorldClim 2.1 (2.5 arc-min)<br>
<b>Method:</b> Presence-background (pseudo-absence)<br><br>
<b>Caveat:</b> Suitability scores reflect
environmental conditions similar to known
Lantana locations, not confirmed invasion
probability or ecological damage.
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────

df_all     = load_occurrence_data()
df_pres    = load_presence_data()
df_suit    = load_suitability_grid()
metrics    = load_metrics()
stats      = load_stats()

pipeline_ready = metrics is not None
grid_ready     = df_suit is not None and "suitability_score" in df_suit.columns


def pipeline_warning():
    st.warning(
        "Model outputs not found. "
        "Run `python train_pipeline.py` in the project directory first, "
        "then refresh this page.",
        icon=None,
    )


def presence_df():
    if df_pres is not None:
        return df_pres
    if df_all is not None:
        return df_all[df_all["species_present"] == 1]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# A. HOME
# ──────────────────────────────────────────────────────────────────────────────

if page == "Home":
    img_uri, is_photo = lantana_image()
    img_class = "hero-img" if is_photo else "hero-img illustration"
    st.markdown(f"""
<div class="hero">
  <div style="position:relative;z-index:1">
    <div class="kicker">BioWatch · Invasive Species Intelligence</div>
    <h1>Where could Lantana spread next?</h1>
    <p><span class="sci">Lantana camara</span> is one of the world's most invasive shrubs and
    now covers large parts of India's forests and grasslands. BioWatch combines thousands of
    GBIF sightings with WorldClim climate data and machine learning to map the zones where
    conditions suit the plant.</p>
    <div class="chips">
      <span class="chip y">Native to tropical America</span>
      <span class="chip o">Spread by fruit-eating birds</span>
      <span class="chip p">Toxic to livestock</span>
      <span class="chip m">IUCN top-100 invader</span>
    </div>
  </div>
  <div>
    <div class="{img_class}" role="img" aria-label="Lantana camara flower cluster"
         style="background-image:url('{img_uri}')"></div>
    {'' if is_photo else '<div class="photo-credit">Illustration — add assets/lantana.jpg to show a photo</div>'}
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Key statistics ──────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    if df_all is not None:
        total_rows  = len(df_all)
        n_presence  = (df_all["species_present"] == 1).sum()
        n_unique    = df_all[df_all["species_present"] == 1][["latitude", "longitude"]].drop_duplicates().shape[0]
        feat_cols   = [c for c in df_all.columns if c not in
                       ("latitude", "longitude", "year", "species_present", "point_type")]
        n_features  = len(feat_cols)
    else:
        total_rows = n_presence = n_unique = n_features = "N/A"

    col1.metric("Total Records", f"{total_rows:,}" if isinstance(total_rows, int) else total_rows)
    col2.metric("Presence Records", f"{n_presence:,}" if isinstance(n_presence, (int, np.integer)) else n_presence)
    col3.metric("Unique Locations", f"{n_unique:,}" if isinstance(n_unique, (int, np.integer)) else n_unique)
    col4.metric("Environmental Features", str(n_features))

    if pipeline_ready:
        best_name = max(metrics, key=lambda k: metrics[k]["roc_auc"])
        best_auc  = metrics[best_name]["roc_auc"]
        col5.metric("Best Model AUC", f"{best_auc:.3f}", help=f"Model: {best_name}")
    else:
        col5.metric("Best Model AUC", "N/A")

    # ── About the species ────────────────────────────────────────────────────
    st.markdown("## Know the invader")
    a1, a2, a3 = st.columns(3)
    a1.markdown(f"""<div class="card"><h4><span class="dot" style="background:{YELLOW}"></span>How to recognise it</h4>
<p>A thorny, aromatic shrub with rough leaves and dense round clusters of small flowers
that change colour as they age — yellow at the centre, turning orange, pink and red.</p></div>""",
                unsafe_allow_html=True)
    a2.markdown(f"""<div class="card"><h4><span class="dot" style="background:{PINK}"></span>Why it matters</h4>
<p>It forms thickets that crowd out native plants, reduces fodder for wildlife,
is toxic to cattle and increases the intensity of forest fires.</p></div>""",
                unsafe_allow_html=True)
    a3.markdown(f"""<div class="card"><h4><span class="dot" style="background:{LEAF}"></span>How BioWatch helps</h4>
<p>The suitability zones show where the climate matches places Lantana already grows,
so field teams can plan surveys and early removal where it matters most.</p></div>""",
                unsafe_allow_html=True)

    # ── Pipeline status ──────────────────────────────────────────────────────
    st.markdown("## Pipeline Status")
    s1, s2, s3 = st.columns(3)
    with s1:
        ok = df_all is not None
        pill = '<span class="status-pill status-ok">Ready</span>' if ok else '<span class="status-pill status-warn">Missing</span>'
        st.markdown(f"**Final Dataset** &nbsp; {pill}", unsafe_allow_html=True)
        if ok:
            st.caption(f"{len(df_all):,} rows — temperature, rainfall, elevation + bioclimatic variables")
    with s2:
        pill = '<span class="status-pill status-ok">Ready</span>' if pipeline_ready else '<span class="status-pill status-warn">Not trained</span>'
        st.markdown(f"**Trained Models** &nbsp; {pill}", unsafe_allow_html=True)
        if pipeline_ready:
            st.caption(", ".join(metrics.keys()))
    with s3:
        pill = '<span class="status-pill status-ok">Ready</span>' if grid_ready else '<span class="status-pill status-warn">Not generated</span>'
        st.markdown(f"**Suitability Grid** &nbsp; {pill}", unsafe_allow_html=True)
        if grid_ready:
            st.caption(f"{len(df_suit):,} grid cells")

    # ── Suitability area summary ─────────────────────────────────────────────
    if grid_ready:
        st.markdown("## Share of India by Suitability Zone")
        st.caption(
            "Share of 0.1° grid cells inside India in each zone. "
            "Not equivalent to hectares without area-weighted projection."
        )
        zones_fc = load_class_zones(df_suit)
        cdf = pd.DataFrame([f["properties"] for f in zones_fc["features"]])
        if not cdf.empty:
            fig = px.bar(
                cdf, x="zone", y="share",
                color="zone", color_discrete_map=CLASS_COLORS,
                text=cdf["share"].map(lambda v: f"{v:.1f}%"),
                category_orders={"zone": CLASS_ORDER},
            )
            fig.update_traces(marker_line_width=0, textposition="outside")
            style_fig(fig, 340).update_layout(showlegend=False, xaxis_title="", yaxis_title="% of India's cells")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="footer">BioWatch · Built with GBIF, WorldClim and scikit-learn</div>',
                unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# B. ZONE MAP
# ──────────────────────────────────────────────────────────────────────────────

elif page == "Zone Map":
    st.markdown("# Lantana Suitability Zones of India (Asia)")
    st.caption(
        "Hover over a zone to see its class and score; click a hotspot for details. "
        "Use the layer control (top right) to switch layers on and off."
    )

    pres_df = presence_df()
    if pres_df is None:
        st.error("No occurrence data found.")
        st.stop()
    if not grid_ready:
        pipeline_warning()
        st.stop()

    # Map controls
    mc1, mc2, mc3, mc4 = st.columns([3, 2, 2, 2])
    with mc1:
        shown_classes = st.multiselect(
            "Suitability zones to show", CLASS_ORDER, default=CLASS_ORDER,
        )
    with mc2:
        hot_threshold = st.select_slider(
            "Hotspot threshold", options=[0.5, 0.6, 0.7, 0.8], value=0.6,
            help="Connected areas with suitability score at or above this value form a hotspot zone.",
        )
    with mc3:
        zone_opacity = st.slider("Zone opacity", 0.2, 1.0, 0.7, 0.05)
    with mc4:
        show_presence = st.toggle("Recorded sightings", value=False)
        show_heatmap = st.toggle("Sightings heatmap", value=False)

    zones_fc = load_class_zones(df_suit)
    hot_fc, hot_df = load_hotspot_zones(df_suit, pres_df, hot_threshold)

    zone_ids = ["All of India"] + [f"{r.Zone} · near {r.Near}" for r in hot_df.itertuples()]

    lcol, rcol = st.columns([3, 1.15])

    with rcol:
        focus = st.selectbox("Zoom to hotspot zone", zone_ids)
        focus_id = None if focus == "All of India" else focus.split(" · ")[0]

        if focus_id:
            z = hot_df[hot_df["Zone"] == focus_id].iloc[0]
            st.markdown(f"""
<div class="zone-card">
<b class="title">Zone {z['Zone']}</b><br>
<b>Near:</b> {z['Near']}<br>
<b>Area:</b> {z['Area (km²)']:,} km²<br>
<b>Mean score:</b> {z['Mean Score']:.3f}<br>
<b>Peak score:</b> {z['Peak Score']:.3f}<br>
<b>Very-high cells:</b> {z['Very High Cells']}<br>
<b>Recorded sightings:</b> {z['Recorded Sightings']:,}
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="zone-card">
<b class="title">{len(hot_df)} hotspot zones</b><br>
Connected areas with score ≥ {hot_threshold}.<br>
<b>Total area:</b> {int(hot_df['Area (km²)'].sum()) if len(hot_df) else 0:,} km²<br>
<b>Largest:</b> {(hot_df.iloc[0]['Zone'] + ' near ' + hot_df.iloc[0]['Near']) if len(hot_df) else '—'}
</div>""", unsafe_allow_html=True)

        legend_html = '<div class="legend-box" style="margin-top:12px"><b>Suitability zones</b><br>'
        for f in zones_fc["features"]:
            p = f["properties"]
            legend_html += (f'<div style="display:flex;justify-content:space-between;margin-top:6px">'
                            f'<span><span class="legend-swatch" style="background:{CLASS_COLORS[p["zone"]]}"></span>{p["zone"]}</span>'
                            f'<span style="color:{MUTED}">{p["share"]:.1f}%</span></div>')
        legend_html += '</div>'
        st.markdown(legend_html, unsafe_allow_html=True)

    # Build Folium map
    m, india = base_india_map()
    Fullscreen(position="topleft").add_to(m)

    # ── Suitability zones ────────────────────────────────────────────────────
    shown_fc = {
        "type": "FeatureCollection",
        "features": [f for f in zones_fc["features"] if f["properties"]["zone"] in shown_classes],
    }
    for f in shown_fc["features"]:
        p = f["properties"]
        p["share_txt"] = f"{p['share']:.1f}%"
        p["mean_txt"] = f"{p['mean_score']:.3f}"
    if shown_fc["features"]:
        folium.GeoJson(
            shown_fc,
            name="Suitability zones",
            style_function=lambda f: {
                "fillColor": CLASS_COLORS[f["properties"]["zone"]],
                "fillOpacity": zone_opacity,
                "color": CLASS_COLORS[f["properties"]["zone"]],
                "weight": 0.4,
            },
            highlight_function=lambda f: {"weight": 2, "color": LEAF_DARK, "fillOpacity": min(1, zone_opacity + 0.15)},
            tooltip=folium.GeoJsonTooltip(
                fields=["zone", "share_txt", "mean_txt"],
                aliases=["Zone", "Share of India", "Mean score"],
                sticky=True,
            ),
        ).add_to(m)

    add_india_outline(m, india)

    # ── Hotspot zones ────────────────────────────────────────────────────────
    if hot_fc["features"]:
        hot_layer = folium.FeatureGroup(name="Hotspot zones", show=True)
        for f in hot_fc["features"]:
            p = f["properties"]
            is_focus = p["zone_id"] == focus_id
            popup = folium.Popup(
                f"<div style='font-family:Inter,sans-serif;font-size:12px;line-height:1.6'>"
                f"<b style='color:{MAGENTA};font-size:14px'>Hotspot {p['zone_id']}</b><br>"
                f"Near <b>{p['near']}</b><br>Area: {p['area']} km²<br>"
                f"Mean score: {p['mean_score']}<br>Peak score: {p['peak_score']}<br>"
                f"Recorded sightings: {p['sightings']}</div>",
                max_width=240,
            )
            folium.GeoJson(
                f,
                style_function=lambda _, foc=is_focus: {
                    "fillColor": MAGENTA,
                    "fillOpacity": 0.15 if foc else 0.0,
                    "color": MAGENTA,
                    "weight": 3.5 if foc else 1.8,
                    "dashArray": None if foc else "5 4",
                },
                highlight_function=lambda _: {"fillOpacity": 0.25, "weight": 3.5},
                tooltip=f"Hotspot {p['zone_id']} · near {p['near']} — click for details",
                popup=popup,
            ).add_to(hot_layer)
        hot_layer.add_to(m)

        label_layer = folium.FeatureGroup(name="Hotspot labels", show=True)
        for r in hot_df.head(15).itertuples():
            folium.Marker(
                [r.lat, r.lon],
                icon=folium.DivIcon(
                    html=f"<div style='font:700 11px Inter,sans-serif;color:#fff;background:{MAGENTA};"
                         f"border:2px solid #fff;border-radius:999px;padding:1px 6px;"
                         f"box-shadow:0 1px 4px rgba(0,0,0,.3);white-space:nowrap;"
                         f"transform:translate(-50%,-50%);display:inline-block'>{r.Zone}</div>",
                    icon_size=(0, 0),
                ),
                tooltip=f"{r.Zone} · near {r.Near}",
            ).add_to(label_layer)
        label_layer.add_to(m)

    # ── Sightings ────────────────────────────────────────────────────────────
    if show_heatmap and not pres_df.empty:
        heat_data = pres_df[["latitude", "longitude"]].dropna().values.tolist()
        hm_layer = folium.FeatureGroup(name="Sightings heatmap", show=True)
        HeatMap(heat_data, radius=10, blur=12, min_opacity=0.35,
                gradient={0.3: YELLOW, 0.6: ORANGE, 0.85: PINK, 1.0: MAGENTA}).add_to(hm_layer)
        hm_layer.add_to(m)

    if show_presence and not pres_df.empty:
        pres_layer = folium.FeatureGroup(name="Recorded sightings", show=True)
        sample = pres_df.dropna(subset=["latitude", "longitude"])
        if len(sample) > 3000:
            sample = sample.sample(3000, random_state=42)
        for row in sample.itertuples():
            folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=2.5,
                color=LEAF_DARK,
                fill=True,
                fill_color="#ffffff",
                fill_opacity=0.9,
                weight=1,
                tooltip=f"Lantana sighting · {row.latitude:.3f}, {row.longitude:.3f}",
            ).add_to(pres_layer)
        pres_layer.add_to(m)

    if focus_id:
        z = next(f for f in hot_fc["features"] if f["properties"]["zone_id"] == focus_id)
        minx, miny, maxx, maxy = shape(z["geometry"]).bounds
        m.fit_bounds([[miny - 0.4, minx - 0.4], [maxy + 0.4, maxx + 0.4]])

    folium.LayerControl(collapsed=False).add_to(m)
    add_map_legend(m, "Lantana suitability", [
        *[(c, CLASS_COLORS[c], "1px solid rgba(0,0,0,.1)") for c in CLASS_ORDER],
        ("Hotspot zone", "transparent", f"2px dashed {MAGENTA}"),
    ])

    with lcol:
        render_map(m, 640)

    # ── Hotspot table ────────────────────────────────────────────────────────
    st.markdown("## Hotspot Zones")
    if hot_df.empty:
        st.info("No hotspot zones at this threshold.")
    else:
        st.dataframe(
            hot_df.drop(columns=["lat", "lon"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Mean Score": st.column_config.ProgressColumn("Mean Score", min_value=0, max_value=1, format="%.3f"),
                "Peak Score": st.column_config.NumberColumn(format="%.3f"),
                "Area (km²)": st.column_config.NumberColumn(format="%d"),
            },
        )
    st.caption(
        "India boundary from Natural Earth 1:110m (coarse; small islands are not shown). "
        "Hotspot names refer to the nearest reference town, not an administrative region."
    )


# ──────────────────────────────────────────────────────────────────────────────
# C. MODEL COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

elif page == "Model Comparison":
    st.markdown("# Model Comparison")
    st.caption(
        "Background points (label=0) are pseudo-absences. Metrics describe how "
        "well each model distinguishes presence records from random background "
        "locations — a habitat suitability signal, not a validated absence accuracy."
    )

    if not pipeline_ready:
        pipeline_warning()
        st.stop()

    # ── Summary table ────────────────────────────────────────────────────────
    st.markdown("## Metrics Summary")
    rows = []
    for name, mt in metrics.items():
        cv = mt.get("spatial_cv", {})
        rows.append({
            "Model": name,
            "Hold-out AUC": round(mt["roc_auc"], 4),
            "Spatial CV AUC": f"{cv.get('mean_auc', float('nan')):.4f}" if cv.get('mean_auc') else "N/A",
            "Precision": round(mt["precision"], 4),
            "Recall": round(mt["recall"], 4),
            "F1-Score": round(mt["f1_score"], 4),
        })
    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    best_model = max(metrics, key=lambda k: metrics[k]["roc_auc"])
    st.success(
        f"**Selected model: {best_model}** — highest hold-out ROC-AUC "
        f"({metrics[best_model]['roc_auc']:.4f}). "
        "Spatial CV AUC is the preferred metric for presence-background data "
        "because it avoids spatial autocorrelation leakage."
    )

    # ── ROC curves ───────────────────────────────────────────────────────────
    st.markdown("## ROC Curves")
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                      line=dict(color="#b8b2a3", dash="dash"))
    for i, (name, mt) in enumerate(metrics.items()):
        rc = mt.get("roc_curve", {})
        if rc:
            fig_roc.add_trace(go.Scatter(
                x=rc["fpr"], y=rc["tpr"],
                name=f"{name} (AUC={mt['roc_auc']:.3f})",
                line=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)], width=2.5),
            ))
    style_fig(fig_roc, 420).update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(bgcolor="rgba(255,255,255,.9)", bordercolor="#eadfca", borderwidth=1),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # ── Spatial CV AUC ───────────────────────────────────────────────────────
    st.markdown("## Spatial Block Cross-Validation AUC")
    cv_rows = []
    for name, mt in metrics.items():
        cv = mt.get("spatial_cv", {})
        for fi, auc in enumerate(cv.get("fold_aucs", [])):
            cv_rows.append({"Model": name, "Fold": fi, "AUC": auc})
    if cv_rows:
        cv_df = pd.DataFrame(cv_rows)
        fig_cv = px.box(
            cv_df, x="Model", y="AUC", color="Model",
            color_discrete_sequence=MODEL_COLORS,
            points="all",
        )
        style_fig(fig_cv, 320).update_layout(showlegend=False)
        st.plotly_chart(fig_cv, use_container_width=True)

    # ── Confusion matrices ───────────────────────────────────────────────────
    st.markdown("## Confusion Matrices")
    cm_cols = st.columns(len(metrics))
    for col, (name, mt) in zip(cm_cols, metrics.items()):
        with col:
            cm = np.array(mt["confusion_matrix"])
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Background", "Presence"],
                y=["Background", "Presence"],
                text_auto=True,
                color_continuous_scale=[[0, "#fff6dc"], [0.5, ORANGE], [1, MAGENTA]],
                title=name,
            )
            style_fig(fig_cm, 280).update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_cm, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# D. FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────

elif page == "Feature Importance":
    st.markdown("# Feature Importance")
    st.caption(
        "Feature importance indicates which environmental variables most "
        "distinguish presence from background. It does not imply causation."
    )

    if not pipeline_ready:
        pipeline_warning()
        st.stop()

    fi_models = {
        k: v for k, v in metrics.items()
        if v.get("feature_importance") and v["feature_importance"].get("features")
    }

    if not fi_models:
        st.warning("No feature importance data available.")
        st.stop()

    model_choice = st.selectbox("Select model", list(fi_models.keys()))
    fi_data = fi_models[model_choice]["feature_importance"]

    feat_labels = {
        "temperature": "Bio 1 — Annual Mean Temperature",
        "rainfall":    "Bio 12 — Annual Precipitation",
        "elevation":   "Elevation (m)",
        "bio_2":  "Bio 2 — Mean Diurnal Range",
        "bio_3":  "Bio 3 — Isothermality",
        "bio_4":  "Bio 4 — Temperature Seasonality",
        "bio_5":  "Bio 5 — Max Temp Warmest Month",
        "bio_6":  "Bio 6 — Min Temp Coldest Month",
        "bio_7":  "Bio 7 — Temperature Annual Range",
        "bio_8":  "Bio 8 — Mean Temp Wettest Quarter",
        "bio_9":  "Bio 9 — Mean Temp Driest Quarter",
        "bio_10": "Bio 10 — Mean Temp Warmest Quarter",
        "bio_11": "Bio 11 — Mean Temp Coldest Quarter",
        "bio_13": "Bio 13 — Precip Wettest Month",
        "bio_14": "Bio 14 — Precip Driest Month",
        "bio_15": "Bio 15 — Precipitation Seasonality",
        "bio_16": "Bio 16 — Precip Wettest Quarter",
        "bio_17": "Bio 17 — Precip Driest Quarter",
        "bio_18": "Bio 18 — Precip Warmest Quarter",
        "bio_19": "Bio 19 — Precip Coldest Quarter",
    }

    fi_df = pd.DataFrame({
        "Feature":    [feat_labels.get(f, f) for f in fi_data["features"]],
        "Importance": fi_data["importances"],
    })

    imp_type = fi_data.get("importance_type", "")
    y_label = {
        "gini_impurity":        "Mean Decrease in Gini Impurity",
        "absolute_coefficient": "Absolute Coefficient",
    }.get(imp_type, "Importance")

    fig_fi = px.bar(
        fi_df.head(15), x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=[[0, YELLOW], [0.5, ORANGE], [0.8, PINK], [1, MAGENTA]],
        labels={"Importance": y_label},
    )
    style_fig(fig_fi, 480).update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_fi, use_container_width=True)
    st.caption(f"Importance type: {imp_type}")

    with st.expander("Full feature importance table"):
        st.dataframe(fi_df, use_container_width=True, hide_index=True)

    # ── Environmental variable distributions ─────────────────────────────────
    if df_all is not None:
        st.markdown("## Environmental Variable Distributions")
        avail_feats = [f for f in fi_data["features"] if f in df_all.columns][:6]
        if avail_feats:
            chosen = st.multiselect(
                "Choose variables to plot",
                options=[feat_labels.get(f, f) for f in fi_data["features"] if f in df_all.columns],
                default=[feat_labels.get(f, f) for f in avail_feats[:3]],
                max_selections=6,
            )
            label_to_col = {v: k for k, v in feat_labels.items()}
            chosen_cols = [label_to_col.get(lbl, lbl) for lbl in chosen]
            chosen_cols = [c for c in chosen_cols if c in df_all.columns]

            if chosen_cols:
                dist_cols = st.columns(min(3, len(chosen_cols)))
                for i, col_name in enumerate(chosen_cols):
                    with dist_cols[i % 3]:
                        fig_dist = px.histogram(
                            df_all,
                            x=col_name,
                            color="point_type",
                            color_discrete_map={
                                "presence": PINK,
                                "background": "#9fb59a",
                            },
                            barmode="overlay",
                            opacity=0.75,
                            title=feat_labels.get(col_name, col_name),
                            nbins=40,
                        )
                        style_fig(fig_dist, 260).update_layout(
                            showlegend=(i == 0),
                            legend=dict(orientation="h", y=1.15, title=""),
                            title_font_size=13,
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# E. MONITORING PRIORITY
# ──────────────────────────────────────────────────────────────────────────────

elif page == "Monitoring Priority":
    st.markdown("# Suitability and Monitoring Priority")
    st.caption(
        "Monitoring priority combines habitat suitability with proximity to "
        "known occurrences. It is a heuristic score, not a validated ecological "
        "threat measure. Formula: Priority = 0.7 × Suitability + 0.3 × Proximity."
    )

    if not grid_ready:
        pipeline_warning()
        st.stop()

    pres_df = presence_df()
    if pres_df is None:
        st.error("Occurrence data not available.")
        st.stop()

    pres_coords = pres_df[["latitude", "longitude"]].dropna().values
    suit_valid = df_suit[df_suit["suitability_score"].notna()].copy()

    if len(suit_valid) == 0:
        st.warning("No valid suitability scores found.")
        st.stop()

    # ── Proximity score (inverse distance to nearest occurrence) ─────────────
    from sklearn.neighbors import BallTree

    with st.spinner("Calculating proximity scores…"):
        pres_rad = np.deg2rad(pres_coords)
        grid_rad = np.deg2rad(suit_valid[["latitude", "longitude"]].values)
        tree = BallTree(pres_rad, metric="haversine")
        dists_rad, _ = tree.query(grid_rad, k=1)
        dists_km = dists_rad[:, 0] * 6371.0

    # Proximity score: 1 at 0 km, ~0 at 500+ km
    proximity = np.exp(-dists_km / 150)
    suit_valid["proximity_score"] = proximity
    suit_valid["monitoring_priority"] = (
        0.7 * suit_valid["suitability_score"] + 0.3 * proximity
    )

    # ── Priority zone map ────────────────────────────────────────────────────
    st.markdown("## Priority Zones")
    top_n = st.slider("Show top N cells by priority score", 500, 5000, 2000, 500)
    top = suit_valid.nlargest(top_n, "monitoring_priority")

    # Bin the selected cells into four priority tiers and dissolve each tier into a zone
    tier_colors = [YELLOW, ORANGE, PINK, MAGENTA]
    tier_names = ["Watch", "Elevated", "High", "Urgent"]
    edges = np.quantile(top["monitoring_priority"], [0, 0.25, 0.5, 0.75, 1.0])
    top["tier"] = np.clip(np.searchsorted(edges, top["monitoring_priority"], side="right") - 1, 0, 3)

    m2, india = base_india_map()
    half = 0.05
    for t in range(4):
        sub = top[top["tier"] == t]
        if sub.empty:
            continue
        geom = unary_union([box(lo - half, la - half, lo + half, la + half)
                            for la, lo in zip(sub["latitude"], sub["longitude"])]).intersection(india)
        if geom.is_empty:
            continue
        folium.GeoJson(
            mapping(geom),
            name=f"{tier_names[t]} priority",
            style_function=lambda _, c=tier_colors[t]: {
                "fillColor": c, "color": c, "fillOpacity": 0.8, "weight": 0.5,
            },
            highlight_function=lambda _: {"weight": 2, "color": LEAF_DARK},
            tooltip=(f"<b>{tier_names[t]} priority</b><br>"
                     f"Score {edges[t]:.3f} – {edges[t + 1]:.3f}<br>{len(sub):,} cells"),
        ).add_to(m2)
    add_india_outline(m2, india)
    folium.LayerControl(collapsed=False).add_to(m2)
    add_map_legend(m2, "Monitoring priority", [
        (n, c, "1px solid rgba(0,0,0,.1)") for n, c in reversed(list(zip(tier_names, tier_colors)))
    ])

    render_map(m2, 580)

    # ── Distribution chart ───────────────────────────────────────────────────
    st.markdown("## Priority Score Distribution")
    fig_p = px.histogram(
        suit_valid, x="monitoring_priority",
        nbins=50,
        color_discrete_sequence=[PINK],
        labels={"monitoring_priority": "Monitoring Priority Score"},
    )
    style_fig(fig_p, 280).update_traces(marker_line_width=0)
    st.plotly_chart(fig_p, use_container_width=True)

    with st.expander("Methodology note"):
        st.markdown("""
**Monitoring Priority Formula**
```
Priority = 0.7 × Suitability Score + 0.3 × Proximity Score
```
- **Suitability Score**: Model probability score in [0, 1] from the best-performing classifier.
- **Proximity Score**: `exp(-d / 150)` where d is the geodesic distance (km) to the nearest known occurrence record. Decays to ~0.19 at 250 km and ~0.04 at 500 km.
- Weights (0.7 / 0.3) are configurable heuristics, not ecologically validated parameters.
- Tiers split the selected top-N cells into quartiles of priority score.
- This score should guide field survey planning, not species-management decisions.
        """)
