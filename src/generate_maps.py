"""
src/generate_maps.py

Generate static and interactive map outputs from the suitability grid.

Outputs:
    outputs/map_occurrence_points.html   – Interactive Folium occurrence map
    outputs/map_suitability.html         – Interactive Folium suitability map
    outputs/figures/suitability_map.png  – Static matplotlib suitability map
"""

import logging
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    HAS_GPD = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parent.parent
OUTPUTS     = ROOT / "outputs"
FIGURES     = OUTPUTS / "figures"
DATA_PRES   = ROOT / "data" / "processed" / "presence_points.csv"
SUITABILITY = OUTPUTS / "suitability_grid.csv"
FIGURES.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = {
    "Very Low":  "#2d6a4f",
    "Low":       "#74c69d",
    "Moderate":  "#f4d03f",
    "High":      "#e67e22",
    "Very High": "#c0392b",
}
CLASS_ORDER = ["Very Low", "Low", "Moderate", "High", "Very High"]


def build_occurrence_map(pres_df: pd.DataFrame, out_path: Path):
    m = folium.Map(location=[20.5, 80.0], zoom_start=5, tiles="OpenStreetMap")
    layer = folium.FeatureGroup(name="Occurrences")
    sample = pres_df.dropna(subset=["latitude", "longitude"])
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)
    for _, row in sample.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color="#58a6ff",
            fill=True,
            fill_color="#58a6ff",
            fill_opacity=0.8,
            weight=0.5,
            popup=f"Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f}",
        ).add_to(layer)
    layer.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(str(out_path))
    log.info("Occurrence map saved -> %s", out_path)


def build_suitability_map(suit_df: pd.DataFrame, pres_df: pd.DataFrame, out_path: Path):
    m = folium.Map(location=[20.5, 80.0], zoom_start=5, tiles="OpenStreetMap", control_scale=True)

    # Suitability layer
    suit_layer = folium.FeatureGroup(name="Habitat Suitability")
    sub = suit_df[suit_df["suitability_class"].isin(CLASS_ORDER)].copy()
    if len(sub) > 40_000:
        sub = sub.sample(40_000, random_state=42)
    for _, row in sub.iterrows():
        cls = row["suitability_class"]
        score = row.get("suitability_score", 0)
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color=CLASS_COLORS.get(cls, "#aaa"),
            fill=True,
            fill_color=CLASS_COLORS.get(cls, "#aaa"),
            fill_opacity=0.6,
            weight=0,
            popup=(
                f"Score: {score:.3f} | Class: {cls}"
                if not (isinstance(score, float) and np.isnan(score)) else cls
            ),
        ).add_to(suit_layer)
    suit_layer.add_to(m)

    # Presence layer
    pres_layer = folium.FeatureGroup(name="Occurrence Points", show=False)
    sample = pres_df.dropna(subset=["latitude", "longitude"])
    if len(sample) > 3000:
        sample = sample.sample(3000, random_state=42)
    for _, row in sample.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            color="#58a6ff",
            fill=True,
            fill_color="#58a6ff",
            fill_opacity=0.9,
            weight=0,
        ).add_to(pres_layer)
    pres_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_path))
    log.info("Suitability map saved -> %s", out_path)


def build_static_map(suit_df: pd.DataFrame, pres_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # Plot each class separately for legend
    for cls in CLASS_ORDER:
        sub = suit_df[suit_df["suitability_class"] == cls]
        if sub.empty:
            continue
        ax.scatter(
            sub["longitude"], sub["latitude"],
            c=CLASS_COLORS[cls], s=1, alpha=0.5,
            label=cls, rasterized=True,
        )

    # Presence points on top
    pres_valid = pres_df.dropna(subset=["latitude", "longitude"])
    ax.scatter(
        pres_valid["longitude"], pres_valid["latitude"],
        c="#58a6ff", s=4, alpha=0.6, label="Occurrence", zorder=5,
    )

    ax.set_xlim(67, 98)
    ax.set_ylim(6, 37)
    ax.set_xlabel("Longitude", color="#c9d1d9")
    ax.set_ylabel("Latitude", color="#c9d1d9")
    ax.set_title("Lantana camara — Habitat Suitability\n(India, WorldClim 2.1 + Random Forest)",
                 color="#e8eaf0", fontsize=13)
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_color("#21262d")

    legend = ax.legend(
        loc="lower left", facecolor="#161b22", edgecolor="#21262d",
        labelcolor="#c9d1d9", fontsize=8, markerscale=4,
        title="Suitability Class", title_fontsize=8,
    )
    legend.get_title().set_color("#8b949e")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info("Static suitability map saved -> %s", out_path)


def main():
    pres_df = None
    if DATA_PRES.exists():
        pres_df = pd.read_csv(DATA_PRES)
        log.info("Presence data: %d records", len(pres_df))
    else:
        log.warning("Presence points file not found: %s", DATA_PRES)

    if SUITABILITY.exists():
        suit_df = pd.read_csv(SUITABILITY)
        log.info("Suitability grid: %d cells", len(suit_df))
    else:
        log.error(
            "Suitability grid not found: %s\n"
            "Run train_pipeline.py first.",
            SUITABILITY,
        )
        return

    if pres_df is not None:
        build_occurrence_map(pres_df, OUTPUTS / "map_occurrence_points.html")
        build_suitability_map(suit_df, pres_df, OUTPUTS / "map_suitability.html")
        build_static_map(suit_df, pres_df, FIGURES / "suitability_map.png")
    else:
        log.warning("Skipping maps that require presence data.")


if __name__ == "__main__":
    main()
