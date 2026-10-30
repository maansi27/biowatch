"""
09_visualize.py

Generates the core exploratory figures and saves them to outputs/figures/:
  1. presence_points_map.png       - GBIF presence points over India
  2. background_points_map.png     - generated background points over India
  3. combined_map.png              - presence vs background overlay
  4. env_variable_distributions.png- histograms of temperature/rainfall/elevation(/NDVI)
  5. records_by_year.png           - presence record counts per year
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FINAL_CSV = ROOT / "data" / "final" / "invasive_species_lantana_india.csv"
BOUNDARY_SHP = ROOT / "data" / "raw" / "natural_earth" / "ne_110m_admin_0_countries.shp"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_india_boundary():
    import geopandas as gpd
    world = gpd.read_file(BOUNDARY_SHP)
    col = "ADMIN" if "ADMIN" in world.columns else "NAME"
    return world[world[col] == "India"]


def plot_points_map(df, title, filename, color):
    india = load_india_boundary()
    fig, ax = plt.subplots(figsize=(7, 8))
    india.boundary.plot(ax=ax, color="black", linewidth=0.8)
    ax.scatter(df["longitude"], df["latitude"], s=6, alpha=0.5, color=color)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_combined_map(df):
    india = load_india_boundary()
    fig, ax = plt.subplots(figsize=(7, 8))
    india.boundary.plot(ax=ax, color="black", linewidth=0.8)
    pres = df[df["point_type"] == "presence"]
    bg = df[df["point_type"] == "background"]
    ax.scatter(bg["longitude"], bg["latitude"], s=6, alpha=0.4, color="gray", label="Background")
    ax.scatter(pres["longitude"], pres["latitude"], s=8, alpha=0.6, color="crimson", label="Presence (GBIF)")
    ax.set_title("Presence vs. Background Points - Lantana camara, India")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "combined_map.png", dpi=150)
    plt.close(fig)
    print("Saved combined_map.png")


def plot_env_distributions(df):
    env_cols = [c for c in ["temperature", "rainfall", "elevation", "NDVI"] if c in df.columns]
    fig, axes = plt.subplots(1, len(env_cols), figsize=(5 * len(env_cols), 4))
    if len(env_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, env_cols):
        for ptype, color in [("presence", "crimson"), ("background", "gray")]:
            subset = df[df["point_type"] == ptype][col].dropna()
            ax.hist(subset, bins=30, alpha=0.5, label=ptype, color=color)
        ax.set_title(col)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "env_variable_distributions.png", dpi=150)
    plt.close(fig)
    print("Saved env_variable_distributions.png")


def plot_records_by_year(df):
    pres = df[df["point_type"] == "presence"]
    counts = pres["year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index.astype(str), counts.values, color="steelblue")
    ax.set_title("GBIF Presence Records by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of records")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "records_by_year.png", dpi=150)
    plt.close(fig)
    print("Saved records_by_year.png")


def main():
    if not FINAL_CSV.exists():
        raise SystemExit("Run 07_build_final_dataset.py first.")
    df = pd.read_csv(FINAL_CSV)

    presence = df[df["point_type"] == "presence"]
    background = df[df["point_type"] == "background"]

    plot_points_map(presence, "GBIF Presence Points - Lantana camara, India", "presence_points_map.png", "crimson")
    plot_points_map(background, "Generated Background Points, India", "background_points_map.png", "gray")
    plot_combined_map(df)
    plot_env_distributions(df)
    plot_records_by_year(df)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
