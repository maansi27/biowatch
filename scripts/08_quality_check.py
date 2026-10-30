"""
08_quality_check.py

Prints an automated data-quality report on the final dataset:
missing values, duplicate coordinates, class counts, per-year counts,
environmental-value ranges, and (if geopandas + a states shapefile are
available) per-state counts.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FINAL_CSV = ROOT / "data" / "final" / "invasive_species_lantana_india.csv"


def main():
    if not FINAL_CSV.exists():
        raise SystemExit("Run 07_build_final_dataset.py first.")

    df = pd.read_csv(FINAL_CSV)

    n_presence = (df["point_type"] == "presence").sum()
    n_background = (df["point_type"] == "background").sum()

    print("=== DATA QUALITY REPORT ===\n")
    print(f"Final ML rows:              {len(df)}")
    print(f"  Presence points:          {n_presence}")
    print(f"  Background points:        {n_background}")

    print("\n--- Missing values per column ---")
    print(df.isna().sum())

    print("\n--- Duplicate coordinate pairs ---")
    dupe_coords = df.duplicated(subset=["latitude", "longitude"]).sum()
    print(f"Duplicate lat/lon pairs: {dupe_coords}")

    print("\n--- Coordinate range check ---")
    print(f"Latitude range:  {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
    print(f"Longitude range: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
    out_of_india_bbox = df[
        ~df["latitude"].between(6, 38) | ~df["longitude"].between(68, 98)
    ]
    print(f"Points outside India's rough bounding box (6-38N, 68-98E): {len(out_of_india_bbox)}")

    print("\n--- Environmental value sanity ranges ---")
    if "temperature" in df.columns:
        print(f"Temperature (deg C): {df['temperature'].min():.2f} to {df['temperature'].max():.2f}")
        weird_temp = df[~df["temperature"].between(-10, 45)]
        print(f"  Suspicious temperature values (outside -10 to 45C): {len(weird_temp)}")
    if "rainfall" in df.columns:
        print(f"Rainfall (mm/yr): {df['rainfall'].min():.1f} to {df['rainfall'].max():.1f}")
        weird_rain = df[~df["rainfall"].between(0, 12000)]
        print(f"  Suspicious rainfall values (outside 0-12000mm): {len(weird_rain)}")
    if "elevation" in df.columns:
        print(f"Elevation (m): {df['elevation'].min():.1f} to {df['elevation'].max():.1f}")
        weird_elev = df[~df["elevation"].between(-100, 9000)]
        print(f"  Suspicious elevation values: {len(weird_elev)}")
    if "NDVI" in df.columns:
        print(f"NDVI: {df['NDVI'].min():.3f} to {df['NDVI'].max():.3f}")
        weird_ndvi = df[~df["NDVI"].between(-1, 1)]
        print(f"  Out-of-range NDVI values: {len(weird_ndvi)}")

    print("\n--- Records by year (presence only) ---")
    presence = df[df["point_type"] == "presence"]
    print(presence["year"].value_counts().sort_index())

    print(
        "\n--- Per-state counts ---\n"
        "Not computed automatically in this script (needs a states/UTs "
        "shapefile, e.g. Natural Earth 10m 'admin-1' or GADM level-1 for "
        "India). To add: spatial-join presence/background points against "
        "that layer with GeoPandas the same way 03_clean_occurrences.py "
        "does the country-level join."
    )


if __name__ == "__main__":
    main()
