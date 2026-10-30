"""
03_clean_occurrences.py

Clean the raw GBIF download into a quality-controlled presence-point table.

Quality control steps (each counted and reported):
  1. Drop rows with missing latitude/longitude
  2. Drop rows with invalid coordinates (out of physical range, or 0,0)
  3. Drop rows whose coordinates fall outside India's geographic extent
     using a REAL polygon boundary (Natural Earth admin-0 country, via
     GeoPandas), not a rectangular bounding box
  4. Drop exact duplicate coordinate+date rows

Input:
    data/raw/gbif_lantana_india.csv

Output:
    data/processed/presence_points.csv
    columns: latitude, longitude, year, month, species, occurrence_id,
             basis_of_record, species_present (=1)

Requires the India boundary shapefile - see 04_generate_background.py /
README for the Natural Earth download step. This script downloads it
automatically (Natural Earth ships a small, low-res 110m file that is fine
for a country-level in/out check) if it isn't already present.
"""

import sys
import zipfile
import io
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "gbif_lantana_india.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = PROCESSED_DIR / "presence_points.csv"

BOUNDARY_DIR = ROOT / "data" / "raw" / "natural_earth"
BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
# 110m resolution admin-0 countries - small file, fine for a country-level
# point-in-polygon test. For a tighter (10m) boundary, follow the manual
# download instructions in the README instead.
NE_URL = (
    "https://naciscdn.org/naturalearth/110m/cultural/"
    "ne_110m_admin_0_countries.zip"
)


def ensure_boundary_shapefile() -> Path:
    shp_path = BOUNDARY_DIR / "ne_110m_admin_0_countries.shp"
    if shp_path.exists():
        return shp_path

    print(f"Downloading Natural Earth country boundaries from {NE_URL} ...")
    try:
        resp = requests.get(NE_URL, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(BOUNDARY_DIR)
        print(f"Extracted boundary files to {BOUNDARY_DIR}")
    except requests.RequestException as e:
        print(
            f"\nCould not auto-download Natural Earth boundaries ({e}).\n"
            "MANUAL FALLBACK:\n"
            "  1. Go to https://www.naturalearthdata.com/downloads/110m-cultural-vectors/\n"
            "  2. Download 'Admin 0 - Countries' (110m)\n"
            "  3. Unzip into: "
            f"{BOUNDARY_DIR}\n"
            "  4. Re-run this script.\n"
        )
        sys.exit(1)

    return shp_path


def main():
    if not RAW_CSV.exists():
        raise SystemExit(f"{RAW_CSV} not found. Run 02_download_gbif.py first.")

    import geopandas as gpd
    from shapely.geometry import Point

    df = pd.read_csv(RAW_CSV, low_memory=False)
    n_total = len(df)
    print(f"1. Total raw GBIF records loaded: {n_total}")

    # --- Step 1: missing lat/lon ---
    df["decimalLatitude"] = pd.to_numeric(df["decimalLatitude"], errors="coerce")
    df["decimalLongitude"] = pd.to_numeric(df["decimalLongitude"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["decimalLatitude", "decimalLongitude"])
    n_missing_coords = before - len(df)
    print(f"2. Records with coordinates: {len(df)} (dropped {n_missing_coords} missing)")

    # --- Step 2: invalid coordinate ranges / null island ---
    before = len(df)
    df = df[
        df["decimalLatitude"].between(-90, 90)
        & df["decimalLongitude"].between(-180, 180)
        & ~((df["decimalLatitude"] == 0) & (df["decimalLongitude"] == 0))
    ]
    n_invalid = before - len(df)
    print(f"3. Records with valid coordinate ranges: {len(df)} (dropped {n_invalid} invalid)")

    # --- Step 3: point-in-polygon test against real India boundary ---
    shp_path = ensure_boundary_shapefile()
    world = gpd.read_file(shp_path)
    india_col = "ADMIN" if "ADMIN" in world.columns else "NAME"
    india = world[world[india_col] == "India"]
    if india.empty:
        raise SystemExit("Could not find India in the Natural Earth boundary file.")
    india_geom = india.geometry.union_all() if hasattr(india.geometry, "union_all") else india.geometry.unary_union

    before = len(df)
    geometry = [Point(xy) for xy in zip(df["decimalLongitude"], df["decimalLatitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    inside_mask = gdf.geometry.within(india_geom)
    gdf = gdf[inside_mask]
    n_outside = before - len(gdf)
    print(f"4. Records inside India polygon: {len(gdf)} (dropped {n_outside} outside boundary)")

    # --- Step 4: duplicates ---
    before = len(gdf)
    gdf["_dupe_key"] = (
        gdf["decimalLatitude"].round(5).astype(str)
        + "_"
        + gdf["decimalLongitude"].round(5).astype(str)
        + "_"
        + gdf.get("eventDate", "").astype(str)
    )
    gdf = gdf.drop_duplicates(subset="_dupe_key")
    n_dupes = before - len(gdf)
    print(f"5. After duplicate removal: {len(gdf)} (dropped {n_dupes} duplicates)")

    # --- Build final presence table ---
    out = pd.DataFrame({
        "latitude": gdf["decimalLatitude"],
        "longitude": gdf["decimalLongitude"],
        "year": pd.to_numeric(gdf.get("year"), errors="coerce"),
        "month": pd.to_numeric(gdf.get("month"), errors="coerce"),
        "species": gdf.get("species", "Lantana camara"),
        "occurrence_id": gdf.get("occurrenceID", gdf.get("gbifID")),
        "basis_of_record": gdf.get("basisOfRecord"),
        "species_present": 1,
    })
    out.to_csv(OUT_CSV, index=False)

    print("\n=== SUMMARY ===")
    print(f"Total raw GBIF records:        {n_total}")
    print(f"Missing coordinates removed:   {n_missing_coords}")
    print(f"Invalid coordinates removed:   {n_invalid}")
    print(f"Outside-India removed:         {n_outside}")
    print(f"Duplicates removed:            {n_dupes}")
    print(f"FINAL usable presence points:  {len(out)}")
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    main()
