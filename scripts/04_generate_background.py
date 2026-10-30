"""
04_generate_background.py

Generate background (pseudo-absence-free) points randomly within India's
real geographic boundary (Natural Earth polygon, not a bounding box).

IMPORTANT - these are NOT confirmed absences. They are locations sampled
from the available geographic space where no GBIF Lantana camara occurrence
was used. Downstream modelling must not treat species_present = 0 as a
biologically confirmed absence. See README "Background-point methodology".

Method:
  - Sample random points inside the India polygon using rejection sampling
    (draw within the polygon's bounding box, keep only points that fall
    inside the actual polygon).
  - Discard any candidate within MIN_DISTANCE_KM of a known presence point,
    to reduce spatial overlap between classes.
  - Generate roughly the same number of background points as presence
    points (configurable).

Output:
    data/processed/background_points.csv
    columns: latitude, longitude, year (NA), species_present (=0), point_type
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRESENCE_CSV = ROOT / "data" / "processed" / "presence_points.csv"
OUT_CSV = ROOT / "data" / "processed" / "background_points.csv"
BOUNDARY_SHP = ROOT / "data" / "raw" / "natural_earth" / "ne_110m_admin_0_countries.shp"

MIN_DISTANCE_KM = 1.0  # minimum distance from any known presence point
RANDOM_SEED = 42


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    if not PRESENCE_CSV.exists():
        raise SystemExit(f"{PRESENCE_CSV} not found. Run 03_clean_occurrences.py first.")
    if not BOUNDARY_SHP.exists():
        raise SystemExit(
            f"{BOUNDARY_SHP} not found. Run 03_clean_occurrences.py first "
            "(it downloads the Natural Earth boundary)."
        )

    import geopandas as gpd
    from shapely.geometry import Point

    presence = pd.read_csv(PRESENCE_CSV)
    n_presence = len(presence)
    print(f"Presence points to match: {n_presence}")

    world = gpd.read_file(BOUNDARY_SHP)
    india_col = "ADMIN" if "ADMIN" in world.columns else "NAME"
    india = world[world[india_col] == "India"]
    india_geom = india.geometry.union_all() if hasattr(india.geometry, "union_all") else india.geometry.unary_union
    minx, miny, maxx, maxy = india_geom.bounds

    rng = np.random.default_rng(RANDOM_SEED)
    presence_lat = presence["latitude"].to_numpy()
    presence_lon = presence["longitude"].to_numpy()

    background = []
    attempts = 0
    max_attempts = n_presence * 200  # safety valve

    print("Generating background points via rejection sampling inside the India polygon...")
    while len(background) < n_presence and attempts < max_attempts:
        batch_size = min(2000, (n_presence - len(background)) * 5)
        attempts += batch_size

        cand_lon = rng.uniform(minx, maxx, batch_size)
        cand_lat = rng.uniform(miny, maxy, batch_size)

        cand_gdf = gpd.GeoDataFrame(
            {"latitude": cand_lat, "longitude": cand_lon},
            geometry=[Point(xy) for xy in zip(cand_lon, cand_lat)],
            crs="EPSG:4326",
        )
        inside = cand_gdf[cand_gdf.geometry.within(india_geom)]

        for _, row in inside.iterrows():
            if len(background) >= n_presence:
                break
            dists = haversine_km(row["latitude"], row["longitude"], presence_lat, presence_lon)
            if dists.min() >= MIN_DISTANCE_KM:
                background.append((row["latitude"], row["longitude"]))

    print(f"Generated {len(background)} background points after {attempts} candidate draws.")
    if len(background) < n_presence:
        print(
            f"WARNING: only generated {len(background)} of {n_presence} requested "
            "background points within the attempt budget. Increase max_attempts "
            "or relax MIN_DISTANCE_KM if you need an exact 1:1 ratio."
        )

    out = pd.DataFrame(background, columns=["latitude", "longitude"])
    out["year"] = pd.NA
    out["species_present"] = 0
    out["point_type"] = "background"

    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(out)} background points to {OUT_CSV}")
    print(
        "\nREMINDER: species_present = 0 here means 'background sampling "
        "point', NOT a confirmed biological absence of Lantana camara."
    )


if __name__ == "__main__":
    main()
