"""
zones.py – Turn the point-based suitability grid into map zones.

Each 0.1-degree grid cell is expanded to a square, clipped to India's
boundary and dissolved by suitability class, so the map shows continuous
zones instead of individual dots. Connected High / Very High cells are
grouped into numbered hotspot zones with summary statistics.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import ndimage
import shapely
from shapely.geometry import box, mapping
from shapely.ops import unary_union

ROOT        = Path(__file__).resolve().parent.parent
NE_COUNTRIES = ROOT / "data" / "raw" / "natural_earth" / "ne_110m_admin_0_countries.shp"

CLASS_ORDER = ["Very Low", "Low", "Moderate", "High", "Very High"]

# Reference towns used to give hotspot zones a readable name
REFERENCE_PLACES = [
    ("Srinagar", 34.08, 74.80), ("Shimla", 31.10, 77.17), ("Dehradun", 30.32, 78.03),
    ("Chandigarh", 30.73, 76.78), ("Delhi", 28.61, 77.21), ("Jaipur", 26.91, 75.79),
    ("Jodhpur", 26.24, 73.02), ("Udaipur", 24.58, 73.71), ("Lucknow", 26.85, 80.95),
    ("Varanasi", 25.32, 82.97), ("Patna", 25.59, 85.14), ("Gangtok", 27.33, 88.61),
    ("Siliguri", 26.73, 88.40), ("Guwahati", 26.14, 91.74), ("Shillong", 25.58, 91.89),
    ("Imphal", 24.82, 93.94), ("Aizawl", 23.73, 92.72), ("Agartala", 23.83, 91.29),
    ("Itanagar", 27.08, 93.61), ("Kohima", 25.67, 94.11), ("Kolkata", 22.57, 88.36),
    ("Ranchi", 23.34, 85.31), ("Bhubaneswar", 20.30, 85.82), ("Raipur", 21.25, 81.63),
    ("Jagdalpur", 19.08, 82.02), ("Bhopal", 23.26, 77.41), ("Jabalpur", 23.18, 79.99),
    ("Indore", 22.72, 75.86), ("Ahmedabad", 23.02, 72.57), ("Rajkot", 22.30, 70.80),
    ("Surat", 21.17, 72.83), ("Nagpur", 21.15, 79.09), ("Mumbai", 19.08, 72.88),
    ("Pune", 18.52, 73.86), ("Aurangabad", 19.88, 75.34), ("Hyderabad", 17.39, 78.49),
    ("Visakhapatnam", 17.69, 83.22), ("Vijayawada", 16.51, 80.65), ("Goa", 15.49, 73.83),
    ("Hubballi", 15.36, 75.12), ("Bengaluru", 12.97, 77.59), ("Mysuru", 12.30, 76.64),
    ("Mangaluru", 12.91, 74.86), ("Chennai", 13.08, 80.27), ("Ooty", 11.41, 76.70),
    ("Coimbatore", 11.02, 76.96), ("Madurai", 9.93, 78.12), ("Kochi", 9.93, 76.27),
    ("Thiruvananthapuram", 8.52, 76.94), ("Kozhikode", 11.26, 75.78),
]


def load_india_boundary():
    """India polygon from Natural Earth (110m, coarse)."""
    countries = gpd.read_file(NE_COUNTRIES)
    return countries.loc[countries["ADMIN"] == "India", "geometry"].iloc[0]


def nearest_place(lat, lon):
    best = min(REFERENCE_PLACES, key=lambda p: (p[1] - lat) ** 2 + (p[2] - lon) ** 2)
    return best[0]


def _cell_box(lat, lon, half):
    return box(lon - half, lat - half, lon + half, lat + half)


def build_class_zones(df_suit, india, resolution=0.1):
    """Dissolve grid cells into one clipped (multi)polygon per suitability class.

    Returns a GeoJSON FeatureCollection dict.
    """
    half = resolution / 2
    valid = df_suit[df_suit["suitability_class"].isin(CLASS_ORDER)]
    total = 0
    features = []
    for cls in CLASS_ORDER:
        sub = valid[valid["suitability_class"] == cls]
        if sub.empty:
            continue
        cells = [_cell_box(la, lo, half) for la, lo in zip(sub["latitude"], sub["longitude"])]
        geom = unary_union(cells).intersection(india)
        if geom.is_empty:
            continue
        # Small simplification keeps the GeoJSON light without visible loss
        geom = geom.simplify(resolution / 4, preserve_topology=True)
        inside = sub[shapely.contains_xy(india, sub["longitude"], sub["latitude"])]
        total += len(inside)
        features.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {
                "zone": cls,
                "cells": int(len(inside)),
                "mean_score": float(inside["suitability_score"].mean()) if len(inside) else float("nan"),
            },
        })
    for f in features:
        f["properties"]["share"] = 100 * f["properties"]["cells"] / total if total else 0
    return {"type": "FeatureCollection", "features": features}


def build_hotspot_zones(df_suit, india, df_pres=None, threshold=0.6,
                        resolution=0.1, min_cells=4):
    """Group connected cells with score >= threshold into numbered hotspot zones.

    Returns (GeoJSON FeatureCollection dict, summary DataFrame).
    """
    half = resolution / 2
    valid = df_suit[df_suit["suitability_score"].notna()].copy()
    valid["ri"] = np.round((valid["latitude"] - valid["latitude"].min()) / resolution).astype(int)
    valid["ci"] = np.round((valid["longitude"] - valid["longitude"].min()) / resolution).astype(int)

    mask = np.zeros((valid["ri"].max() + 1, valid["ci"].max() + 1), dtype=bool)
    hot = valid[valid["suitability_score"] >= threshold]
    mask[hot["ri"], hot["ci"]] = True
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    hot = hot.assign(label=labels[hot["ri"], hot["ci"]])

    pres = None
    if df_pres is not None and not df_pres.empty:
        pres = df_pres[["latitude", "longitude"]].dropna()

    # Approximate cell area: 0.1 deg ≈ 11.1 km, scaled by cos(latitude)
    km_per_deg = 111.32

    rows, features = [], []
    for lab, grp in hot.groupby("label"):
        if len(grp) < min_cells:
            continue
        cells = [_cell_box(la, lo, half) for la, lo in zip(grp["latitude"], grp["longitude"])]
        geom = unary_union(cells).intersection(india)
        if geom.is_empty or geom.area < (resolution ** 2) * min_cells / 2:
            continue
        geom = geom.simplify(resolution / 4, preserve_topology=True)
        c = geom.centroid
        area_km2 = float(sum((resolution * km_per_deg) ** 2 * np.cos(np.deg2rad(la))
                             for la in grp["latitude"]))
        n_occ = 0
        if pres is not None:
            minx, miny, maxx, maxy = geom.bounds
            cand = pres[(pres["longitude"].between(minx, maxx)) & (pres["latitude"].between(miny, maxy))]
            n_occ = int(shapely.contains_xy(geom, cand["longitude"], cand["latitude"]).sum())
        rows.append({
            "label": int(lab),
            "Near": nearest_place(c.y, c.x),
            "Area (km²)": round(area_km2),
            "Mean Score": float(grp["suitability_score"].mean()),
            "Peak Score": float(grp["suitability_score"].max()),
            "Very High Cells": int((grp["suitability_score"] >= 0.8).sum()),
            "Recorded Sightings": n_occ,
            "lat": c.y, "lon": c.x,
            "geometry": geom,
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return {"type": "FeatureCollection", "features": []}, summary

    summary = summary.sort_values("Area (km²)", ascending=False).reset_index(drop=True)
    summary.insert(0, "Zone", [f"Z{i + 1:02d}" for i in range(len(summary))])
    for _, r in summary.iterrows():
        features.append({
            "type": "Feature",
            "geometry": mapping(r["geometry"]),
            "properties": {
                "zone_id": r["Zone"],
                "near": r["Near"],
                "area": f"{r['Area (km²)']:,}",
                "mean_score": round(r["Mean Score"], 3),
                "peak_score": round(r["Peak Score"], 3),
                "sightings": int(r["Recorded Sightings"]),
            },
        })
    summary = summary.drop(columns=["geometry", "label"])
    return {"type": "FeatureCollection", "features": features}, summary


def india_mask_geojson(india):
    """A world-sized polygon with India cut out, to dim everything outside India."""
    world = box(40, -10, 130, 50)
    return mapping(world.difference(india))
