"""
11_export_web_data.py – Export pipeline outputs as JSON for the React website.

Writes everything the BioWatch web app needs into web/public/data/:
  summary.json        headline counts and best model
  india.json          India outline and outside-India mask
  zones.json          suitability-class zones (dissolved polygons)
  hotspots.json       hotspot zones for each threshold (0.5 – 0.8)
  sightings.json      occurrence coordinates
  metrics.json        model metrics, ROC curves, CV folds, feature importance
  distributions.json  presence vs background histograms per feature
  priority.json       monitoring-priority tier zones for several top-N values

Run after train_pipeline.py:
    python scripts/11_export_web_data.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import box, mapping
from shapely.ops import unary_union
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.zones import (  # noqa: E402
    build_class_zones,
    build_hotspot_zones,
    india_mask_geojson,
    load_india_boundary,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

OUTPUTS     = ROOT / "outputs"
DATA_FINAL  = ROOT / "data" / "final" / "invasive_species_lantana_india.csv"
DATA_PRES   = ROOT / "data" / "processed" / "presence_points.csv"
SUITABILITY = OUTPUTS / "suitability_grid.csv"
WEB_DATA    = ROOT / "web" / "public" / "data"

HOTSPOT_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
PRIORITY_TOP_N     = list(range(500, 5001, 500))
TIER_NAMES         = ["Watch", "Elevated", "High", "Urgent"]


def round_coords(obj, nd=3):
    """Round every coordinate in a GeoJSON-like structure to keep files small."""
    if isinstance(obj, dict):
        return {k: round_coords(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (int, float)):
            return [round(float(v), nd) for v in obj]
        return [round_coords(v, nd) for v in obj]
    return obj


def write(name, payload):
    path = WEB_DATA / name
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"), allow_nan=False)
    log.info("Wrote %-20s %6.0f KB", name, path.stat().st_size / 1024)


def clean(v):
    """Replace NaN / inf with None so the JSON stays valid."""
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def downsample(xs, ys, n=150):
    if len(xs) <= n:
        return xs, ys
    idx = np.unique(np.linspace(0, len(xs) - 1, n).astype(int))
    return [xs[i] for i in idx], [ys[i] for i in idx]


def main():
    WEB_DATA.mkdir(parents=True, exist_ok=True)

    df_all  = pd.read_csv(DATA_FINAL)
    df_pres = pd.read_csv(DATA_PRES)
    df_suit = pd.read_csv(SUITABILITY)
    with open(OUTPUTS / "model_metrics.json") as f:
        metrics = json.load(f)
    india = load_india_boundary()

    # ── India outline and mask ──────────────────────────────────────────────
    write("india.json", round_coords({
        "outline": mapping(india),
        "mask": india_mask_geojson(india),
    }))

    # ── Suitability zones ───────────────────────────────────────────────────
    zones = build_class_zones(df_suit, india)
    for f in zones["features"]:
        f["properties"] = {k: clean(v) for k, v in f["properties"].items()}
    write("zones.json", round_coords(zones))

    # ── Hotspot zones per threshold ─────────────────────────────────────────
    hotspots = {}
    for t in HOTSPOT_THRESHOLDS:
        fc, table = build_hotspot_zones(df_suit, india, df_pres, threshold=t)
        hotspots[str(t)] = {
            "features": round_coords(fc)["features"],
            "table": [
                {
                    "id": r["Zone"], "near": r["Near"], "area": int(r["Area (km²)"]),
                    "mean": round(r["Mean Score"], 4), "peak": round(r["Peak Score"], 4),
                    "veryHigh": int(r["Very High Cells"]), "sightings": int(r["Recorded Sightings"]),
                    "lat": round(r["lat"], 4), "lon": round(r["lon"], 4),
                }
                for _, r in table.iterrows()
            ],
        }
        log.info("Threshold %.1f: %d hotspot zones", t, len(table))
    write("hotspots.json", hotspots)

    # ── Sightings ───────────────────────────────────────────────────────────
    pts = df_pres[["latitude", "longitude"]].dropna().round(4).values.tolist()
    write("sightings.json", pts)

    # ── Model metrics ───────────────────────────────────────────────────────
    models = []
    for name, m in metrics.items():
        fpr, tpr = downsample(m["roc_curve"]["fpr"], m["roc_curve"]["tpr"])
        cv = m.get("spatial_cv", {})
        fi = m.get("feature_importance") or {}
        models.append({
            "name": name,
            "auc": m["roc_auc"], "precision": m["precision"],
            "recall": m["recall"], "f1": m["f1_score"],
            "cvMean": clean(cv.get("mean_auc")), "cvStd": clean(cv.get("std_auc")),
            "cvFolds": [clean(v) for v in cv.get("fold_aucs", [])],
            "confusion": m["confusion_matrix"],
            "roc": [{"fpr": round(a, 4), "tpr": round(b, 4)} for a, b in zip(fpr, tpr)],
            "importance": {
                "type": fi.get("importance_type", ""),
                "items": [{"feature": f, "value": v}
                          for f, v in zip(fi.get("features", []), fi.get("importances", []))],
            },
        })
    write("metrics.json", models)

    # ── Feature distributions ───────────────────────────────────────────────
    feat_cols = [c for c in df_all.columns
                 if c not in ("latitude", "longitude", "year", "species_present", "point_type")]
    dists = {}
    for col in feat_cols:
        vals = df_all[col].dropna()
        edges = np.histogram_bin_edges(vals, bins=30)
        pres = np.histogram(df_all.loc[df_all["point_type"] == "presence", col].dropna(), bins=edges)[0]
        back = np.histogram(df_all.loc[df_all["point_type"] == "background", col].dropna(), bins=edges)[0]
        dists[col] = [
            {"x": round(float((edges[i] + edges[i + 1]) / 2), 2),
             "presence": int(pres[i]), "background": int(back[i])}
            for i in range(len(pres))
        ]
    write("distributions.json", dists)

    # ── Monitoring priority ─────────────────────────────────────────────────
    suit = df_suit[df_suit["suitability_score"].notna()].copy()
    tree = BallTree(np.deg2rad(df_pres[["latitude", "longitude"]].dropna().values), metric="haversine")
    d_km = tree.query(np.deg2rad(suit[["latitude", "longitude"]].values), k=1)[0][:, 0] * 6371.0
    suit["proximity"] = np.exp(-d_km / 150)
    suit["priority"] = 0.7 * suit["suitability_score"] + 0.3 * suit["proximity"]

    import shapely
    suit = suit[shapely.contains_xy(india, suit["longitude"], suit["latitude"])]
    ranked = suit.sort_values("priority", ascending=False)

    hist_edges = np.linspace(0, 1, 41)
    hist = np.histogram(suit["priority"], bins=hist_edges)[0]

    half = 0.05
    by_n = {}
    for n in PRIORITY_TOP_N:
        top = ranked.head(n)
        edges = np.quantile(top["priority"], [0, 0.25, 0.5, 0.75, 1.0])
        tiers = np.clip(np.searchsorted(edges, top["priority"], side="right") - 1, 0, 3)
        feats = []
        for t in range(4):
            sub = top[tiers == t]
            if sub.empty:
                continue
            geom = unary_union([box(lo - half, la - half, lo + half, la + half)
                                for la, lo in zip(sub["latitude"], sub["longitude"])]).intersection(india)
            feats.append({
                "type": "Feature",
                "geometry": round_coords(mapping(geom.simplify(0.02, preserve_topology=True))),
                "properties": {"tier": TIER_NAMES[t], "rank": t, "cells": int(len(sub)),
                               "min": round(float(edges[t]), 3), "max": round(float(edges[t + 1]), 3)},
            })
        by_n[str(n)] = feats
    write("priority.json", {
        "histogram": [{"x": round(float((hist_edges[i] + hist_edges[i + 1]) / 2), 3), "count": int(hist[i])}
                      for i in range(len(hist))],
        "byTopN": by_n,
    })

    # ── Summary ─────────────────────────────────────────────────────────────
    best = max(models, key=lambda m: m["auc"])
    pres_rows = df_all[df_all["species_present"] == 1]
    write("summary.json", {
        "totalRecords": int(len(df_all)),
        "presenceRecords": int(len(pres_rows)),
        "uniqueLocations": int(pres_rows[["latitude", "longitude"]].drop_duplicates().shape[0]),
        "features": len(feat_cols),
        "gridCells": int(len(df_suit)),
        "bestModel": best["name"],
        "bestAuc": best["auc"],
        "models": [m["name"] for m in models],
        "classShare": [{"zone": f["properties"]["zone"], "share": round(f["properties"]["share"], 2),
                        "cells": f["properties"]["cells"]} for f in zones["features"]],
    })


if __name__ == "__main__":
    main()
