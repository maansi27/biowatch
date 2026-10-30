"""
src/predict_suitability.py

Generate a habitat suitability raster and prediction grid for India
using the best trained model and WorldClim environmental rasters.

Suitability classes (configurable thresholds):
    Very Low    [0.00, 0.20)
    Low         [0.20, 0.40)
    Moderate    [0.40, 0.60)
    High        [0.60, 0.80)
    Very High   [0.80, 1.00]

Output files:
    outputs/suitability_grid.csv        – point grid with scores
    outputs/suitability_raster.tif      – GeoTIFF raster (if rasterio write works)

Notes
-----
* Predictions are model suitability scores in [0, 1], not calibrated
  ecological probabilities. Do not interpret them as invasion likelihoods.
* Thresholds are heuristic quantile breaks documented below and are
  configurable via THRESHOLDS dict.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR  = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
WC_DIR      = ROOT / "data" / "raw" / "worldclim"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# India bounding box
INDIA_LAT = (6.0, 37.0)
INDIA_LON = (67.0, 98.0)

# Grid resolution in degrees (0.1 ~ 11 km; lower = faster but coarser)
GRID_RES = 0.1

# Configurable suitability class thresholds
THRESHOLDS = {
    "Very Low":   (0.00, 0.20),
    "Low":        (0.20, 0.40),
    "Moderate":   (0.40, 0.60),
    "High":       (0.60, 0.80),
    "Very High":  (0.80, 1.01),
}

RASTER_MAP = {
    "temperature": "wc2.1_2.5m_bio_1.tif",
    "bio_2":       "wc2.1_2.5m_bio_2.tif",
    "bio_3":       "wc2.1_2.5m_bio_3.tif",
    "bio_4":       "wc2.1_2.5m_bio_4.tif",
    "bio_5":       "wc2.1_2.5m_bio_5.tif",
    "bio_6":       "wc2.1_2.5m_bio_6.tif",
    "bio_7":       "wc2.1_2.5m_bio_7.tif",
    "bio_8":       "wc2.1_2.5m_bio_8.tif",
    "bio_9":       "wc2.1_2.5m_bio_9.tif",
    "bio_10":      "wc2.1_2.5m_bio_10.tif",
    "bio_11":      "wc2.1_2.5m_bio_11.tif",
    "rainfall":    "wc2.1_2.5m_bio_12.tif",
    "bio_13":      "wc2.1_2.5m_bio_13.tif",
    "bio_14":      "wc2.1_2.5m_bio_14.tif",
    "bio_15":      "wc2.1_2.5m_bio_15.tif",
    "bio_16":      "wc2.1_2.5m_bio_16.tif",
    "bio_17":      "wc2.1_2.5m_bio_17.tif",
    "bio_18":      "wc2.1_2.5m_bio_18.tif",
    "bio_19":      "wc2.1_2.5m_bio_19.tif",
    "elevation":   "wc2.1_2.5m_elev.tif",
}


def load_best_model() -> tuple:
    """Load the best model pipeline and its feature list."""
    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}.\n"
            "Run train_pipeline.py first."
        )
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    return artifact["pipeline"], artifact["features"], artifact.get("model_name", "Unknown")


def build_india_grid(resolution: float = GRID_RES) -> pd.DataFrame:
    """Create a regular lat/lon grid covering India's bounding box."""
    lats = np.arange(INDIA_LAT[0], INDIA_LAT[1], resolution)
    lons = np.arange(INDIA_LON[0], INDIA_LON[1], resolution)
    lon_g, lat_g = np.meshgrid(lons, lats)
    grid = pd.DataFrame({
        "latitude":  lat_g.ravel(),
        "longitude": lon_g.ravel(),
    })
    log.info("India grid: %d cells at %.2f-degree resolution", len(grid), resolution)
    return grid


def extract_grid_features(
    grid: pd.DataFrame, features: list, batch_size: int = 50_000
) -> pd.DataFrame:
    """Sample WorldClim rasters at every grid cell in memory-efficient batches."""
    raster_handles = {}
    for feat in features:
        fname = RASTER_MAP.get(feat)
        if fname is None:
            log.warning("No raster mapping found for feature '%s'", feat)
            continue
        fpath = WC_DIR / fname
        if not fpath.exists():
            log.warning("Raster file missing for feature '%s': %s", feat, fpath)
            continue
        raster_handles[feat] = rasterio.open(fpath)

    if not raster_handles:
        raise RuntimeError(
            "No valid raster files found for any of the model's features. "
            "Cannot generate suitability predictions."
        )

    n = len(grid)
    result_cols = {feat: np.full(n, np.nan) for feat in raster_handles}

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = grid.iloc[start:end]
        coords = list(zip(batch["longitude"], batch["latitude"]))
        for feat, src in raster_handles.items():
            sampled = [v[0] for v in src.sample(coords)]
            arr = np.array(sampled, dtype=float)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            result_cols[feat][start:end] = arr

        if start % 200_000 == 0:
            log.info("  Extracted %d / %d cells ...", end, n)

    for src in raster_handles.values():
        src.close()

    for feat, vals in result_cols.items():
        grid[feat] = vals

    log.info("Feature extraction complete. Missing values per feature:")
    for feat in raster_handles:
        log.info("  %-15s  %d NaN", feat, grid[feat].isna().sum())

    return grid


def assign_suitability_class(score: float) -> str:
    """Map a suitability score to a class label."""
    for cls, (lo, hi) in THRESHOLDS.items():
        if lo <= score < hi:
            return cls
    return "Very High"


def predict_grid(
    resolution: float = GRID_RES,
    save_raster: bool = True,
) -> pd.DataFrame:
    """Full pipeline: build grid -> extract features -> predict -> save outputs."""
    pipe, features, model_name = load_best_model()
    log.info("Loaded model: %s  |  features: %s", model_name, features)

    grid = build_india_grid(resolution)
    grid = extract_grid_features(grid, features)

    # Rows where all required features are available
    valid_mask = grid[features].notna().all(axis=1)
    n_valid = valid_mask.sum()
    log.info(
        "%d / %d grid cells have complete feature data (%.1f%%)",
        n_valid, len(grid), 100 * n_valid / len(grid),
    )

    grid["suitability_score"] = np.nan
    grid["suitability_class"] = "No Data"

    if n_valid > 0:
        X_pred = grid.loc[valid_mask, features].values
        scores = pipe.predict_proba(X_pred)[:, 1]
        grid.loc[valid_mask, "suitability_score"] = scores
        grid.loc[valid_mask, "suitability_class"] = [
            assign_suitability_class(s) for s in scores
        ]

    # Save CSV
    out_csv = OUTPUTS_DIR / "suitability_grid.csv"
    grid.to_csv(out_csv, index=False)
    log.info("Suitability grid saved -> %s", out_csv)

    # Save class distribution
    class_dist = grid["suitability_class"].value_counts().to_dict()
    log.info("Suitability class distribution: %s", class_dist)

    with open(OUTPUTS_DIR / "suitability_stats.json", "w") as f:
        json.dump({
            "model_name": model_name,
            "features": features,
            "thresholds": THRESHOLDS,
            "class_distribution": class_dist,
            "grid_resolution_deg": resolution,
            "total_cells": int(len(grid)),
            "valid_cells": int(n_valid),
        }, f, indent=2)

    # Save GeoTIFF raster
    if save_raster and n_valid > 0:
        _save_geotiff(grid, resolution)

    return grid


def _save_geotiff(grid: pd.DataFrame, resolution: float):
    """Write the suitability scores as a GeoTIFF."""
    lats = np.arange(INDIA_LAT[0], INDIA_LAT[1], resolution)
    lons = np.arange(INDIA_LON[0], INDIA_LON[1], resolution)
    n_rows, n_cols = len(lats), len(lons)

    raster_array = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    for _, row in grid.iterrows():
        ri = int((row["latitude"]  - INDIA_LAT[0]) / resolution)
        ci = int((row["longitude"] - INDIA_LON[0]) / resolution)
        if 0 <= ri < n_rows and 0 <= ci < n_cols:
            raster_array[ri, ci] = row["suitability_score"]

    # Flip so north is up
    raster_array = np.flipud(raster_array)

    transform = from_bounds(
        INDIA_LON[0], INDIA_LAT[0], INDIA_LON[1], INDIA_LAT[1],
        n_cols, n_rows,
    )
    out_tif = OUTPUTS_DIR / "suitability_raster.tif"
    with rasterio.open(
        out_tif, "w",
        driver="GTiff",
        height=n_rows,
        width=n_cols,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(raster_array, 1)
    log.info("GeoTIFF suitability raster saved -> %s", out_tif)


if __name__ == "__main__":
    predict_grid()
