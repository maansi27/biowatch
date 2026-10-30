"""
src/feature_extraction.py

Extract all available WorldClim bioclimatic variables and elevation
for every presence and background point, and rebuild the final dataset
with the full feature set.

WorldClim 2.1 bioclimatic variables (bio_1 through bio_19):
    bio_1   Annual Mean Temperature
    bio_2   Mean Diurnal Range
    bio_3   Isothermality
    bio_4   Temperature Seasonality
    bio_5   Max Temperature of Warmest Month
    bio_6   Min Temperature of Coldest Month
    bio_7   Temperature Annual Range
    bio_8   Mean Temperature of Wettest Quarter
    bio_9   Mean Temperature of Driest Quarter
    bio_10  Mean Temperature of Warmest Quarter
    bio_11  Mean Temperature of Coldest Quarter
    bio_12  Annual Precipitation
    bio_13  Precipitation of Wettest Month
    bio_14  Precipitation of Driest Month
    bio_15  Precipitation Seasonality
    bio_16  Precipitation of Wettest Quarter
    bio_17  Precipitation of Driest Quarter
    bio_18  Precipitation of Warmest Quarter
    bio_19  Precipitation of Coldest Quarter

No values are fabricated. If a raster file is absent, that variable is
skipped with a warning.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import rasterio

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
WC_DIR = ROOT / "data" / "raw" / "worldclim"
FINAL_DIR = ROOT / "data" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = FINAL_DIR / "invasive_species_lantana_india.csv"

# Map column name -> raster filename
RASTER_MAP: Dict[str, str] = {
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


def extract_raster_values(
    df: pd.DataFrame, raster_path: Path, col_name: str
) -> pd.Series:
    """Sample a raster at each (longitude, latitude) in df."""
    with rasterio.open(raster_path) as src:
        coords = list(zip(df["longitude"], df["latitude"]))
        sampled = [v[0] for v in src.sample(coords)]
        nodata = src.nodata
    vals = pd.Series(sampled, index=df.index, dtype="float64")
    if nodata is not None:
        vals = vals.replace(nodata, np.nan)
    return vals


def build_full_dataset(
    presence_csv: Optional[Path] = None,
    background_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load presence and background points, extract all available
    WorldClim variables, and save the result to data/final/.
    """
    if presence_csv is None:
        presence_csv = PROCESSED / "presence_points.csv"
    if background_csv is None:
        background_csv = PROCESSED / "background_points.csv"

    for p in [presence_csv, background_csv]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Run the earlier data-preparation scripts first."
            )

    presence = pd.read_csv(presence_csv)
    presence["point_type"] = "presence"
    background = pd.read_csv(background_csv)

    combined = pd.concat(
        [
            presence[["latitude", "longitude", "year", "species_present", "point_type"]],
            background[["latitude", "longitude", "year", "species_present", "point_type"]],
        ],
        ignore_index=True,
    )
    log.info("Combined points to extract: %d", len(combined))

    available_rasters = {}
    missing_rasters = []
    for col, fname in RASTER_MAP.items():
        path = WC_DIR / fname
        if path.exists():
            available_rasters[col] = path
        else:
            missing_rasters.append(fname)

    if missing_rasters:
        log.warning("Raster files not found (skipped): %s", missing_rasters)

    for col, rpath in available_rasters.items():
        log.info("Extracting %-15s from %s ...", col, rpath.name)
        combined[col] = extract_raster_values(combined, rpath, col)
        n_nan = combined[col].isna().sum()
        if n_nan:
            log.info("  -> %d NaN values (ocean / no-data pixels)", n_nan)

    combined.to_csv(OUT_CSV, index=False)
    log.info("Full feature dataset saved -> %s  (%d rows)", OUT_CSV, len(combined))

    feature_cols = list(available_rasters.keys())
    log.info("Feature columns: %s", feature_cols)
    log.info("\nFeature summary:\n%s", combined[feature_cols].describe().to_string())
    return combined


if __name__ == "__main__":
    build_full_dataset()
