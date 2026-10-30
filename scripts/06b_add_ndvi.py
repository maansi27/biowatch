"""
06b_add_ndvi.py  (OPTIONAL - run after 06_extract_environmental_variables.py)

NDVI cannot be auto-downloaded unattended (Sentinel-2/MODIS both require
authenticated, area-and-date-specific queries), so this script does not
run automatically as part of the core pipeline. It documents the exact
manual workflow and then does the (real, non-fabricated) raster extraction
once you have the file.

RECOMMENDED SOURCE: MODIS MOD13Q1 v6.1 (250 m, 16-day NDVI composite)
via NASA Earthdata / AppEEARS, because it is a single authenticated
download rather than mosaicking many individual Sentinel-2 scenes:

  1. Create a free NASA Earthdata account: https://urs.earthdata.nasa.gov/
  2. Go to AppEEARS: https://appeears.earthdatacloud.nasa.gov/
  3. Submit an "Area" request:
       - Product: MOD13Q1.061, layer "_250m_16_days_NDVI"
       - Region: India boundary (upload the Natural Earth shapefile from
         data/raw/natural_earth/, or draw the bounding box
         lat 6-37N, lon 68-97E)
       - Date range: match your presence-point years, e.g. 2015-01-01 to
         2024-12-31, or a single representative dry-season composite
       - Output format: GeoTIFF
  4. Download the resulting GeoTIFF(s) and place under:
       data/raw/ndvi/modis_ndvi_india.tif
     (If you downloaded multiple dates, either pick one representative
     composite or extend the extraction below to average across dates.)

ALTERNATIVE: Sentinel-2 via Copernicus Data Space Ecosystem
  https://dataspace.copernicus.eu/ - browse/download L2A tiles covering
  India, compute NDVI = (NIR - Red) / (NIR + Red) per tile with rasterio,
  then mosaic. This gives higher resolution (10m) but requires stitching
  many tiles/dates - more effort than MODIS, so only recommended for a
  later, higher-resolution iteration of this project.

NDVI values here range roughly -1 to 1 (unitless); MOD13Q1 stores it
scaled by 10000, which this script corrects for automatically.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parent.parent
BASE_CSV = ROOT / "data" / "processed" / "base_dataset.csv"
NDVI_RASTER = ROOT / "data" / "raw" / "ndvi" / "modis_ndvi_india.tif"
OUT_CSV = ROOT / "data" / "processed" / "base_dataset_with_ndvi.csv"

# Set True if your raster stores NDVI as MODIS's integer-scaled format
# (value = NDVI * 10000). MOD13Q1 uses this scaling; a raw Sentinel-2-derived
# NDVI GeoTIFF you computed yourself typically will NOT need this.
MODIS_SCALED = True


def main():
    if not BASE_CSV.exists():
        raise SystemExit("Run 06_extract_environmental_variables.py first.")
    if not NDVI_RASTER.exists():
        raise SystemExit(
            f"{NDVI_RASTER} not found.\n"
            "Follow the manual download instructions in this file's "
            "docstring, place the GeoTIFF at the path above, then re-run."
        )

    df = pd.read_csv(BASE_CSV)

    with rasterio.open(NDVI_RASTER) as src:
        coords = list(zip(df["longitude"], df["latitude"]))
        sampled = [v[0] for v in src.sample(coords)]
        nodata = src.nodata

    ndvi = pd.Series(sampled, index=df.index, dtype="float64")
    if nodata is not None:
        ndvi = ndvi.replace(nodata, np.nan)
    if MODIS_SCALED:
        ndvi = ndvi / 10000.0

    df["NDVI"] = ndvi
    n_missing = df["NDVI"].isna().sum()
    print(f"Extracted NDVI for {len(df)} points; {n_missing} missing (NaN).")
    print(df["NDVI"].describe())

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved dataset with NDVI to {OUT_CSV}")
    print(
        "Point 07_build_final_dataset.py at this file (instead of "
        "base_dataset.csv) if you want NDVI included in the final CSV."
    )


if __name__ == "__main__":
    main()
