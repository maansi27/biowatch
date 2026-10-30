"""
06_extract_environmental_variables.py

For every presence and background point, extract the real raster value
(no synthetic/fabricated values) at that lat/lon from the WorldClim rasters
downloaded in step 05.

Input:
    data/processed/presence_points.csv
    data/processed/background_points.csv
    data/raw/worldclim/wc2.1_2.5m_bio_1.tif   (temperature)
    data/raw/worldclim/wc2.1_2.5m_bio_12.tif  (precipitation)
    data/raw/worldclim/wc2.1_2.5m_elev.tif    (elevation)

Output:
    data/processed/base_dataset.csv
    columns: latitude, longitude, year, species_present, point_type,
             temperature, rainfall, elevation
    (NDVI column is added separately by 06b_add_ndvi.py once available)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
WC_DIR = ROOT / "data" / "raw" / "worldclim"
OUT_CSV = PROCESSED / "base_dataset.csv"

RASTERS = {
    "temperature": WC_DIR / "wc2.1_2.5m_bio_1.tif",
    "rainfall": WC_DIR / "wc2.1_2.5m_bio_12.tif",
    "elevation": WC_DIR / "wc2.1_2.5m_elev.tif",
}


def extract_values(df: pd.DataFrame, raster_path: Path, col_name: str) -> pd.Series:
    with rasterio.open(raster_path) as src:
        coords = list(zip(df["longitude"], df["latitude"]))
        sampled = [v[0] for v in src.sample(coords)]
        nodata = src.nodata
    vals = pd.Series(sampled, index=df.index, dtype="float64")
    if nodata is not None:
        vals = vals.replace(nodata, np.nan)
    return vals


def main():
    presence_csv = PROCESSED / "presence_points.csv"
    background_csv = PROCESSED / "background_points.csv"
    if not presence_csv.exists() or not background_csv.exists():
        raise SystemExit("Run 03_clean_occurrences.py and 04_generate_background.py first.")

    missing = [name for name, path in RASTERS.items() if not path.exists()]
    if missing:
        raise SystemExit(
            f"Missing raster(s) for: {missing}. Run 05_download_environmental_data.py "
            "(or place the files manually) first."
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

    print(f"Total points to extract environmental values for: {len(combined)}")

    for col_name, raster_path in RASTERS.items():
        print(f"Extracting {col_name} from {raster_path.name} ...")
        combined[col_name] = extract_values(combined, raster_path, col_name)
        n_missing = combined[col_name].isna().sum()
        print(f"  -> {n_missing} points had no data at this raster location (NaN).")

    # WorldClim bio_1 is in deg C already (v2.1); no scaling needed.
    combined.to_csv(OUT_CSV, index=False)
    print(f"\nSaved combined base dataset ({len(combined)} rows) to {OUT_CSV}")
    print("\nColumn summary:")
    print(combined[["temperature", "rainfall", "elevation"]].describe())


if __name__ == "__main__":
    main()
