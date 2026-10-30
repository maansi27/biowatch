"""
07_build_final_dataset.py

Assembles the final ML-ready CSV from the base dataset (+ NDVI if present).

Input:
    data/processed/base_dataset_with_ndvi.csv  (preferred, if it exists)
    data/processed/base_dataset.csv            (fallback, no NDVI column)

Output:
    data/final/invasive_species_lantana_india.csv
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FINAL_DIR = ROOT / "data" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = FINAL_DIR / "invasive_species_lantana_india.csv"

WITH_NDVI = PROCESSED / "base_dataset_with_ndvi.csv"
WITHOUT_NDVI = PROCESSED / "base_dataset.csv"


def main():
    if WITH_NDVI.exists():
        src = WITH_NDVI
        print(f"Using dataset WITH NDVI: {src}")
    elif WITHOUT_NDVI.exists():
        src = WITHOUT_NDVI
        print(f"Using dataset WITHOUT NDVI: {src}")
        print("(Run 06b_add_ndvi.py first if you want NDVI included.)")
    else:
        raise SystemExit("No base dataset found. Run 06_extract_environmental_variables.py first.")

    df = pd.read_csv(src)

    cols = ["latitude", "longitude", "year", "temperature", "rainfall", "elevation"]
    if "NDVI" in df.columns:
        cols.append("NDVI")
    cols += ["species_present", "point_type"]

    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[cols]
    df.to_csv(OUT_CSV, index=False)

    print(f"\nFinal dataset saved to {OUT_CSV}")
    print(f"Rows: {len(df)}   Columns: {list(df.columns)}")
    print(f"\nClass balance:\n{df['point_type'].value_counts()}")


if __name__ == "__main__":
    main()
