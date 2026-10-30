"""
src/data_processing.py

Reproducible preprocessing pipeline for Lantana camara occurrence data.
Loads the final dataset, validates coordinates, removes duplicates,
drops NaN rows in features, and returns train/test splits.

Design notes
------------
- Background points are pseudo-absences, NOT confirmed absences.
- We use stratified splits on the label, plus spatial block CV in
  train_models.py for a less optimistic AUC estimate.
- All feature scaling is done here via a Pipeline so the scaler is
  bundled with the model artifact and cannot leak test statistics.
"""

import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# India bounding box (conservative)
INDIA_LAT = (6.0, 37.0)
INDIA_LON = (67.0, 98.0)

# All 19 WorldClim bioclimatic variables + elevation available in this project
ALL_FEATURES = [
    "temperature",   # bio_1  – Annual Mean Temperature
    "rainfall",      # bio_12 – Annual Precipitation
    "elevation",     # elevation
    # Extended bioclimatic features (populated by feature_extraction.py)
    "bio_2", "bio_3", "bio_4", "bio_5", "bio_6", "bio_7",
    "bio_8", "bio_9", "bio_10", "bio_11",
    "bio_13", "bio_14", "bio_15", "bio_16", "bio_17", "bio_18", "bio_19",
]

SEED = 42


def load_dataset(csv_path: Path = None) -> pd.DataFrame:
    """Load the final ML dataset and return a validated DataFrame."""
    if csv_path is None:
        csv_path = ROOT / "data" / "final" / "invasive_species_lantana_india.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Run scripts/07_build_final_dataset.py first."
        )
    df = pd.read_csv(csv_path)
    log.info("Loaded %d rows from %s", len(df), csv_path.name)
    return df


def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing or out-of-India coordinates."""
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    in_india = (
        df["latitude"].between(*INDIA_LAT)
        & df["longitude"].between(*INDIA_LON)
    )
    df = df[in_india].copy()
    log.info(
        "Coordinate validation: %d -> %d rows (removed %d)",
        before, len(df), before - len(df),
    )
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows (same lat, lon, label)."""
    before = len(df)
    df = df.drop_duplicates(subset=["latitude", "longitude"])
    log.info(
        "Duplicate removal: %d -> %d rows (removed %d)",
        before, len(df), before - len(df),
    )
    return df


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Return the subset of ALL_FEATURES that are actually present in df."""
    available = [f for f in ALL_FEATURES if f in df.columns]
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        log.info("Features not found in dataset (will be skipped): %s", missing)
    log.info("Using features: %s", available)
    return df, available


def prepare_modelling_data(
    csv_path: Path = None,
    test_size: float = 0.2,
) -> dict:
    """
    Full preprocessing pipeline.

    Returns a dict with:
        X_train, X_test, y_train, y_test  – numpy arrays
        features                           – list of feature names
        df_clean                           – cleaned DataFrame
        df_train, df_test                  – DataFrames with lat/lon retained
    """
    df = load_dataset(csv_path)
    df = validate_coordinates(df)
    df = remove_duplicates(df)
    df, features = select_features(df)

    # Drop rows where any selected feature is NaN
    before = len(df)
    df = df.dropna(subset=features + ["species_present"]).reset_index(drop=True)
    log.info(
        "NaN drop: %d -> %d rows (removed %d with missing feature values)",
        before, len(df), before - len(df),
    )

    if len(df) < 50:
        raise ValueError(
            f"Only {len(df)} rows remain after cleaning – too few to train."
        )

    X = df[features].values
    y = df["species_present"].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(df)),
        test_size=test_size,
        random_state=SEED,
        stratify=y,
    )

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)

    log.info(
        "Train: %d rows (presence=%d, background=%d)",
        len(y_train), y_train.sum(), (y_train == 0).sum(),
    )
    log.info(
        "Test:  %d rows (presence=%d, background=%d)",
        len(y_test), y_test.sum(), (y_test == 0).sum(),
    )

    return dict(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        features=features,
        df_clean=df,
        df_train=df_train,
        df_test=df_test,
    )


def make_spatial_blocks(df: pd.DataFrame, block_size: float = 1.0) -> pd.Series:
    """Assign each row to a 1-degree spatial block (for cross-validation)."""
    bx = (df["longitude"] // block_size).astype(int)
    by = (df["latitude"] // block_size).astype(int)
    return bx.astype(str) + "_" + by.astype(str)


if __name__ == "__main__":
    data = prepare_modelling_data()
    print(f"Features: {data['features']}")
    print(f"Train shape: {data['X_train'].shape}")
    print(f"Test shape:  {data['X_test'].shape}")
