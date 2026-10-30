"""
tests/test_preprocessing.py

Basic tests for data_processing.py and predict_suitability.py.
Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_processing import validate_coordinates, remove_duplicates, make_spatial_blocks


# ── validate_coordinates ─────────────────────────────────────────────────────

def test_validate_coordinates_removes_nan():
    df = pd.DataFrame({
        "latitude":  [20.0, None, 15.0],
        "longitude": [80.0, 75.0, None],
        "species_present": [1, 0, 1],
    })
    result = validate_coordinates(df)
    assert len(result) == 1
    assert result.iloc[0]["latitude"] == 20.0


def test_validate_coordinates_removes_out_of_india():
    df = pd.DataFrame({
        "latitude":  [20.0, 5.0, 40.0, 20.0],   # 5 and 40 outside India
        "longitude": [80.0, 80.0, 80.0, 60.0],  # 60 outside India lon
        "species_present": [1, 1, 1, 1],
    })
    result = validate_coordinates(df)
    assert len(result) == 1
    assert result.iloc[0]["latitude"] == 20.0


def test_validate_coordinates_preserves_valid():
    df = pd.DataFrame({
        "latitude":  [8.5, 20.0, 35.0],
        "longitude": [68.0, 78.5, 97.0],
        "species_present": [1, 0, 1],
    })
    result = validate_coordinates(df)
    assert len(result) == 3


# ── remove_duplicates ─────────────────────────────────────────────────────────

def test_remove_duplicates_basic():
    df = pd.DataFrame({
        "latitude":  [20.0, 20.0, 15.0],
        "longitude": [80.0, 80.0, 75.0],
        "species_present": [1, 1, 0],
    })
    result = remove_duplicates(df)
    assert len(result) == 2


def test_remove_duplicates_different_labels_kept():
    """Same coordinates but different labels -> not a duplicate."""
    df = pd.DataFrame({
        "latitude":  [20.0, 20.0],
        "longitude": [80.0, 80.0],
        "species_present": [1, 0],
    })
    result = remove_duplicates(df)
    assert len(result) == 2


# ── make_spatial_blocks ───────────────────────────────────────────────────────

def test_spatial_blocks_shape():
    df = pd.DataFrame({
        "latitude":  [20.1, 20.9, 21.5],
        "longitude": [80.2, 80.8, 81.0],
    })
    blocks = make_spatial_blocks(df)
    assert len(blocks) == 3


def test_spatial_blocks_same_block():
    df = pd.DataFrame({
        "latitude":  [20.1, 20.9],
        "longitude": [80.1, 80.9],
    })
    blocks = make_spatial_blocks(df, block_size=1.0)
    # Both fall in the 20_80 block
    assert blocks.iloc[0] == blocks.iloc[1]


def test_spatial_blocks_different_blocks():
    df = pd.DataFrame({
        "latitude":  [20.5, 21.5],
        "longitude": [80.5, 81.5],
    })
    blocks = make_spatial_blocks(df, block_size=1.0)
    assert blocks.iloc[0] != blocks.iloc[1]


# ── Integration: load real dataset (skip if not present) ─────────────────────

@pytest.mark.skipif(
    not (ROOT / "data" / "final" / "invasive_species_lantana_india.csv").exists(),
    reason="Final dataset not found",
)
def test_load_and_validate_real_data():
    from src.data_processing import load_dataset, validate_coordinates
    df = load_dataset()
    assert len(df) > 0
    assert "latitude" in df.columns
    assert "longitude" in df.columns
    assert "species_present" in df.columns

    df_val = validate_coordinates(df)
    # All remaining points should be within India's bounds
    assert (df_val["latitude"] >= 6.0).all()
    assert (df_val["latitude"] <= 37.0).all()
    assert (df_val["longitude"] >= 67.0).all()
    assert (df_val["longitude"] <= 98.0).all()


@pytest.mark.skipif(
    not (ROOT / "data" / "final" / "invasive_species_lantana_india.csv").exists(),
    reason="Final dataset not found",
)
def test_feature_columns_present():
    from src.data_processing import load_dataset
    df = load_dataset()
    # At minimum these three should be present from the existing pipeline
    for col in ["temperature", "rainfall", "elevation"]:
        assert col in df.columns, f"Missing expected column: {col}"


@pytest.mark.skipif(
    not (ROOT / "models" / "best_model.pkl").exists(),
    reason="No trained model found",
)
def test_model_loads_and_predicts():
    import pickle
    import numpy as np
    with open(ROOT / "models" / "best_model.pkl", "rb") as f:
        artifact = pickle.load(f)
    pipe     = artifact["pipeline"]
    features = artifact["features"]
    # Dummy input: all-zero feature vector
    X_dummy = np.zeros((1, len(features)))
    proba = pipe.predict_proba(X_dummy)
    assert proba.shape == (1, 2)
    assert abs(proba.sum(axis=1)[0] - 1.0) < 1e-6
