"""
train_pipeline.py

End-to-end training script. Run this once before launching the dashboard.

Phases:
  1. Extract all WorldClim features (builds full feature dataset)
  2. Preprocess and train LR, RF, XGBoost models
  3. Run suitability prediction grid for India

Usage:
    python train_pipeline.py
    python train_pipeline.py --skip-features   # if features already extracted
    python train_pipeline.py --skip-predict    # skip grid prediction step
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Lantana camara ML training pipeline")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature extraction (use existing final dataset)")
    parser.add_argument("--skip-predict",  action="store_true",
                        help="Skip suitability grid prediction")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Phase 1 – Feature extraction                                         #
    # ------------------------------------------------------------------ #
    if not args.skip_features:
        log.info("=" * 60)
        log.info("PHASE 1: Extracting environmental features from WorldClim rasters")
        log.info("=" * 60)
        from src.feature_extraction import build_full_dataset
        build_full_dataset()
    else:
        log.info("Skipping feature extraction (--skip-features flag set).")

    # ------------------------------------------------------------------ #
    # Phase 2 – Preprocessing + model training                             #
    # ------------------------------------------------------------------ #
    log.info("=" * 60)
    log.info("PHASE 2: Preprocessing data and training models")
    log.info("=" * 60)
    from src.data_processing import prepare_modelling_data
    from src.train_models import train_all_models

    data = prepare_modelling_data()
    log.info("Features used: %s", data["features"])
    results = train_all_models(data)
    log.info("Best model: %s", results["best_model_name"])

    # ------------------------------------------------------------------ #
    # Phase 3 – Suitability grid prediction                                #
    # ------------------------------------------------------------------ #
    if not args.skip_predict:
        log.info("=" * 60)
        log.info("PHASE 3: Generating habitat suitability prediction grid")
        log.info("=" * 60)
        from src.predict_suitability import predict_grid
        predict_grid()
    else:
        log.info("Skipping suitability prediction (--skip-predict flag set).")

    log.info("=" * 60)
    log.info("Training pipeline complete.")
    log.info("Run: streamlit run app.py  to launch the dashboard.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
