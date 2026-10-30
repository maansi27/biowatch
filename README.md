# Lantana camara Habitat Suitability Assessment

ML-based system that identifies areas in India where environmental conditions are similar to known *Lantana camara* occurrence locations, visualised on an interactive map.

## Project Overview

This project uses presence-background modelling with WorldClim bioclimatic rasters to estimate habitat suitability for *Lantana camara* across India. Three classifiers (Logistic Regression, Random Forest, XGBoost) are trained and compared using spatial block cross-validation.

**Important caveat**: Suitability scores reflect similarity to known occurrence environments, not confirmed invasion probability or ecological damage. Background points are pseudo-absences, not confirmed absences.

## Project Structure

```
invasive_species_ml/
├── data/
│   ├── raw/
│   │   ├── gbif_lantana_india.csv          GBIF occurrence download
│   │   ├── worldclim/                      WorldClim 2.1 rasters (2.5 arc-min)
│   │   └── natural_earth/                  Country boundary shapefiles
│   ├── processed/
│   │   ├── presence_points.csv             Cleaned occurrence records
│   │   └── background_points.csv           Pseudo-absence points
│   └── final/
│       └── invasive_species_lantana_india.csv  Full ML-ready dataset
├── src/
│   ├── data_processing.py                  Preprocessing pipeline
│   ├── feature_extraction.py               WorldClim raster sampling
│   ├── train_models.py                     LR / RF / XGBoost training
│   └── predict_suitability.py             India-wide suitability grid
├── models/
│   ├── best_model.pkl                      Best-performing pipeline
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── features.json                       Feature list used in training
├── outputs/
│   ├── model_metrics.json                  All evaluation metrics
│   ├── model_summary.csv                   Comparison table
│   ├── suitability_grid.csv                India prediction grid
│   ├── suitability_raster.tif              GeoTIFF output
│   └── suitability_stats.json             Class distribution stats
├── scripts/                                Original data-download scripts
├── tests/
│   └── test_preprocessing.py              Pytest unit tests
├── train_pipeline.py                       End-to-end training script
├── app.py                                  Streamlit dashboard
└── requirements.txt
```

## Data

| File | Description |
|------|-------------|
| `data/raw/gbif_lantana_india.csv` | Raw GBIF download |
| `data/processed/presence_points.csv` | 4,317 cleaned occurrence records |
| `data/processed/background_points.csv` | 4,317 pseudo-absence points (geographically random within India) |
| `data/final/invasive_species_lantana_india.csv` | Full feature dataset (temperature, rainfall, elevation, bio_2–bio_19) |
| `data/raw/worldclim/` | 20 WorldClim 2.1 rasters at 2.5 arc-minute resolution |

### Environmental Features

All features are extracted from WorldClim 2.1 (no fabricated values):

| Column | Source | Description |
|--------|--------|-------------|
| `temperature` | bio_1 | Annual Mean Temperature (°C) |
| `rainfall` | bio_12 | Annual Precipitation (mm) |
| `elevation` | elev | Elevation (m) |
| `bio_2` – `bio_19` | bio_2–19 | Full bioclimatic variable set |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify raster data

WorldClim rasters must be in `data/raw/worldclim/`. The project ships with all 20 required `.tif` files.

## Running the Pipeline

### Option A — Full pipeline (feature extraction + training + prediction)

```bash
python train_pipeline.py
```

This runs three phases:
1. Extract all WorldClim features from rasters for every presence and background point
2. Train Logistic Regression, Random Forest, and XGBoost with spatial block cross-validation
3. Generate a 0.1-degree resolution habitat suitability grid for India

Expected runtime: 10–30 minutes depending on hardware (grid prediction is the slowest step).

### Option B — Skip feature extraction (if final dataset already built)

```bash
python train_pipeline.py --skip-features
```

### Option C — Skip grid prediction

```bash
python train_pipeline.py --skip-predict
```

## BioWatch Website (React)

The main website lives in `web/` (React + Vite + Leaflet + Recharts). It is a static site that reads JSON exported from the pipeline, so it needs no Python server.

```bash
# 1. Export pipeline outputs to web/public/data/ (re-run after retraining)
python scripts/11_export_web_data.py

# 2. Install and start the dev server (http://localhost:5173)
cd web
npm install
npm run dev

# 3. Build a static site into web/dist/ for hosting
npm run build
```

Pages:
- **Home** — species profile, key statistics, share of India per zone, largest hotspots
- **Zone Map** — suitability zones and hotspot zones of India; click a zone for details, search and zoom to any hotspot
- **Priority** — monitoring-priority tier zones for the top 500–5,000 cells
- **Models** — model cards, ROC curves, spatial CV AUC, confusion matrices
- **Climate Drivers** — feature importance per model and presence vs background distributions

The home-page photo is `web/public/lantana.webp`.

## Streamlit Dashboard (legacy)

```bash
streamlit run app.py
```

The Streamlit version has the same five sections and is kept for quick local analysis.

## Running Tests

```bash
python -m pytest tests/ -v
```

## Modelling Approach

### Presence-background design

GBIF records represent known locations. Randomly sampled background points serve as contrast. This is a presence-background model (similar in spirit to MaxEnt), not a presence-absence model.

### Validation strategy

1. **Stratified hold-out split** (80/20): primary reported metrics
2. **Spatial block cross-validation** (5 folds, 1° blocks): preferred metric because nearby points share similar environmental conditions, making random splits over-optimistic

### Suitability classes

| Class | Score Range |
|-------|-------------|
| Very Low | [0.00, 0.20) |
| Low | [0.20, 0.40) |
| Moderate | [0.40, 0.60) |
| High | [0.60, 0.80) |
| Very High | [0.80, 1.00] |

Thresholds are equal-width quantile breaks configurable in `src/predict_suitability.py`.

## Limitations

- Background points are geographically random, not target-group sampled. Reporting bias (observations concentrated near roads, cities, protected areas) may inflate suitability estimates in accessible areas.
- Three features in the baseline dataset (temperature, rainfall, elevation) cover most of the explainable variance but the full 19 bioclimatic variable run provides a richer feature space.
- The prediction grid at 0.1° resolution (~11 km) is coarser than the 2.5 arc-minute (~5 km) raster resolution. Increase `GRID_RES` in `predict_suitability.py` for finer predictions (at the cost of compute time).
- Suitability scores are not calibrated ecological probabilities. Do not use them as invasion likelihood estimates without further validation.

## Citation

Occurrence data: GBIF.org — *Lantana camara* occurrences in India  
Environmental data: Fick, S.E. & Hijmans, R.J. (2017). WorldClim 2. *International Journal of Climatology* 37(12): 4302–4315.
