"""
src/train_models.py

Train and compare three models for Lantana camara habitat suitability:
  1. Logistic Regression  (baseline)
  2. Random Forest        (primary)
  3. XGBoost              (gradient-boosting comparison)

Validation strategy
-------------------
* Stratified 80/20 hold-out split (primary reported metrics).
* Spatial block cross-validation (5-fold, 1-degree blocks) to give a
  less optimistic AUC estimate that accounts for spatial autocorrelation.

Interpretation caveat
---------------------
Background points (label=0) are pseudo-absences, not confirmed absences.
Metrics reflect the model's ability to separate known presence records
from random background locations — a habitat suitability signal, not
a validated absence-prediction accuracy.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5
BLOCK_SIZE = 1.0   # degrees (~111 km at equator)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_logistic_regression() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=SEED,
            solver="lbfgs",
        )),
    ])


def build_random_forest() -> Pipeline:
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=-1,
        )),
    ])


def build_xgboost() -> Pipeline:
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Run: pip install xgboost")
    return Pipeline([
        ("clf", XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])


# ---------------------------------------------------------------------------
# Spatial block cross-validation
# ---------------------------------------------------------------------------

def spatial_block_cv(
    df: pd.DataFrame,
    features: List[str],
    pipeline_fn,
    n_folds: int = N_FOLDS,
    block_size: float = BLOCK_SIZE,
) -> dict:
    """Run spatial block CV and return per-fold AUCs."""
    df = df.copy()
    bx = (df["longitude"] // block_size).astype(int)
    by = (df["latitude"] // block_size).astype(int)
    df["_block"] = bx.astype(str) + "_" + by.astype(str)

    blocks = df["_block"].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(blocks)
    fold_map = {b: i % n_folds for i, b in enumerate(blocks)}
    df["_fold"] = df["_block"].map(fold_map)

    aucs = []
    for fold in range(n_folds):
        train = df[df["_fold"] != fold]
        test  = df[df["_fold"] == fold]
        if test["species_present"].nunique() < 2 or train["species_present"].nunique() < 2:
            log.warning("Fold %d: skipped (single class in train or test)", fold)
            continue
        pipe = pipeline_fn()
        pipe.fit(train[features], train["species_present"])
        probs = pipe.predict_proba(test[features])[:, 1]
        auc = roc_auc_score(test["species_present"], probs)
        aucs.append(auc)
        log.info("  Spatial CV fold %d: n_train=%d n_test=%d AUC=%.4f",
                 fold, len(train), len(test), auc)
    return {
        "fold_aucs": aucs,
        "mean_auc": float(np.mean(aucs)) if aucs else None,
        "std_auc":  float(np.std(aucs))  if aucs else None,
    }


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    pipe: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict:
    """Compute and return evaluation metrics for a fitted pipeline."""
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred).tolist()
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    report = classification_report(y_test, y_pred, output_dict=True)

    log.info("=== %s ===", model_name)
    log.info("  ROC-AUC:   %.4f", auc)
    log.info("  Precision: %.4f", precision)
    log.info("  Recall:    %.4f", recall)
    log.info("  F1-score:  %.4f", f1)

    return {
        "model_name":  model_name,
        "roc_auc":     float(auc),
        "precision":   float(precision),
        "recall":      float(recall),
        "f1_score":    float(f1),
        "confusion_matrix": cm,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(pipe: Pipeline, features: List[str], model_name: str) -> dict:
    """Extract feature importances where available."""
    clf = pipe.named_steps.get("clf")
    if clf is None:
        # single-step pipelines
        clf = pipe[-1]

    importances = None
    importance_type = None

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        importance_type = "gini_impurity"
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        importance_type = "absolute_coefficient"

    if importances is None:
        return {}

    order = np.argsort(importances)[::-1]
    return {
        "importance_type": importance_type,
        "features": [features[i] for i in order],
        "importances": [float(importances[i]) for i in order],
    }


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train_all_models(data: dict) -> dict:
    """
    Train LR, RF, XGBoost on data dict from data_processing.prepare_modelling_data().
    Returns results dict and saves artifacts to models/ and outputs/.
    """
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    features = data["features"]
    df_clean = data["df_clean"]

    model_defs = [
        ("Logistic Regression", build_logistic_regression),
        ("Random Forest",       build_random_forest),
    ]
    if XGBOOST_AVAILABLE:
        model_defs.append(("XGBoost", build_xgboost))
    else:
        log.warning("XGBoost not installed – skipping. Run: pip install xgboost")

    all_results = {}
    best_auc = -1
    best_name = None
    best_pipe = None

    for model_name, build_fn in model_defs:
        log.info("\nTraining: %s", model_name)
        pipe = build_fn()
        pipe.fit(X_train, y_train)

        # Hold-out evaluation
        metrics = evaluate_model(pipe, X_test, y_test, model_name)

        # Spatial CV (on full clean dataset so both train & test blocks are covered)
        log.info("  Running spatial block CV for %s ...", model_name)
        cv_results = spatial_block_cv(df_clean, features, build_fn)
        metrics["spatial_cv"] = cv_results
        log.info(
            "  Spatial CV AUC: %.4f +/- %.4f",
            cv_results.get("mean_auc", float("nan")),
            cv_results.get("std_auc", float("nan")),
        )

        # Feature importance
        metrics["feature_importance"] = get_feature_importance(pipe, features, model_name)

        all_results[model_name] = metrics

        # Track best by hold-out AUC
        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_name = model_name
            best_pipe = pipe

        # Save model artifact
        safe_name = model_name.lower().replace(" ", "_")
        model_path = MODELS_DIR / f"{safe_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"pipeline": pipe, "features": features}, f)
        log.info("  Saved model -> %s", model_path)

    # Save the best model separately for use in prediction
    if best_pipe is not None:
        with open(MODELS_DIR / "best_model.pkl", "wb") as f:
            pickle.dump({
                "pipeline": best_pipe,
                "features": features,
                "model_name": best_name,
            }, f)
        log.info("\nBest model: %s (hold-out AUC=%.4f) saved as best_model.pkl", best_name, best_auc)

    # Save feature list
    with open(MODELS_DIR / "features.json", "w") as f:
        json.dump(features, f, indent=2)

    # Save all metrics
    results_path = OUTPUTS_DIR / "model_metrics.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("All metrics saved -> %s", results_path)

    # Summary table
    summary = []
    for name, m in all_results.items():
        cv = m.get("spatial_cv", {})
        summary.append({
            "Model": name,
            "Hold-out AUC": f"{m['roc_auc']:.4f}",
            "Spatial CV AUC": f"{cv.get('mean_auc', float('nan')):.4f} "
                              f"(+/-{cv.get('std_auc', float('nan')):.4f})",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1": f"{m['f1_score']:.4f}",
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUTS_DIR / "model_summary.csv", index=False)
    print("\n" + summary_df.to_string(index=False))
    print(f"\nSelected model: {best_name} (highest hold-out AUC)")

    return {"results": all_results, "best_model_name": best_name}


if __name__ == "__main__":
    from src.data_processing import prepare_modelling_data

    log.info("Loading and preparing data...")
    data = prepare_modelling_data()
    log.info("Features: %s", data["features"])
    train_all_models(data)
