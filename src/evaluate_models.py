"""
src/evaluate_models.py

Load saved model artifacts and produce evaluation reports and figures.
Run after train_pipeline.py has completed.

Outputs (to outputs/figures/):
    roc_curves.png
    confusion_matrices.png
    feature_importance.png
    spatial_cv_auc.png
"""

import json
import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parent.parent
OUTPUTS     = ROOT / "outputs"
FIGURES     = OUTPUTS / "figures"
MODELS_DIR  = ROOT / "models"
METRICS_JSON = OUTPUTS / "model_metrics.json"
FIGURES.mkdir(parents=True, exist_ok=True)

# Color palette: blue, green, orange (no purple)
MODEL_COLORS = {
    "Logistic Regression": "#4a90d9",
    "Random Forest":       "#3aa86a",
    "XGBoost":             "#e07b39",
}


def load_metrics() -> dict:
    if not METRICS_JSON.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_JSON}\n"
            "Run train_pipeline.py first."
        )
    with open(METRICS_JSON) as f:
        return json.load(f)


def plot_roc_curves(metrics: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#161b22")
    ax.plot([0, 1], [0, 1], "--", color="#555", linewidth=1, label="Random")

    for name, m in metrics.items():
        rc = m.get("roc_curve", {})
        if not rc:
            continue
        color = MODEL_COLORS.get(name, "#aaa")
        auc = m["roc_auc"]
        ax.plot(rc["fpr"], rc["tpr"], color=color, linewidth=2,
                label=f"{name} (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate", color="#c9d1d9")
    ax.set_ylabel("True Positive Rate", color="#c9d1d9")
    ax.set_title("ROC Curves — Hold-out Test Set", color="#e8eaf0", fontsize=13)
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_color("#21262d")
    legend = ax.legend(loc="lower right", facecolor="#161b22", edgecolor="#21262d",
                       labelcolor="#c9d1d9", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info("ROC curve figure saved -> %s", out_path)


def plot_confusion_matrices(metrics: dict, out_path: Path):
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.patch.set_facecolor("#0f1117")
    if n == 1:
        axes = [axes]

    for ax, (name, m) in zip(axes, metrics.items()):
        ax.set_facecolor("#161b22")
        cm = np.array(m["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Background", "Presence"], color="#c9d1d9", fontsize=8)
        ax.set_yticklabels(["Background", "Presence"], color="#c9d1d9", fontsize=8)
        ax.set_xlabel("Predicted", color="#c9d1d9")
        ax.set_ylabel("Actual", color="#c9d1d9")
        ax.set_title(name, color="#e8eaf0", fontsize=10)
        ax.spines[:].set_color("#21262d")
        ax.tick_params(colors="#8b949e")
        for (r, c), val in np.ndenumerate(cm):
            ax.text(c, r, str(val), ha="center", va="center",
                    color="#e8eaf0", fontsize=12, fontweight="bold")

    fig.suptitle("Confusion Matrices", color="#e8eaf0", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info("Confusion matrix figure saved -> %s", out_path)


def plot_feature_importance(metrics: dict, out_path: Path, top_n: int = 12):
    fi_models = {
        k: v["feature_importance"] for k, v in metrics.items()
        if v.get("feature_importance") and v["feature_importance"].get("features")
    }
    if not fi_models:
        log.warning("No feature importance data available to plot.")
        return

    n = len(fi_models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    fig.patch.set_facecolor("#0f1117")
    if n == 1:
        axes = [axes]

    feat_labels = {
        "temperature": "Bio 1 — Annual Mean Temp",
        "rainfall":    "Bio 12 — Annual Precip",
        "elevation":   "Elevation",
        **{f"bio_{i}": f"Bio {i}" for i in range(2, 20)},
    }

    for ax, (name, fi) in zip(axes, fi_models.items()):
        ax.set_facecolor("#161b22")
        features   = [feat_labels.get(f, f) for f in fi["features"][:top_n]]
        importances = fi["importances"][:top_n]
        color = MODEL_COLORS.get(name, "#aaa")
        y_pos = range(len(features))
        ax.barh(y_pos, importances, color=color, alpha=0.85, edgecolor="#21262d")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(features, color="#c9d1d9", fontsize=8)
        ax.set_xlabel(fi.get("importance_type", "Importance"), color="#c9d1d9")
        ax.set_title(f"{name}\nFeature Importance", color="#e8eaf0", fontsize=10)
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#21262d")
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info("Feature importance figure saved -> %s", out_path)


def plot_spatial_cv(metrics: dict, out_path: Path):
    data = []
    for name, m in metrics.items():
        cv = m.get("spatial_cv", {})
        for auc in cv.get("fold_aucs", []):
            data.append({"Model": name, "AUC": auc})

    if not data:
        log.warning("No spatial CV data to plot.")
        return

    df = pd.DataFrame(data)
    models = df["Model"].unique()

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#161b22")

    for i, model in enumerate(models):
        sub = df[df["Model"] == model]["AUC"].values
        color = MODEL_COLORS.get(model, "#aaa")
        x = np.full(len(sub), i)
        ax.scatter(x + np.random.uniform(-0.08, 0.08, len(sub)), sub,
                   color=color, alpha=0.7, s=60, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [sub.mean(), sub.mean()],
                color=color, linewidth=2.5, zorder=4)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, color="#c9d1d9")
    ax.set_ylabel("ROC-AUC", color="#c9d1d9")
    ax.set_title("Spatial Block Cross-Validation AUC (5 folds)", color="#e8eaf0", fontsize=13)
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_color("#21262d")
    ax.grid(axis="y", color="#21262d", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info("Spatial CV figure saved -> %s", out_path)


def print_report(metrics: dict):
    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    for name, m in metrics.items():
        cv = m.get("spatial_cv", {})
        print(f"\n  {name}")
        print(f"    Hold-out AUC : {m['roc_auc']:.4f}")
        if cv.get("mean_auc"):
            print(f"    Spatial CV   : {cv['mean_auc']:.4f} +/- {cv['std_auc']:.4f}")
        print(f"    Precision    : {m['precision']:.4f}")
        print(f"    Recall       : {m['recall']:.4f}")
        print(f"    F1-Score     : {m['f1_score']:.4f}")
        cm = m["confusion_matrix"]
        print(f"    Conf. Matrix : TN={cm[0][0]}  FP={cm[0][1]}  "
              f"FN={cm[1][0]}  TP={cm[1][1]}")


def main():
    metrics = load_metrics()
    print_report(metrics)
    plot_roc_curves(metrics, FIGURES / "roc_curves.png")
    plot_confusion_matrices(metrics, FIGURES / "confusion_matrices.png")
    plot_feature_importance(metrics, FIGURES / "feature_importance.png")
    plot_spatial_cv(metrics, FIGURES / "spatial_cv_auc.png")
    log.info("All evaluation figures saved to %s", FIGURES)


if __name__ == "__main__":
    main()
