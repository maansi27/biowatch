"""
10_baseline_model.py  (OPTIONAL - Stage 2 of the project, run after the CSV exists)

Baseline Random Forest presence/background classifier, WITH spatial
cross-validation (not just random split), because nearby points are
spatially autocorrelated and random splitting inflates apparent accuracy.

IMPORTANT INTERPRETATION CAVEAT (read before trusting any number this
script prints): species_present=0 rows are background samples, not
confirmed absences. A high accuracy here reflects how well the model
separates "known GBIF-reported locations" from "random unreported
locations" - which is a mix of genuine environmental suitability signal
and reporting/sampling bias (e.g. Lantana is disproportionately reported
near roads, protected areas, and cities with more observers). Treat
outputs as a habitat-suitability / relative-likelihood surface, not a
validated absence-prediction accuracy.

Validation strategy implemented:
  1. Spatial block cross-validation: India is split into a grid of blocks;
     folds are built from blocks (not individual points), so nearby
     points can't leak between train and test.
  2. Temporal split (only run if enough dated presence records exist):
     train on 2015-2020, validate 2021-2022, test 2023-2024.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

ROOT = Path(__file__).resolve().parent.parent
FINAL_CSV = ROOT / "data" / "final" / "invasive_species_lantana_india.csv"

FEATURES = ["temperature", "rainfall", "elevation"]  # + "NDVI" if present
BLOCK_SIZE_DEG = 1.0  # ~111km blocks at the equator; adjust as needed


def make_spatial_blocks(df, block_size=BLOCK_SIZE_DEG):
    block_x = (df["longitude"] // block_size).astype(int)
    block_y = (df["latitude"] // block_size).astype(int)
    return block_x.astype(str) + "_" + block_y.astype(str)


def spatial_cv(df, features, n_folds=5, seed=42):
    df = df.copy()
    df["block"] = make_spatial_blocks(df)
    blocks = df["block"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    fold_assign = {b: i % n_folds for i, b in enumerate(blocks)}
    df["fold"] = df["block"].map(fold_assign)

    aucs = []
    for fold in range(n_folds):
        train = df[df["fold"] != fold]
        test = df[df["fold"] == fold]
        if test["species_present"].nunique() < 2 or train["species_present"].nunique() < 2:
            continue
        X_train, y_train = train[features], train["species_present"]
        X_test, y_test = test[features], test["species_present"]

        clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        aucs.append(auc)
        print(f"  Fold {fold}: n_train={len(train)}, n_test={len(test)}, AUC={auc:.3f}")

    if aucs:
        print(f"\nSpatial CV mean AUC: {np.mean(aucs):.3f} (+/- {np.std(aucs):.3f}) over {len(aucs)} folds")
    else:
        print("Not enough class balance in any fold to compute AUC.")
    return aucs


def temporal_split(df, features):
    presence = df[df["point_type"] == "presence"].dropna(subset=["year"])
    if presence.empty:
        print("No dated presence records; skipping temporal validation.")
        return

    train_years = (2015, 2020)
    val_years = (2021, 2022)
    test_years = (2023, 2024)

    def year_mask(d, lo, hi):
        return d["year"].between(lo, hi)

    background = df[df["point_type"] == "background"]

    train = pd.concat([presence[year_mask(presence, *train_years)], background])
    val = presence[year_mask(presence, *val_years)]
    test = presence[year_mask(presence, *test_years)]

    print(f"\nTemporal split sizes -> train:{len(train)}  val(presence-only):{len(val)}  test(presence-only):{len(test)}")

    if len(train[train["species_present"] == 1]) < 10 or len(val) < 5:
        print("Too few dated records for a meaningful temporal validation; skipping.")
        return

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(train[features], train["species_present"])

    if len(val):
        val_probs = clf.predict_proba(val[features])[:, 1]
        print(f"Mean predicted suitability on 2021-2022 presence points: {val_probs.mean():.3f}")
        print("(Interpretation: how confidently the model, trained on 2015-2020 data, "
              "flags LATER known Lantana locations as suitable - not a full AUC since "
              "we only have presence-labelled points here, no contemporaneous background.)")


def main():
    if not FINAL_CSV.exists():
        raise SystemExit("Run 07_build_final_dataset.py first.")

    df = pd.read_csv(FINAL_CSV)
    features = [f for f in FEATURES + ["NDVI"] if f in df.columns]
    df = df.dropna(subset=features + ["species_present", "latitude", "longitude"])
    print(f"Rows available for modelling after dropping NaNs in {features}: {len(df)}")

    print("\n=== Spatial block cross-validation ===")
    spatial_cv(df, features)

    print("\n=== Temporal validation (2015-2020 train / 2021-22 val / 2023-24 test) ===")
    temporal_split(df, features)

    print(
        "\nReminder: do not report these numbers as 'biological prediction "
        "accuracy' without the sampling-bias caveat above. Consider "
        "target-group background sampling (background points drawn from "
        "other plant occurrence records, not uniformly at random) as a "
        "next step to reduce reporting-bias contamination."
    )


if __name__ == "__main__":
    main()
