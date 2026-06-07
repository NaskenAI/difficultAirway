"""
Ultrasound model (Weeks 6-7).

WHAT THIS DOES
--------------
Trains and cross-validates TWO classifiers on the cleaned ultrasound features:

  - Logistic regression with L2 regularisation
  - XGBoost

It reuses the face-model machinery (same patient-level stratified 5x2 CV, same
CVResult / metrics / ROC helpers, same outcome = difficult airway / CL 3-4) and
differs only in the preprocessing the task asks for:

  - WITHIN-FOLD MEAN imputation (SimpleImputer(strategy="mean")) fitted on the
    training fold only — never on the full dataset before CV.
  - the derived hyomental distance ratio, computed safely in
    ultrasound_features.clean_ultrasound_features.

LEAKAGE DISCIPLINE
------------------
Cleaning (ultrasound_features) is deterministic and learns nothing from the
data. Everything that learns from data — mean imputation, standardisation, the
classifier, XGBoost's scale_pos_weight — lives inside the sklearn Pipeline and
is therefore fitted inside each CV fold on training patients only.

ALL-MISSING FEATURES
--------------------
A feature that is entirely missing across the dataset is dropped up front by
ultrasound_features.usable_feature_cols (with a warning). Within a fold, the
mean imputer handles any remaining gaps; if a feature were all-NaN in a single
training fold, SimpleImputer drops that column for that fold (documented).

OUTPUTS (reports/)
------------------
  us_model.pkl, us_cv_metrics.csv, us_roc.png,
  us_feature_importance.csv, us_feature_importance.png
plus data/processed/cleaned_ultrasound_features.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from airway import config, face_model, splits, ultrasound_features

SEED = config.RANDOM_SEED


def make_us_logreg_pipeline(y_train: np.ndarray) -> Pipeline:
    """Within-fold MEAN impute -> standardise -> L2 logistic regression."""
    return Pipeline(steps=[
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
        # L2 is LogisticRegression's default penalty; left implicit to stay
        # warning-free across sklearn versions.
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        )),
    ])


def make_us_xgb_pipeline(y_train: np.ndarray) -> Pipeline:
    """Within-fold MEAN impute -> XGBoost with training-fold scale_pos_weight."""
    from xgboost import XGBClassifier

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    spw = (n_neg / n_pos) if n_pos > 0 else 1.0
    return Pipeline(steps=[
        ("impute", SimpleImputer(strategy="mean")),
        ("model", XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=spw,
            eval_metric="logloss",
            importance_type="gain",     # so feature_importances_ is gain-based
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        )),
    ])


# model registry in the same shape the face model uses
US_MODELS = {
    "logreg_l2": make_us_logreg_pipeline,
    "xgboost": make_us_xgb_pipeline,
}


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
def compute_feature_importance(data: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Two complementary importances for the ultrasound features:

      - PERMUTATION importance: model-agnostic, computed leakage-safely on each
        held-out TEST fold (fit on train, permute test features, measure the AUC
        drop), then averaged across folds. Uses the XGBoost pipeline.
      - XGBoost GAIN importance: from the XGBoost model refit on all patients
        (gain = average improvement in the split criterion from each feature).

    Returns one row per feature with both scores.
    """
    from sklearn.inspection import permutation_importance

    folds = splits.patient_level_folds(data)
    splits.assert_no_leakage(folds)

    perm_per_fold: dict[str, list[float]] = {c: [] for c in feature_cols}
    n_used = 0
    for fold in folds:
        train = splits.select_rows(data, fold.train_ids)
        test = splits.select_rows(data, fold.test_ids)
        y_te = test[config.LABEL_COL].to_numpy()
        if len(np.unique(y_te)) < 2:
            continue  # AUC (and so permutation importance) undefined on one class
        x_tr, y_tr = train[feature_cols].to_numpy(), train[config.LABEL_COL].to_numpy()
        x_te = test[feature_cols].to_numpy()

        pipe = make_us_xgb_pipeline(y_tr)
        pipe.fit(x_tr, y_tr)
        r = permutation_importance(
            pipe, x_te, y_te, scoring="roc_auc",
            n_repeats=10, random_state=SEED)
        for i, c in enumerate(feature_cols):
            perm_per_fold[c].append(float(r.importances_mean[i]))
        n_used += 1

    if n_used == 0:
        print("compute_feature_importance: no fold had both classes in test; "
              "permutation importance is undefined (reporting NaN).")

    # XGBoost gain from the model refit on all data
    x_all = data[feature_cols].to_numpy()
    y_all = data[config.LABEL_COL].to_numpy()
    final_xgb = make_us_xgb_pipeline(y_all)
    final_xgb.fit(x_all, y_all)
    gain = final_xgb.named_steps["model"].feature_importances_

    rows = []
    for i, c in enumerate(feature_cols):
        vals = perm_per_fold[c]
        rows.append({
            "feature": c,
            "perm_importance_mean": float(np.mean(vals)) if vals else float("nan"),
            "perm_importance_std": float(np.std(vals)) if vals else float("nan"),
            "xgb_gain": float(gain[i]),
        })
    out = pd.DataFrame(rows).sort_values("perm_importance_mean", ascending=False)
    return out.reset_index(drop=True)


def _save_importance_plot(imp: pd.DataFrame, out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    perm = imp.sort_values("perm_importance_mean")
    axes[0].barh(perm["feature"], perm["perm_importance_mean"],
                 xerr=perm["perm_importance_std"].fillna(0.0))
    axes[0].set_title("Permutation importance (AUC drop, OOF)")
    axes[0].set_xlabel("mean decrease in ROC-AUC")

    gain = imp.sort_values("xgb_gain")
    axes[1].barh(gain["feature"], gain["xgb_gain"])
    axes[1].set_title("XGBoost gain importance")
    axes[1].set_xlabel("gain")

    fig.suptitle("Ultrasound model — feature importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    import joblib

    from airway import loaders

    config.ensure_dirs()
    patient_table = loaders.build_patient_table()

    print("[1/4] cleaning ultrasound features ...")
    cleaned = ultrasound_features.build_and_save_cleaned_ultrasound()
    feature_cols = ultrasound_features.usable_feature_cols(cleaned)
    print(f"      {len(cleaned)} patients x {len(feature_cols)} usable features: {feature_cols}")

    print("[2/4] cross-validating classifiers (patient-level 5x2, mean imputation) ...")
    results = [face_model.cross_validate(
        cleaned, patient_table, feature_cols, name,
        models=US_MODELS, modality_prefix="ultrasound") for name in US_MODELS]

    print("[3/4] computing feature importance ...")
    data = cleaned.merge(patient_table[[config.ID_COL, config.LABEL_COL]],
                         on=config.ID_COL, how="inner")
    importance = compute_feature_importance(data, feature_cols)

    print("[4/4] writing outputs ...")
    summary = pd.DataFrame([r.summary_row() for r in results])
    csv_out = config.REPORTS_DIR / "us_cv_metrics.csv"
    summary.to_csv(csv_out, index=False)

    roc_out = config.REPORTS_DIR / "us_roc.png"
    face_model._save_roc(results, roc_out,
                         title="Ultrasound model — pooled out-of-fold ROC")

    imp_csv = config.REPORTS_DIR / "us_feature_importance.csv"
    importance.to_csv(imp_csv, index=False)
    imp_png = config.REPORTS_DIR / "us_feature_importance.png"
    _save_importance_plot(importance, imp_png)

    bundle = {
        "models": {name: face_model._fit_final_model(name, data, feature_cols, models=US_MODELS)
                   for name in US_MODELS},
        "feature_cols": feature_cols,
        "outcome": "difficult airway (CL 3-4)",
        "imputation": "within-fold mean (SimpleImputer strategy='mean')",
        "cv": {r.modality: r.summary_row() for r in results},
        "n_patients": len(data),
        "seed": SEED,
    }
    pkl_out = config.REPORTS_DIR / "us_model.pkl"
    joblib.dump(bundle, pkl_out)

    print(f"\ncleaned features -> {config.CLEANED_US_CSV}")
    print(f"metrics          -> {csv_out}")
    print(f"ROC              -> {roc_out}")
    print(f"importance       -> {imp_csv} , {imp_png}")
    print(f"model            -> {pkl_out}\n")
    print(summary.to_string(index=False))
    print("\nfeature importance:")
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
