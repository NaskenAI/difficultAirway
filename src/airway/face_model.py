"""
Face-model classifiers (Week 7) + persisted outputs (Week 8).

WHAT THIS DOES
--------------
Trains and cross-validates TWO classifiers on the per-patient 1024-d face
features (from face_embeddings):

  - Logistic regression with L2 regularisation
  - XGBoost

Both use the project's patient-level 5x2 cross-validation (splits.py), so a
patient is never split across train and test. The rare difficult-airway class
is handled with class weighting (LogReg `class_weight='balanced'`) and
`scale_pos_weight` (XGBoost), the latter computed from the TRAINING fold only.

LEAKAGE DISCIPLINE
------------------
- Embeddings are produced by a frozen network and computed ONCE, outside CV.
- Everything supervised — imputation, scaling, the classifier, and XGBoost's
  scale_pos_weight — is fitted INSIDE each fold on training patients only.

OUTPUTS (written to reports/)
-----------------------------
  - face_model.pkl        : both classifiers refit on all data + metadata
  - face_cv_metrics.csv   : cross-validated metrics, one row per classifier
  - face_roc.png          : pooled out-of-fold ROC curves for both classifiers
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from airway import config, splits
from airway.baseline_model import CVResult, _classification_metrics


def make_logreg_pipeline(y_train: np.ndarray) -> Pipeline:
    """Leakage-safe L2 logistic regression: impute -> scale -> LogReg."""
    return Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        # LogisticRegression uses L2 regularisation by default (penalty="l2");
        # we leave it implicit so this stays warning-free across sklearn versions.
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=config.RANDOM_SEED,
        )),
    ])


def make_xgb_pipeline(y_train: np.ndarray) -> Pipeline:
    """
    XGBoost with scale_pos_weight set from the TRAINING fold's class balance.
    Tree models do not need scaling, so the pipeline is just impute -> XGB.
    """
    from xgboost import XGBClassifier

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    spw = (n_neg / n_pos) if n_pos > 0 else 1.0
    return Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=config.RANDOM_SEED,
            n_jobs=1,
            verbosity=0,
        )),
    ])


# factory registry: name -> pipeline builder taking y_train
MODELS = {
    "logreg_l2": make_logreg_pipeline,
    "xgboost": make_xgb_pipeline,
}


def cross_validate(feature_table: pd.DataFrame, patient_table: pd.DataFrame,
                   feature_cols: list[str], model_name: str,
                   models: dict | None = None,
                   modality_prefix: str = "face") -> CVResult:
    """
    Run patient-level 5x2 CV for one classifier and return pooled metrics.

    Reusable across modalities: pass a `models` registry (name -> factory(y_train))
    and a `modality_prefix` (e.g. "ultrasound"). Defaults to the face MODELS so
    existing face callers are unchanged.
    """
    models = MODELS if models is None else models
    factory = models[model_name]
    data = feature_table.merge(
        patient_table[[config.ID_COL, config.LABEL_COL]],
        on=config.ID_COL, how="inner")
    if data.empty:
        raise ValueError(f"cross_validate({model_name}): no patients after join.")

    folds = splits.patient_level_folds(data)
    splits.assert_no_leakage(folds)

    per_fold_auc, oof_true, oof_prob = [], [], []
    for fold in folds:
        train = splits.select_rows(data, fold.train_ids)
        test = splits.select_rows(data, fold.test_ids)
        x_tr, y_tr = train[feature_cols].to_numpy(), train[config.LABEL_COL].to_numpy()
        x_te, y_te = test[feature_cols].to_numpy(), test[config.LABEL_COL].to_numpy()

        pipe = factory(y_tr)
        pipe.fit(x_tr, y_tr)
        prob = pipe.predict_proba(x_te)[:, 1]

        if len(np.unique(y_te)) == 2:
            per_fold_auc.append(roc_auc_score(y_te, prob))
        oof_true.extend(y_te.tolist())
        oof_prob.extend(prob.tolist())

    oof_true_arr, oof_prob_arr = np.array(oof_true), np.array(oof_prob)
    pooled_auc = (roc_auc_score(oof_true_arr, oof_prob_arr)
                  if len(np.unique(oof_true_arr)) == 2 else float("nan"))
    m = _classification_metrics(oof_true_arr, oof_prob_arr)
    return CVResult(
        modality=f"{modality_prefix}:{model_name}",
        n_patients=len(data), n_features=len(feature_cols),
        auc_mean=float(np.mean(per_fold_auc)) if per_fold_auc else pooled_auc,
        auc_std=float(np.std(per_fold_auc)) if per_fold_auc else 0.0,
        sensitivity=m["sensitivity"], specificity=m["specificity"],
        accuracy=m["accuracy"], ppv=m["ppv"], npv=m["npv"],
        per_fold_auc=per_fold_auc, oof_true=oof_true, oof_prob=oof_prob,
    )


def _fit_final_model(model_name: str, data: pd.DataFrame, feature_cols: list[str],
                     models: dict | None = None):
    """Refit a classifier on ALL patients — this is what we persist for reuse."""
    models = MODELS if models is None else models
    x = data[feature_cols].to_numpy()
    y = data[config.LABEL_COL].to_numpy()
    pipe = models[model_name](y)
    pipe.fit(x, y)
    return pipe


def _save_roc(results: list[CVResult], out_path,
              title: str = "Face model — pooled out-of-fold ROC") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    plt.figure(figsize=(6, 6))
    for res in results:
        y, p = np.array(res.oof_true), np.array(res.oof_prob)
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{res.modality} (AUC={res.auc_mean:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    plt.xlabel("False positive rate (1 - specificity)")
    plt.ylabel("True positive rate (sensitivity)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _ensure_features(force_embeddings: bool) -> None:
    """
    Make sure the per-patient face-feature parquet exists.

    Embeddings need torch; XGBoost and torch cannot safely share one process on
    macOS (OpenMP clash -> segfault). So we build the features in a SEPARATE
    subprocess and keep this (model-training) process torch-free.
    """
    if config.FACE_FEATURES_PARQUET.exists() and not force_embeddings:
        return
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "airway.face_embeddings"]
    if force_embeddings:
        cmd.append("--force")
    print(f"      building face features in a subprocess: {' '.join(cmd)}")
    env = {**os.environ, "PYTHONPATH": str(config.PROJECT_ROOT / "src")}
    subprocess.run(cmd, check=True, env=env)
    if not config.FACE_FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"face features still missing at {config.FACE_FEATURES_PARQUET}; "
            f"run `python -m airway.face_embeddings` manually.")


def main(force_embeddings: bool = False) -> None:
    import joblib

    from airway import loaders

    config.ensure_dirs()
    patient_table = loaders.build_patient_table()

    print("[1/3] loading per-patient face features ...")
    _ensure_features(force_embeddings)
    face_tbl = pd.read_parquet(config.FACE_FEATURES_PARQUET)
    face_cols = [c for c in face_tbl.columns if c.startswith(config.FACE_FEATURE_PREFIX)]
    print(f"      {len(face_tbl)} patients x {len(face_cols)} features")

    print("[2/3] cross-validating classifiers (patient-level 5x2) ...")
    results = [cross_validate(face_tbl, patient_table, face_cols, name)
               for name in MODELS]

    print("[3/3] writing outputs ...")
    summary = pd.DataFrame([r.summary_row() for r in results])
    csv_out = config.REPORTS_DIR / "face_cv_metrics.csv"
    summary.to_csv(csv_out, index=False)

    roc_out = config.REPORTS_DIR / "face_roc.png"
    _save_roc(results, roc_out)

    # refit both on all data and persist together with metadata
    data = face_tbl.merge(patient_table[[config.ID_COL, config.LABEL_COL]],
                          on=config.ID_COL, how="inner")
    bundle = {
        "models": {name: _fit_final_model(name, data, face_cols) for name in MODELS},
        "feature_cols": face_cols,
        "outcome": "difficult airway (CL 3-4)",
        "cv": {r.modality: r.summary_row() for r in results},
        "n_patients": len(data),
        "seed": config.RANDOM_SEED,
    }
    pkl_out = config.REPORTS_DIR / "face_model.pkl"
    joblib.dump(bundle, pkl_out)

    print(f"\nmetrics -> {csv_out}")
    print(f"ROC     -> {roc_out}")
    print(f"model   -> {pkl_out}\n")
    print(summary.to_string(index=False))


def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description="Train + CV the face-model classifiers.")
    p.add_argument("--force-embeddings", action="store_true",
                   help="recompute face features before training")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    main(force_embeddings=args.force_embeddings)
