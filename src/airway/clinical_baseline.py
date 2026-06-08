"""
Clinical baseline — a logistic-regression model on routine bedside variables.

WHAT THIS IS
------------
A pre-specified secondary comparator: a model that uses ONLY information an
anaesthetist already has at the bedside (age, sex, BMI, Mallampati class, and
thyromental distance when recorded). Its purpose is to let the paper answer the
question that matters clinically — does adding facial imaging and ultrasound
improve discrimination *beyond standard clinical assessment*? — rather than
only beating individual bedside scores.

HOW IT STAYS COMPARABLE AND LEAKAGE-FREE
----------------------------------------
It is evaluated on the EXACT same patients and patient-level 5x2 folds as every
other model: the cohort and fold membership are read from
reports/fusion_fold_predictions.csv (one row per patient per repeat). Within
each fold the model is fitted on the training patients only — imputation,
scaling, and the classifier all live in a single scikit-learn Pipeline — and
scored on the held-out patients. Metrics reuse the shared helper
(airway.predictions.full_metrics) so they line up with per_model_metrics.csv.

SEX ENCODING
------------
sex is mapped M -> 1, F -> 0; any other/blank value becomes NaN and is imputed
(median) inside the fold, like any other missing value.

OUTPUTS (reports/)
------------------
  clinical_baseline_probs.csv    out-of-fold probabilities (one row/patient/repeat)
  clinical_baseline_metrics.csv  AUC + operating-point metrics (fixed 0.5 + Youden),
                                 with a `columns_used` note
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from airway import clinical_comparison, config, predictions

# Bedside variables we will use if present (subset taken when some are missing).
CANDIDATE_COLS = ["age_years", "sex", "bmi", "mallampati_class", "thyromental_mm"]
SEX_MAP = {"M": 1, "m": 1, "F": 0, "f": 0}

PROBS_CSV = config.REPORTS_DIR / "clinical_baseline_probs.csv"
METRICS_CSV = config.REPORTS_DIR / "clinical_baseline_metrics.csv"
PROB_COL = "clinical_prob"


def make_pipeline() -> Pipeline:
    """Leakage-safe bedside model: median impute -> standardise -> balanced L2 LR."""
    return Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000,
                                     random_state=config.RANDOM_SEED)),
    ])


def _bedside_table():
    """Per-patient bedside feature frame + the list of columns actually used."""
    from airway import loaders

    preop = loaders.preop_loader()
    used = [c for c in CANDIDATE_COLS if c in preop.columns]
    missing = [c for c in CANDIDATE_COLS if c not in preop.columns]
    if missing:
        print(f"clinical_baseline: WARNING — bedside column(s) {missing} not in "
              f"{config.PREOP_CSV.name}; proceeding with {used}.")
    if not used:
        raise ValueError("clinical_baseline: none of the bedside columns "
                         f"{CANDIDATE_COLS} are present; nothing to model.")

    feat = preop[[config.ID_COL, *used]].copy()
    if "sex" in used:
        feat["sex"] = feat["sex"].map(SEX_MAP)          # M->1, F->0, else NaN
    for c in used:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")
    return feat, used


def cross_val_probs() -> tuple[pd.DataFrame, list[str]]:
    """
    Out-of-fold clinical-baseline probabilities on the shared folds.

    Returns (probs_df, columns_used). probs_df columns:
    study_id, repeat, fold_index, label, clinical_prob.
    """
    preds = predictions.load_fusion_predictions()      # cohort + folds + label
    feat, used = _bedside_table()
    labels = preds[[config.ID_COL, config.LABEL_COL]].drop_duplicates(config.ID_COL)
    data = feat.merge(labels, on=config.ID_COL, how="inner")

    rows = []
    for repeat in sorted(preds["repeat"].unique()):
        rep = preds[preds["repeat"] == repeat]
        for fold in sorted(rep["fold_index"].unique()):
            val_ids = set(rep.loc[rep["fold_index"] == fold, config.ID_COL])
            train_ids = set(rep.loc[rep["fold_index"] != fold, config.ID_COL])
            train = data[data[config.ID_COL].isin(train_ids)]
            val = data[data[config.ID_COL].isin(val_ids)]
            if train.empty or val.empty:
                continue
            pipe = make_pipeline()
            pipe.fit(train[used].to_numpy(), train[config.LABEL_COL].to_numpy())
            prob = pipe.predict_proba(val[used].to_numpy())[:, 1]
            for (sid, y), p in zip(
                    val[[config.ID_COL, config.LABEL_COL]].itertuples(index=False), prob):
                rows.append({config.ID_COL: sid, "repeat": int(repeat),
                             "fold_index": int(fold), config.LABEL_COL: int(y),
                             PROB_COL: float(p)})
    return pd.DataFrame(rows), used


def metrics_table(probs: pd.DataFrame, columns_used: list[str]) -> pd.DataFrame:
    """AUC (mean±SD over folds) + pooled operating-point metrics at 0.5 and Youden."""
    per_fold_auc = []
    for _, g in probs.groupby(["repeat", "fold_index"]):
        y = g[config.LABEL_COL].to_numpy()
        if len(np.unique(y)) == 2:
            per_fold_auc.append(roc_auc_score(y, g[PROB_COL].to_numpy()))

    y_all = probs[config.LABEL_COL].to_numpy()
    p_all = probs[PROB_COL].to_numpy()
    auc_pooled = (roc_auc_score(y_all, p_all)
                  if len(np.unique(y_all)) == 2 else float("nan"))
    auc_mean = float(np.mean(per_fold_auc)) if per_fold_auc else auc_pooled
    auc_std = float(np.std(per_fold_auc)) if per_fold_auc else 0.0

    youden = clinical_comparison.youden_threshold(y_all, p_all)
    rows = []
    for ttype, thr in [("fixed_0.5", 0.5),
                       ("youden", 0.5 if np.isnan(youden) else youden)]:
        m = predictions.full_metrics(y_all, p_all, thr, is_probability=True)
        rows.append({
            "model": "clinical_baseline", "threshold_type": ttype,
            "n": int(probs[config.ID_COL].nunique()),
            "auc_mean": round(auc_mean, 4) if not np.isnan(auc_mean) else np.nan,
            "auc_std": round(auc_std, 4),
            "auc_pooled": round(auc_pooled, 4) if not np.isnan(auc_pooled) else np.nan,
            "threshold": round(float(thr), 4),
            "sensitivity": round(m["sensitivity"], 4),
            "specificity": round(m["specificity"], 4),
            "ppv": round(m["ppv"], 4), "npv": round(m["npv"], 4),
            "accuracy": round(m["accuracy"], 4),
            "balanced_accuracy": round(m["balanced_accuracy"], 4),
            "f1": round(m["f1"], 4),
            "columns_used": ",".join(columns_used),
        })
    return pd.DataFrame(rows)


def main() -> None:
    config.ensure_dirs()
    probs, used = cross_val_probs()
    probs.to_csv(PROBS_CSV, index=False)
    metrics = metrics_table(probs, used)
    metrics.to_csv(METRICS_CSV, index=False)

    print(f"clinical baseline: bedside columns used = {used}")
    print(f"out-of-fold probs -> {PROBS_CSV}  ({len(probs)} rows, "
          f"{probs[config.ID_COL].nunique()} patients)")
    print(f"metrics           -> {METRICS_CSV}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
