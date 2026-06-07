"""
Block D shared helpers: load the fusion predictions and assemble the master
per-patient/per-fold table used by bootstrap CIs, subgroup analysis,
explainability, and error analysis.

EVERYTHING REUSES BLOCK C OUTPUTS
---------------------------------
The single source of fold-level predictions is reports/fusion_fold_predictions.csv
(written by airway.fusion). We never regenerate folds or models here; we join the
existing predictions with the clinical scores, demographics, and cleaned
ultrasound features. Patient-level units throughout — `per_patient` averages a
patient's probabilities across the CV repeats (each patient is a validation
patient once per repeat).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from airway import config, fusion, scores, ultrasound_features
from airway.baseline_model import _classification_metrics

PROB_COLS = ["face_prob", "us_prob", "fused_prob", "avg_prob"]
SCORE_COLS = ["mallampati_class", "lemon_score", "wilson_score"]
DEFAULT_THRESHOLD = 0.5

# All metrics Block D reports (order is stable for CSV output).
METRIC_KEYS = ["auc", "sensitivity", "specificity", "ppv", "npv",
               "accuracy", "balanced_accuracy", "f1", "brier"]


def load_fusion_predictions() -> pd.DataFrame:
    """Read reports/fusion_fold_predictions.csv or fail clearly."""
    path = fusion.FUSION_FOLD_PRED_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"predictions: {path} not found. Run `make block-c` "
            f"(python -m airway.fusion) before the Block D steps."
        )
    return pd.read_csv(path)


def build_master_table() -> pd.DataFrame:
    """
    Fold-level fusion predictions joined with comparator scores, demographics,
    surgery type, and cleaned ultrasound features. One row per patient per fold.
    """
    from airway import loaders

    rows = load_fusion_predictions()

    comp = scores.compute_comparator_scores(loaders.preop_loader())
    rows = rows.merge(comp[[config.ID_COL, *SCORE_COLS]], on=config.ID_COL, how="left")

    preop = loaders.preop_loader()
    demo_cols = [c for c in (config.DEMOGRAPHIC_COLS + [config.SURGERY_TYPE_COL])
                 if c in preop.columns]
    if demo_cols:
        rows = rows.merge(preop[[config.ID_COL, *demo_cols]], on=config.ID_COL, how="left")

    us = ultrasound_features.clean_ultrasound_features()
    us_cols = [c for c in ultrasound_features.US_FEATURE_COLS if c in us.columns]
    rows = rows.merge(us[[config.ID_COL, *us_cols]], on=config.ID_COL, how="left")
    return rows


def per_patient(master: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to one row per patient: probabilities averaged across repeats;
    static columns (label, scores, demographics, ultrasound) taken once. Adds a
    `folds` column listing the validation fold indices the patient appeared in.
    """
    static = [c for c in master.columns
              if c not in PROB_COLS + ["repeat", "fold_index", config.ID_COL]]
    agg = {c: "mean" for c in PROB_COLS if c in master.columns}
    for c in static:
        agg[c] = "first"
    out = master.groupby(config.ID_COL, as_index=False).agg(agg)
    folds = (master.groupby(config.ID_COL)["fold_index"]
             .agg(lambda s: ",".join(str(x) for x in sorted(set(s)))))
    out = out.merge(folds.rename("folds"), on=config.ID_COL)
    return out


def predicted_class(prob, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    return (np.asarray(prob, dtype=float) >= threshold).astype(int)


def confusion_category(label: int, pred: int) -> str:
    """TP / TN / FP / FN for a single (label, predicted-class) pair."""
    if label == 1 and pred == 1:
        return "TP"
    if label == 0 and pred == 0:
        return "TN"
    if label == 0 and pred == 1:
        return "FP"
    return "FN"   # label == 1 and pred == 0


def full_metrics(y_true, y_score, threshold: float = DEFAULT_THRESHOLD,
                 is_probability: bool = True) -> dict:
    """
    All Block D metrics for one set of paired (label, score) values.

    AUC uses the raw score; the operating-point metrics use (score >= threshold).
    Brier is only meaningful for probabilities (NaN otherwise). Any metric that
    is undefined (e.g. AUC with one class present) is returned as NaN.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    out = {k: float("nan") for k in METRIC_KEYS}

    # drop entries with a missing score (e.g. a clinical comparator that could
    # not be computed for a patient); metrics are over the patients with a score
    keep = ~np.isnan(y_score)
    y_true, y_score = y_true[keep], y_score[keep]
    if len(y_true) == 0:
        return out

    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, y_score))
        if is_probability:
            out["brier"] = float(brier_score_loss(y_true, y_score))

    y_pred = predicted_class(y_score, threshold).astype(float)
    m = _classification_metrics(y_true, y_pred)
    out.update({k: float(m[k]) for k in
                ["sensitivity", "specificity", "ppv", "npv", "accuracy"]})
    out["balanced_accuracy"] = 0.5 * (m["sensitivity"] + m["specificity"])
    denom = m["ppv"] + m["sensitivity"]
    out["f1"] = (2 * m["ppv"] * m["sensitivity"] / denom) if denom else 0.0
    return out


# The models Block D reports on: (display name, score column, is_probability).
def model_specs(master: pd.DataFrame) -> list[tuple[str, str, bool]]:
    specs = [("face", "face_prob", True), ("ultrasound", "us_prob", True),
             ("fusion:logreg", "fused_prob", True), ("fusion:average", "avg_prob", True)]
    for col in SCORE_COLS:
        if col in master.columns:
            name = col.replace("_class", "").replace("_score", "")
            specs.append((name, col, False))
    return specs
