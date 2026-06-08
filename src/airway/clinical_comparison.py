"""
Block C / Week 11 — clinical comparison and DeLong tests.

WHAT THIS DOES
--------------
1. Computes the bedside clinical comparators (Mallampati, LEMON, Wilson) for the
   fusion cohort, reusing airway.scores.
2. Evaluates every model on the SAME folds as the fusion model (read from
   fusion_fold_predictions.csv): per-fold and pooled AUC, plus operating-point
   metrics (sensitivity, specificity, PPV, NPV, accuracy, balanced accuracy, F1)
   at deterministic thresholds.
3. Runs six DeLong tests comparing the fused model against Mallampati, LEMON,
   Wilson, the face model, the ultrasound model, and the average-probability
   baseline, with a Bonferroni-adjusted alpha of 0.0083 (= 0.05 / 6).

DETERMINISTIC OPERATING-POINT THRESHOLDS (documented)
-----------------------------------------------------
- Probability models (face, ultrasound, fused, average): threshold 0.5.
- Mallampati: class >= 3 (scores.MALLAMPATI_DIFFICULT_CLASS).
- LEMON:      score >= 2 (scores.LEMON_DIFFICULT_THRESHOLD).
- Wilson:     score >= 2 (scores.WILSON_DIFFICULT_THRESHOLD).
AUC uses the raw (continuous/ordinal) score; the threshold only sets the
operating point for the sensitivity/specificity-type metrics.

FOLD REUSE
----------
Folds are NOT regenerated here. The per-patient/per-fold rows come from
fusion_fold_predictions.csv, which carries the calibration/fusion fold
membership; clinical scores are joined on study_id. DeLong is run on one value
per patient (probabilities averaged across the two CV repeats; clinical scores
are constant per patient).

OUTPUTS (reports/)
------------------
  per_model_metrics.csv, delong_comparisons.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from airway import config, delong, fusion, scores
from airway.baseline_model import _classification_metrics

BONFERRONI_ALPHA = 0.0083   # 0.05 / 6 comparisons

PER_MODEL_CSV = config.REPORTS_DIR / "per_model_metrics.csv"
DELONG_CSV = config.REPORTS_DIR / "delong_comparisons.csv"

# model column -> (operating-point threshold, display name)
PROB_THRESHOLD = 0.5
SCORE_COLS = {
    "mallampati_class": scores.MALLAMPATI_DIFFICULT_CLASS,
    "lemon_score": scores.LEMON_DIFFICULT_THRESHOLD,
    "wilson_score": scores.WILSON_DIFFICULT_THRESHOLD,
}


def _load_fold_rows() -> pd.DataFrame:
    """Fusion fold predictions joined with the clinical comparator scores."""
    if not fusion.FUSION_FOLD_PRED_CSV.exists():
        raise FileNotFoundError(
            f"clinical_comparison: {fusion.FUSION_FOLD_PRED_CSV} not found. "
            f"Run `python -m airway.fusion` first."
        )
    rows = pd.read_csv(fusion.FUSION_FOLD_PRED_CSV)

    from airway import loaders
    comp = scores.compute_comparator_scores(loaders.preop_loader())
    keep = [config.ID_COL] + list(SCORE_COLS)
    missing = [c for c in keep if c not in comp.columns]
    if missing:
        raise ValueError(f"clinical_comparison: comparator score columns missing: {missing}")
    merged = rows.merge(comp[keep], on=config.ID_COL, how="left")
    return merged


def youden_threshold(y_true, score) -> float:
    """
    Threshold that maximises Youden's J = sensitivity + specificity - 1, read
    off the ROC curve. Returns NaN if only one outcome class is present (J is
    undefined). The returned value lies within the range of `score`.
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, thresholds = roc_curve(y_true, score)
    j = tpr - fpr
    best = thresholds[int(np.argmax(j))]
    # sklearn prepends an +inf threshold; clamp into the observed score range
    lo, hi = float(np.min(score)), float(np.max(score))
    return float(min(max(best, lo), hi))


def _metrics_at_threshold(y_true, score, threshold) -> dict:
    """Operating-point metrics treating (score >= threshold) as the positive call."""
    y_pred = (np.asarray(score) >= threshold).astype(int)
    # reuse the shared metric core by passing the hard call as a 0/1 "probability"
    m = _classification_metrics(np.asarray(y_true), y_pred.astype(float))
    sens, spec, ppv = m["sensitivity"], m["specificity"], m["ppv"]
    m["balanced_accuracy"] = 0.5 * (sens + spec)
    m["f1"] = (2 * ppv * sens / (ppv + sens)) if (ppv + sens) else 0.0
    return m


def _per_model_row(fold_rows: pd.DataFrame, col: str, threshold: float,
                   modality: str, threshold_type: str = "fixed_0.5") -> dict:
    """Per-fold AUC (mean±SD) + pooled operating-point metrics for one model."""
    per_fold_auc = []
    for _, grp in fold_rows.groupby(["repeat", "fold_index"]):
        sub = grp.dropna(subset=[col])
        y = sub[config.LABEL_COL].to_numpy()
        if len(sub) and len(np.unique(y)) == 2:
            per_fold_auc.append(roc_auc_score(y, sub[col].to_numpy()))

    pooled = fold_rows.dropna(subset=[col])
    y_all = pooled[config.LABEL_COL].to_numpy()
    auc_pooled = (roc_auc_score(y_all, pooled[col].to_numpy())
                  if len(np.unique(y_all)) == 2 else float("nan"))
    m = _metrics_at_threshold(y_all, pooled[col].to_numpy(), threshold)
    return {
        "model": modality,
        "threshold_type": threshold_type,
        "n": int(pooled[config.ID_COL].nunique()),
        "auc_mean": round(float(np.mean(per_fold_auc)) if per_fold_auc else auc_pooled, 4),
        "auc_std": round(float(np.std(per_fold_auc)) if per_fold_auc else 0.0, 4),
        "auc_pooled": round(auc_pooled, 4),
        "threshold": round(float(threshold), 4),
        "sensitivity": round(m["sensitivity"], 4),
        "specificity": round(m["specificity"], 4),
        "ppv": round(m["ppv"], 4),
        "npv": round(m["npv"], 4),
        "accuracy": round(m["accuracy"], 4),
        "balanced_accuracy": round(m["balanced_accuracy"], 4),
        "f1": round(m["f1"], 4),
    }


# probability models: (column, display name)
PROB_MODELS = [
    ("face_prob", "face"), ("us_prob", "ultrasound"),
    ("fused_prob", "fusion:logreg"), ("avg_prob", "fusion:average"),
]


def per_model_metrics(fold_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # probability models at the fixed 0.5 threshold
    for col, name in PROB_MODELS:
        rows.append(_per_model_row(fold_rows, col, PROB_THRESHOLD, name, "fixed_0.5"))
    # the same probability models at their Youden-optimal threshold (operating
    # point chosen to maximise sensitivity + specificity - 1, pooled)
    pooled = fold_rows
    for col, name in PROB_MODELS:
        sub = pooled.dropna(subset=[col])
        thr = youden_threshold(sub[config.LABEL_COL].to_numpy(), sub[col].to_numpy())
        if np.isnan(thr):
            thr = PROB_THRESHOLD
        rows.append(_per_model_row(fold_rows, col, thr, name, "youden"))
    # clinical comparators keep their fixed clinical cut-points (unchanged)
    for col, thr in SCORE_COLS.items():
        rows.append(_per_model_row(fold_rows, col, thr,
                                   col.replace("_class", "").replace("_score", ""),
                                   "fixed_clinical"))
    return pd.DataFrame(rows)


ALPHA_FAMILYWISE = 0.05   # Bonferroni-corrected per-comparison alpha = this / n_comparisons


def _per_patient(fold_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: probs averaged across repeats; scores are constant."""
    candidate = ["face_prob", "us_prob", "fused_prob", "avg_prob",
                 *SCORE_COLS, "clinical_prob"]
    agg = {c: "mean" for c in candidate if c in fold_rows.columns}
    agg[config.LABEL_COL] = "first"
    return fold_rows.groupby(config.ID_COL, as_index=False).agg(agg)


def delong_comparisons(fold_rows: pd.DataFrame) -> pd.DataFrame:
    """
    DeLong tests: fused model vs each comparator (one value per patient).

    The comparator set is the six core models plus, when its per-patient column
    is present, the bedside clinical baseline (`clinical_prob`) — making seven
    comparisons. The Bonferroni-adjusted alpha is ALPHA_FAMILYWISE / n_comparisons,
    recomputed from the actual number of comparisons in the set.
    """
    pp = _per_patient(fold_rows)
    comparators = [
        ("mallampati", "mallampati_class"),
        ("lemon", "lemon_score"),
        ("wilson", "wilson_score"),
        ("face", "face_prob"),
        ("ultrasound", "us_prob"),
        ("average", "avg_prob"),
    ]
    if "clinical_prob" in pp.columns:
        comparators.append(("clinical_baseline", "clinical_prob"))

    alpha = round(ALPHA_FAMILYWISE / len(comparators), 4)
    rows = []
    for name, col in comparators:
        sub = pp.dropna(subset=["fused_prob", col])
        y = sub[config.LABEL_COL].to_numpy()
        if len(np.unique(y)) < 2:
            rows.append({"comparison": f"fused_vs_{name}", "reference": "fused_prob",
                         "comparator": col, "n": int(len(sub)),
                         "auc_fused": float("nan"), "auc_comparator": float("nan"),
                         "auc_diff": float("nan"), "z": float("nan"),
                         "p_value": float("nan"), "alpha_bonferroni": alpha,
                         "significant": False})
            continue
        res = delong.delong_test(y, sub["fused_prob"].to_numpy(), sub[col].to_numpy())
        rows.append({
            "comparison": f"fused_vs_{name}", "reference": "fused_prob",
            "comparator": col, "n": int(len(sub)),
            "auc_fused": round(res["auc_1"], 4), "auc_comparator": round(res["auc_2"], 4),
            "auc_diff": round(res["auc_diff"], 4), "z": round(res["z"], 4),
            "p_value": round(res["p_value"], 5), "alpha_bonferroni": alpha,
            "significant": bool(res["p_value"] < alpha),
        })
    return pd.DataFrame(rows)


def main() -> None:
    config.ensure_dirs()
    fold_rows = _load_fold_rows()

    # Include the bedside clinical baseline as a comparator if it has been run.
    cb_path = config.REPORTS_DIR / "clinical_baseline_probs.csv"
    if cb_path.exists():
        cb = pd.read_csv(cb_path)[[config.ID_COL, "repeat", "fold_index", "clinical_prob"]]
        fold_rows = fold_rows.merge(cb, on=[config.ID_COL, "repeat", "fold_index"], how="left")
        print("clinical_comparison: including clinical baseline as a 7th comparator.")
    else:
        print("clinical_comparison: clinical baseline not found "
              "(run `make clinical-baseline`); using the six core comparators.")

    print(f"clinical_comparison: {fold_rows[config.ID_COL].nunique()} patients, "
          f"reusing {fold_rows.groupby(['repeat', 'fold_index']).ngroups} fusion folds.")

    metrics = per_model_metrics(fold_rows)
    metrics.to_csv(PER_MODEL_CSV, index=False)

    comparisons = delong_comparisons(fold_rows)
    comparisons.to_csv(DELONG_CSV, index=False)

    n_comp = len(comparisons)
    alpha = comparisons["alpha_bonferroni"].iloc[0] if n_comp else float("nan")
    print(f"\nper-model metrics -> {PER_MODEL_CSV}")
    print(f"DeLong comparisons -> {DELONG_CSV}")
    print(f"(Bonferroni alpha = {alpha} for {n_comp} comparisons)\n")
    print(metrics.to_string(index=False))
    print("\nDeLong (fused vs comparator):")
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
