"""
Block C / Weeks 9-10 — late fusion of the calibrated modality probabilities.

WHAT THIS DOES
--------------
Reads the out-of-fold calibrated probabilities written by `airway.calibration`
(one per patient per fold, for face and ultrasound) and fuses them two ways:

  1. LEARNED fusion: a logistic-regression meta-learner with inputs
     [calibrated_face_prob, calibrated_us_prob].
  2. AVERAGE baseline (no learning): (face_prob + ultrasound_prob) / 2.

Both are evaluated with the SAME patient-level stratified 5x2 folds used for
calibration — the fold membership is read straight from the calibrated-prob
files, so the folds are identical by construction (no recomputation).

LEAKAGE DISCIPLINE
------------------
The meta-features are out-of-fold calibrated probabilities: each patient's
calibrated prob came from a base model that never saw that patient. Within each
fusion fold, the meta-learner is trained on the TRAIN patients' calibrated probs
and evaluated on the VAL patients' calibrated probs; validation labels never
enter meta-training.

SANITY CHECK
------------
A learned fusion that cannot beat the trivial average baseline is suspicious
(modalities not complementary, or a bug). We do NOT fail the pipeline, but print
and persist a strong warning when learned AUC <= average AUC.

OUTPUTS (reports/)
------------------
  fused_model.pkl, fusion_cv_metrics.csv, fusion_roc.png,
  fusion_fold_predictions.csv, fusion_average_baseline_metrics.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from airway import calibration, config, face_model
from airway.baseline_model import CVResult, _classification_metrics

FUSED_PKL = config.REPORTS_DIR / "fused_model.pkl"
FUSION_METRICS_CSV = config.REPORTS_DIR / "fusion_cv_metrics.csv"
FUSION_ROC_PNG = config.REPORTS_DIR / "fusion_roc.png"
FUSION_FOLD_PRED_CSV = config.REPORTS_DIR / "fusion_fold_predictions.csv"
FUSION_AVG_CSV = config.REPORTS_DIR / "fusion_average_baseline_metrics.csv"

FACE_PROB = "face_prob"
US_PROB = "us_prob"
FUSION_INPUTS = [FACE_PROB, US_PROB]


def _meta_learner() -> LogisticRegression:
    """The fusion meta-learner: balanced L2 logistic regression on 2 inputs."""
    return LogisticRegression(class_weight="balanced", max_iter=1000,
                              random_state=config.RANDOM_SEED)


def load_calibrated_probs() -> pd.DataFrame:
    """
    Join the face + ultrasound calibrated-prob files into one aligned table.

    Returns columns: study_id, repeat, fold_index, label, face_prob, us_prob.
    Fails clearly if a calibration output is missing.
    """
    for path in (calibration.CAL_FACE_CSV, calibration.CAL_US_CSV):
        if not path.exists():
            raise FileNotFoundError(
                f"fusion: calibrated probabilities not found at {path}. "
                f"Run `python -m airway.calibration` first."
            )
    face = pd.read_csv(calibration.CAL_FACE_CSV)
    us = pd.read_csv(calibration.CAL_US_CSV)
    keys = [config.ID_COL, "repeat", "fold_index", config.LABEL_COL]
    merged = face.merge(us, on=keys, suffixes=("_face", "_us"))
    if len(merged) != len(face) or len(merged) != len(us):
        raise ValueError(
            "fusion: face and ultrasound calibrated-prob rows do not align on "
            f"{keys}. face={len(face)} us={len(us)} merged={len(merged)}. "
            "Re-run calibration so both modalities share the same cohort/folds."
        )
    merged = merged.rename(columns={
        f"{calibration.PROB_COL}_face": FACE_PROB,
        f"{calibration.PROB_COL}_us": US_PROB,
    })
    return merged[keys + [FACE_PROB, US_PROB]]


def run_fusion(merged: pd.DataFrame):
    """
    Train/evaluate the meta-learner and the average baseline over the shared
    folds. Returns (fold_predictions_df, fused_result, avg_result).
    """
    rows = []
    for repeat, rep in merged.groupby("repeat"):
        for fold_index, val in rep.groupby("fold_index"):
            # train = every OTHER patient in this repeat (each is val exactly once)
            train = rep[rep["fold_index"] != fold_index]
            x_tr = train[FUSION_INPUTS].to_numpy()
            y_tr = train[config.LABEL_COL].to_numpy()
            x_val = val[FUSION_INPUTS].to_numpy()

            meta = _meta_learner()
            meta.fit(x_tr, y_tr)
            fused = meta.predict_proba(x_val)[:, 1]
            avg = val[FUSION_INPUTS].mean(axis=1).to_numpy()

            for (_, r), fp, ap in zip(val.iterrows(), fused, avg):
                rows.append({
                    config.ID_COL: r[config.ID_COL], "repeat": int(repeat),
                    "fold_index": int(fold_index), config.LABEL_COL: int(r[config.LABEL_COL]),
                    FACE_PROB: float(r[FACE_PROB]), US_PROB: float(r[US_PROB]),
                    "fused_prob": float(fp), "avg_prob": float(ap),
                })

    preds = pd.DataFrame(rows)
    fused_result = _result_from_oof(preds, "fused_prob", "fusion:logreg")
    avg_result = _result_from_oof(preds, "avg_prob", "fusion:average")
    return preds, fused_result, avg_result


def _result_from_oof(preds: pd.DataFrame, prob_col: str, modality: str) -> CVResult:
    """Build a CVResult (per-fold AUC + pooled metrics) from OOF predictions."""
    per_fold_auc = []
    for (_, _), grp in preds.groupby(["repeat", "fold_index"]):
        y = grp[config.LABEL_COL].to_numpy()
        if len(np.unique(y)) == 2:
            per_fold_auc.append(roc_auc_score(y, grp[prob_col].to_numpy()))

    y_all = preds[config.LABEL_COL].to_numpy()
    p_all = preds[prob_col].to_numpy()
    pooled = roc_auc_score(y_all, p_all) if len(np.unique(y_all)) == 2 else float("nan")
    m = _classification_metrics(y_all, p_all)
    return CVResult(
        modality=modality, n_patients=int(preds[config.ID_COL].nunique()),
        n_features=len(FUSION_INPUTS),
        auc_mean=float(np.mean(per_fold_auc)) if per_fold_auc else pooled,
        auc_std=float(np.std(per_fold_auc)) if per_fold_auc else 0.0,
        sensitivity=m["sensitivity"], specificity=m["specificity"],
        accuracy=m["accuracy"], ppv=m["ppv"], npv=m["npv"],
        per_fold_auc=per_fold_auc,
        oof_true=y_all.tolist(), oof_prob=p_all.tolist(),
    )


def main() -> None:
    import joblib

    config.ensure_dirs()
    merged = load_calibrated_probs()
    print(f"fusion: {len(merged)} patient-fold rows "
          f"({merged[config.ID_COL].nunique()} patients).")

    preds, fused_result, avg_result = run_fusion(merged)

    # --- sanity check: learned fusion vs average baseline -------------------
    note = ""
    if not (fused_result.auc_mean > avg_result.auc_mean):
        note = (
            "WARNING: the learned fusion did NOT beat the unweighted-average "
            f"baseline (learned AUC={fused_result.auc_mean:.3f} <= "
            f"average AUC={avg_result.auc_mean:.3f}). The modalities may not be "
            "complementary, the signal may be too weak (e.g. dummy data), or "
            "there may be a bug. This is a warning, not a failure."
        )
        print("\n" + note + "\n")
    else:
        print(f"\nsanity check OK: learned fusion AUC={fused_result.auc_mean:.3f} > "
              f"average AUC={avg_result.auc_mean:.3f}\n")

    # --- write outputs ------------------------------------------------------
    preds.to_csv(FUSION_FOLD_PRED_CSV, index=False)

    summary = pd.DataFrame([fused_result.summary_row(), avg_result.summary_row()])
    if note:
        summary["note"] = ["", ""]
        summary.loc[summary["modality"] == "fusion:logreg", "note"] = note
    summary.to_csv(FUSION_METRICS_CSV, index=False)

    pd.DataFrame([avg_result.summary_row()]).to_csv(FUSION_AVG_CSV, index=False)
    face_model._save_roc([fused_result, avg_result], FUSION_ROC_PNG,
                         title="Late fusion — pooled out-of-fold ROC")

    # refit the meta-learner on ALL out-of-fold calibrated probs and persist
    meta = _meta_learner()
    meta.fit(merged[FUSION_INPUTS].to_numpy(), merged[config.LABEL_COL].to_numpy())
    bundle = {
        "meta_learner": meta,
        "inputs": FUSION_INPUTS,
        "outcome": "difficult airway (CL 3-4)",
        "cv": {fused_result.modality: fused_result.summary_row(),
               avg_result.modality: avg_result.summary_row()},
        "beats_average_baseline": bool(fused_result.auc_mean > avg_result.auc_mean),
        "n_patients": int(merged[config.ID_COL].nunique()),
        "seed": config.RANDOM_SEED,
    }
    joblib.dump(bundle, FUSED_PKL)

    print(f"fold predictions -> {FUSION_FOLD_PRED_CSV}")
    print(f"metrics          -> {FUSION_METRICS_CSV}")
    print(f"average baseline -> {FUSION_AVG_CSV}")
    print(f"ROC              -> {FUSION_ROC_PNG}")
    print(f"fused model      -> {FUSED_PKL}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
