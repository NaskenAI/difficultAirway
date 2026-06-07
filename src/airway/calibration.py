"""
Block C / Week 8 — isotonic calibration of the single-modality models.

WHAT THIS DOES
--------------
For each modality (face, ultrasound) it refits the PRIMARY single-modality model
(L2 logistic regression — see PRIMARY_MODEL) wrapped in isotonic calibration,
using the project's patient-level stratified 5x2 CV, and persists the calibrated
probability for every patient in every validation fold.

WHY A COMMON COHORT + SHARED FOLDS
----------------------------------
Late fusion and the clinical comparison need the two modalities' calibrated
probabilities aligned per patient and per fold. So we build ONE common cohort
(patients with face features AND ultrasound features AND a label) and generate
the folds ONCE on that cohort; both modalities are calibrated on those exact
folds. The fold membership is written into the output CSVs so downstream steps
(fusion, clinical comparison) reuse the identical folds without recomputation.

LEAKAGE DISCIPLINE
------------------
Within each outer fold, CalibratedClassifierCV fits the base model and the
isotonic calibrator using an INNER cross-validation on the training patients
only; the validation patients are scored out-of-sample. Nothing about a
validation patient influences their own calibrated probability.

OUTPUTS (reports/ unless noted)
-------------------------------
  calibrated_face_probs.csv, calibrated_us_probs.csv   (one row per patient per fold)
  calibration_metrics.csv                              (Brier per fold + overall)
  face_calibration.png, us_calibration.png             (reliability diagrams)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from airway import config, face_model, splits, ultrasound_features, ultrasound_model

# The primary single-modality model used for calibration and fusion. L2 logistic
# regression is chosen for both modalities: it calibrates cleanly and is a
# natural, deterministic choice. (XGBoost remains available in the face/US model
# modules for the standalone per-modality reports.)
PRIMARY_MODEL = "logreg_l2"

PROB_COL = "calibrated_prob"
FOLD_COLS = ["study_id", "repeat", "fold_index"]

CAL_FACE_CSV = config.REPORTS_DIR / "calibrated_face_probs.csv"
CAL_US_CSV = config.REPORTS_DIR / "calibrated_us_probs.csv"
CAL_METRICS_CSV = config.REPORTS_DIR / "calibration_metrics.csv"
FACE_CAL_PNG = config.REPORTS_DIR / "face_calibration.png"
US_CAL_PNG = config.REPORTS_DIR / "us_calibration.png"


def _safe_inner_cv(y_train: np.ndarray, max_splits: int = 3) -> int:
    """Largest inner CV (<= max_splits) with each class present in every split."""
    min_class = int(min(np.sum(y_train == 0), np.sum(y_train == 1)))
    return min(max_splits, min_class)   # 0 or 1 -> not calibratable


def load_common_cohort():
    """
    Assemble the common cohort and the shared folds.

    Returns
    -------
    dict with: face_data, face_cols, us_data, us_cols, folds, n_common
    Each *_data is one row per patient with study_id + features + label.

    Raises clearly if the face features are missing (run the Weeks 4-5 pipeline).
    """
    from airway import loaders

    if not config.FACE_FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"calibration: face features not found at {config.FACE_FEATURES_PARQUET}. "
            f"Run `make week45` (or `python -m airway.face_embeddings`) first."
        )

    face = pd.read_parquet(config.FACE_FEATURES_PARQUET)
    face_cols = [c for c in face.columns if c.startswith(config.FACE_FEATURE_PREFIX)]
    if not face_cols:
        raise ValueError("calibration: no face feature columns found in the parquet.")

    us = ultrasound_features.clean_ultrasound_features()
    us_cols = ultrasound_features.usable_feature_cols(us)

    patient = loaders.build_patient_table()[[config.ID_COL, config.LABEL_COL]]

    common = (set(face[config.ID_COL]) & set(us[config.ID_COL])
              & set(patient[config.ID_COL]))
    if not common:
        raise ValueError("calibration: no patients have face + ultrasound + label.")
    common = sorted(common)

    face_data = (face[face[config.ID_COL].isin(common)]
                 .merge(patient, on=config.ID_COL, how="inner").reset_index(drop=True))
    us_data = (us[us[config.ID_COL].isin(common)]
               .merge(patient, on=config.ID_COL, how="inner").reset_index(drop=True))

    cohort_table = (patient[patient[config.ID_COL].isin(common)]
                    .sort_values(config.ID_COL).reset_index(drop=True))
    folds = splits.patient_level_folds(cohort_table)
    splits.assert_no_leakage(folds)

    print(f"calibration: common cohort of {len(common)} patients "
          f"(face+ultrasound+label); {len(folds)} folds.")
    return {"face_data": face_data, "face_cols": face_cols,
            "us_data": us_data, "us_cols": us_cols,
            "folds": folds, "n_common": len(common)}


def calibrate_modality(data: pd.DataFrame, feature_cols: list[str],
                       make_pipeline, folds, modality: str):
    """
    Produce out-of-fold isotonic-calibrated probabilities for one modality.

    Returns (probs_df, brier_df):
      probs_df : study_id, repeat, fold_index, label, calibrated_prob  (val rows)
      brier_df : modality, scope, repeat, fold_index, n, brier
    """
    prob_rows, brier_rows = [], []
    for fold in folds:
        train = splits.select_rows(data, fold.train_ids)
        test = splits.select_rows(data, fold.test_ids)
        x_tr, y_tr = train[feature_cols].to_numpy(), train[config.LABEL_COL].to_numpy()
        x_te, y_te = test[feature_cols].to_numpy(), test[config.LABEL_COL].to_numpy()
        ids_te = test[config.ID_COL].tolist()

        inner = _safe_inner_cv(y_tr)
        base = make_pipeline(y_tr)
        if inner >= 2:
            clf = CalibratedClassifierCV(base, method="isotonic", cv=inner)
            calibrated = True
        else:
            print(f"  WARNING [{modality} fold {fold.fold_index}]: only "
                  f"{inner} usable inner split(s); using UNCALIBRATED probabilities "
                  f"for this fold (too few minority cases to calibrate).")
            clf = base
            calibrated = False
        clf.fit(x_tr, y_tr)
        prob = clf.predict_proba(x_te)[:, 1]

        for pid, p, y in zip(ids_te, prob, y_te):
            prob_rows.append({config.ID_COL: pid, "repeat": fold.repeat,
                              "fold_index": fold.fold_index, config.LABEL_COL: int(y),
                              PROB_COL: float(p)})
        brier = float(brier_score_loss(y_te, prob)) if len(y_te) else float("nan")
        brier_rows.append({"modality": modality, "scope": "fold",
                           "repeat": fold.repeat, "fold_index": fold.fold_index,
                           "n": int(len(y_te)), "brier": round(brier, 5),
                           "calibrated": calibrated})

    probs_df = pd.DataFrame(prob_rows)
    brier_df = pd.DataFrame(brier_rows)

    # overall (pooled) Brier across all validation predictions
    pooled = float(brier_score_loss(probs_df[config.LABEL_COL], probs_df[PROB_COL]))
    fold_briers = brier_df["brier"].to_numpy()
    brier_df = pd.concat([brier_df, pd.DataFrame([{
        "modality": modality, "scope": "overall", "repeat": "all", "fold_index": "all",
        "n": int(len(probs_df)), "brier": round(pooled, 5),
        "calibrated": bool(brier_df["calibrated"].all()),
    }])], ignore_index=True)
    print(f"  {modality}: pooled Brier={pooled:.4f}, "
          f"mean-of-folds={np.nanmean(fold_briers):.4f} ± {np.nanstd(fold_briers):.4f}")
    return probs_df, brier_df


def _save_calibration_plot(probs_df: pd.DataFrame, out_path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    y = probs_df[config.LABEL_COL].to_numpy()
    p = probs_df[PROB_COL].to_numpy()
    n_bins = min(10, max(3, len(np.unique(p))))
    plt.figure(figsize=(6, 6))
    if len(np.unique(y)) == 2:
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        plt.plot(mean_pred, frac_pos, "o-", label="calibrated model")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction of difficult airways")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main() -> None:
    config.ensure_dirs()
    cohort = load_common_cohort()

    print("[1/2] calibrating face model (isotonic, within-fold) ...")
    face_probs, face_brier = calibrate_modality(
        cohort["face_data"], cohort["face_cols"],
        face_model.make_logreg_pipeline, cohort["folds"], "face")

    print("[2/2] calibrating ultrasound model (isotonic, within-fold) ...")
    us_probs, us_brier = calibrate_modality(
        cohort["us_data"], cohort["us_cols"],
        ultrasound_model.make_us_logreg_pipeline, cohort["folds"], "ultrasound")

    face_probs.to_csv(CAL_FACE_CSV, index=False)
    us_probs.to_csv(CAL_US_CSV, index=False)
    pd.concat([face_brier, us_brier], ignore_index=True).to_csv(CAL_METRICS_CSV, index=False)

    _save_calibration_plot(face_probs, FACE_CAL_PNG, "Face model — calibration (OOF)")
    _save_calibration_plot(us_probs, US_CAL_PNG, "Ultrasound model — calibration (OOF)")

    print(f"\ncalibrated face probs -> {CAL_FACE_CSV}")
    print(f"calibrated US probs   -> {CAL_US_CSV}")
    print(f"calibration metrics   -> {CAL_METRICS_CSV}")
    print(f"calibration plots     -> {FACE_CAL_PNG} , {US_CAL_PNG}")


if __name__ == "__main__":
    main()
