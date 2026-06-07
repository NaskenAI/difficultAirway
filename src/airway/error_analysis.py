"""
Block D / Week 14 — error-analysis export (support for MANUAL review).

WHAT THIS DOES
--------------
Exports two tables for the clinician to review by hand: all false negatives and
all false positives of the fused model (at threshold 0.5, one row per patient,
probabilities averaged across CV repeats). Each row carries everything a
reviewer needs in one place: patient id, the validation fold(s), true label,
predicted probability/class, available demographics + surgery type, the clinical
comparator scores, every modality probability, and the available ultrasound
features.

This module ONLY assembles tables. It does no interpretation.

OUTPUTS
-------
reports/false_negatives_for_manual_review.csv
reports/false_positives_for_manual_review.csv
"""

from __future__ import annotations

from airway import config, predictions, ultrasound_features

FN_CSV = config.REPORTS_DIR / "false_negatives_for_manual_review.csv"
FP_CSV = config.REPORTS_DIR / "false_positives_for_manual_review.csv"


def build_review_tables(threshold: float = predictions.DEFAULT_THRESHOLD):
    """Return (false_negatives_df, false_positives_df), one row per patient."""
    master = predictions.build_master_table()
    pp = predictions.per_patient(master)

    pp = pp.copy()
    pp["predicted_class"] = predictions.predicted_class(pp["fused_prob"], threshold)
    pp["error_type"] = [predictions.confusion_category(int(y), int(p))
                        for y, p in zip(pp[config.LABEL_COL], pp["predicted_class"])]
    pp = pp.rename(columns={"fused_prob": "predicted_prob"})

    demo_cols = [c for c in (config.DEMOGRAPHIC_COLS + [config.SURGERY_TYPE_COL])
                 if c in pp.columns]
    us_cols = [c for c in ultrasound_features.US_FEATURE_COLS if c in pp.columns]
    ordered = ([config.ID_COL, "folds", config.LABEL_COL, "predicted_prob",
                "predicted_class", "error_type"]
               + demo_cols + predictions.SCORE_COLS
               + ["face_prob", "us_prob", "avg_prob"] + us_cols)
    ordered = [c for c in ordered if c in pp.columns]

    fn = pp[pp["error_type"] == "FN"][ordered].sort_values(config.ID_COL).reset_index(drop=True)
    fp = pp[pp["error_type"] == "FP"][ordered].sort_values(config.ID_COL).reset_index(drop=True)
    return fn, fp


def main() -> None:
    config.ensure_dirs()
    fn, fp = build_review_tables()
    fn.to_csv(FN_CSV, index=False)
    fp.to_csv(FP_CSV, index=False)
    print(f"false negatives -> {FN_CSV}  ({len(fn)} patients)")
    print(f"false positives -> {FP_CSV}  ({len(fp)} patients)")
    print("These tables are for MANUAL clinical review; no interpretation is included.")


if __name__ == "__main__":
    main()
