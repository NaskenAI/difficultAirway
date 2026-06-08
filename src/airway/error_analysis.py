"""
Block D — error analysis for manual review.

WHAT THIS DOES
--------------
Classifies every patient as TP / TN / FP / FN from the fused model (one row per
patient: fused_prob averaged across folds, thresholded at 0.5), joins back
demographics (age, sex, BMI) and the Cormack-Lehane grade, and writes:

  - error_analysis.csv          : every patient with its category + context
  - error_analysis_summary.md   : category counts, plus the FALSE NEGATIVE and
                                  FALSE POSITIVE tables (the clinically important
                                  cases) rendered as markdown for hand review.

This module only assembles the cases; it does no clinical interpretation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airway import config, predictions

THRESHOLD = 0.5
OUT_CSV = config.REPORTS_DIR / "error_analysis.csv"
OUT_MD = config.REPORTS_DIR / "error_analysis_summary.md"

OUT_COLS = ["study_id", "label", "fused_prob", "predicted", "category",
            "age_years", "sex", "bmi", "cl_grade"]


def build_error_table() -> pd.DataFrame:
    """One row per patient with predicted class, confusion category, and context."""
    preds = predictions.load_fusion_predictions()
    pp = predictions.per_patient(preds)[[config.ID_COL, "fused_prob", config.LABEL_COL]].copy()
    pp["predicted"] = predictions.predicted_class(pp["fused_prob"], THRESHOLD)
    pp["category"] = [predictions.confusion_category(int(y), int(p))
                      for y, p in zip(pp[config.LABEL_COL], pp["predicted"])]

    # demographics
    from airway import loaders
    preop = loaders.preop_loader()
    demo = [c for c in ["age_years", "sex", "bmi"] if c in preop.columns]
    if demo:
        pp = pp.merge(preop[[config.ID_COL, *demo]], on=config.ID_COL, how="left")

    # CL grade from labels
    labels = loaders.label_loader()
    if config.CL_GRADE_COL in labels.columns:
        pp = pp.merge(labels[[config.ID_COL, config.CL_GRADE_COL]],
                      on=config.ID_COL, how="left")

    pp = pp.rename(columns={config.LABEL_COL: "label"})
    pp["fused_prob"] = pp["fused_prob"].round(4)
    for col in OUT_COLS:
        if col not in pp.columns:
            pp[col] = np.nan
    return pp[OUT_COLS].sort_values("study_id").reset_index(drop=True)


def _write_summary(table: pd.DataFrame) -> None:
    counts = table["category"].value_counts().to_dict()
    fn = table[table["category"] == "FN"]
    fp = table[table["category"] == "FP"]

    def _md(df: pd.DataFrame) -> str:
        if df.empty:
            return "_none_"
        head = "| " + " | ".join(df.columns) + " |"
        sep = "| " + " | ".join("---" for _ in df.columns) + " |"
        body = ["| " + " | ".join(str(v) for v in row) + " |"
                for row in df.itertuples(index=False, name=None)]
        return "\n".join([head, sep, *body])

    lines = [
        "# Error Analysis Summary",
        "",
        "_Cases for MANUAL clinical review. No interpretation is included._",
        "",
        "## Category counts",
        "",
        f"- True positives (TP): {counts.get('TP', 0)}",
        f"- True negatives (TN): {counts.get('TN', 0)}",
        f"- False positives (FP): {counts.get('FP', 0)}",
        f"- False negatives (FN): {counts.get('FN', 0)}",
        f"- Total patients: {len(table)}",
        "",
        "## False negatives (missed difficult airways)",
        "",
        _md(fn),
        "",
        "## False positives (flagged but not difficult)",
        "",
        _md(fp),
        "",
    ]
    OUT_MD.write_text("\n".join(lines))


def main() -> None:
    config.ensure_dirs()
    table = build_error_table()
    table.to_csv(OUT_CSV, index=False)
    _write_summary(table)
    counts = table["category"].value_counts().to_dict()
    print(f"error analysis -> {OUT_CSV}  ({len(table)} patients: {counts})")
    print(f"summary        -> {OUT_MD}")


if __name__ == "__main__":
    main()
