"""
Pilot report -- Week 2 version.

WHAT THIS DOES NOW
------------------
Week 1's version only summarised the cross-validation folds. This version runs
the real Week 2 pipeline end-to-end:

  1. load data (labels + ultrasound + face index)
  2. build the face feature table (ResNet-18 embeddings)
  3. build the ultrasound feature table
  4. cross-validate the baseline model for EACH modality
  5. write a metrics CSV and a ROC plot to reports/

The command stays the same:  make pilot-report

REPRODUCIBILITY
---------------
Every random seed is fixed (config.RANDOM_SEED). Running this twice must
produce an identical metrics CSV. That is the Block A / ongoing check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airway import (
    baseline_model,
    config,
    face_features,
    loaders,
    ultrasound_features,
)


def _save_roc_plot(results: list, out_path) -> None:
    """Draw one ROC curve per modality onto a single figure."""
    # import matplotlib lazily so the rest of the module loads even if a
    # headless environment has plotting issues
    import matplotlib
    matplotlib.use("Agg")              # 'Agg' = write to file, no screen needed
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    plt.figure(figsize=(6, 6))
    for res in results:
        y_true = np.array(res.oof_true)
        y_prob = np.array(res.oof_prob)
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{res.modality} (AUC={res.auc_mean:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    plt.xlabel("False positive rate (1 - specificity)")
    plt.ylabel("True positive rate (sensitivity)")
    plt.title("Baseline model -- ROC by modality")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main() -> None:
    config.ensure_dirs()

    # --- 1. load -----------------------------------------------------------
    patient_table = loaders.build_patient_table()

    # --- 2 & 3. build feature tables --------------------------------------
    print("\n[1/3] building face features (ResNet-18 embeddings) ...")
    face_tbl = face_features.build_and_save_face_features()
    face_cols = [c for c in face_tbl.columns if c.startswith("face_")]

    print("\n[2/3] building ultrasound features ...")
    us_tbl = ultrasound_features.build_and_save_ultrasound_features()
    us_cols = ultrasound_features.US_FEATURE_COLS

    # --- 4. cross-validate the baseline per modality ----------------------
    print("\n[3/3] cross-validating baseline models ...")
    results = []
    results.append(baseline_model.evaluate_modality(
        face_tbl, patient_table, face_cols, modality="face"))
    results.append(baseline_model.evaluate_modality(
        us_tbl, patient_table, us_cols, modality="ultrasound"))

    # --- 5. write outputs --------------------------------------------------
    summary = pd.DataFrame([r.summary_row() for r in results])
    csv_out = config.REPORTS_DIR / "baseline_metrics.csv"
    summary.to_csv(csv_out, index=False)

    roc_out = config.REPORTS_DIR / "baseline_roc.png"
    _save_roc_plot(results, roc_out)

    print(f"\nmetrics  -> {csv_out}")
    print(f"ROC plot -> {roc_out}\n")
    print(summary.to_string(index=False))
    print(
        "\nReminder: these numbers are from DUMMY data. They only prove the "
        "pipeline runs.\nRun this twice -- baseline_metrics.csv must be "
        "identical both times."
    )


if __name__ == "__main__":
    main()
