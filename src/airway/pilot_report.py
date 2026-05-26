"""
Pilot report — Week 1 placeholder version.

Right now this just proves the pipeline runs end-to-end: load data, make
folds, confirm no leakage, write a tiny CSV. Week 2 replaces the middle of
this with real feature extraction and a baseline model.

The point of having it NOW: `make pilot-report` must run and produce
identical output on two consecutive runs. That is Block A's exit criterion.
"""

from __future__ import annotations

import pandas as pd

from airway import config, loaders, splits


def main() -> None:
    config.ensure_dirs()

    # 1. Load
    patient_table = loaders.build_patient_table()

    # 2. Split (patient-level)
    folds = splits.patient_level_folds(patient_table)
    splits.assert_no_leakage(folds)

    # 3. Summarise each fold into a small table
    rows = []
    for f in folds:
        train = splits.select_rows(patient_table, f.train_ids)
        test = splits.select_rows(patient_table, f.test_ids)
        rows.append({
            "fold": f.fold_index,
            "repeat": f.repeat,
            "n_train": len(train),
            "n_test": len(test),
            "train_difficult": int(train[config.LABEL_COL].sum()),
            "test_difficult": int(test[config.LABEL_COL].sum()),
        })
    summary = pd.DataFrame(rows)

    # 4. Write the report
    out = config.REPORTS_DIR / "fold_summary.csv"
    summary.to_csv(out, index=False)
    print(f"\nPilot report written: {out}")
    print(summary.to_string(index=False))
    print("\nBlock A check: run this twice; the CSV must be byte-identical.")


if __name__ == "__main__":
    main()
