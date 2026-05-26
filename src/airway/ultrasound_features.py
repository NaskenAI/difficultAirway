"""
Ultrasound feature table.

WHAT THIS DOES
--------------
The ultrasound loader (loaders.us_loader) already reads the raw measurements
and adds the hyomental distance ratio. This module turns those measurements
into a clean, model-ready feature table:

  - keeps the four measured variables + the derived ratio
  - documents exactly which columns are the features
  - leaves the values in their natural units (millimetres)

WHY SO SIMPLE
-------------
Ultrasound data is already tabular numbers -- unlike images, it needs no
embedding. The only real work is being explicit and consistent about which
columns count as features, so the model code never has to guess.

A NOTE ON SCALING AND IMPUTATION
--------------------------------
You might expect standardisation (mean 0, variance 1) and missing-value
imputation here. They are deliberately NOT done in this file. They must be
done INSIDE cross-validation, fitted on the training fold only, to avoid
leakage. The baseline model module handles that with a scikit-learn Pipeline.
This file just delivers clean raw features.

OUTPUT
------
A pandas DataFrame, one row per patient: study_id + the feature columns.
Saved to data/processed/ultrasound_features.parquet.
"""

from __future__ import annotations

import pandas as pd

from airway import config

# The exact ultrasound feature columns, named once here so every other file
# imports this list instead of hard-coding column names.
US_FEATURE_COLS = [
    "dstvc_mm",          # anterior neck soft tissue thickness at vocal cords
    "hmd_neutral_mm",    # hyomental distance, neutral head position
    "hmd_extended_mm",   # hyomental distance, head extended
    "dse_mm",            # distance skin-to-epiglottis
    "hmdr",              # derived: hmd_extended_mm / hmd_neutral_mm
]


def build_ultrasound_features() -> pd.DataFrame:
    """
    Build the per-patient ultrasound feature table.

    Returns
    -------
    DataFrame
        Columns: study_id + US_FEATURE_COLS. One row per patient.
    """
    from airway import loaders

    us = loaders.us_loader()

    missing = [c for c in US_FEATURE_COLS if c not in us.columns]
    if missing:
        raise ValueError(
            f"build_ultrasound_features: expected columns {missing} not found. "
            f"Available: {list(us.columns)}"
        )

    features = us[[config.ID_COL] + US_FEATURE_COLS].copy()

    # report missing values so you are never surprised by them
    n_missing = features[US_FEATURE_COLS].isna().sum().sum()
    if n_missing:
        print(f"build_ultrasound_features: {n_missing} missing values present "
              f"(imputed later, inside cross-validation).")

    return features.reset_index(drop=True)


def build_and_save_ultrasound_features() -> pd.DataFrame:
    """
    Build the ultrasound feature table and save it to
    data/processed/ultrasound_features.parquet.
    """
    config.ensure_dirs()
    features = build_ultrasound_features()
    out = config.PROCESSED_DIR / "ultrasound_features.parquet"
    features.to_parquet(out, index=False)
    print(f"saved ultrasound features -> {out}  (shape {features.shape})")
    return features


if __name__ == "__main__":
    build_and_save_ultrasound_features()
