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

import numpy as np
import pandas as pd

from airway import config

# The measured (raw) ultrasound columns, in millimetres.
US_MEASURED_COLS = [
    "dstvc_mm",          # anterior neck soft tissue thickness at vocal cords
    "hmd_neutral_mm",    # hyomental distance, neutral head position
    "hmd_extended_mm",   # hyomental distance, head extended
    "dse_mm",            # distance skin-to-epiglottis
]

# The derived feature.
HMDR_COL = "hmdr"        # hyomental distance ratio = hmd_extended_mm / hmd_neutral_mm

# The full ultrasound feature schema (measured + derived). Named once here so
# every other file imports this list instead of hard-coding column names.
US_FEATURE_COLS = US_MEASURED_COLS + [HMDR_COL]

# ---------------------------------------------------------------------------
# Schema placeholder mappings.
# If your real export names a column differently, map it here (raw -> schema).
# Extend this dict as you discover the actual column names in your data; the
# cleaning step renames raw -> schema BEFORE anything else runs. Any measured
# column still missing after aliasing is added as an all-NaN placeholder with a
# clear warning, so downstream code always sees a consistent set of columns.
# ---------------------------------------------------------------------------
US_COLUMN_ALIASES = {
    "dstvc": "dstvc_mm", "dstv_cm": "dstvc_mm", "ant_neck_mm": "dstvc_mm",
    "hmd_neutral": "hmd_neutral_mm", "hmdn_mm": "hmd_neutral_mm",
    "hmd_extended": "hmd_extended_mm", "hmde_mm": "hmd_extended_mm",
    "dse": "dse_mm", "skin_epiglottis_mm": "dse_mm",
}


def compute_hmdr(extended, neutral) -> np.ndarray:
    """
    Hyomental distance ratio = extended / neutral, computed safely.

    Returns NaN wherever the ratio is undefined or non-physical:
      - neutral missing / non-numeric
      - neutral == 0 (division by zero)
      - neutral < 0  (non-physical distance)
      - extended missing / non-numeric

    Parameters accept any array-like (Series, list, ndarray) and the result is a
    float ndarray of the same length.
    """
    ext = pd.to_numeric(pd.Series(extended).reset_index(drop=True), errors="coerce").to_numpy(dtype=float)
    neu = pd.to_numeric(pd.Series(neutral).reset_index(drop=True), errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ext / neu
    invalid = ~np.isfinite(neu) | (neu <= 0) | ~np.isfinite(ext)
    return np.where(invalid, np.nan, ratio)


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


# ===========================================================================
# WEEK 6: cleaning + standardisation of the ultrasound numeric features
# ===========================================================================
def clean_ultrasound_features(path=None) -> pd.DataFrame:
    """
    Read, alias, numerically clean, and derive the ultrasound feature table.

    Steps (all deterministic, no fitting — fitting belongs inside CV):
      1. Read the raw CSV (config.ULTRASOUND_CSV). Fail clearly if the patient
         id column is missing.
      2. Apply schema aliases (US_COLUMN_ALIASES): raw column names -> schema.
      3. For any measured column still missing, add an all-NaN placeholder and
         print a clear warning (so columns are consistent downstream).
      4. Coerce measured columns to numeric with errors='coerce'; any value that
         is not a valid number becomes NaN (treated as missing) and is counted.
      5. Compute the derived hyomental distance ratio (hmdr) safely (see
         compute_hmdr): zero, negative, or missing neutral -> NaN.
      6. Fail clearly if EVERY measured value is missing (nothing to model).

    INVALID / MISSING VALUE HANDLING
    --------------------------------
    Non-numeric strings and blanks become NaN. NaNs are NOT filled here — they
    are imputed inside each cross-validation fold (mean imputation) so no
    information leaks across the train/test boundary.

    Returns
    -------
    DataFrame, one row per patient: study_id + US_FEATURE_COLS.
    """
    path = config.ULTRASOUND_CSV if path is None else path
    df = pd.read_csv(path)

    if config.ID_COL not in df.columns:
        raise ValueError(
            f"clean_ultrasound_features: required column '{config.ID_COL}' not "
            f"found in {path}. Columns present: {list(df.columns)}"
        )

    # 2. aliases: only rename when the target schema name is not already present
    rename = {raw: schema for raw, schema in US_COLUMN_ALIASES.items()
              if raw in df.columns and schema not in df.columns}
    if rename:
        print(f"clean_ultrasound_features: applied schema aliases {rename}")
        df = df.rename(columns=rename)

    # 3. placeholders for measured columns still missing from the schema
    for col in US_MEASURED_COLS:
        if col not in df.columns:
            print(f"  WARNING: ultrasound column '{col}' missing; adding an "
                  f"all-NaN placeholder. Map it in US_COLUMN_ALIASES if your "
                  f"export uses a different name.")
            df[col] = np.nan

    # 4. safe numeric conversion
    for col in US_MEASURED_COLS:
        before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        coerced = int(df[col].isna().sum() - before)
        if coerced:
            print(f"  '{col}': {coerced} non-numeric value(s) coerced to NaN "
                  f"(treated as missing).")

    # 5. derived feature
    df[HMDR_COL] = compute_hmdr(df["hmd_extended_mm"], df["hmd_neutral_mm"])

    out = df[[config.ID_COL] + US_FEATURE_COLS].copy()

    if out[config.ID_COL].duplicated().any():
        dups = out.loc[out[config.ID_COL].duplicated(), config.ID_COL].tolist()
        raise ValueError(f"clean_ultrasound_features: duplicate study_id rows: {dups}")

    # 6. fail clearly if there is nothing to model
    if int(out[US_MEASURED_COLS].notna().to_numpy().sum()) == 0:
        raise ValueError(
            "clean_ultrasound_features: every ultrasound measurement is missing; "
            "nothing to model. Check the raw file and US_COLUMN_ALIASES."
        )

    n_missing = int(out[US_FEATURE_COLS].isna().to_numpy().sum())
    if n_missing:
        print(f"clean_ultrasound_features: {n_missing} missing cell(s) remain "
              f"(imputed later, inside cross-validation).")
    return out.reset_index(drop=True)


def usable_feature_cols(cleaned: pd.DataFrame) -> list[str]:
    """
    Return the feature columns that are not ENTIRELY missing across the dataset.

    A column that is all-NaN cannot be imputed or learned from, so it is dropped
    with a clear warning (documented behaviour). Raises if nothing usable remains.
    """
    usable, dropped = [], []
    for col in US_FEATURE_COLS:
        if col in cleaned.columns and cleaned[col].notna().any():
            usable.append(col)
        else:
            dropped.append(col)
    if dropped:
        print(f"usable_feature_cols: dropping entirely-missing feature(s) "
              f"{dropped} (cannot be imputed or modelled).")
    if not usable:
        raise ValueError("usable_feature_cols: no usable ultrasound features remain.")
    return usable


def build_and_save_cleaned_ultrasound() -> pd.DataFrame:
    """Clean the ultrasound features and persist to config.CLEANED_US_CSV."""
    config.ensure_dirs()
    cleaned = clean_ultrasound_features()
    out = config.CLEANED_US_CSV
    cleaned.to_csv(out, index=False)
    print(f"saved cleaned ultrasound features -> {out}  (shape {cleaned.shape})")
    return cleaned


if __name__ == "__main__":
    build_and_save_cleaned_ultrasound()
