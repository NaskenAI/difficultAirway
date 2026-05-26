"""
Data loaders.

WHAT A "LOADER" IS
------------------
A loader is just a function that reads one raw file from disk and returns a
clean pandas DataFrame. Nothing more. It does NOT train models, does NOT
compute features. One job: read a file, hand back a tidy table keyed by
study_id.

WHY SEPARATE LOADERS
--------------------
Each modality (labels, ultrasound, face) lives in its own file with its own
quirks. Isolating the "read it from disk" step means that when your raw data
format changes, you fix ONE loader and the rest of the project is untouched.

THE GOLDEN RULE
---------------
Every loader returns a DataFrame that:
  1. has a column named exactly `study_id` (config.ID_COL)
  2. has one row per study_id (labels, ultrasound) OR is clearly documented
     as multi-row (face_index has several image rows per patient)
"""

from __future__ import annotations

import pandas as pd

from airway import config


# ---------------------------------------------------------------------------
# A small helper used by every loader: confirm the id column exists.
# ---------------------------------------------------------------------------
def _check_id_column(df: pd.DataFrame, source: str) -> None:
    if config.ID_COL not in df.columns:
        raise ValueError(
            f"{source}: expected a column named '{config.ID_COL}' but found "
            f"columns: {list(df.columns)}. "
            f"Rename the patient-identifier column to '{config.ID_COL}'."
        )


# ---------------------------------------------------------------------------
# 1. LABEL LOADER
# ---------------------------------------------------------------------------
def label_loader() -> pd.DataFrame:
    """
    Read the outcome labels.

    Expected raw file (config.LABELS_CSV) has at least:
        study_id   : patient identifier
        cl_grade   : Cormack-Lehane grade, integer 1-4

    Returns
    -------
    DataFrame with columns: study_id, cl_grade, label
        label = 1 if cl_grade in {3, 4}  (difficult airway)
        label = 0 if cl_grade in {1, 2}  (not difficult)
    One row per patient.
    """
    df = pd.read_csv(config.LABELS_CSV)
    _check_id_column(df, "label_loader")

    if config.CL_GRADE_COL not in df.columns:
        raise ValueError(
            f"label_loader: expected column '{config.CL_GRADE_COL}'. "
            f"Found: {list(df.columns)}"
        )

    # Drop rows with no CL grade — you cannot use a patient with no label.
    before = len(df)
    df = df.dropna(subset=[config.CL_GRADE_COL]).copy()
    dropped = before - len(df)
    if dropped:
        print(f"label_loader: dropped {dropped} rows with missing CL grade.")

    df[config.CL_GRADE_COL] = df[config.CL_GRADE_COL].astype(int)

    # Sanity check: CL grade must be 1-4.
    bad = ~df[config.CL_GRADE_COL].isin([1, 2, 3, 4])
    if bad.any():
        raise ValueError(
            f"label_loader: found CL grades outside 1-4: "
            f"{sorted(df.loc[bad, config.CL_GRADE_COL].unique())}"
        )

    # Derive the binary label.
    df[config.LABEL_COL] = df[config.CL_GRADE_COL].isin([3, 4]).astype(int)

    # Confirm one row per patient.
    if df[config.ID_COL].duplicated().any():
        dups = df.loc[df[config.ID_COL].duplicated(), config.ID_COL].tolist()
        raise ValueError(f"label_loader: duplicate study_id rows: {dups}")

    return df[[config.ID_COL, config.CL_GRADE_COL, config.LABEL_COL]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. ULTRASOUND LOADER
# ---------------------------------------------------------------------------
def us_loader() -> pd.DataFrame:
    """
    Read the ultrasound measurements.

    Expected raw file (config.ULTRASOUND_CSV) has at least:
        study_id
        dstvc_mm         : anterior neck soft tissue thickness at vocal cords
        hmd_neutral_mm   : hyomental distance, neutral head position
        hmd_extended_mm  : hyomental distance, head extended
        dse_mm           : distance skin-to-epiglottis

    Returns
    -------
    DataFrame, one row per patient, with the numeric ultrasound columns
    plus a derived `hmdr` (hyomental distance ratio).
    """
    df = pd.read_csv(config.ULTRASOUND_CSV)
    _check_id_column(df, "us_loader")

    expected = ["dstvc_mm", "hmd_neutral_mm", "hmd_extended_mm", "dse_mm"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"us_loader: missing expected ultrasound columns {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Derived feature: hyomental distance ratio = extended / neutral.
    # Guard against division by zero.
    safe_neutral = df["hmd_neutral_mm"].replace(0, pd.NA)
    df["hmdr"] = df["hmd_extended_mm"] / safe_neutral

    if df[config.ID_COL].duplicated().any():
        dups = df.loc[df[config.ID_COL].duplicated(), config.ID_COL].tolist()
        raise ValueError(f"us_loader: duplicate study_id rows: {dups}")

    keep = [config.ID_COL] + expected + ["hmdr"]
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. FACE INDEX LOADER
# ---------------------------------------------------------------------------
def face_loader() -> pd.DataFrame:
    """
    Read the INDEX of facial images (not the pixels — just the catalogue).

    Expected raw file (config.FACE_INDEX_CSV) has at least:
        study_id
        view_code : e.g. 'frontal_rest', 'left_profile'
        file_path : path to the image file, relative to config.FACE_IMAGE_DIR

    Returns
    -------
    DataFrame with MULTIPLE rows per patient (one per image), columns:
        study_id, view_code, abs_path
    `abs_path` is the resolved absolute path to the image on disk.

    NOTE: this loader does NOT open the images. Feature extraction in Block B
    will do that. Week 1 only needs the catalogue to exist and be valid.
    """
    df = pd.read_csv(config.FACE_INDEX_CSV)
    _check_id_column(df, "face_loader")

    for col in ["view_code", "file_path"]:
        if col not in df.columns:
            raise ValueError(
                f"face_loader: missing column '{col}'. Found: {list(df.columns)}"
            )

    # Resolve each file_path to an absolute path under FACE_IMAGE_DIR.
    df["abs_path"] = df["file_path"].apply(
        lambda rel: str((config.FACE_IMAGE_DIR / rel).resolve())
    )

    return df[[config.ID_COL, "view_code", "file_path", "abs_path"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. MERGE HELPER — build the patient-level table
# ---------------------------------------------------------------------------
def build_patient_table() -> pd.DataFrame:
    """
    Join labels + ultrasound into ONE patient-level table.

    Face images stay separate (multi-row), so they are NOT merged here.
    This merged table is what the train/test split utility operates on.

    Returns
    -------
    DataFrame, one row per patient, with study_id, cl_grade, label,
    and all ultrasound columns. Only patients present in BOTH labels and
    ultrasound are kept (an inner join), and a note is printed about any
    patients dropped.
    """
    labels = label_loader()
    us = us_loader()

    merged = labels.merge(us, on=config.ID_COL, how="inner")

    n_labels_only = len(set(labels[config.ID_COL]) - set(us[config.ID_COL]))
    n_us_only = len(set(us[config.ID_COL]) - set(labels[config.ID_COL]))
    if n_labels_only:
        print(f"build_patient_table: {n_labels_only} patients have labels but no ultrasound.")
    if n_us_only:
        print(f"build_patient_table: {n_us_only} patients have ultrasound but no label.")

    print(f"build_patient_table: {len(merged)} patients in the final joined table.")
    return merged
