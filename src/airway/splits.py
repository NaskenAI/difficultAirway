"""
Patient-level cross-validation splitter.

============================================================================
THIS IS THE MOST IMPORTANT FILE IN THE PROJECT. READ THIS COMMENT.
============================================================================

THE PROBLEM IT SOLVES
---------------------
Each patient has SEVERAL facial images. A naive train/test split shuffles
ROWS. If patient P's frontal image lands in the training set and patient P's
profile image lands in the test set, the model has effectively "seen" the
test patient during training. The reported accuracy is then a lie — inflated,
not real. This is called DATA LEAKAGE and it is the single most common reason
medical-AI pilot results fail to reproduce.

THE RULE
--------
Every image, every measurement, every row belonging to one patient must be
ENTIRELY in the training set OR entirely in the test set. Never split a
patient across the two.

HOW THIS FILE ENFORCES IT
-------------------------
`patient_level_folds()` splits by UNIQUE study_id, not by row. It returns,
for each fold, the list of patient IDs for train and the list for test.
You then select rows by those IDs.

DO NOT, ANYWHERE IN THIS PROJECT, CALL sklearn.train_test_split DIRECTLY ON
THE DATA ROWS. Always go through this file. The test in tests/ checks this
function never lets a patient appear in both halves.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from airway import config


@dataclass
class Fold:
    """One cross-validation fold, described by patient IDs (not row indices)."""

    fold_index: int          # 0, 1, 2, ... across all repeats
    repeat: int              # which repeat this fold belongs to
    train_ids: list          # study_id values for training
    test_ids: list           # study_id values for testing


def patient_level_folds(
    patient_table: pd.DataFrame,
    n_splits: int = config.N_SPLITS,
    n_repeats: int = config.N_REPEATS,
    seed: int = config.RANDOM_SEED,
) -> list[Fold]:
    """
    Produce repeated, stratified, PATIENT-LEVEL cross-validation folds.

    Parameters
    ----------
    patient_table : DataFrame
        One row per patient. Must contain config.ID_COL and config.LABEL_COL.
    n_splits : int
        Folds per repeat (default 5).
    n_repeats : int
        How many times to repeat the whole CV (default 2 -> "5x2 CV").
    seed : int
        Random seed. Fixed seed = identical folds every run = reproducible.

    Returns
    -------
    list[Fold]
        Length n_splits * n_repeats. Each Fold carries train/test study_ids.

    Notes
    -----
    Stratified means each fold keeps roughly the same proportion of difficult
    airways as the whole dataset — important because difficult airway is rare.
    """
    # --- Validate input -----------------------------------------------------
    for col in (config.ID_COL, config.LABEL_COL):
        if col not in patient_table.columns:
            raise ValueError(
                f"patient_level_folds: patient_table needs column '{col}'."
            )

    # Confirm one row per patient. If not, the caller passed the wrong table.
    if patient_table[config.ID_COL].duplicated().any():
        raise ValueError(
            "patient_level_folds: patient_table has duplicate study_id rows. "
            "Pass a table with exactly one row per patient (use "
            "loaders.build_patient_table())."
        )

    # One id, one label.
    ids = patient_table[config.ID_COL].to_numpy()
    labels = patient_table[config.LABEL_COL].to_numpy()

    folds: list[Fold] = []
    fold_counter = 0

    for repeat in range(n_repeats):
        # A different seed per repeat so the two repeats are not identical,
        # but still fully determined by the master seed.
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + repeat,
        )
        # skf splits POSITIONS in the patient array. Because the array has one
        # entry per patient, splitting positions == splitting patients.
        for train_pos, test_pos in skf.split(ids, labels):
            folds.append(
                Fold(
                    fold_index=fold_counter,
                    repeat=repeat,
                    train_ids=ids[train_pos].tolist(),
                    test_ids=ids[test_pos].tolist(),
                )
            )
            fold_counter += 1

    return folds


def select_rows(df: pd.DataFrame, ids: list) -> pd.DataFrame:
    """
    Return the subset of `df` whose study_id is in `ids`.

    Works for BOTH the patient-level table (one row per patient) and the
    face index (many rows per patient) — that is the whole point: you select
    by patient, and every row for that patient comes along.
    """
    if config.ID_COL not in df.columns:
        raise ValueError(f"select_rows: df needs column '{config.ID_COL}'.")
    return df[df[config.ID_COL].isin(set(ids))].copy()


def assert_no_leakage(folds: list[Fold]) -> None:
    """
    Raise an error if ANY patient appears in both train and test of any fold.

    Call this defensively before training. It is cheap insurance.
    """
    for f in folds:
        overlap = set(f.train_ids) & set(f.test_ids)
        if overlap:
            raise AssertionError(
                f"LEAKAGE in fold {f.fold_index}: {len(overlap)} patient(s) "
                f"in both train and test: {sorted(overlap)[:5]}..."
            )
