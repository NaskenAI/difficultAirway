"""
Test suite.

The single most important test is `test_no_patient_appears_in_train_and_test`.
If that test ever fails, STOP — your results would be invalid.

RUN ALL TESTS
-------------
    pytest

Tests assume dummy data exists. If they error with "file not found", run:
    python -m airway.make_dummy_data
"""

from __future__ import annotations

import pandas as pd
import pytest

from airway import config, loaders, splits


# ---------------------------------------------------------------------------
# Fixture: build the patient table once, reuse it across tests.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def patient_table() -> pd.DataFrame:
    if not config.LABELS_CSV.exists():
        pytest.skip(
            "Dummy data missing. Run: python -m airway.make_dummy_data"
        )
    return loaders.build_patient_table()


@pytest.fixture(scope="module")
def face_index() -> pd.DataFrame:
    if not config.FACE_INDEX_CSV.exists():
        pytest.skip("Dummy data missing. Run: python -m airway.make_dummy_data")
    return loaders.face_loader()


# ===========================================================================
# LOADER TESTS
# ===========================================================================
def test_label_loader_has_required_columns(patient_table):
    for col in (config.ID_COL, config.CL_GRADE_COL, config.LABEL_COL):
        assert col in patient_table.columns


def test_label_is_binary_and_correct(patient_table):
    # label must be 0 or 1 only
    assert set(patient_table[config.LABEL_COL].unique()).issubset({0, 1})
    # label must match the CL-grade rule: CL 3-4 -> 1, CL 1-2 -> 0
    expected = patient_table[config.CL_GRADE_COL].isin([3, 4]).astype(int)
    assert (patient_table[config.LABEL_COL] == expected).all()


def test_patient_table_one_row_per_patient(patient_table):
    assert not patient_table[config.ID_COL].duplicated().any()


def test_ultrasound_has_derived_ratio(patient_table):
    assert "hmdr" in patient_table.columns
    # ratio of extended/neutral hyomental distance should be positive
    assert (patient_table["hmdr"] > 0).all()


def test_face_loader_resolves_paths(face_index):
    assert "abs_path" in face_index.columns
    # every catalogued image file should actually exist on disk
    from pathlib import Path
    for p in face_index["abs_path"]:
        assert Path(p).exists(), f"missing image file: {p}"


def test_face_index_is_multi_row_per_patient(face_index):
    # face index is expected to have several images per patient
    counts = face_index.groupby(config.ID_COL).size()
    assert (counts > 1).all()


# ===========================================================================
# SPLIT TESTS  --  the ones that protect your science
# ===========================================================================
def test_folds_count_matches_config(patient_table):
    folds = splits.patient_level_folds(patient_table)
    assert len(folds) == config.N_SPLITS * config.N_REPEATS


def test_no_patient_appears_in_train_and_test(patient_table):
    """THE critical test. A patient in both halves = data leakage = invalid results."""
    folds = splits.patient_level_folds(patient_table)
    for f in folds:
        overlap = set(f.train_ids) & set(f.test_ids)
        assert not overlap, (
            f"LEAKAGE in fold {f.fold_index}: patients in both "
            f"train and test: {sorted(overlap)}"
        )


def test_every_patient_used_for_test_once_per_repeat(patient_table):
    """Across one repeat's 5 folds, each patient is tested exactly once."""
    folds = splits.patient_level_folds(patient_table)
    all_ids = set(patient_table[config.ID_COL])
    for repeat in range(config.N_REPEATS):
        repeat_folds = [f for f in folds if f.repeat == repeat]
        tested = []
        for f in repeat_folds:
            tested.extend(f.test_ids)
        # no patient tested twice within a repeat
        assert len(tested) == len(set(tested))
        # every patient was tested
        assert set(tested) == all_ids


def test_train_and_test_cover_all_patients(patient_table):
    folds = splits.patient_level_folds(patient_table)
    all_ids = set(patient_table[config.ID_COL])
    for f in folds:
        assert set(f.train_ids) | set(f.test_ids) == all_ids


def test_folds_are_reproducible(patient_table):
    """Same seed -> identical folds. This is what makes the project reproducible."""
    folds_a = splits.patient_level_folds(patient_table, seed=123)
    folds_b = splits.patient_level_folds(patient_table, seed=123)
    for a, b in zip(folds_a, folds_b):
        assert a.train_ids == b.train_ids
        assert a.test_ids == b.test_ids


def test_select_rows_brings_all_patient_rows(face_index):
    """Selecting a patient must bring ALL their image rows, not just one."""
    some_ids = face_index[config.ID_COL].unique()[:3].tolist()
    subset = splits.select_rows(face_index, some_ids)
    assert set(subset[config.ID_COL]) == set(some_ids)
    # every selected row belongs to a requested patient
    assert subset[config.ID_COL].isin(some_ids).all()


def test_assert_no_leakage_passes_on_clean_folds(patient_table):
    folds = splits.patient_level_folds(patient_table)
    # should not raise
    splits.assert_no_leakage(folds)
