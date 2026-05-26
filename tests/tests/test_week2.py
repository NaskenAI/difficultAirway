"""
Week 2 test suite.

Tests the new Week 2 modules: face alignment, face features, ultrasound
features, and the baseline model. As in Week 1, these run against dummy data.

RUN
---
    pytest                       # runs Week 1 + Week 2 tests
    pytest tests/test_week2.py   # just these

If tests skip with "Dummy data missing", run:
    python -m airway.make_dummy_data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airway import (
    baseline_model,
    config,
    face_align,
    face_features,
    loaders,
    ultrasound_features,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def patient_table() -> pd.DataFrame:
    if not config.LABELS_CSV.exists():
        pytest.skip("Dummy data missing. Run: python -m airway.make_dummy_data")
    return loaders.build_patient_table()


@pytest.fixture(scope="module")
def face_index() -> pd.DataFrame:
    if not config.FACE_INDEX_CSV.exists():
        pytest.skip("Dummy data missing. Run: python -m airway.make_dummy_data")
    return loaders.face_loader()


# ===========================================================================
# FACE ALIGNMENT
# ===========================================================================
def test_align_face_returns_224_rgb(face_index):
    path = face_index["abs_path"].iloc[0]
    img = face_align.align_face(path)
    assert img.size == (224, 224)      # PIL size is (width, height)
    assert img.mode == "RGB"


def test_align_face_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        face_align.align_face("/no/such/image/file.jpg")


def test_face_was_detected_handles_bad_path():
    # should return False, not raise, on a non-existent file
    assert face_align.face_was_detected("/no/such/file.jpg") is False


# ===========================================================================
# FACE FEATURES
# ===========================================================================
@pytest.fixture(scope="module")
def face_feature_table(face_index) -> pd.DataFrame:
    # embed only the first 4 patients to keep the test fast
    small = face_index[face_index[config.ID_COL].isin(
        sorted(face_index[config.ID_COL].unique())[:4]
    )]
    return face_features.extract_face_features(small)


def test_face_features_one_row_per_patient(face_feature_table):
    assert not face_feature_table[config.ID_COL].duplicated().any()
    assert len(face_feature_table) == 4


def test_face_features_have_1024_columns(face_feature_table):
    feat_cols = [c for c in face_feature_table.columns if c.startswith("face_")]
    # 512 mean-pooled + 512 max-pooled
    assert len(feat_cols) == 1024


def test_face_features_are_numeric_and_finite(face_feature_table):
    feat_cols = [c for c in face_feature_table.columns if c.startswith("face_")]
    values = face_feature_table[feat_cols].to_numpy()
    assert np.isfinite(values).all()


def test_face_features_reproducible(face_index):
    """Same images -> identical embeddings. Critical for reproducibility."""
    small = face_index[face_index[config.ID_COL].isin(
        sorted(face_index[config.ID_COL].unique())[:3]
    )]
    a = face_features.extract_face_features(small)
    b = face_features.extract_face_features(small)
    feat_cols = [c for c in a.columns if c.startswith("face_")]
    np.testing.assert_allclose(
        a[feat_cols].to_numpy(), b[feat_cols].to_numpy(), rtol=1e-5
    )


# ===========================================================================
# ULTRASOUND FEATURES
# ===========================================================================
def test_ultrasound_features_columns():
    tbl = ultrasound_features.build_ultrasound_features()
    for col in ultrasound_features.US_FEATURE_COLS:
        assert col in tbl.columns
    assert config.ID_COL in tbl.columns


def test_ultrasound_features_one_row_per_patient():
    tbl = ultrasound_features.build_ultrasound_features()
    assert not tbl[config.ID_COL].duplicated().any()


# ===========================================================================
# BASELINE MODEL
# ===========================================================================
def test_baseline_pipeline_has_three_steps():
    pipe = baseline_model.make_baseline_pipeline()
    step_names = [name for name, _ in pipe.steps]
    assert step_names == ["impute", "scale", "model"]


def test_evaluate_ultrasound_returns_valid_metrics(patient_table):
    us_tbl = ultrasound_features.build_ultrasound_features()
    result = baseline_model.evaluate_modality(
        us_tbl, patient_table, ultrasound_features.US_FEATURE_COLS, "ultrasound"
    )
    # AUC must be a probability-like number in [0, 1]
    assert 0.0 <= result.auc_mean <= 1.0
    # every rate metric must be in [0, 1]
    for value in (result.sensitivity, result.specificity, result.accuracy,
                   result.ppv, result.npv):
        assert 0.0 <= value <= 1.0
    # out-of-fold predictions: each patient is in a test fold once PER repeat,
    # so with N_REPEATS repeats there are n_patients * N_REPEATS predictions
    expected = result.n_patients * config.N_REPEATS
    assert len(result.oof_true) == expected
    assert len(result.oof_prob) == expected


def test_evaluate_is_reproducible(patient_table):
    """Same data + same seed -> identical AUC."""
    us_tbl = ultrasound_features.build_ultrasound_features()
    r1 = baseline_model.evaluate_modality(
        us_tbl, patient_table, ultrasound_features.US_FEATURE_COLS, "ultrasound")
    r2 = baseline_model.evaluate_modality(
        us_tbl, patient_table, ultrasound_features.US_FEATURE_COLS, "ultrasound")
    assert r1.auc_mean == r2.auc_mean
    assert r1.oof_prob == r2.oof_prob
