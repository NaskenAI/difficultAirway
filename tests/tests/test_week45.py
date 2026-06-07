"""
Weeks 4-5 tests: face crops (alignment + idempotency), embeddings (512/1024),
and the two face classifiers under patient-level CV.

Run against dummy data. If data is missing:
    python -m airway.make_dummy_data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airway import (
    config,
    face_crops,
    face_embeddings,
    face_model,
    loaders,
    splits,
)


@pytest.fixture(scope="module")
def face_index() -> pd.DataFrame:
    if not config.FACE_INDEX_CSV.exists():
        pytest.skip("Dummy data missing. Run: python -m airway.make_dummy_data")
    return loaders.face_loader()


@pytest.fixture(scope="module")
def patient_table() -> pd.DataFrame:
    if not config.LABELS_CSV.exists():
        pytest.skip("Dummy data missing. Run: python -m airway.make_dummy_data")
    return loaders.build_patient_table()


# ===========================================================================
# CROPS
# ===========================================================================
def test_align_face_returns_224_rgb(face_index):
    img = face_crops.align_face(face_index["abs_path"].iloc[0])
    assert img.size == (224, 224)
    assert img.mode == "RGB"


def test_align_face_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        face_crops.align_face("/no/such/image.jpg")


def test_crop_path_is_deterministic():
    p1 = face_crops.crop_path_for("P001", "frontal_rest")
    p2 = face_crops.crop_path_for("P001", "frontal_rest")
    assert p1 == p2
    assert p1.parent == config.FACE_CROPS_DIR


def test_generate_crops_is_idempotent(face_index, tmp_path, monkeypatch):
    # redirect crop output to a temp dir so we control the "already exists" state
    monkeypatch.setattr(config, "FACE_CROPS_DIR", tmp_path)
    monkeypatch.setattr(face_crops.config, "FACE_CROPS_DIR", tmp_path)

    first = face_crops.generate_crops(force=False, respect_quarantine=False)
    assert first["written"] > 0
    assert first["skipped"] == 0

    second = face_crops.generate_crops(force=False, respect_quarantine=False)
    assert second["written"] == 0                  # nothing recomputed
    assert second["skipped"] == first["written"]   # all skipped


# ===========================================================================
# EMBEDDINGS
# ===========================================================================
@pytest.fixture(scope="module")
def small_embeddings(face_index) -> pd.DataFrame:
    # embed two images straight from raw (align on the fly) to stay fast
    sub = face_index.head(4)
    rows = []
    for _, row in sub.iterrows():
        emb = face_embeddings._embed_one(
            row[config.ID_COL], row["view_code"], row["abs_path"])
        rec = {config.ID_COL: row[config.ID_COL], "view_code": row["view_code"],
               "file_path": row["file_path"]}
        for j, v in enumerate(emb):
            rec[f"{face_embeddings.IMG_PREFIX}{j:03d}"] = v
        rows.append(rec)
    return pd.DataFrame(rows)


def test_per_image_embedding_is_512_and_finite(small_embeddings):
    emb_cols = [c for c in small_embeddings.columns
                if c.startswith(face_embeddings.IMG_PREFIX)]
    assert len(emb_cols) == 512
    assert np.isfinite(small_embeddings[emb_cols].to_numpy()).all()


def test_aggregate_per_patient_is_1024(small_embeddings):
    agg = face_embeddings.aggregate_per_patient(small_embeddings)
    feat_cols = [c for c in agg.columns if c.startswith(face_embeddings.FACE_PREFIX)]
    assert len(feat_cols) == 1024
    assert not agg[config.ID_COL].duplicated().any()


def test_embedding_reproducible(face_index):
    row = face_index.iloc[0]
    a = face_embeddings._embed_one(row[config.ID_COL], row["view_code"], row["abs_path"])
    b = face_embeddings._embed_one(row[config.ID_COL], row["view_code"], row["abs_path"])
    np.testing.assert_allclose(a, b, rtol=1e-5)


# ===========================================================================
# FACE MODEL (both classifiers, patient-level CV)
# ===========================================================================
@pytest.fixture(scope="module")
def tiny_features(patient_table) -> pd.DataFrame:
    """Cheap synthetic 1024-d features so the CV test does not run ResNet on all."""
    rng = np.random.default_rng(0)
    ids = patient_table[config.ID_COL].tolist()
    data = {config.ID_COL: ids}
    label = patient_table.set_index(config.ID_COL)[config.LABEL_COL]
    for j in range(1024):
        # give a little signal so AUC is defined, plus noise
        data[f"{face_embeddings.FACE_PREFIX}{j:03d}"] = (
            label.loc[ids].to_numpy() * (0.1 if j < 5 else 0.0)
            + rng.normal(0, 1, len(ids)))
    return pd.DataFrame(data)


@pytest.mark.parametrize("model_name", ["logreg_l2", "xgboost"])
def test_cross_validate_metrics_in_range(tiny_features, patient_table, model_name):
    res = face_model.cross_validate(
        tiny_features, patient_table,
        [c for c in tiny_features.columns if c.startswith(face_embeddings.FACE_PREFIX)],
        model_name)
    assert 0.0 <= res.auc_mean <= 1.0
    for v in (res.sensitivity, res.specificity, res.accuracy, res.ppv, res.npv):
        assert 0.0 <= v <= 1.0
    # each patient tested once per repeat -> n_patients * N_REPEATS predictions
    assert len(res.oof_true) == res.n_patients * config.N_REPEATS


def test_face_model_uses_patient_level_folds_without_leakage(tiny_features, patient_table):
    data = tiny_features.merge(
        patient_table[[config.ID_COL, config.LABEL_COL]], on=config.ID_COL)
    folds = splits.patient_level_folds(data)
    splits.assert_no_leakage(folds)   # raises on leakage
