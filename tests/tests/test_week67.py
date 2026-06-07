"""
Weeks 6-7 tests: ultrasound cleaning, hyomental distance ratio, within-fold
mean imputation, and leakage safety.

Run against dummy data. If data is missing:
    python -m airway.make_dummy_data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from airway import config, face_model, splits, ultrasound_features, ultrasound_model


@pytest.fixture(scope="module")
def patient_table() -> pd.DataFrame:
    if not config.LABELS_CSV.exists():
        pytest.skip("Dummy data missing. Run: python -m airway.make_dummy_data")
    from airway import loaders
    return loaders.build_patient_table()


# ===========================================================================
# HYOMENTAL DISTANCE RATIO
# ===========================================================================
def test_hmdr_basic_ratio():
    out = ultrasound_features.compute_hmdr([60.0, 50.0], [40.0, 25.0])
    np.testing.assert_allclose(out, [1.5, 2.0])


def test_hmdr_zero_negative_and_missing_neutral_are_nan():
    # neutral = 0 (div by zero), neutral < 0 (non-physical), neutral NaN
    out = ultrasound_features.compute_hmdr(
        [50.0, 50.0, 50.0, np.nan],
        [0.0, -10.0, np.nan, 40.0])
    assert np.isnan(out[0])   # zero neutral
    assert np.isnan(out[1])   # negative neutral
    assert np.isnan(out[2])   # missing neutral
    assert np.isnan(out[3])   # missing extended


def test_hmdr_handles_non_numeric_strings():
    out = ultrasound_features.compute_hmdr(["60", "bad"], ["40", "20"])
    np.testing.assert_allclose(out[0], 1.5)
    assert np.isnan(out[1])   # "bad" -> NaN


# ===========================================================================
# NUMERIC CLEANING
# ===========================================================================
def test_cleaning_coerces_non_numeric_to_nan(tmp_path):
    csv = tmp_path / "us.csv"
    pd.DataFrame({
        config.ID_COL: ["P1", "P2"],
        "dstvc_mm": ["18.0", "not_a_number"],
        "hmd_neutral_mm": [45.0, 40.0],
        "hmd_extended_mm": [55.0, 50.0],
        "dse_mm": [27.0, 25.0],
    }).to_csv(csv, index=False)

    cleaned = ultrasound_features.clean_ultrasound_features(path=csv)
    assert pd.api.types.is_numeric_dtype(cleaned["dstvc_mm"])
    assert pd.isna(cleaned.loc[cleaned[config.ID_COL] == "P2", "dstvc_mm"].iloc[0])
    # derived ratio still computed for valid rows
    np.testing.assert_allclose(
        cleaned.loc[cleaned[config.ID_COL] == "P1", "hmdr"].iloc[0], 55.0 / 45.0)


def test_cleaning_adds_placeholder_for_missing_column(tmp_path):
    csv = tmp_path / "us.csv"
    # 'dse_mm' deliberately absent
    pd.DataFrame({
        config.ID_COL: ["P1", "P2"],
        "dstvc_mm": [18.0, 19.0],
        "hmd_neutral_mm": [45.0, 40.0],
        "hmd_extended_mm": [55.0, 50.0],
    }).to_csv(csv, index=False)

    cleaned = ultrasound_features.clean_ultrasound_features(path=csv)
    assert "dse_mm" in cleaned.columns          # placeholder added
    assert cleaned["dse_mm"].isna().all()
    # an entirely-missing feature is dropped by usable_feature_cols (with warning)
    usable = ultrasound_features.usable_feature_cols(cleaned)
    assert "dse_mm" not in usable
    assert "hmdr" in usable


def test_cleaning_fails_clearly_without_id_column(tmp_path):
    csv = tmp_path / "us.csv"
    pd.DataFrame({"dstvc_mm": [18.0]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match=config.ID_COL):
        ultrasound_features.clean_ultrasound_features(path=csv)


def test_cleaning_applies_aliases(tmp_path):
    csv = tmp_path / "us.csv"
    pd.DataFrame({
        config.ID_COL: ["P1"],
        "dstvc": [18.0],          # alias -> dstvc_mm
        "hmd_neutral": [45.0],    # alias -> hmd_neutral_mm
        "hmd_extended": [55.0],   # alias -> hmd_extended_mm
        "dse": [27.0],            # alias -> dse_mm
    }).to_csv(csv, index=False)
    cleaned = ultrasound_features.clean_ultrasound_features(path=csv)
    for col in ultrasound_features.US_MEASURED_COLS:
        assert col in cleaned.columns
        assert cleaned[col].notna().all()


# ===========================================================================
# WITHIN-FOLD MEAN IMPUTATION + NO LEAKAGE
# ===========================================================================
def test_pipeline_uses_mean_imputation():
    pipe = ultrasound_model.make_us_logreg_pipeline(np.array([0, 1, 0, 1]))
    assert pipe.named_steps["impute"].strategy == "mean"


def test_within_fold_imputation_uses_train_mean_only():
    """
    The imputed value for a missing TEST cell must equal the TRAIN-fold mean,
    not a mean that has seen the test rows. A large test value that would shift
    the global mean must NOT influence the imputed value.
    """
    train = np.array([[10.0], [20.0], [30.0]])     # train mean = 20
    test = np.array([[1000.0], [np.nan]])          # one present (large), one missing

    imp = SimpleImputer(strategy="mean")
    imp.fit(train)                                  # fit on TRAIN only
    assert imp.statistics_[0] == 20.0

    imputed = imp.transform(test)
    assert imputed[1, 0] == 20.0                    # filled with TRAIN mean

    # had imputation leaked test rows in, the mean would be (10+20+30+1000)/4
    global_mean = np.array([10.0, 20.0, 30.0, 1000.0]).mean()
    assert imputed[1, 0] != global_mean


def test_full_pipeline_imputer_fits_on_training_fold(patient_table):
    """End-to-end: the pipeline's imputer statistics come from the train fold."""
    cleaned = ultrasound_features.clean_ultrasound_features()
    feature_cols = ultrasound_features.usable_feature_cols(cleaned)
    data = cleaned.merge(patient_table[[config.ID_COL, config.LABEL_COL]],
                         on=config.ID_COL, how="inner")
    folds = splits.patient_level_folds(data)
    fold = folds[0]
    train = splits.select_rows(data, fold.train_ids)
    x_tr = train[feature_cols].to_numpy()

    pipe = ultrasound_model.make_us_logreg_pipeline(train[config.LABEL_COL].to_numpy())
    pipe.fit(x_tr, train[config.LABEL_COL].to_numpy())
    stats = pipe.named_steps["impute"].statistics_
    expected = np.nanmean(x_tr, axis=0)
    np.testing.assert_allclose(stats, expected, rtol=1e-6)


# ===========================================================================
# MODEL CV (reusing the face-model machinery, patient-level, no leakage)
# ===========================================================================
@pytest.mark.parametrize("model_name", ["logreg_l2", "xgboost"])
def test_us_cross_validate_metrics_in_range(patient_table, model_name):
    cleaned = ultrasound_features.clean_ultrasound_features()
    feature_cols = ultrasound_features.usable_feature_cols(cleaned)
    res = face_model.cross_validate(
        cleaned, patient_table, feature_cols, model_name,
        models=ultrasound_model.US_MODELS, modality_prefix="ultrasound")
    assert res.modality == f"ultrasound:{model_name}"
    assert 0.0 <= res.auc_mean <= 1.0
    assert len(res.oof_true) == res.n_patients * config.N_REPEATS


def test_us_cv_no_patient_leakage(patient_table):
    cleaned = ultrasound_features.clean_ultrasound_features()
    data = cleaned.merge(patient_table[[config.ID_COL, config.LABEL_COL]],
                         on=config.ID_COL, how="inner")
    folds = splits.patient_level_folds(data)
    splits.assert_no_leakage(folds)


def test_feature_importance_has_both_metrics(patient_table):
    cleaned = ultrasound_features.clean_ultrasound_features()
    feature_cols = ultrasound_features.usable_feature_cols(cleaned)
    data = cleaned.merge(patient_table[[config.ID_COL, config.LABEL_COL]],
                         on=config.ID_COL, how="inner")
    imp = ultrasound_model.compute_feature_importance(data, feature_cols)
    assert set(imp["feature"]) == set(feature_cols)
    for col in ("perm_importance_mean", "perm_importance_std", "xgb_gain"):
        assert col in imp.columns
    assert (imp["xgb_gain"] >= 0).all()
