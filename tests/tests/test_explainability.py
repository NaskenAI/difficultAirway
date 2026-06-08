"""Tests for airway.explainability (SHAP on the ultrasound XGBoost)."""

from __future__ import annotations

import joblib
import pytest

from airway import config, explainability


@pytest.fixture(scope="module")
def feature_cols():
    if not explainability.US_MODEL_PKL.exists() or not config.CLEANED_US_CSV.exists():
        pytest.skip("us_model.pkl / cleaned ultrasound missing; run `make us-model`.")
    return joblib.load(explainability.US_MODEL_PKL)["feature_cols"]


@pytest.fixture(scope="module")
def importance(feature_cols):
    return explainability.ultrasound_shap()


def test_shap_importance_one_row_per_feature(importance, feature_cols):
    assert set(importance["feature"]) == set(feature_cols)
    assert len(importance) == len(feature_cols)


def test_shap_importance_non_negative(importance):
    assert (importance["mean_abs_shap"] >= 0).all()


def test_summary_png_created_and_nonempty(importance):
    assert explainability.SHAP_SUMMARY_PNG.exists()
    assert explainability.SHAP_SUMMARY_PNG.stat().st_size > 0


def test_shap_importance_csv_written(importance):
    assert explainability.SHAP_IMPORTANCE_CSV.exists()
