"""Tests for airway.decision_curve (net-benefit decision-curve analysis)."""

from __future__ import annotations

import numpy as np
import pytest

from airway import decision_curve, fusion


def test_fails_clearly_without_fusion_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(fusion, "FUSION_FOLD_PRED_CSV", tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError, match="block-c"):
        decision_curve.build_curve()


@pytest.fixture(scope="module")
def curve_data():
    if not fusion.FUSION_FOLD_PRED_CSV.exists() or not decision_curve.PER_MODEL_CSV.exists():
        pytest.skip("fusion predictions / per_model_metrics missing; run `make block-c`.")
    curve, best, prevalence = decision_curve.build_curve()
    return curve, best, prevalence


def test_treat_none_is_zero_everywhere(curve_data):
    curve, _, _ = curve_data
    assert (curve["nb_treat_none"] == 0.0).all()


def test_treat_all_approx_prevalence_at_low_threshold(curve_data):
    curve, _, prevalence = curve_data
    # at the lowest threshold, treat-all net benefit ≈ prevalence
    assert abs(curve.iloc[0]["nb_treat_all"] - prevalence) < 0.02


def test_one_row_per_grid_point_and_fused_finite(curve_data):
    curve, _, _ = curve_data
    expected = len(decision_curve._threshold_grid())
    assert len(curve) == expected
    assert np.isfinite(curve["nb_fused"].to_numpy()).all()


def test_png_created_and_nonempty(curve_data):
    curve, best, _ = curve_data
    decision_curve._save_plot(curve, best, decision_curve.OUT_PNG)
    assert decision_curve.OUT_PNG.exists()
    assert decision_curve.OUT_PNG.stat().st_size > 0
