"""Tests for airway.clinical_baseline (bedside-variables-only logistic model)."""

from __future__ import annotations

import pandas as pd
import pytest

from airway import clinical_baseline, config, fusion, predictions


def test_pipeline_has_impute_scale_model():
    steps = [name for name, _ in clinical_baseline.make_pipeline().steps]
    assert steps == ["impute", "scale", "model"]


def test_fails_clearly_without_fusion_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(fusion, "FUSION_FOLD_PRED_CSV", tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError, match="block-c"):
        clinical_baseline.cross_val_probs()


@pytest.fixture(scope="module")
def have_preds() -> bool:
    if not fusion.FUSION_FOLD_PRED_CSV.exists():
        pytest.skip("fusion_fold_predictions.csv missing; run `make block-c` first.")
    return True


def test_one_oof_row_per_patient_per_repeat(have_preds):
    probs, _ = clinical_baseline.cross_val_probs()
    preds = predictions.load_fusion_predictions()
    n_patients = preds[config.ID_COL].nunique()
    assert len(probs) == n_patients * config.N_REPEATS
    # exactly one row per (patient, repeat)
    assert not probs.duplicated(subset=[config.ID_COL, "repeat"]).any()
    assert set(probs.columns) == {config.ID_COL, "repeat", "fold_index",
                                  config.LABEL_COL, clinical_baseline.PROB_COL}


def test_metrics_in_unit_interval(have_preds):
    probs, used = clinical_baseline.cross_val_probs()
    metrics = clinical_baseline.metrics_table(probs, used)
    for col in ("auc_mean", "auc_pooled", "sensitivity", "specificity",
                "ppv", "npv", "accuracy", "balanced_accuracy", "f1"):
        vals = pd.to_numeric(metrics[col], errors="coerce").dropna().to_numpy()
        assert ((vals >= -1e-9) & (vals <= 1 + 1e-9)).all()
    assert set(metrics["threshold_type"]) == {"fixed_0.5", "youden"}


def test_reproducible_metrics(have_preds):
    p1, u1 = clinical_baseline.cross_val_probs()
    p2, u2 = clinical_baseline.cross_val_probs()
    pd.testing.assert_frame_equal(p1, p2)
    pd.testing.assert_frame_equal(
        clinical_baseline.metrics_table(p1, u1),
        clinical_baseline.metrics_table(p2, u2))


def test_uses_subset_when_thyromental_absent(have_preds, monkeypatch):
    from airway import loaders
    real = loaders.preop_loader()
    trimmed = real.drop(columns=[c for c in ["thyromental_mm"] if c in real.columns])
    monkeypatch.setattr(loaders, "preop_loader", lambda: trimmed)
    probs, used = clinical_baseline.cross_val_probs()
    assert "thyromental_mm" not in used
    assert len(used) >= 1
    assert len(probs) == predictions.load_fusion_predictions()[config.ID_COL].nunique() * config.N_REPEATS
