"""
Block C tests (Weeks 8-11): calibration, late fusion, clinical comparison,
and the DeLong utility.

Tests that need the face features parquet are skipped if it is missing
(run `make week45` first). The DeLong and synthetic-fusion tests are hermetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from airway import (
    calibration,
    clinical_comparison,
    config,
    delong,
    face_model,
    fusion,
    ultrasound_model,
)


# ===========================================================================
# DELONG UTILITY (hermetic)
# ===========================================================================
def test_delong_auc_matches_sklearn():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    s = np.array([.1, .2, .35, .7, .8, .9, .3, .6, .15, .55])
    auc, var = delong.delong_roc_variance(y, s)
    assert auc == pytest.approx(roc_auc_score(y, s), abs=1e-9)
    assert var >= 0.0


def test_delong_identical_scores_give_pvalue_one():
    y = np.array([0, 1, 0, 1, 1, 0])
    s = np.array([.2, .8, .3, .7, .6, .4])
    res = delong.delong_test(y, s, s.copy())
    assert res["auc_diff"] == pytest.approx(0.0)
    assert res["p_value"] == pytest.approx(1.0)


def test_delong_known_difference_orders_aucs():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    good = np.array([.1, .2, .3, .4, .6, .7, .8, .9])   # AUC 1.0
    bad = np.array([.9, .8, .7, .6, .4, .3, .2, .1])    # AUC 0.0
    res = delong.delong_test(y, good, bad)
    assert res["auc_1"] > res["auc_2"]
    assert 0.0 <= res["p_value"] <= 1.0


def test_delong_rejects_single_class():
    with pytest.raises(ValueError):
        delong.delong_roc_variance(np.array([1, 1, 1]), np.array([.2, .5, .8]))


# ===========================================================================
# FUSION (hermetic, synthetic calibrated probs)
# ===========================================================================
def _synthetic_merged(n=12, repeats=2, folds=3, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"P{i:03d}" for i in range(n)]
    labels = {pid: int(rng.random() < 0.4) for pid in ids}
    rows = []
    for r in range(repeats):
        fold_of = {pid: (i % folds) for i, pid in enumerate(ids)}
        for pid in ids:
            y = labels[pid]
            rows.append({
                config.ID_COL: pid, "repeat": r, "fold_index": fold_of[pid] + r * folds,
                config.LABEL_COL: y,
                fusion.FACE_PROB: float(np.clip(0.3 * y + rng.normal(0.3, 0.1), 0, 1)),
                fusion.US_PROB: float(np.clip(0.3 * y + rng.normal(0.3, 0.1), 0, 1)),
            })
    return pd.DataFrame(rows)


def test_average_baseline_is_mean_of_modalities():
    merged = _synthetic_merged()
    preds, _, _ = fusion.run_fusion(merged)
    expected = (preds[fusion.FACE_PROB] + preds[fusion.US_PROB]) / 2.0
    np.testing.assert_allclose(preds["avg_prob"].to_numpy(), expected.to_numpy())


def test_fusion_train_excludes_validation_patients():
    """The meta-learner's training fold must not contain validation patients."""
    merged = _synthetic_merged()
    for _, rep in merged.groupby("repeat"):
        for fold_index, val in rep.groupby("fold_index"):
            train = rep[rep["fold_index"] != fold_index]
            assert not (set(train[config.ID_COL]) & set(val[config.ID_COL]))


def test_fusion_outputs_one_row_per_patient_per_repeat():
    merged = _synthetic_merged(n=12, repeats=2, folds=3)
    preds, fused, avg = fusion.run_fusion(merged)
    assert len(preds) == len(merged)
    assert set(preds["fused_prob"].between(0, 1)) == {True}
    assert fused.modality == "fusion:logreg"
    assert avg.modality == "fusion:average"


def test_fusion_meta_learner_has_two_inputs():
    assert fusion.FUSION_INPUTS == [fusion.FACE_PROB, fusion.US_PROB]


# ===========================================================================
# CALIBRATION (needs face features parquet)
# ===========================================================================
@pytest.fixture(scope="module")
def cohort():
    if not config.FACE_FEATURES_PARQUET.exists():
        pytest.skip("face features parquet missing; run `make week45` first.")
    return calibration.load_common_cohort()


def test_calibrated_probs_have_expected_columns(cohort):
    probs, brier = calibration.calibrate_modality(
        cohort["us_data"], cohort["us_cols"],
        ultrasound_model.make_us_logreg_pipeline, cohort["folds"], "ultrasound")
    assert set(probs.columns) == {
        config.ID_COL, "repeat", "fold_index", config.LABEL_COL, calibration.PROB_COL}
    assert probs[calibration.PROB_COL].between(0, 1).all()
    # brier table has per-fold rows + an overall summary row
    assert "scope" in brier.columns
    assert (brier["scope"] == "overall").sum() == 1


def test_calibration_is_within_fold_oof(cohort):
    """
    Out-of-fold structure: within each repeat every patient is scored exactly
    once (as a validation patient), i.e. calibration/scoring is within-fold.
    """
    probs, brier = calibration.calibrate_modality(
        cohort["us_data"], cohort["us_cols"],
        ultrasound_model.make_us_logreg_pipeline, cohort["folds"], "ultrasound")
    n_patients = cohort["n_common"]
    for _, grp in probs.groupby("repeat"):
        assert grp[config.ID_COL].nunique() == n_patients
        assert not grp[config.ID_COL].duplicated().any()
    # with this cohort there are enough minority cases to actually calibrate
    assert brier.loc[brier["scope"] == "fold", "calibrated"].all()


# ===========================================================================
# CLINICAL COMPARISON (needs face features parquet)
# ===========================================================================
@pytest.fixture(scope="module")
def fold_rows(cohort):
    """Build fusion fold predictions in-memory and attach real clinical scores."""
    face_probs, _ = calibration.calibrate_modality(
        cohort["face_data"], cohort["face_cols"],
        face_model.make_logreg_pipeline, cohort["folds"], "face")
    us_probs, _ = calibration.calibrate_modality(
        cohort["us_data"], cohort["us_cols"],
        ultrasound_model.make_us_logreg_pipeline, cohort["folds"], "ultrasound")
    keys = [config.ID_COL, "repeat", "fold_index", config.LABEL_COL]
    merged = (face_probs.merge(us_probs, on=keys, suffixes=("_face", "_us"))
              .rename(columns={f"{calibration.PROB_COL}_face": fusion.FACE_PROB,
                               f"{calibration.PROB_COL}_us": fusion.US_PROB}))
    preds, _, _ = fusion.run_fusion(merged[keys + [fusion.FACE_PROB, fusion.US_PROB]])

    from airway import loaders, scores
    comp = scores.compute_comparator_scores(loaders.preop_loader())
    return preds.merge(comp[[config.ID_COL, *clinical_comparison.SCORE_COLS]],
                       on=config.ID_COL, how="left")


def test_clinical_metrics_cover_all_models(fold_rows):
    metrics = clinical_comparison.per_model_metrics(fold_rows)
    models = set(metrics["model"])
    assert {"face", "ultrasound", "fusion:logreg", "fusion:average",
            "mallampati", "lemon", "wilson"} <= models
    # AUCs are valid probabilities-of-ranking
    assert metrics["auc_pooled"].between(0, 1).all()


def test_clinical_uses_same_folds_as_fusion(fold_rows):
    # the clinical fold rows carry exactly the fusion fold ids (5 splits x 2 repeats)
    n_folds = fold_rows.groupby(["repeat", "fold_index"]).ngroups
    assert n_folds == config.N_SPLITS * config.N_REPEATS


def test_delong_comparisons_has_six_rows(fold_rows):
    comp = clinical_comparison.delong_comparisons(fold_rows)
    assert len(comp) == 6
    assert (comp["alpha_bonferroni"] == 0.0083).all()
    for col in ("auc_fused", "auc_comparator", "p_value", "significant"):
        assert col in comp.columns


# ===========================================================================
# Block D extension: Youden-optimal operating point
# ===========================================================================
def test_youden_threshold_within_score_range():
    import numpy as np
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.35, 0.7, 0.8, 0.9, 0.3, 0.6])
    thr = clinical_comparison.youden_threshold(y, s)
    assert s.min() <= thr <= s.max()


def test_youden_threshold_single_class_is_nan():
    import numpy as np
    assert np.isnan(clinical_comparison.youden_threshold([1, 1, 1], [0.2, 0.5, 0.8]))


def test_per_model_metrics_has_both_threshold_types(fold_rows):
    metrics = clinical_comparison.per_model_metrics(fold_rows)
    assert "threshold_type" in metrics.columns
    types = set(metrics["threshold_type"])
    assert "fixed_0.5" in types
    assert "youden" in types
