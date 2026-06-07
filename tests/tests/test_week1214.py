"""
Block D tests (Weeks 12-14): bootstrap CIs, subgroup analysis, confusion
categories, explanation case selection, and FN/FP export logic.

These are hermetic: they build small synthetic tables or monkeypatch
predictions.build_master_table, so they do not depend on a prior pipeline run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airway import (
    bootstrap_ci,
    config,
    error_analysis,
    explainability,
    predictions,
    subgroups,
)


# ===========================================================================
# CONFUSION CATEGORIES
# ===========================================================================
def test_confusion_category_all_four():
    assert predictions.confusion_category(1, 1) == "TP"
    assert predictions.confusion_category(0, 0) == "TN"
    assert predictions.confusion_category(0, 1) == "FP"
    assert predictions.confusion_category(1, 0) == "FN"


def test_predicted_class_threshold():
    out = predictions.predicted_class([0.2, 0.5, 0.8], threshold=0.5)
    assert out.tolist() == [0, 1, 1]


# ===========================================================================
# PER-PATIENT COLLAPSE (patient-level units)
# ===========================================================================
def test_per_patient_averages_across_repeats():
    master = pd.DataFrame({
        config.ID_COL: ["P1", "P1", "P2", "P2"],
        "repeat": [0, 1, 0, 1], "fold_index": [0, 5, 1, 6],
        config.LABEL_COL: [1, 1, 0, 0],
        "face_prob": [0.4, 0.6, 0.1, 0.3], "us_prob": [0.5, 0.5, 0.2, 0.2],
        "fused_prob": [0.2, 0.8, 0.1, 0.1], "avg_prob": [0.45, 0.55, 0.15, 0.25],
    })
    pp = predictions.per_patient(master)
    assert len(pp) == 2
    p1 = pp[pp[config.ID_COL] == "P1"].iloc[0]
    assert p1["face_prob"] == pytest.approx(0.5)   # mean of 0.4, 0.6
    assert p1["fused_prob"] == pytest.approx(0.5)   # mean of 0.2, 0.8
    assert p1["folds"] == "0,5"


# ===========================================================================
# BOOTSTRAP CIs
# ===========================================================================
def _synthetic_scores(n=40, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    score = np.clip(0.25 * y + rng.normal(0.4, 0.15, n), 0, 1)
    return y, score


def test_bootstrap_ci_output_shape():
    y, s = _synthetic_scores()
    cis = bootstrap_ci.bootstrap_metric_cis(y, s, 0.5, True, n_boot=100, seed=1)
    assert set(cis) == set(predictions.METRIC_KEYS)
    for k, c in cis.items():
        assert set(c) == {"estimate", "ci_lower", "ci_upper", "n_valid"}
        assert c["n_valid"] <= 100


def test_bootstrap_is_deterministic_with_seed():
    y, s = _synthetic_scores()
    a = bootstrap_ci.bootstrap_metric_cis(y, s, 0.5, True, n_boot=100, seed=7)
    b = bootstrap_ci.bootstrap_metric_cis(y, s, 0.5, True, n_boot=100, seed=7)
    assert a["auc"] == b["auc"]


def test_bootstrap_ci_brackets_estimate_for_auc():
    y, s = _synthetic_scores(n=60, seed=3)
    cis = bootstrap_ci.bootstrap_metric_cis(y, s, 0.5, True, n_boot=300, seed=3)
    auc = cis["auc"]
    assert auc["ci_lower"] <= auc["estimate"] <= auc["ci_upper"]


def test_bootstrap_table_shape_monkeypatched(monkeypatch):
    master = _synthetic_master(seed=2)
    monkeypatch.setattr(predictions, "build_master_table", lambda: master)
    table = bootstrap_ci.build_table(n_boot=50)
    n_models = len(predictions.model_specs(master))
    assert len(table) == n_models * len(predictions.METRIC_KEYS)
    assert {"model", "metric", "estimate", "ci_lower", "ci_upper",
            "n_boot_valid", "n_patients"} <= set(table.columns)


# ===========================================================================
# SUBGROUPS
# ===========================================================================
def test_tertiles_split_into_three_balanced_levels():
    pp = pd.DataFrame({"bmi": np.arange(30, dtype=float)})
    tert = subgroups.add_tertiles(pp, "bmi")
    counts = tert.value_counts()
    assert set(counts.index) == {"low", "mid", "high"}
    assert counts.tolist() == [10, 10, 10]


def test_small_subgroup_is_flagged_underpowered():
    sub = pd.DataFrame({
        config.ID_COL: ["P1", "P2", "P3"],
        config.LABEL_COL: [1, 0, 0],
        "fused_prob": [0.8, 0.2, 0.3],
    })
    row = subgroups._subgroup_row(sub, "surgery_type", "rare")
    assert row["underpowered"] is True
    assert "underpowered" in row["note"]


def test_large_balanced_subgroup_not_flagged():
    rng = np.random.default_rng(0)
    n = 40
    sub = pd.DataFrame({
        config.ID_COL: [f"P{i}" for i in range(n)],
        config.LABEL_COL: ([1] * 12 + [0] * 28),
        "fused_prob": rng.random(n),
    })
    row = subgroups._subgroup_row(sub, "overall", "all")
    assert row["underpowered"] is False


# ===========================================================================
# EXPLANATION CASE SELECTION
# ===========================================================================
def test_case_selection_up_to_two_per_category():
    # 3 of each category -> selector keeps at most 2 each
    rows = []
    specs = [(1, 0.9, "TP"), (0, 0.1, "TN"), (0, 0.9, "FP"), (1, 0.1, "FN")]
    i = 0
    for label, prob, _ in specs:
        for _ in range(3):
            rows.append({config.ID_COL: f"P{i:02d}", config.LABEL_COL: label,
                         "fused_prob": prob})
            i += 1
    pp = pd.DataFrame(rows)
    cases = explainability.select_cases(pp)
    counts = cases["error_type"].value_counts().to_dict()
    for cat in ("TP", "TN", "FP", "FN"):
        assert counts.get(cat, 0) == 2
    # error types are assigned consistently with label/threshold
    for _, c in cases.iterrows():
        assert c["error_type"] == predictions.confusion_category(
            c["true_label"], c["predicted_class"])


def test_case_selection_handles_missing_category():
    pp = pd.DataFrame({                       # only TP and TN present
        config.ID_COL: ["P1", "P2", "P3"],
        config.LABEL_COL: [1, 0, 0],
        "fused_prob": [0.9, 0.1, 0.2],
    })
    cases = explainability.select_cases(pp)
    assert set(cases["error_type"]) <= {"TP", "TN"}
    assert len(cases) == 3


# ===========================================================================
# ERROR-ANALYSIS EXPORT (FN / FP)
# ===========================================================================
def _synthetic_master(seed=0, with_errors=True):
    rng = np.random.default_rng(seed)
    ids = [f"P{i:03d}" for i in range(30)]
    labels = {pid: int(rng.random() < 0.4) for pid in ids}
    rows = []
    for r in range(config.N_REPEATS):
        for i, pid in enumerate(ids):
            y = labels[pid]
            fused = float(np.clip((0.4 * y + rng.normal(0.35, 0.2)) if with_errors
                                  else (0.9 if y else 0.1), 0, 1))
            rows.append({
                config.ID_COL: pid, "repeat": r, "fold_index": i % config.N_SPLITS + r * config.N_SPLITS,
                config.LABEL_COL: y,
                "face_prob": fused, "us_prob": fused,
                "fused_prob": fused, "avg_prob": fused,
                "mallampati_class": rng.integers(1, 5), "lemon_score": float(rng.integers(0, 6)),
                "wilson_score": float(rng.integers(0, 10)),
                "age_years": int(rng.integers(20, 80)), "sex": "M", "bmi": float(rng.normal(27, 4)),
                config.SURGERY_TYPE_COL: rng.choice(["general", "ent"]),
            })
    return pd.DataFrame(rows)


def test_error_analysis_splits_fn_and_fp(monkeypatch):
    # explicit per-fold master with one guaranteed FN and one guaranteed FP
    master = pd.DataFrame({
        config.ID_COL: ["FN1", "FP1", "TP1", "TN1"],
        "repeat": [0, 0, 0, 0], "fold_index": [0, 1, 2, 3],
        config.LABEL_COL: [1, 0, 1, 0],
        "face_prob": [0.2, 0.8, 0.9, 0.1], "us_prob": [0.2, 0.8, 0.9, 0.1],
        "fused_prob": [0.2, 0.8, 0.9, 0.1], "avg_prob": [0.2, 0.8, 0.9, 0.1],
    })
    monkeypatch.setattr(predictions, "build_master_table", lambda: master)
    fn, fp = error_analysis.build_review_tables()
    assert fn[config.ID_COL].tolist() == ["FN1"]
    assert fp[config.ID_COL].tolist() == ["FP1"]
    assert (fn["error_type"] == "FN").all()
    assert (fp["error_type"] == "FP").all()
    # required review columns present
    for col in ("folds", "predicted_prob", "predicted_class", "face_prob", "us_prob"):
        assert col in fn.columns
