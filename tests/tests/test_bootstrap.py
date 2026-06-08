"""Tests for airway.bootstrap_ci (patient-level bootstrap CIs)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airway import bootstrap_ci, fusion


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    if not fusion.FUSION_FOLD_PRED_CSV.exists():
        pytest.skip("fusion_fold_predictions.csv missing; run `make block-c` first.")
    return bootstrap_ci.build_table(n_boot=200)


def test_runs_on_pipeline_output(table):
    assert len(table) > 0
    assert set(table.columns) == {"model", "metric", "point_estimate",
                                  "ci_lower", "ci_upper", "n_valid_iterations"}


def test_cis_are_ordered(table):
    for _, r in table.iterrows():
        if not (np.isnan(r["ci_lower"]) or np.isnan(r["point_estimate"])
                or np.isnan(r["ci_upper"])):
            assert r["ci_lower"] <= r["point_estimate"] + 1e-9
            assert r["point_estimate"] <= r["ci_upper"] + 1e-9


def test_all_values_in_unit_interval(table):
    for col in ("point_estimate", "ci_lower", "ci_upper"):
        vals = table[col].dropna().to_numpy()
        assert ((vals >= -1e-9) & (vals <= 1 + 1e-9)).all()


def test_reproducible_same_seed():
    a = bootstrap_ci.build_table(n_boot=200)
    b = bootstrap_ci.build_table(n_boot=200)
    pd.testing.assert_frame_equal(a, b)


def test_bootstrap_model_is_deterministic():
    rng = np.random.default_rng(0)
    y = (rng.random(40) < 0.4).astype(int)
    s = np.clip(0.25 * y + rng.normal(0.4, 0.15, 40), 0, 1)
    r1 = bootstrap_ci.bootstrap_model(y, s, n_boot=100, seed=3)
    r2 = bootstrap_ci.bootstrap_model(y, s, n_boot=100, seed=3)
    assert r1["auc"] == r2["auc"]
    # AUC may skip one-class resamples, so valid iters can be < n_boot
    assert r1["auc"]["n_valid_iterations"] <= 100
