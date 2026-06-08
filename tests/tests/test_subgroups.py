"""Tests for airway.subgroups (descriptive subgroup AUC)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airway import fusion, subgroups


def test_tertiles_cover_all_values():
    s = pd.Series(np.arange(30, dtype=float))
    tert = subgroups.tertiles(s)
    assert tert.notna().all()                       # every value assigned a level
    assert set(tert.unique()) == {"low", "mid", "high"}


def test_tertiles_keep_missing_as_nan():
    s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    tert = subgroups.tertiles(s)
    assert pd.isna(tert.iloc[2])
    assert tert.notna().sum() == 5


@pytest.fixture(scope="module")
def table():
    if not fusion.FUSION_FOLD_PRED_CSV.exists():
        pytest.skip("fusion_fold_predictions.csv missing; run `make block-c` first.")
    return subgroups.build_table()


def test_auc_in_unit_interval_or_nan(table):
    for v in table["auc"].to_numpy():
        assert np.isnan(v) or (0.0 <= v <= 1.0)


def test_subgroup_levels_sum_to_total(table):
    from airway import loaders, predictions
    pp = predictions.per_patient(predictions.load_fusion_predictions())
    total = pp[predictions.config.ID_COL].nunique()
    # within each subgroup type, level counts sum to the number of patients with
    # that variable present; with the synthetic data every patient has sex/bmi/age
    preop = loaders.preop_loader()
    for subgroup, grp in table.groupby("subgroup"):
        var = {"sex": "sex", "bmi_tertile": "bmi", "age_tertile": "age_years"}[subgroup]
        n_present = preop[preop[predictions.config.ID_COL].isin(pp[predictions.config.ID_COL])][var].notna().sum()
        assert grp["n"].sum() == n_present == total
