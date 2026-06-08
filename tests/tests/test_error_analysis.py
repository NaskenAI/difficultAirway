"""Tests for airway.error_analysis (per-patient TP/TN/FP/FN)."""

from __future__ import annotations

import pytest

from airway import config, error_analysis, fusion


@pytest.fixture(scope="module")
def table():
    if not fusion.FUSION_FOLD_PRED_CSV.exists():
        pytest.skip("fusion_fold_predictions.csv missing; run `make block-c` first.")
    return error_analysis.build_error_table()


def test_every_patient_has_exactly_one_category(table):
    assert table[config.ID_COL].is_unique
    assert table["category"].isin(["TP", "TN", "FP", "FN"]).all()


def test_categories_sum_to_n_patients(table):
    counts = table["category"].value_counts()
    total = sum(counts.get(c, 0) for c in ["TP", "TN", "FP", "FN"])
    assert total == len(table)


def test_category_matches_rule(table):
    for _, r in table.iterrows():
        expected = {
            (1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN",
        }[(int(r["label"]), int(r["predicted"]))]
        assert r["category"] == expected


def test_required_columns_present(table):
    for col in error_analysis.OUT_COLS:
        assert col in table.columns
