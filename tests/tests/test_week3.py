"""
Week 3 tests: comparator scores, quarantine rules, data audit.

Run against dummy data. If data is missing:
    python -m airway.make_dummy_data
"""

from __future__ import annotations

import pandas as pd
import pytest

from airway import config, data_audit, loaders, quarantine, scores


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def have_data() -> bool:
    if not config.LABELS_CSV.exists() or not config.PREOP_CSV.exists():
        pytest.skip("Dummy data missing. Run: python -m airway.make_dummy_data")
    return True


# ===========================================================================
# SCORES — known-input / known-output (deterministic)
# ===========================================================================
def test_mallampati_known_values():
    df = pd.DataFrame({config.ID_COL: ["a", "b"], "mallampati_class": [2, 3]})
    out = scores.mallampati_scores(df)
    assert out["mallampati_difficult"].tolist() == [0, 1]


def test_lemon_full_and_zero():
    df = pd.DataFrame({
        config.ID_COL: ["hard", "easy"],
        "buck_teeth": [1, 0],
        "mouth_opening_mm": [30, 50],
        "thyromental_mm": [50, 80],
        "mallampati_class": [4, 1],
        "obstructed_airway": [1, 0],
        "neck_movement_deg": [70, 100],
    })
    out = scores.lemon_scores(df)
    assert out["lemon_score"].tolist() == [6.0, 0.0]
    assert out["lemon_difficult"].tolist() == [1, 0]


def test_wilson_full_and_zero_and_jaw():
    df = pd.DataFrame({
        config.ID_COL: ["hard", "easy", "mid"],
        "weight_class": [2, 0, 0],
        "head_neck_class": [2, 0, 0],
        "receding_mandible": [2, 0, 0],
        "mouth_opening_mm": [30, 60, 60],   # mid: not small opening
        "jaw_subluxation": [1, 0, 1],       # mid: cannot protrude -> jaw=1
        "buck_teeth": [1, 0, 0],
    })
    out = scores.wilson_scores(df)
    assert out["wilson_score"].tolist() == [10.0, 0.0, 1.0]
    assert out["wilson_jaw"].tolist() == [2.0, 0.0, 1.0]
    assert out["wilson_difficult"].tolist() == [1, 0, 0]


def test_scores_missing_column_degrades_to_nan():
    # no mallampati_class column at all -> Mallampati NaN, LEMON total NaN
    df = pd.DataFrame({
        config.ID_COL: ["x"],
        "buck_teeth": [0], "mouth_opening_mm": [50], "thyromental_mm": [80],
        "obstructed_airway": [0], "neck_movement_deg": [100],
    })
    mall = scores.mallampati_scores(df)
    assert pd.isna(mall["mallampati_class"].iloc[0])
    lem = scores.lemon_scores(df)
    assert pd.isna(lem["lemon_M"].iloc[0])
    assert pd.isna(lem["lemon_score"].iloc[0])


def test_compute_comparator_scores_deterministic(have_data):
    preop = loaders.preop_loader()
    a = scores.compute_comparator_scores(preop)
    b = scores.compute_comparator_scores(preop)
    pd.testing.assert_frame_equal(a, b)


def test_comparator_scores_one_row_per_patient(have_data):
    preop = loaders.preop_loader()
    out = scores.compute_comparator_scores(preop)
    assert not out[config.ID_COL].duplicated().any()
    for col in ("mallampati_class", "lemon_score", "wilson_score"):
        assert col in out.columns


# ===========================================================================
# QUARANTINE
# ===========================================================================
def test_quarantine_is_deterministic(have_data):
    a = quarantine.compute_quarantine(check_faces=False)
    b = quarantine.compute_quarantine(check_faces=False)
    assert a == b


def test_quarantine_imputed_cells_match_missing_ultrasound(have_data):
    decisions = quarantine.compute_quarantine(check_faces=False)
    us = loaders.us_loader()
    # every flagged cell must actually be missing in the ultrasound table
    for cell in decisions["imputed_us_cells"]:
        val = us.loc[us[config.ID_COL] == cell["study_id"], cell["column"]]
        assert val.isna().all()


def test_quarantine_save_and_load_roundtrip(have_data, tmp_path, monkeypatch):
    monkeypatch.setattr(quarantine, "DECISIONS_JSON", tmp_path / "q.json")
    monkeypatch.setattr(quarantine, "RULES_MD", tmp_path / "q.md")
    decisions = quarantine.compute_quarantine(check_faces=False)
    quarantine.save_quarantine(decisions)
    loaded = quarantine.load_quarantine()
    assert loaded["counts"] == decisions["counts"]
    assert (tmp_path / "q.md").exists()


# ===========================================================================
# DATA AUDIT
# ===========================================================================
def test_audit_report_has_all_sections(have_data):
    report = data_audit.build_report()
    for heading in ("Per-modality usability", "Missingness",
                    "Cormack-Lehane grade distribution", "Demographics",
                    "Inter-observer agreement"):
        assert heading in report


def test_audit_kappa_present_when_second_observer_exists(have_data):
    labels = loaders.label_loader()
    if config.CL_GRADE_OBS2_COL not in labels.columns:
        pytest.skip("no second observer in dummy data")
    k = data_audit._interobserver_kappa()
    assert k is not None
    assert -1.0 <= k["kappa_grade_quadratic"] <= 1.0


# ===========================================================================
# Week-12 extension: ultrasound inter-rater ICC(2,1)
# ===========================================================================
def test_icc_high_when_obs2_matches_obs1():
    import numpy as np
    rng = np.random.default_rng(0)
    obs1 = rng.normal(40, 8, 20)
    obs2 = obs1 + rng.normal(0, 0.3, 20)        # near-identical second reading
    icc = data_audit.icc_2_1(np.column_stack([obs1, obs2]))
    assert icc > 0.8


def test_icc_low_when_obs2_is_noise():
    import numpy as np
    rng = np.random.default_rng(1)
    obs1 = rng.normal(40, 8, 30)
    obs2 = rng.normal(40, 8, 30)                # unrelated second reading
    icc = data_audit.icc_2_1(np.column_stack([obs1, obs2]))
    assert icc < 0.5


def test_icc_in_sane_range():
    import numpy as np
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (15, 2))
    icc = data_audit.icc_2_1(x)
    assert -1.0 <= icc <= 1.0


def test_ultrasound_icc_absent_returns_none(tmp_path, monkeypatch):
    import pandas as pd
    csv = tmp_path / "us_no_obs2.csv"
    pd.DataFrame({
        config.ID_COL: ["P1", "P2"],
        "dstvc_mm": [18.0, 19.0], "hmd_neutral_mm": [45.0, 40.0],
        "hmd_extended_mm": [55.0, 50.0], "dse_mm": [27.0, 25.0],
    }).to_csv(csv, index=False)
    monkeypatch.setattr(config, "ULTRASOUND_CSV", csv)
    assert data_audit._ultrasound_icc() is None


def test_audit_reports_icc_not_available_when_absent(monkeypatch):
    # when no repeat measurements exist, the report still builds and says so
    monkeypatch.setattr(data_audit, "_ultrasound_icc", lambda: None)
    report = data_audit.build_report()
    assert "Ultrasound inter-rater reliability (ICC 2,1)" in report
    assert "not available" in report
