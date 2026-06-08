"""
Block E — beginner-runnable demonstrator (NOT a notebook dependency).

==============================================================================
 ⚠️  NOT FOR CLINICAL USE — RESEARCH DEMONSTRATOR ONLY  ⚠️
 Trained on a tiny synthetic pilot dataset. The numbers are illustrative and
 must never inform patient care.
==============================================================================

WHAT THIS SHOWS
---------------
How the persisted models turn one patient's features into a fused difficult-
airway probability, a risk tier, and the ultrasound features that pushed the
ultrasound score up or down.

    raw face features (1024-d)  --[face logistic model]-->  face probability
    raw ultrasound features (5) --[ultrasound logistic]-->  ultrasound probability
            face prob + ultrasound prob --[fused meta-learner]--> fused probability

Run it:   PYTHONPATH=src python notebooks/demo.py

NOTE ON CALIBRATION: the calibrated single-modality MODELS are not persisted
(calibration is done within CV folds). This demo therefore uses the persisted
primary (logistic) single-modality models as the base scorers and feeds them to
the fused meta-learner — fine for a demonstrator, not for deployment.
"""

from __future__ import annotations

import sys
from pathlib import Path

# allow running as `python notebooks/demo.py` without PYTHONPATH=src set
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd

from airway import config

RISK_TIERS = [("Low", 0.20), ("Moderate", 0.60), ("High", 1.01)]


def risk_tier(prob: float) -> str:
    """Low < 0.20, Moderate 0.20-0.60, High > 0.60."""
    for name, upper in RISK_TIERS:
        if prob < upper:
            return name
    return "High"


def _load_models():
    for pkl in (config.REPORTS_DIR / "fused_model.pkl",
                config.REPORTS_DIR / "us_model.pkl",
                config.REPORTS_DIR / "face_model.pkl"):
        if not pkl.exists():
            raise FileNotFoundError(
                f"demo: {pkl} not found. Run `make week45` and `make block-c` first."
            )
    fused = joblib.load(config.REPORTS_DIR / "fused_model.pkl")
    us = joblib.load(config.REPORTS_DIR / "us_model.pkl")
    face = joblib.load(config.REPORTS_DIR / "face_model.pkl")
    return fused, us, face


def _top_us_contributions(us_bundle, us_row: pd.Series, k: int = 3):
    """
    Rank ultrasound features by their signed contribution to the logistic score
    for THIS patient: coefficient x standardised feature value.
    """
    pipe = us_bundle["models"]["logreg_l2"]
    cols = us_bundle["feature_cols"]
    x = us_row[cols].to_numpy(dtype=float).reshape(1, -1)
    steps = dict(pipe.named_steps)
    x = steps["impute"].transform(x)
    x = steps["scale"].transform(x)
    coef = np.ravel(steps["model"].coef_)
    contrib = (coef * x.ravel())
    order = np.argsort(-np.abs(contrib))[:k]
    return [(cols[i], round(float(contrib[i]), 4)) for i in order]


def predict_patient(face_feature_row: pd.Series, us_feature_row: pd.Series,
                    models=None) -> dict:
    """
    Predict the fused difficult-airway probability for one patient.

    Parameters
    ----------
    face_feature_row : Series with the 1024 face_* features (from face_features).
    us_feature_row   : Series with the ultrasound features (from cleaned US table).

    Returns dict: face_prob, us_prob, fused_prob, risk_tier, top_us_features.
    """
    fused, us, face = models if models is not None else _load_models()

    face_cols = face["feature_cols"]
    us_cols = us["feature_cols"]
    face_x = face_feature_row[face_cols].to_numpy(dtype=float).reshape(1, -1)
    us_x = us_feature_row[us_cols].to_numpy(dtype=float).reshape(1, -1)

    face_prob = float(face["models"]["logreg_l2"].predict_proba(face_x)[:, 1][0])
    us_prob = float(us["models"]["logreg_l2"].predict_proba(us_x)[:, 1][0])

    meta = fused["meta_learner"]
    fused_prob = float(meta.predict_proba(np.array([[face_prob, us_prob]]))[:, 1][0])

    return {
        "face_prob": round(face_prob, 4),
        "us_prob": round(us_prob, 4),
        "fused_prob": round(fused_prob, 4),
        "risk_tier": risk_tier(fused_prob),
        "top_us_features": _top_us_contributions(us, us_feature_row),
    }


def main() -> None:
    print(__doc__.split("WHAT THIS SHOWS")[0])   # print the banner

    face_tbl = pd.read_parquet(config.FACE_FEATURES_PARQUET)
    us_tbl = pd.read_csv(config.CLEANED_US_CSV)
    common = sorted(set(face_tbl[config.ID_COL]) & set(us_tbl[config.ID_COL]))
    if not common:
        raise SystemExit("demo: no patient has both face and ultrasound features.")
    pid = common[0]

    face_row = face_tbl[face_tbl[config.ID_COL] == pid].iloc[0]
    us_row = us_tbl[us_tbl[config.ID_COL] == pid].iloc[0]

    result = predict_patient(face_row, us_row)
    print(f"Example patient: {pid}")
    for k, v in result.items():
        print(f"  {k:16s}: {v}")
    print("\n⚠️  NOT FOR CLINICAL USE — research demonstrator on synthetic data.")


if __name__ == "__main__":
    main()
