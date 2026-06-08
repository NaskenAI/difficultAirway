"""
Block D — explainability.

WHAT THIS DOES
--------------
1. ULTRASOUND model (XGBoost): runs SHAP (TreeExplainer) on the fitted XGBoost
   from reports/us_model.pkl, over the cleaned ultrasound feature matrix, and
   writes a beeswarm summary plot plus a mean-|SHAP| importance table.
2. FACE model (1024-d logistic regression): a SHAP beeswarm over 1024 dimensions
   is not informative, so instead we export the top-30 absolute logistic-
   regression coefficients from the refit face model as a feature view.

WHY TWO DIFFERENT VIEWS
-----------------------
SHAP/TreeExplainer is exact and cheap for tree models, so it is the right tool
for the 5-feature ultrasound XGBoost. The face model is a linear model on 1024
embedding dimensions where individual dims are not interpretable; the magnitude
of the standardised coefficients is the honest, simple summary.

OUTPUTS (reports/)
------------------
  shap_ultrasound_summary.png    beeswarm summary of ultrasound SHAP values
  shap_ultrasound_importance.csv mean |SHAP| per ultrasound feature (sorted)
  face_importance.csv            top-30 |coef| of the face logistic model
                                 (skipped with a warning if face_model.pkl absent)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airway import config

US_MODEL_PKL = config.REPORTS_DIR / "us_model.pkl"
FACE_MODEL_PKL = config.REPORTS_DIR / "face_model.pkl"
SHAP_SUMMARY_PNG = config.REPORTS_DIR / "shap_ultrasound_summary.png"
SHAP_IMPORTANCE_CSV = config.REPORTS_DIR / "shap_ultrasound_importance.csv"
FACE_IMPORTANCE_CSV = config.REPORTS_DIR / "face_importance.csv"

FACE_TOP_N = 30


def _extract_pipeline_parts(pipeline):
    """Return (fitted_imputer_or_None, final_estimator) from an sklearn Pipeline."""
    steps = dict(pipeline.named_steps)
    estimator = pipeline.steps[-1][1]
    imputer = steps.get("impute")
    return imputer, estimator


def ultrasound_shap() -> pd.DataFrame:
    """
    SHAP for the ultrasound XGBoost model. Writes the beeswarm PNG and returns
    the mean-|SHAP| importance table (also written to CSV).
    """
    import joblib

    if not US_MODEL_PKL.exists():
        raise FileNotFoundError(
            f"explainability: {US_MODEL_PKL} not found. Run `make us-model` "
            f"(python -m airway.ultrasound_model) first."
        )
    bundle = joblib.load(US_MODEL_PKL)
    feature_cols = bundle["feature_cols"]
    pipeline = bundle["models"]["xgboost"]
    imputer, xgb = _extract_pipeline_parts(pipeline)

    if not config.CLEANED_US_CSV.exists():
        raise FileNotFoundError(
            f"explainability: {config.CLEANED_US_CSV} not found. Run "
            f"`make us-clean` (python -m airway.ultrasound_features) first."
        )
    cleaned = pd.read_csv(config.CLEANED_US_CSV)
    X = cleaned[feature_cols].to_numpy()
    # SHAP needs the same numeric matrix the model saw: apply the fitted imputer
    # (pass a bare array so the imputer does not warn about feature names).
    X_in = pd.DataFrame(imputer.transform(X) if imputer is not None else X,
                        columns=feature_cols)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_in)
    shap_values = np.asarray(shap_values)

    plt.figure()
    shap.summary_plot(shap_values, X_in, show=False)
    plt.title("Ultrasound XGBoost — SHAP summary")
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PNG, dpi=120, bbox_inches="tight")
    plt.close()

    importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance["mean_abs_shap"] = importance["mean_abs_shap"].round(6)
    importance.to_csv(SHAP_IMPORTANCE_CSV, index=False)
    return importance


def face_importance() -> pd.DataFrame | None:
    """
    Top-FACE_TOP_N absolute logistic-regression coefficients of the face model.
    Returns None (with a printed warning) if the face model pickle is missing.
    """
    import joblib

    if not FACE_MODEL_PKL.exists():
        print(f"explainability: WARNING — {FACE_MODEL_PKL} not found; "
              f"skipping face importance (run `make face-model` to produce it).")
        return None
    bundle = joblib.load(FACE_MODEL_PKL)
    feature_cols = bundle["feature_cols"]
    logreg = bundle["models"]["logreg_l2"].steps[-1][1]
    coef = np.ravel(logreg.coef_)

    imp = pd.DataFrame({"feature": feature_cols, "abs_coef": np.abs(coef),
                        "coef": coef})
    imp = imp.sort_values("abs_coef", ascending=False).head(FACE_TOP_N).reset_index(drop=True)
    imp["abs_coef"] = imp["abs_coef"].round(6)
    imp["coef"] = imp["coef"].round(6)
    imp.to_csv(FACE_IMPORTANCE_CSV, index=False)
    return imp


def main() -> None:
    config.ensure_dirs()
    us_imp = ultrasound_shap()
    print(f"ultrasound SHAP summary -> {SHAP_SUMMARY_PNG}")
    print(f"ultrasound SHAP import. -> {SHAP_IMPORTANCE_CSV}")
    print(us_imp.to_string(index=False))

    face_imp = face_importance()
    if face_imp is not None:
        print(f"\nface importance (top {FACE_TOP_N}) -> {FACE_IMPORTANCE_CSV}")
        print(face_imp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
