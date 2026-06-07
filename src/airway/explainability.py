"""
Block D / Week 13 — explainability for the fused model.

WHAT THIS DOES
--------------
1. SHAP summary for the fused-model inputs (the calibrated face and ultrasound
   probabilities — the meta-learner's only inputs; any ultrasound features
   present in the fusion input table would be included automatically).
   If SHAP is unavailable or the explainer fails, falls back to a
   coefficient-based feature-importance summary and records a note.
2. Automatic per-prediction case selection: up to two each of true positives,
   true negatives, false positives, false negatives (at threshold 0.5).
3. Per-case SHAP force plots when SHAP supports it in this environment;
   otherwise writes a clear note instead of failing.

OUTPUTS
-------
  reports/shap_summary_fused.png            (or skipped -> see notes)
  reports/explainability_feature_summary.csv
  reports/explanation_case_selection.csv
  outputs/explainability/force_plots/*.png  (or force_plot_notes.md)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airway import config, fusion, predictions

SHAP_SUMMARY_PNG = config.REPORTS_DIR / "shap_summary_fused.png"
FEATURE_SUMMARY_CSV = config.REPORTS_DIR / "explainability_feature_summary.csv"
CASE_CSV = config.REPORTS_DIR / "explanation_case_selection.csv"
FORCE_NOTES_MD = config.EXPLAIN_DIR / "force_plot_notes.md"

CATEGORIES = ["TP", "TN", "FP", "FN"]
MAX_PER_CATEGORY = 2


def _try_import_shap():
    try:
        import shap  # noqa: PLC0415
        return shap
    except Exception as err:  # noqa: BLE001 - any import failure -> fallback
        print(f"explainability: SHAP unavailable ({type(err).__name__}: {err}); "
              f"using coefficient-based fallback.")
        return None


def _load_fused_model():
    import joblib
    if not fusion.FUSED_PKL.exists():
        raise FileNotFoundError(
            f"explainability: {fusion.FUSED_PKL} not found. Run "
            f"`python -m airway.fusion` first."
        )
    bundle = joblib.load(fusion.FUSED_PKL)
    return bundle["meta_learner"], bundle["inputs"]


def select_cases(pp: pd.DataFrame, prob_col: str = "fused_prob",
                 threshold: float = predictions.DEFAULT_THRESHOLD,
                 max_per_category: int = MAX_PER_CATEGORY) -> pd.DataFrame:
    """Up to `max_per_category` patients per confusion category (deterministic)."""
    df = pp.copy()
    df["predicted_class"] = predictions.predicted_class(df[prob_col], threshold)
    df["error_type"] = [predictions.confusion_category(int(y), int(p))
                        for y, p in zip(df[config.LABEL_COL], df["predicted_class"])]

    rows = []
    for cat in CATEGORIES:
        # deterministic: sort by study_id, take the first up-to-N
        sub = df[df["error_type"] == cat].sort_values(config.ID_COL).head(max_per_category)
        for _, r in sub.iterrows():
            rows.append({
                config.ID_COL: r[config.ID_COL],
                "true_label": int(r[config.LABEL_COL]),
                "predicted_prob": round(float(r[prob_col]), 4),
                "predicted_class": int(r["predicted_class"]),
                "error_type": cat,
            })
    return pd.DataFrame(rows, columns=[config.ID_COL, "true_label",
                                       "predicted_prob", "predicted_class", "error_type"])


def _coefficient_summary(meta, inputs, X: pd.DataFrame) -> pd.DataFrame:
    """Fallback importance when SHAP is unavailable: |standardised coefficient|."""
    coef = np.ravel(meta.coef_)
    std = X[inputs].std(axis=0).to_numpy()
    importance = np.abs(coef * std)
    return pd.DataFrame({
        "feature": inputs,
        "importance": np.round(importance, 6),
        "model_coef": np.round(coef, 6),
        "method": "logreg_standardised_coef (SHAP unavailable)",
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def _shap_summary(shap, meta, inputs, X: pd.DataFrame):
    """Return (shap_values ndarray, expected_value, feature_summary_df)."""
    explainer = shap.LinearExplainer(meta, X[inputs])
    sv = explainer.shap_values(X[inputs])
    sv = np.asarray(sv)
    expected = explainer.expected_value
    expected = float(np.ravel(expected)[0]) if np.ndim(expected) else float(expected)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap.summary_plot(sv, X[inputs], show=False)
    plt.title("Fused model — SHAP summary")
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PNG, dpi=120, bbox_inches="tight")
    plt.close()

    summary = pd.DataFrame({
        "feature": inputs,
        "mean_abs_shap": np.round(np.abs(sv).mean(axis=0), 6),
        "model_coef": np.round(np.ravel(meta.coef_), 6),
        "method": "shap_linear",
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return sv, expected, summary


def _force_plots(shap, explainer_vals, expected, X, cases: pd.DataFrame,
                 inputs) -> list[str]:
    """One SHAP force plot per selected case; returns notes about any failures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.FORCE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    notes = []
    id_to_pos = {pid: i for i, pid in enumerate(X.index)}
    for _, c in cases.iterrows():
        pid = c[config.ID_COL]
        pos = id_to_pos.get(pid)
        if pos is None:
            notes.append(f"- `{pid}`: no feature row found; skipped.")
            continue
        try:
            shap.force_plot(expected, explainer_vals[pos], X[inputs].iloc[pos],
                            matplotlib=True, show=False)
            out = config.FORCE_PLOTS_DIR / f"force_{c['error_type']}_{pid}.png"
            plt.savefig(out, dpi=120, bbox_inches="tight")
            plt.close()
        except Exception as err:  # noqa: BLE001
            notes.append(f"- `{pid}` ({c['error_type']}): force plot failed "
                         f"({type(err).__name__}: {err}).")
    return notes


def _write_force_notes(lines: list[str]) -> None:
    config.EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
    FORCE_NOTES_MD.write_text("\n".join(["# Force plot notes", "", *lines, ""]))


def main() -> None:
    config.ensure_dirs()
    config.EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)

    meta, inputs = _load_fused_model()
    master = predictions.build_master_table()
    pp = predictions.per_patient(master).set_index(config.ID_COL)
    # keep the index as study_id but also a column for downstream selection
    pp_reset = pp.reset_index()

    missing = [c for c in inputs if c not in pp.columns]
    if missing:
        raise ValueError(f"explainability: fused inputs missing from predictions: {missing}")

    X = pp[inputs].copy()

    # --- case selection (independent of SHAP availability) ------------------
    cases = select_cases(pp_reset)
    cases.to_csv(CASE_CSV, index=False)
    counts = cases["error_type"].value_counts().to_dict()
    print(f"explanation cases -> {CASE_CSV}  ({len(cases)} cases: {counts})")

    shap = _try_import_shap()
    force_notes: list[str] = []

    if shap is None:
        summary = _coefficient_summary(meta, inputs, X)
        force_notes = ["SHAP is not available in this environment, so SHAP summary "
                       "and force plots were not generated. "
                       "`explainability_feature_summary.csv` falls back to "
                       "standardised logistic-regression coefficients."]
    else:
        try:
            sv, expected, summary = _shap_summary(shap, meta, inputs, X)
            print(f"SHAP summary plot -> {SHAP_SUMMARY_PNG}")
            force_notes = _force_plots(shap, sv, expected, X, cases, inputs)
            if not force_notes:
                force_notes = [f"Force plots generated for {len(cases)} cases in "
                               f"`{config.FORCE_PLOTS_DIR}`."]
        except Exception as err:  # noqa: BLE001
            print(f"explainability: SHAP explainer failed ({type(err).__name__}: {err}); "
                  f"using coefficient fallback.")
            summary = _coefficient_summary(meta, inputs, X)
            force_notes = [f"SHAP failed at runtime ({type(err).__name__}: {err}); "
                           f"used coefficient-based fallback; no force plots."]

    summary.to_csv(FEATURE_SUMMARY_CSV, index=False)
    _write_force_notes(force_notes)
    print(f"feature summary   -> {FEATURE_SUMMARY_CSV}")
    print(f"force-plot notes  -> {FORCE_NOTES_MD}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
