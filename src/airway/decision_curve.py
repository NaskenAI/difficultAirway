"""
Decision-curve analysis (clinical utility) for the fused model.

WHAT THIS IS
------------
Discrimination (AUC) tells you how well a model ranks patients; it does not tell
you whether *acting* on the model helps. Decision-curve analysis answers the
latter by plotting net benefit against the threshold probability at which a
clinician would act (here, escalate airway preparation). It compares the fused
model and the best bedside score against the two trivial strategies — treat
everyone and treat no one (Vickers & Elkin).

NET BENEFIT
-----------
At threshold probability p_t, with patients flagged when predicted risk ≥ p_t:

    NB = TP/n − (FP/n) · (p_t / (1 − p_t))

    treat-all  = prevalence − (1 − prevalence) · (p_t / (1 − p_t))
    treat-none = 0

The threshold grid runs from 0.01 to MAX_THRESHOLD (0.50, because difficult
airway is uncommon) in 0.01 steps.

BEDSIDE SCORE MAPPING
---------------------
The best bedside score (chosen by AUC from reports/per_model_metrics.csv) is
ordinal, not a probability. To place it on the same p_t axis we min–max
normalise it to [0, 1] across the cohort and flag patients when the normalised
score ≥ p_t. This is a monotonic re-scaling (it does not change the score's
ranking), made explicit here.

LEAKAGE / COHORT
----------------
Per-patient values come from reports/fusion_fold_predictions.csv reduced via
airway.predictions.per_patient (one row per patient). All curves are computed on
the same complete-case cohort (patients with both a fused probability and the
selected bedside score) so the comparison and the prevalence line are consistent.

OUTPUTS (reports/)
------------------
  decision_curve.csv : threshold, nb_fused, nb_best_bedside, nb_treat_all, nb_treat_none
  decision_curve.png : the four net-benefit curves
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airway import config, predictions

MAX_THRESHOLD = 0.50
STEP = 0.01
PER_MODEL_CSV = config.REPORTS_DIR / "per_model_metrics.csv"
OUT_CSV = config.REPORTS_DIR / "decision_curve.csv"
OUT_PNG = config.REPORTS_DIR / "decision_curve.png"

# bedside model name (per_model_metrics.csv) -> its per-patient score column
BEDSIDE_SCORE_COL = {
    "mallampati": "mallampati_class",
    "lemon": "lemon_score",
    "wilson": "wilson_score",
}


def _threshold_grid() -> np.ndarray:
    return np.round(np.arange(STEP, MAX_THRESHOLD + 1e-9, STEP), 2)


def _net_benefit(label: np.ndarray, flagged: np.ndarray, p_t: float, n: int) -> float:
    tp = int(np.sum(flagged & (label == 1)))
    fp = int(np.sum(flagged & (label == 0)))
    w = p_t / (1.0 - p_t)
    return tp / n - (fp / n) * w


def _best_bedside() -> str:
    """Name of the highest-AUC bedside score from per_model_metrics.csv."""
    if not PER_MODEL_CSV.exists():
        raise FileNotFoundError(
            f"decision_curve: {PER_MODEL_CSV} not found. Run `make block-c` "
            f"(python -m airway.clinical_comparison) first."
        )
    pm = pd.read_csv(PER_MODEL_CSV)
    bedside = pm[pm["model"].isin(BEDSIDE_SCORE_COL)]
    if bedside.empty:
        raise ValueError("decision_curve: no bedside models found in per_model_metrics.csv.")
    return str(bedside.loc[bedside["auc_mean"].idxmax(), "model"])


def _cohort() -> tuple[pd.DataFrame, str]:
    """Per-patient fused_prob, normalised best-bedside score, and label."""
    from airway import loaders, scores

    pp = predictions.per_patient(predictions.load_fusion_predictions())
    best = _best_bedside()
    score_col = BEDSIDE_SCORE_COL[best]

    comp = scores.compute_comparator_scores(loaders.preop_loader())
    pp = pp.merge(comp[[config.ID_COL, score_col]], on=config.ID_COL, how="left")
    data = pp.dropna(subset=["fused_prob", score_col, config.LABEL_COL]).copy()

    s = data[score_col].to_numpy(dtype=float)
    lo, hi = s.min(), s.max()
    data["bedside_norm"] = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)
    return data, best


def build_curve() -> tuple[pd.DataFrame, str, float]:
    """Return (decision_curve_df, best_bedside_name, prevalence)."""
    data, best = _cohort()
    y = data[config.LABEL_COL].to_numpy()
    fused = data["fused_prob"].to_numpy()
    bedside = data["bedside_norm"].to_numpy()
    n = len(data)
    prevalence = float(np.mean(y))

    rows = []
    for p_t in _threshold_grid():
        w = p_t / (1.0 - p_t)
        rows.append({
            "threshold": float(p_t),
            "nb_fused": _net_benefit(y, fused >= p_t, p_t, n),
            "nb_best_bedside": _net_benefit(y, bedside >= p_t, p_t, n),
            "nb_treat_all": prevalence - (1.0 - prevalence) * w,
            "nb_treat_none": 0.0,
        })
    return pd.DataFrame(rows), best, prevalence


def _save_plot(curve: pd.DataFrame, best: str, out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.plot(curve["threshold"], curve["nb_fused"], label="Fused model")
    plt.plot(curve["threshold"], curve["nb_best_bedside"],
             label=f"Best bedside ({best})")
    plt.plot(curve["threshold"], curve["nb_treat_all"], "--", label="Treat all")
    plt.plot(curve["threshold"], curve["nb_treat_none"], ":", label="Treat none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision-curve analysis — difficult airway (CL 3–4)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main() -> None:
    config.ensure_dirs()
    curve, best, prevalence = build_curve()
    curve.to_csv(OUT_CSV, index=False)
    _save_plot(curve, best, OUT_PNG)
    print(f"decision curve -> {OUT_CSV}  ({len(curve)} thresholds; "
          f"best bedside = {best}; prevalence = {prevalence:.3f})")
    print(f"plot           -> {OUT_PNG}")


if __name__ == "__main__":
    main()
