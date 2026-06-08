"""
Block D — patient-level bootstrap confidence intervals.

WHAT THIS DOES
--------------
Reports 95% confidence intervals for every model's headline metrics by
resampling PATIENTS (not rows or images) with replacement, 1000 times. For each
probability model in the fusion fold-predictions table (face, ultrasound, fused,
average) it computes AUC, sensitivity, specificity, PPV and NPV at the 0.5
threshold on each bootstrap sample, then takes the mean and the 2.5 / 97.5
percentiles.

WHY PATIENT-LEVEL
-----------------
A patient contributes several fold rows (one per CV repeat). Resampling rows
would treat the same patient as independent observations and understate the
uncertainty. So we first collapse to one row per patient (averaging that
patient's probabilities across folds) and bootstrap over patients.

AUC ON A DEGENERATE RESAMPLE
----------------------------
A bootstrap resample can occasionally contain only one outcome class; AUC is
undefined there, so that iteration is SKIPPED for AUC (the other metrics are
still defined and kept). `n_valid_iterations` records how many iterations
contributed to each metric.

OUTPUT
------
reports/bootstrap_ci.csv : model, metric, point_estimate, ci_lower, ci_upper,
                           n_valid_iterations
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airway import config, predictions

N_BOOTSTRAP = 1000
THRESHOLD = 0.5
METRICS = ["auc", "sensitivity", "specificity", "ppv", "npv"]
MODEL_COLS = ["face_prob", "us_prob", "fused_prob", "avg_prob"]

OUT_CSV = config.REPORTS_DIR / "bootstrap_ci.csv"


def _metrics(y_true: np.ndarray, score: np.ndarray) -> dict:
    """The five reported metrics for one (label, score) vector at THRESHOLD."""
    full = predictions.full_metrics(y_true, score, THRESHOLD, is_probability=True)
    return {m: full[m] for m in METRICS}


def bootstrap_model(y_true: np.ndarray, score: np.ndarray,
                    n_boot: int = N_BOOTSTRAP, seed: int = config.RANDOM_SEED) -> dict:
    """
    Point estimate + percentile bootstrap CI for one model's metrics.

    Returns {metric: {"point_estimate", "ci_lower", "ci_upper", "n_valid_iterations"}}.
    """
    point = _metrics(y_true, score)
    samples = {m: [] for m in METRICS}

    rng = np.random.default_rng(seed)
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)            # resample PATIENTS with replacement
        m = _metrics(y_true[idx], score[idx])
        for key, val in m.items():
            if not np.isnan(val):                   # AUC skipped on one-class resamples
                samples[key].append(val)

    out = {}
    for m in METRICS:
        vals = np.array(samples[m], dtype=float)
        if len(vals):
            lo, hi = np.percentile(vals, [2.5, 97.5])
        else:
            lo = hi = float("nan")
        out[m] = {"point_estimate": point[m], "ci_lower": float(lo),
                  "ci_upper": float(hi), "n_valid_iterations": int(len(vals))}
    return out


def build_table(n_boot: int = N_BOOTSTRAP) -> pd.DataFrame:
    """Bootstrap CIs for every model column, as a long-format table."""
    preds = predictions.load_fusion_predictions()
    pp = predictions.per_patient(preds)
    y = pp[config.LABEL_COL].to_numpy()

    rows = []
    for col in MODEL_COLS:
        if col not in pp.columns:
            print(f"  bootstrap_ci: column '{col}' absent; skipping.")
            continue
        cis = bootstrap_model(y, pp[col].to_numpy(), n_boot)
        for metric in METRICS:
            c = cis[metric]
            rows.append({
                "model": col, "metric": metric,
                "point_estimate": round(c["point_estimate"], 4)
                if not np.isnan(c["point_estimate"]) else np.nan,
                "ci_lower": round(c["ci_lower"], 4) if not np.isnan(c["ci_lower"]) else np.nan,
                "ci_upper": round(c["ci_upper"], 4) if not np.isnan(c["ci_upper"]) else np.nan,
                "n_valid_iterations": c["n_valid_iterations"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    config.ensure_dirs()
    table = build_table()
    table.to_csv(OUT_CSV, index=False)
    print(f"patient-level bootstrap CIs ({N_BOOTSTRAP} iters) -> {OUT_CSV}\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
