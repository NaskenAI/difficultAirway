"""
Block D / Week 12 — patient-level bootstrap 95% confidence intervals.

WHAT THIS DOES
--------------
For every model (face, ultrasound, fused, average baseline, and the clinical
comparators) it computes 95% bootstrap confidence intervals for all reported
metrics (AUC, sensitivity, specificity, PPV, NPV, accuracy, balanced accuracy,
F1, and Brier for probability models).

PATIENT-LEVEL RESAMPLING
------------------------
Resampling is over PATIENTS, not rows or images: we collapse to one row per
patient first (predictions averaged across CV repeats) and draw patients with
replacement. The seed is fixed (config.RANDOM_SEED) so the intervals are
reproducible. Bootstrap iterations in which a metric is undefined (e.g. AUC when
a resample contains a single outcome class) are skipped and counted.

OUTPUT
------
reports/bootstrap_metric_cis.csv  (one row per model x metric)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airway import config, predictions

N_BOOT = 1000
CI_LOW, CI_HIGH = 2.5, 97.5
OUT_CSV = config.REPORTS_DIR / "bootstrap_metric_cis.csv"


def bootstrap_metric_cis(y_true, y_score, threshold: float, is_probability: bool,
                         n_boot: int = N_BOOT, seed: int = config.RANDOM_SEED) -> dict:
    """
    Point estimate + percentile bootstrap CI for every metric of one model.

    Returns {metric: {"estimate", "ci_lower", "ci_upper", "n_valid"}}.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)

    point = predictions.full_metrics(y_true, y_score, threshold, is_probability)
    samples = {k: [] for k in predictions.METRIC_KEYS}

    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)        # resample PATIENTS with replacement
        m = predictions.full_metrics(y_true[idx], y_score[idx], threshold, is_probability)
        for k, v in m.items():
            if not np.isnan(v):
                samples[k].append(v)

    out = {}
    for k in predictions.METRIC_KEYS:
        vals = np.array(samples[k], dtype=float)
        if len(vals):
            lo, hi = np.percentile(vals, [CI_LOW, CI_HIGH])
        else:
            lo = hi = float("nan")
        out[k] = {"estimate": point[k], "ci_lower": float(lo),
                  "ci_upper": float(hi), "n_valid": int(len(vals))}
    return out


def build_table(n_boot: int = N_BOOT) -> pd.DataFrame:
    master = predictions.build_master_table()
    pp = predictions.per_patient(master)
    y = pp[config.LABEL_COL].to_numpy()

    rows = []
    for name, col, is_prob in predictions.model_specs(master):
        cis = bootstrap_metric_cis(
            y, pp[col].to_numpy(), predictions.DEFAULT_THRESHOLD, is_prob, n_boot)
        for metric in predictions.METRIC_KEYS:
            c = cis[metric]
            rows.append({
                "model": name, "metric": metric,
                "estimate": round(c["estimate"], 4) if not np.isnan(c["estimate"]) else np.nan,
                "ci_lower": round(c["ci_lower"], 4) if not np.isnan(c["ci_lower"]) else np.nan,
                "ci_upper": round(c["ci_upper"], 4) if not np.isnan(c["ci_upper"]) else np.nan,
                "n_boot_valid": c["n_valid"], "n_boot": n_boot,
                "n_patients": int(len(pp)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    config.ensure_dirs()
    table = build_table()
    table.to_csv(OUT_CSV, index=False)
    print(f"bootstrap metric CIs ({N_BOOT} iters, patient-level) -> {OUT_CSV}")
    # show AUC rows as a quick look
    auc = table[table["metric"] == "auc"]
    print(auc[["model", "estimate", "ci_lower", "ci_upper", "n_boot_valid"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
