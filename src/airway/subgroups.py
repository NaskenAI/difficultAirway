"""
Block D / Week 12 — descriptive subgroup analyses.

WHAT THIS DOES (DESCRIPTIVE ONLY)
---------------------------------
Summarises the fused model's performance within subgroups:
  - BMI tertile (low / mid / high)
  - age tertile (low / mid / high)
  - surgery type (categorical)

For each subgroup level it reports n, event count/prevalence, and the fused
model's metrics. It also reports simple effect sizes (difference between the two
extreme tertiles / vs the most common surgery type) with patient-level bootstrap
confidence intervals.

NO HYPOTHESIS TESTING. These are descriptive summaries with uncertainty
intervals; no p-values are produced. Subgroups that are too small are still
reported but flagged `underpowered` (descriptive wording only — small samples
give wide, unreliable estimates).

OUTPUTS
-------
reports/subgroup_metrics.csv, reports/subgroup_effect_sizes.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airway import bootstrap_ci, config, predictions

# Below these, a subgroup is flagged underpowered (descriptive, not a hard gate).
MIN_SUBGROUP_N = 10
MIN_SUBGROUP_EVENTS = 5

FUSED_COL = "fused_prob"
OUT_METRICS = config.REPORTS_DIR / "subgroup_metrics.csv"
OUT_EFFECTS = config.REPORTS_DIR / "subgroup_effect_sizes.csv"


def add_tertiles(pp: pd.DataFrame, col: str) -> pd.Series:
    """
    Label each patient low/mid/high by tertile of `col`. Uses rank-based qcut so
    ties and small samples degrade gracefully; returns a string Series (NaN where
    the value is missing). Falls back to fewer bins if there are too few uniques.
    """
    values = pd.to_numeric(pp[col], errors="coerce")
    valid = values.dropna()
    if valid.nunique() < 3:
        # not enough distinct values for tertiles -> single 'all' bin
        return pd.Series(np.where(values.isna(), np.nan, "all"), index=pp.index)
    labels = ["low", "mid", "high"]
    try:
        binned = pd.qcut(values.rank(method="first"), q=3, labels=labels)
    except ValueError:
        return pd.Series(np.where(values.isna(), np.nan, "all"), index=pp.index)
    return binned.astype("object").where(values.notna(), other=np.nan)


def _subgroup_row(sub: pd.DataFrame, variable: str, level: str) -> dict:
    y = sub[config.LABEL_COL].to_numpy()
    n = len(sub)
    n_events = int(np.sum(y == 1))
    m = predictions.full_metrics(y, sub[FUSED_COL].to_numpy(),
                                 predictions.DEFAULT_THRESHOLD, is_probability=True)
    underpowered = (n < MIN_SUBGROUP_N) or (n_events < MIN_SUBGROUP_EVENTS) \
        or (n - n_events < MIN_SUBGROUP_EVENTS)
    note = ""
    if underpowered:
        note = (f"underpowered: n={n}, events={n_events}, non-events={n - n_events} "
                f"(below n>={MIN_SUBGROUP_N} / class>={MIN_SUBGROUP_EVENTS}); "
                f"estimates are unreliable")
    row = {"subgroup": variable, "level": str(level), "n": n,
           "n_events": n_events,
           "prevalence": round(n_events / n, 4) if n else np.nan,
           "underpowered": underpowered, "note": note}
    for k in predictions.METRIC_KEYS:
        row[k] = round(m[k], 4) if not np.isnan(m[k]) else np.nan
    return row


def subgroup_metrics(pp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # overall, as a reference row
    rows.append(_subgroup_row(pp, "overall", "all"))

    for col in config.TERTILE_SUBGROUP_COLS:
        if col not in pp.columns:
            print(f"  subgroups: column '{col}' absent; skipping tertile subgroup.")
            continue
        tert = add_tertiles(pp, col)
        for level in ["low", "mid", "high", "all"]:
            sub = pp[tert == level]
            if len(sub):
                rows.append(_subgroup_row(sub, f"{col}_tertile", level))

    for col in config.CATEGORICAL_SUBGROUP_COLS:
        if col not in pp.columns:
            print(f"  subgroups: column '{col}' absent; skipping categorical subgroup.")
            continue
        for level, sub in pp.groupby(col):
            rows.append(_subgroup_row(sub, col, level))
    return pd.DataFrame(rows)


def _effect_size(pp: pd.DataFrame, mask_a, mask_b, variable: str,
                 label_a: str, label_b: str, n_boot: int) -> dict:
    """
    Effect size = difference in fused-model AUC and in event prevalence between
    two subgroups (a - b), with a patient-level bootstrap 95% CI (resampling
    within each subgroup). Descriptive only.
    """
    a, b = pp[mask_a], pp[mask_b]
    ya, sa = a[config.LABEL_COL].to_numpy(), a[FUSED_COL].to_numpy()
    yb, sb = b[config.LABEL_COL].to_numpy(), b[FUSED_COL].to_numpy()

    def _auc(y, s):
        return predictions.full_metrics(y, s)["auc"]

    def _prev(y):
        return float(np.mean(y)) if len(y) else float("nan")

    auc_diff = _auc(ya, sa) - _auc(yb, sb)
    prev_diff = _prev(ya) - _prev(yb)

    rng = np.random.default_rng(config.RANDOM_SEED)
    auc_samples, prev_samples = [], []
    for _ in range(n_boot):
        ia = rng.integers(0, len(a), len(a))
        ib = rng.integers(0, len(b), len(b))
        da, dpa = _auc(ya[ia], sa[ia]), _prev(ya[ia])
        db, dpb = _auc(yb[ib], sb[ib]), _prev(yb[ib])
        if not (np.isnan(da) or np.isnan(db)):
            auc_samples.append(da - db)
        prev_samples.append(dpa - dpb)

    def _ci(vals):
        vals = np.array(vals, dtype=float)
        if not len(vals):
            return (np.nan, np.nan)
        return tuple(np.percentile(vals, [2.5, 97.5]))

    auc_lo, auc_hi = _ci(auc_samples)
    prev_lo, prev_hi = _ci(prev_samples)
    underpowered = min(len(a), len(b)) < MIN_SUBGROUP_N
    return {
        "subgroup": variable, "comparison": f"{label_a} - {label_b}",
        "n_a": len(a), "n_b": len(b),
        "auc_diff": round(auc_diff, 4) if not np.isnan(auc_diff) else np.nan,
        "auc_diff_ci_lower": round(auc_lo, 4) if not np.isnan(auc_lo) else np.nan,
        "auc_diff_ci_upper": round(auc_hi, 4) if not np.isnan(auc_hi) else np.nan,
        "prevalence_diff": round(prev_diff, 4) if not np.isnan(prev_diff) else np.nan,
        "prevalence_diff_ci_lower": round(prev_lo, 4) if not np.isnan(prev_lo) else np.nan,
        "prevalence_diff_ci_upper": round(prev_hi, 4) if not np.isnan(prev_hi) else np.nan,
        "n_boot": n_boot, "underpowered": underpowered,
        "note": "descriptive effect size; no hypothesis test performed",
    }


def subgroup_effect_sizes(pp: pd.DataFrame, n_boot: int = bootstrap_ci.N_BOOT) -> pd.DataFrame:
    rows = []
    for col in config.TERTILE_SUBGROUP_COLS:
        if col not in pp.columns:
            continue
        tert = add_tertiles(pp, col)
        if (tert == "high").any() and (tert == "low").any():
            rows.append(_effect_size(pp, tert == "high", tert == "low",
                                     f"{col}_tertile", "high", "low", n_boot))

    for col in config.CATEGORICAL_SUBGROUP_COLS:
        if col not in pp.columns:
            continue
        counts = pp[col].value_counts()
        if len(counts) >= 2:
            ref = counts.idxmax()                 # most common level as reference
            other = counts.index[1] if counts.index[0] == ref else counts.index[0]
            rows.append(_effect_size(pp, pp[col] == other, pp[col] == ref,
                                     col, str(other), str(ref), n_boot))
    return pd.DataFrame(rows)


def main() -> None:
    config.ensure_dirs()
    master = predictions.build_master_table()
    pp = predictions.per_patient(master)

    metrics = subgroup_metrics(pp)
    metrics.to_csv(OUT_METRICS, index=False)
    effects = subgroup_effect_sizes(pp)
    effects.to_csv(OUT_EFFECTS, index=False)

    n_under = int(metrics["underpowered"].sum())
    print(f"subgroup metrics  -> {OUT_METRICS}  ({len(metrics)} rows, "
          f"{n_under} flagged underpowered)")
    print(f"subgroup effects  -> {OUT_EFFECTS}  ({len(effects)} rows)")
    if n_under:
        print("  NOTE: underpowered subgroups are reported but their estimates "
              "are unreliable (small n / few events). Descriptive only.")
    print(metrics[["subgroup", "level", "n", "n_events", "prevalence", "auc",
                   "underpowered"]].to_string(index=False))


if __name__ == "__main__":
    main()
