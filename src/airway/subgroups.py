"""
Block D — descriptive subgroup AUC of the fused model.

WHAT THIS DOES
--------------
Reports the fused model's AUC within subgroups (one row per patient, fused_prob
averaged across folds), for:
  - sex (as recorded)
  - BMI tertile (low / mid / high, computed here)
  - age tertile (low / mid / high, computed here)

For each subgroup level: n, n_difficult (CL 3-4), and AUC (NaN if the level has
a single outcome class).

IMPORTANT — DESCRIPTIVE / EXPLORATORY ONLY
------------------------------------------
These subgroups are small, so the AUCs are unstable and are NOT a hypothesis
test. The header of the CSV and the printout say so explicitly. Do not draw
subgroup conclusions from a pilot of this size.

OUTPUT
------
reports/subgroup_auc.csv : subgroup, level, n, n_difficult, auc
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from airway import config, predictions

OUT_CSV = config.REPORTS_DIR / "subgroup_auc.csv"
DISCLAIMER = ("DESCRIPTIVE/EXPLORATORY ONLY — small per-subgroup n; AUCs are "
              "unstable and are not hypothesis tests.")


def tertiles(values: pd.Series) -> pd.Series:
    """
    Rank-based tertile labels (low/mid/high) for a numeric Series. Rank-based
    qcut keeps the three bins balanced and handles ties; values that are missing
    stay NaN. Falls back to a single 'all' bin if there are <3 distinct values.
    """
    v = pd.to_numeric(values, errors="coerce")
    if v.dropna().nunique() < 3:
        return pd.Series(np.where(v.isna(), np.nan, "all"), index=v.index)
    labels = ["low", "mid", "high"]
    binned = pd.qcut(v.rank(method="first"), q=3, labels=labels)
    return binned.astype("object").where(v.notna(), other=np.nan)


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def _rows_for(df: pd.DataFrame, subgroup: str, level_series: pd.Series) -> list[dict]:
    rows = []
    for level in pd.unique(level_series.dropna()):
        sub = df[level_series == level]
        y = sub[config.LABEL_COL].to_numpy()
        auc = _auc(y, sub["fused_prob"].to_numpy())
        rows.append({
            "subgroup": subgroup, "level": str(level), "n": int(len(sub)),
            "n_difficult": int(np.sum(y == 1)),
            "auc": round(auc, 4) if not np.isnan(auc) else np.nan,
        })
    return rows


def build_table() -> pd.DataFrame:
    preds = predictions.load_fusion_predictions()
    pp = predictions.per_patient(preds)[[config.ID_COL, "fused_prob", config.LABEL_COL]]

    from airway import loaders
    preop = loaders.preop_loader()
    keep = [c for c in ["sex", "bmi", "age_years"] if c in preop.columns]
    pp = pp.merge(preop[[config.ID_COL, *keep]], on=config.ID_COL, how="left")

    rows = []
    if "sex" in pp.columns:
        rows += _rows_for(pp, "sex", pp["sex"].astype("object"))
    if "bmi" in pp.columns:
        rows += _rows_for(pp, "bmi_tertile", tertiles(pp["bmi"]))
    if "age_years" in pp.columns:
        rows += _rows_for(pp, "age_tertile", tertiles(pp["age_years"]))
    return pd.DataFrame(rows)


def main() -> None:
    config.ensure_dirs()
    table = build_table()
    # write the disclaimer as a leading comment line, then the CSV
    with open(OUT_CSV, "w") as fh:
        fh.write(f"# {DISCLAIMER}\n")
        table.to_csv(fh, index=False)
    print(f"subgroup AUC ({DISCLAIMER}) -> {OUT_CSV}\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
