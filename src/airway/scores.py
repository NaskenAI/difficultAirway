"""
Clinical comparator scores: Mallampati, LEMON, Wilson.

WHAT THIS DOES
--------------
These are the bedside scores an anaesthetist would record. They are the
COMPARATORS the ML model must beat. This module computes them deterministically
from the pre-op exam columns (config.PREOP_COLS) and writes one tidy row per
patient to reports/computed_baselines.csv.

WHY "DETERMINISTIC" MATTERS
---------------------------
Every score here is pure arithmetic on the input columns — no randomness, no
fitting, no thresholds learned from the data. The same input row always yields
the same score. The clinical cut-points are module-level constants below; if a
site uses different thresholds, change the constant (one place) and re-run.

GRACEFUL DEGRADATION
--------------------
Real pre-op sheets have gaps. Each score component reads its inputs with
`_col()`, which returns an all-NaN column if that field is entirely absent.
A component with a missing input becomes NaN for that row; the total score is
NaN if ANY of its components is NaN (we never silently treat "unknown" as
"normal"). The audit / model code can then decide how to handle the NaNs.

REFERENCES (clinical definitions encoded here)
----------------------------------------------
- Mallampati: class >= 3 flags a difficult view.
- LEMON: Look, Evaluate 3-3-2, Mallampati, Obstruction, Neck mobility.
- Wilson risk sum: weight, head/neck movement, jaw, receding mandible, buck
  teeth — each 0/1/2; sum >= 2 flags difficulty.
These are simplified, well-documented encodings suitable for a pilot, not a
substitute for a validated clinical scoring instrument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airway import config

# ---------------------------------------------------------------------------
# Clinical cut-points (edit here, in ONE place, if your site differs).
# ---------------------------------------------------------------------------
MALLAMPATI_DIFFICULT_CLASS = 3        # class >= this is "difficult"
INCISOR_GAP_MIN_MM = 40.0             # LEMON 3-3-2: mouth opening below this fails
THYROMENTAL_MIN_MM = 60.0             # LEMON 3-3-2 / classic difficult threshold
NECK_MOVEMENT_MIN_DEG = 80.0          # LEMON N: range of motion below this is limited
LEMON_DIFFICULT_THRESHOLD = 2         # LEMON total >= this flags difficulty
WILSON_JAW_SMALL_OPENING_MM = 50.0    # Wilson jaw component uses this opening cut
WILSON_DIFFICULT_THRESHOLD = 2        # Wilson sum >= this flags difficulty


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """
    Return column `name` as a float Series, or an all-NaN Series of the right
    length if the column is absent. This is what lets every score degrade
    gracefully when a pre-op field was not collected.
    """
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


# ===========================================================================
# MALLAMPATI
# ===========================================================================
def mallampati_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Pass through the Mallampati class and derive the difficult flag."""
    mc = _col(df, "mallampati_class")
    difficult = (mc >= MALLAMPATI_DIFFICULT_CLASS).astype("Int64")
    difficult[mc.isna()] = pd.NA
    return pd.DataFrame({
        "mallampati_class": mc,
        "mallampati_difficult": difficult,
    })


# ===========================================================================
# LEMON
# ===========================================================================
def lemon_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    LEMON, component by component. Each component is a 0/1 point (the E block
    contributes up to 2 points here: incisor gap + thyromental distance).
    Total ranges 0-6 with the inputs this pilot collects.
    """
    buck = _col(df, "buck_teeth")
    incisor = _col(df, "mouth_opening_mm")
    thyromental = _col(df, "thyromental_mm")
    mc = _col(df, "mallampati_class")
    obstruction = _col(df, "obstructed_airway")
    neck = _col(df, "neck_movement_deg")

    L = (buck >= 1).astype("float64")                          # Look: prominent incisors
    E_incisor = (incisor < INCISOR_GAP_MIN_MM).astype("float64")
    E_thyromental = (thyromental < THYROMENTAL_MIN_MM).astype("float64")
    M = (mc >= MALLAMPATI_DIFFICULT_CLASS).astype("float64")
    Ob = (obstruction >= 1).astype("float64")                  # Obstruction / obesity
    N = (neck < NECK_MOVEMENT_MIN_DEG).astype("float64")       # Neck mobility limited

    # propagate NaN: an unknown input must not be scored as "normal"
    for comp, src in [(L, buck), (E_incisor, incisor), (E_thyromental, thyromental),
                      (M, mc), (Ob, obstruction), (N, neck)]:
        comp[src.isna()] = np.nan

    components = pd.DataFrame({
        "lemon_L": L, "lemon_E_incisor": E_incisor,
        "lemon_E_thyromental": E_thyromental, "lemon_M": M,
        "lemon_O": Ob, "lemon_N": N,
    })
    total = components.sum(axis=1, skipna=False)               # NaN if any component NaN
    difficult = (total >= LEMON_DIFFICULT_THRESHOLD).astype("Int64")
    difficult[total.isna()] = pd.NA

    components["lemon_score"] = total
    components["lemon_difficult"] = difficult
    return components


# ===========================================================================
# WILSON
# ===========================================================================
def _wilson_jaw(incisor: pd.Series, subluxation: pd.Series) -> pd.Series:
    """
    Wilson jaw component (0/1/2):
      0 = good opening OR lower teeth protrude past upper
      1 = small opening, teeth meet edge-to-edge (subluxation == 0 here means
          'cannot protrude'); 2 = small opening AND cannot protrude.
    `subluxation` convention (see config): 1 = cannot protrude, 0 = can.
    """
    can_protrude = subluxation == 0
    small_opening = incisor < WILSON_JAW_SMALL_OPENING_MM
    jaw = pd.Series(np.nan, index=incisor.index, dtype="float64")
    jaw[can_protrude] = 0.0
    jaw[(~can_protrude) & (~small_opening)] = 1.0
    jaw[(~can_protrude) & (small_opening)] = 2.0
    # unknown inputs -> NaN
    jaw[incisor.isna() | subluxation.isna()] = np.nan
    return jaw


def wilson_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wilson risk sum (0-10): weight + head/neck + jaw + mandible + buck teeth.
    weight_class / head_neck_class / receding_mandible are expected pre-scored
    0/1/2. Jaw is derived from opening + subluxation. Buck teeth (a 0/1 flag)
    maps to 0 or 2 points (prominent incisors -> severe).
    """
    weight = _col(df, "weight_class")
    headneck = _col(df, "head_neck_class")
    mandible = _col(df, "receding_mandible")
    jaw = _wilson_jaw(_col(df, "mouth_opening_mm"), _col(df, "jaw_subluxation"))
    buck = _col(df, "buck_teeth")
    buck_pts = (buck >= 1).astype("float64") * 2.0
    buck_pts[buck.isna()] = np.nan

    components = pd.DataFrame({
        "wilson_weight": weight, "wilson_headneck": headneck,
        "wilson_jaw": jaw, "wilson_mandible": mandible,
        "wilson_buckteeth": buck_pts,
    })
    total = components.sum(axis=1, skipna=False)
    difficult = (total >= WILSON_DIFFICULT_THRESHOLD).astype("Int64")
    difficult[total.isna()] = pd.NA

    components["wilson_score"] = total
    components["wilson_difficult"] = difficult
    return components


# ===========================================================================
# DRIVER
# ===========================================================================
def compute_comparator_scores(preop: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all three comparator scores for a pre-op table.

    Parameters
    ----------
    preop : DataFrame
        One row per patient, must contain config.ID_COL plus whatever pre-op
        columns are available (any subset of config.PREOP_COLS).

    Returns
    -------
    DataFrame, one row per patient: study_id + every score / component column.
    """
    if config.ID_COL not in preop.columns:
        raise ValueError(f"compute_comparator_scores: need column '{config.ID_COL}'.")

    out = pd.DataFrame({config.ID_COL: preop[config.ID_COL].to_numpy()},
                       index=preop.index)
    out = pd.concat(
        [out, mallampati_scores(preop), lemon_scores(preop), wilson_scores(preop)],
        axis=1,
    )
    return out.reset_index(drop=True)


def build_and_save_comparator_scores() -> pd.DataFrame:
    """
    Load pre-op data, compute the comparator scores, and save to
    reports/computed_baselines.csv. Also prints each comparator's AUC against
    the difficult-airway outcome when labels are available — a quick sanity
    check on whether the scores carry signal.
    """
    from airway import loaders

    config.ensure_dirs()
    preop = loaders.preop_loader()
    if preop.empty or preop.columns.tolist() == [config.ID_COL]:
        print("build_and_save_comparator_scores: no pre-op columns available; "
              "writing an id-only baseline file.")
    scores = compute_comparator_scores(preop)

    out = config.REPORTS_DIR / "computed_baselines.csv"
    scores.to_csv(out, index=False)
    print(f"saved comparator scores -> {out}  (shape {scores.shape})")

    _print_comparator_auc(scores)
    return scores


def _print_comparator_auc(scores: pd.DataFrame) -> None:
    """If labels exist, report the AUC of each raw comparator score."""
    from airway import loaders

    try:
        labels = loaders.label_loader()
    except Exception as err:  # pragma: no cover - labels missing is fine
        print(f"  (skipping comparator AUC: {type(err).__name__})")
        return

    from sklearn.metrics import roc_auc_score

    merged = scores.merge(labels[[config.ID_COL, config.LABEL_COL]],
                          on=config.ID_COL, how="inner")
    y = merged[config.LABEL_COL].to_numpy()
    if len(np.unique(y)) < 2:
        print("  (skipping comparator AUC: only one outcome class present)")
        return

    print("  comparator AUC vs difficult airway (CL 3-4):")
    for col in ("mallampati_class", "lemon_score", "wilson_score"):
        s = pd.to_numeric(merged[col], errors="coerce")
        ok = s.notna().to_numpy()
        if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
            print(f"    {col:18s}: n/a")
            continue
        print(f"    {col:18s}: {roc_auc_score(y[ok], s[ok].to_numpy()):.3f} "
              f"(n={int(ok.sum())})")


if __name__ == "__main__":
    build_and_save_comparator_scores()
