"""
Week-3 data audit -> reports/data_audit_report.md.

WHAT THIS DOES
--------------
Produces a single, one-page Markdown audit of the dataset before any modelling:

  1. Per-modality usability rates (labels / ultrasound / face).
  2. Missingness table (per column, count + %).
  3. Cormack-Lehane grade distribution + difficult-airway prevalence.
  4. Demographics summary (age / sex / BMI, if those columns exist).
  5. Inter-observer Cohen's kappa, computed ONLY if a second-observer CL grade
     column (config.CL_GRADE_OBS2_COL) is present.

Everything degrades gracefully: a modality or column that is absent is reported
as such rather than crashing the audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airway import config, ultrasound_features


def _md_table(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-flavoured Markdown table."""
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *body])


def _usability_rates() -> pd.DataFrame:
    """One row per modality: n patients/images and the usable fraction."""
    from airway import loaders

    rows = []

    # labels: usable = CL grade present
    raw_labels = pd.read_csv(config.LABELS_CSV)
    n = len(raw_labels)
    usable = int(raw_labels[config.CL_GRADE_COL].notna().sum())
    rows.append({"modality": "labels (CL grade)", "unit": "patients",
                 "n": n, "usable": usable, "usable_pct": round(100 * usable / n, 1)})

    # ultrasound: usable = all measured columns present for that patient
    us = loaders.us_loader()
    us_cols = [c for c in ultrasound_features.US_FEATURE_COLS if c in us.columns]
    complete = int(us[us_cols].notna().all(axis=1).sum())
    rows.append({"modality": "ultrasound (complete rows)", "unit": "patients",
                 "n": len(us), "usable": complete,
                 "usable_pct": round(100 * complete / len(us), 1) if len(us) else 0.0})

    # face: prefer quarantine decisions (already computed); else detect faces
    face_index = loaders.face_loader()
    n_img = len(face_index)
    excluded = _face_excluded_count(face_index)
    usable_img = n_img - excluded
    rows.append({"modality": "face images (face detected)", "unit": "images",
                 "n": n_img, "usable": usable_img,
                 "usable_pct": round(100 * usable_img / n_img, 1) if n_img else 0.0})

    return pd.DataFrame(rows)


def _face_excluded_count(face_index: pd.DataFrame) -> int:
    """Use frozen quarantine decisions if present, else run face detection."""
    from airway import quarantine
    try:
        decisions = quarantine.load_quarantine()
        return decisions["counts"]["n_excluded_images"]
    except FileNotFoundError:
        from airway import face_align
        return int(sum(not face_align.face_was_detected(p)
                       for p in face_index["abs_path"]))


def _missingness() -> pd.DataFrame:
    """Per-column missing count + % across ultrasound and pre-op tables."""
    from airway import loaders

    frames = []
    us = loaders.us_loader()
    frames.append(("ultrasound", us))
    preop = loaders.preop_loader()
    if not preop.empty and preop.columns.tolist() != [config.ID_COL]:
        frames.append(("preop", preop))

    rows = []
    for source, df in frames:
        n = len(df)
        for col in df.columns:
            if col == config.ID_COL:
                continue
            miss = int(df[col].isna().sum())
            rows.append({"source": source, "column": col, "n": n,
                         "missing": miss,
                         "missing_pct": round(100 * miss / n, 1) if n else 0.0})
    return pd.DataFrame(rows)


def _cl_distribution() -> tuple[pd.DataFrame, float]:
    from airway import loaders
    labels = loaders.label_loader()
    counts = labels[config.CL_GRADE_COL].value_counts().sort_index()
    total = int(counts.sum())
    df = pd.DataFrame({
        "cl_grade": counts.index.astype(int),
        "n": counts.values,
        "pct": (100 * counts.values / total).round(1),
    })
    prevalence = float(labels[config.LABEL_COL].mean()) if total else float("nan")
    return df, prevalence


def _demographics() -> pd.DataFrame | None:
    from airway import loaders
    preop = loaders.preop_loader()
    present = [c for c in config.DEMOGRAPHIC_COLS if c in preop.columns]
    if not present:
        return None
    rows = []
    for col in present:
        s = preop[col]
        if pd.api.types.is_numeric_dtype(s):
            rows.append({"variable": col, "summary":
                         f"mean {s.mean():.1f} ± {s.std():.1f} "
                         f"(min {s.min():.0f}, max {s.max():.0f}), "
                         f"n={int(s.notna().sum())}"})
        else:
            vc = s.value_counts()
            rows.append({"variable": col, "summary":
                         ", ".join(f"{k}: {v}" for k, v in vc.items())})
    return pd.DataFrame(rows)


def _interobserver_kappa() -> dict | None:
    """Cohen's kappa between two observers' CL grades, if a 2nd observer exists."""
    from airway import loaders
    labels = loaders.label_loader()
    if config.CL_GRADE_OBS2_COL not in labels.columns:
        return None
    from sklearn.metrics import cohen_kappa_score

    pair = labels.dropna(subset=[config.CL_GRADE_COL, config.CL_GRADE_OBS2_COL])
    o1 = pair[config.CL_GRADE_COL].astype(int).to_numpy()
    o2 = pair[config.CL_GRADE_OBS2_COL].astype(int).to_numpy()
    # binary difficult flag for each observer
    d1 = np.isin(o1, [3, 4]).astype(int)
    d2 = np.isin(o2, [3, 4]).astype(int)
    return {
        "n": len(pair),
        "kappa_grade_quadratic": round(cohen_kappa_score(o1, o2, weights="quadratic"), 3),
        "kappa_grade_unweighted": round(cohen_kappa_score(o1, o2), 3),
        "kappa_difficult_binary": round(cohen_kappa_score(d1, d2), 3),
        "pct_exact_agreement": round(100 * float(np.mean(o1 == o2)), 1),
    }


def build_report() -> str:
    """Assemble the full Markdown audit report as a string."""
    usability = _usability_rates()
    missing = _missingness()
    cl_df, prevalence = _cl_distribution()
    demo = _demographics()
    kappa = _interobserver_kappa()

    parts = [
        "# Data Audit Report",
        "",
        "_One-page pre-modelling audit. Generated deterministically by "
        "`python -m airway.data_audit`._",
        "",
        "## 1. Per-modality usability",
        "",
        _md_table(usability),
        "",
        "## 2. Missingness",
        "",
        _md_table(missing) if not missing.empty else "_no tabular columns found_",
        "",
        "## 3. Cormack-Lehane grade distribution",
        "",
        _md_table(cl_df),
        "",
        f"**Difficult-airway prevalence (CL 3-4): {100 * prevalence:.1f}%**",
        "",
        "## 4. Demographics",
        "",
        _md_table(demo) if demo is not None
        else "_no demographic columns present_",
        "",
        "## 5. Inter-observer agreement (Cormack-Lehane)",
        "",
    ]
    if kappa is None:
        parts.append("_no second-observer column "
                     f"(`{config.CL_GRADE_OBS2_COL}`) present — kappa not computed._")
    else:
        parts.append(_md_table(pd.DataFrame([kappa])))
        parts.append("")
        parts.append("Quadratic-weighted kappa treats a 1-grade disagreement as "
                     "milder than a 3-grade one (appropriate for ordinal CL grades).")
    parts.append("")
    return "\n".join(parts)


def main() -> None:
    config.ensure_dirs()
    report = build_report()
    out = config.REPORTS_DIR / "data_audit_report.md"
    out.write_text(report)
    print(f"saved data audit -> {out}")
    print("\n" + report)


if __name__ == "__main__":
    main()
