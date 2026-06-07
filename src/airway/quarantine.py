"""
Data quarantine rules.

WHY THIS FILE EXISTS
--------------------
"Quarantine" = the written, frozen decisions about what data we will and will
NOT use, and why. In a pilot it is fatally easy for these decisions to drift:
one script drops a patient, another keeps them, and the cohort silently differs
between the audit and the model. This module makes the rules:

  1. EXPLICIT  — every rule is a named constant at the top of this file.
  2. DETERMINISTIC — the same raw data always yields the same decisions.
  3. PERSISTED — decisions are written to disk (a JSON the pipeline reads, plus
     a human-readable Markdown record), so downstream code consumes the SAME
     decisions instead of recomputing its own.

THE THREE DECISIONS (exactly what the Week-3 task asks for)
-----------------------------------------------------------
  - excluded_patients : patients removed from the whole pilot cohort, + reason.
  - excluded_images   : individual face images removed, + reason.
  - imputed_us_cells  : ultrasound (patient, column) cells that are missing and
                        will be imputed. IMPORTANT: this file only RECORDS which
                        cells get imputed. The actual imputation is median-fill
                        fitted INSIDE each cross-validation fold (baseline /
                        face model Pipelines) so it never leaks across folds.

DO NOT change these decisions silently downstream. Downstream code calls
load_quarantine() and uses what is written here.
"""

from __future__ import annotations

import json

import pandas as pd

from airway import config, ultrasound_features

# ---------------------------------------------------------------------------
# THE RULES. Edit here, in ONE place. Changing a rule changes the cohort, so
# re-run the audit and the models afterwards.
# ---------------------------------------------------------------------------
EXCLUDE_PATIENT_IF_NO_CL_GRADE = True   # no outcome -> cannot train or evaluate
MIN_USABLE_FACE_IMAGES = 1              # below this, patient is face-ineligible
EXCLUDE_IMAGE_IF_UNREADABLE = True      # file missing or not a valid image
EXCLUDE_IMAGE_IF_NO_FACE = True         # no detectable face in the image
US_IMPUTE_STRATEGY = "median (fitted inside each CV fold)"   # recorded, not applied here

DECISIONS_JSON = config.PROCESSED_DIR / "quarantine_decisions.json"
RULES_MD = config.REPORTS_DIR / "quarantine_rules.md"


def compute_quarantine(check_faces: bool = True) -> dict:
    """
    Apply the rules above to the raw data and return the decision record.

    Parameters
    ----------
    check_faces : bool
        If True, run face detection on every catalogued image to apply the
        "no detectable face" rule. Set False to skip that (faster; only the
        unreadable-file rule then applies to images).

    Returns
    -------
    dict with keys: excluded_patients, excluded_images, face_ineligible_patients,
    imputed_us_cells, plus a `rules` echo and simple counts. All lists are
    sorted for determinism.
    """
    from pathlib import Path

    from airway import loaders

    # ---- patients without an outcome ------------------------------------
    raw_labels = pd.read_csv(config.LABELS_CSV)
    excluded_patients = []
    if EXCLUDE_PATIENT_IF_NO_CL_GRADE:
        no_grade = raw_labels[raw_labels[config.CL_GRADE_COL].isna()]
        for pid in no_grade[config.ID_COL].tolist():
            excluded_patients.append({"study_id": pid, "reason": "missing CL grade"})

    excluded_ids = {e["study_id"] for e in excluded_patients}

    # ---- images: unreadable / no face -----------------------------------
    face_index = loaders.face_loader()
    excluded_images = []
    usable_counts: dict[str, int] = {}
    for _, row in face_index.iterrows():
        pid = row[config.ID_COL]
        path = row["abs_path"]
        usable = True
        reason = None

        exists = Path(path).exists()
        if EXCLUDE_IMAGE_IF_UNREADABLE and not exists:
            usable, reason = False, "file missing/unreadable"
        elif check_faces and EXCLUDE_IMAGE_IF_NO_FACE:
            from airway import face_align
            if not face_align.face_was_detected(path):
                usable, reason = False, "no detectable face"

        if not usable:
            excluded_images.append({
                "study_id": pid, "view_code": row["view_code"],
                "file_path": row["file_path"], "reason": reason,
            })
        else:
            usable_counts[pid] = usable_counts.get(pid, 0) + 1

    # ---- patients with too few usable images (face-ineligible) ----------
    all_face_patients = set(face_index[config.ID_COL].unique())
    face_ineligible = sorted(
        pid for pid in all_face_patients
        if usable_counts.get(pid, 0) < MIN_USABLE_FACE_IMAGES and pid not in excluded_ids
    )

    # ---- ultrasound cells to impute -------------------------------------
    us = loaders.us_loader()
    imputed_us_cells = []
    for col in ultrasound_features.US_FEATURE_COLS:
        if col not in us.columns:
            continue
        miss = us[us[col].isna()]
        for pid in miss[config.ID_COL].tolist():
            if pid in excluded_ids:
                continue
            imputed_us_cells.append({"study_id": pid, "column": col})

    decisions = {
        "rules": {
            "EXCLUDE_PATIENT_IF_NO_CL_GRADE": EXCLUDE_PATIENT_IF_NO_CL_GRADE,
            "MIN_USABLE_FACE_IMAGES": MIN_USABLE_FACE_IMAGES,
            "EXCLUDE_IMAGE_IF_UNREADABLE": EXCLUDE_IMAGE_IF_UNREADABLE,
            "EXCLUDE_IMAGE_IF_NO_FACE": EXCLUDE_IMAGE_IF_NO_FACE and check_faces,
            "US_IMPUTE_STRATEGY": US_IMPUTE_STRATEGY,
        },
        "excluded_patients": sorted(excluded_patients, key=lambda d: d["study_id"]),
        "face_ineligible_patients": face_ineligible,
        "excluded_images": sorted(
            excluded_images, key=lambda d: (d["study_id"], d["file_path"])),
        "imputed_us_cells": sorted(
            imputed_us_cells, key=lambda d: (d["study_id"], d["column"])),
        "counts": {
            "n_excluded_patients": len(excluded_patients),
            "n_face_ineligible_patients": len(face_ineligible),
            "n_excluded_images": len(excluded_images),
            "n_imputed_us_cells": len(imputed_us_cells),
        },
    }
    return decisions


def save_quarantine(decisions: dict) -> None:
    """Persist decisions to JSON (machine) and Markdown (human)."""
    config.ensure_dirs()
    with open(DECISIONS_JSON, "w") as fh:
        json.dump(decisions, fh, indent=2, sort_keys=True)
    _write_rules_md(decisions)
    print(f"saved quarantine decisions -> {DECISIONS_JSON}")
    print(f"saved quarantine rules     -> {RULES_MD}")


def load_quarantine() -> dict:
    """
    Read the frozen decisions. Downstream code MUST use this rather than
    recomputing, so the cohort is identical everywhere. Raises if the file does
    not exist yet (run `python -m airway.quarantine` first).
    """
    if not DECISIONS_JSON.exists():
        raise FileNotFoundError(
            f"quarantine decisions not found at {DECISIONS_JSON}. "
            f"Run: python -m airway.quarantine"
        )
    with open(DECISIONS_JSON) as fh:
        return json.load(fh)


def excluded_image_paths(decisions: dict) -> set[str]:
    """Return the set of file_path values that are quarantined out."""
    return {img["file_path"] for img in decisions["excluded_images"]}


def excluded_patient_ids(decisions: dict) -> set[str]:
    """Return the set of study_ids excluded from the cohort entirely."""
    return {p["study_id"] for p in decisions["excluded_patients"]}


def _write_rules_md(decisions: dict) -> None:
    r = decisions["rules"]
    c = decisions["counts"]
    lines = [
        "# Data Quarantine Rules",
        "",
        "These rules are **frozen and deterministic**. The same raw data always",
        "produces the same decisions. Downstream code reads",
        f"`{DECISIONS_JSON.name}` (written alongside this file) and must not",
        "re-derive its own cohort. To change a rule, edit `src/airway/quarantine.py`",
        "and re-run `python -m airway.quarantine`, then re-run the models.",
        "",
        "## Rules in force",
        "",
        f"- Exclude a patient with no Cormack-Lehane grade: **{r['EXCLUDE_PATIENT_IF_NO_CL_GRADE']}**",
        f"- Minimum usable face images to be face-eligible: **{r['MIN_USABLE_FACE_IMAGES']}**",
        f"- Exclude an image if the file is missing/unreadable: **{r['EXCLUDE_IMAGE_IF_UNREADABLE']}**",
        f"- Exclude an image if no face is detected: **{r['EXCLUDE_IMAGE_IF_NO_FACE']}**",
        f"- Ultrasound missing-value handling: **{r['US_IMPUTE_STRATEGY']}**",
        "  (cells are listed below; imputation is fitted inside each CV fold, never globally)",
        "",
        "## Resulting decisions (counts)",
        "",
        f"- Excluded patients: **{c['n_excluded_patients']}**",
        f"- Face-ineligible patients (kept, but no face features): **{c['n_face_ineligible_patients']}**",
        f"- Excluded images: **{c['n_excluded_images']}**",
        f"- Ultrasound cells flagged for imputation: **{c['n_imputed_us_cells']}**",
        "",
    ]

    def _section(title: str, rows: list[str]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if rows:
            lines.extend(rows)
        else:
            lines.append("_none_")
        lines.append("")

    _section("Excluded patients", [
        f"- `{p['study_id']}` — {p['reason']}" for p in decisions["excluded_patients"]])
    _section("Face-ineligible patients", [
        f"- `{pid}`" for pid in decisions["face_ineligible_patients"]])
    _section("Excluded images", [
        f"- `{img['study_id']}` / `{img['view_code']}` (`{img['file_path']}`) — {img['reason']}"
        for img in decisions["excluded_images"]])
    _section("Ultrasound cells imputed", [
        f"- `{cell['study_id']}` — `{cell['column']}`"
        for cell in decisions["imputed_us_cells"]])

    with open(RULES_MD, "w") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    decisions = compute_quarantine(check_faces=True)
    save_quarantine(decisions)
    c = decisions["counts"]
    print(f"\nquarantine summary: {c['n_excluded_patients']} patients excluded, "
          f"{c['n_excluded_images']} images excluded, "
          f"{c['n_imputed_us_cells']} ultrasound cells imputed.")


if __name__ == "__main__":
    main()
