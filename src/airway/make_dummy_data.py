"""
Dummy-data generator (Week 2 version).

WHY THIS WAS UPGRADED
---------------------
The Week 1 version wrote tiny TEXT placeholders for the face images. That was
fine when no code actually opened them. Week 2 DOES open them (face alignment,
ResNet-18). So this version writes real, valid JPEG files instead.

The images are simple coloured rectangles with a face-like blob. They are NOT
realistic faces -- they are not meant to be. Their only job is to be valid
images so the whole pipeline runs end-to-end before you touch real data.

When your real data is ready, replace the files in data/raw/ with the real
ones (same column names) and nothing else changes.

RUN IT
------
    python -m airway.make_dummy_data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from airway import config

N_PATIENTS = 30          # small on purpose -- this is plumbing, not science
IMAGES_PER_PATIENT = 6
IMG_SIZE = 256           # pixels, square
SEED = 0
EMIT_US_OBS2 = True      # emit second-observer ultrasound columns (for ICC audit)


def _make_fake_face(rng: np.random.Generator, difficult: bool) -> Image.Image:
    """
    Build a simple synthetic 'face' image.

    To give the model SOMETHING learnable, difficult-airway fakes are drawn a
    little darker. This is purely so the dummy pipeline produces non-random
    metrics; it has no clinical meaning whatsoever.
    """
    base = 90 if difficult else 150
    arr = rng.integers(base - 20, base + 20, size=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    # draw a lighter oval 'face' in the middle
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    cy, cx = IMG_SIZE // 2, IMG_SIZE // 2
    oval = ((xx - cx) / 70) ** 2 + ((yy - cy) / 95) ** 2 < 1
    face_tone = 200 if not difficult else 160
    arr[oval] = np.clip(arr[oval].astype(int) + (face_tone - base), 0, 255).astype(np.uint8)

    # two dark 'eyes'
    for ex in (cx - 30, cx + 30):
        eye = ((xx - ex) / 12) ** 2 + ((yy - (cy - 25)) / 8) ** 2 < 1
        arr[eye] = 40

    return Image.fromarray(arr, mode="RGB")


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.FACE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    ids = [f"P{i:03d}" for i in range(1, N_PATIENTS + 1)]

    # --- labels.csv ---------------------------------------------------------
    # difficult airway rare (~20%), like real life
    cl_grades = rng.choice([1, 2, 3, 4], size=N_PATIENTS, p=[0.45, 0.35, 0.12, 0.08])
    is_difficult = np.isin(cl_grades, [3, 4])

    # A second observer mostly agrees but disagrees by one grade ~25% of the
    # time, so the audit's inter-observer kappa is high but not perfect.
    noise = rng.choice([-1, 0, 1], size=N_PATIENTS, p=[0.12, 0.75, 0.13])
    cl_obs2 = np.clip(cl_grades + noise, 1, 4)

    labels = pd.DataFrame({
        config.ID_COL: ids,
        config.CL_GRADE_COL: cl_grades,
        config.CL_GRADE_OBS2_COL: cl_obs2,
        "ids_score": rng.integers(0, 12, size=N_PATIENTS),
    })
    labels.to_csv(config.LABELS_CSV, index=False)
    print(f"wrote {config.LABELS_CSV}  ({len(labels)} rows)")

    # --- preop.csv : demographics + pre-op airway exam ---------------------
    # Difficult airways are nudged toward worse exam findings so the computed
    # Mallampati / LEMON / Wilson comparators carry real (if synthetic) signal.
    d = is_difficult.astype(float)
    sex = rng.choice(["M", "F"], size=N_PATIENTS, p=[0.55, 0.45])
    bmi = (rng.normal(27, 4, N_PATIENTS) + 4.0 * d).round(1)
    # surgery_type uses a SEPARATE rng so adding it does not perturb the main
    # rng stream (labels, ultrasound, and face images stay byte-identical).
    sep_rng = np.random.default_rng(12345)
    surgery_type = sep_rng.choice(
        ["general", "ent", "cardiac", "obstetric"],
        size=N_PATIENTS, p=[0.5, 0.2, 0.15, 0.15])
    preop = pd.DataFrame({
        config.ID_COL: ids,
        "age_years": rng.integers(20, 80, size=N_PATIENTS),
        "sex": sex,
        "bmi": bmi,
        config.SURGERY_TYPE_COL: surgery_type,
        # clinician Mallampati class: difficult patients skew toward 3-4
        "mallampati_class": np.clip(
            rng.choice([1, 2, 3, 4], size=N_PATIENTS, p=[0.35, 0.4, 0.18, 0.07])
            + rng.integers(0, 2, size=N_PATIENTS) * is_difficult, 1, 4),
        "mouth_opening_mm": (rng.normal(45, 6, N_PATIENTS) - 8.0 * d).round(0),
        "thyromental_mm": (rng.normal(70, 8, N_PATIENTS) - 12.0 * d).round(0),
        "neck_movement_deg": (rng.normal(90, 12, N_PATIENTS) - 20.0 * d).round(0),
        "jaw_subluxation": (rng.random(N_PATIENTS) < (0.1 + 0.4 * d)).astype(int),
        "buck_teeth": (rng.random(N_PATIENTS) < (0.1 + 0.25 * d)).astype(int),
        "neck_circumference_cm": (rng.normal(38, 4, N_PATIENTS) + 5.0 * d).round(1),
        "obstructed_airway": (rng.random(N_PATIENTS) < (0.05 + 0.2 * d)).astype(int),
        # Wilson sub-scores (0/1/2 each), correlated with difficulty
        "weight_class": np.clip(rng.integers(0, 2, N_PATIENTS) + (d).astype(int), 0, 2),
        "head_neck_class": np.clip(rng.integers(0, 2, N_PATIENTS) + (d).astype(int), 0, 2),
        "receding_mandible": np.clip(rng.integers(0, 2, N_PATIENTS) + (d * (rng.random(N_PATIENTS) < 0.5)).astype(int), 0, 2),
    })
    # leave a few realistic gaps so the missingness table is non-trivial
    gap_idx = rng.choice(N_PATIENTS, size=max(1, N_PATIENTS // 12), replace=False)
    preop.loc[gap_idx, "thyromental_mm"] = np.nan
    preop.to_csv(config.PREOP_CSV, index=False)
    print(f"wrote {config.PREOP_CSV}  ({len(preop)} rows)")

    # --- ultrasound.csv -----------------------------------------------------
    # shift difficult patients a little so the model has real signal
    shift = np.where(is_difficult, 1.0, 0.0)
    ultrasound = pd.DataFrame({
        config.ID_COL: ids,
        "dstvc_mm": (rng.normal(18, 4, N_PATIENTS) + 3.0 * shift).round(1),
        "hmd_neutral_mm": (rng.normal(45, 6, N_PATIENTS) - 4.0 * shift).round(1),
        "hmd_extended_mm": (rng.normal(55, 7, N_PATIENTS) - 3.0 * shift).round(1),
        "dse_mm": (rng.normal(27, 5, N_PATIENTS) + 2.0 * shift).round(1),
    })
    # a couple of missing ultrasound readings so imputation rules have something
    # to act on (quarantine.py imputes these inside cross-validation downstream)
    us_gap = rng.choice(N_PATIENTS, size=max(1, N_PATIENTS // 15), replace=False)
    ultrasound.loc[us_gap, "dse_mm"] = np.nan

    # Optional second-observer ('_obs2') ultrasound readings for ~15% of patients,
    # so the data audit can exercise the inter-rater ICC path. A SEPARATE rng is
    # used so the primary measurements (and everything else) stay byte-identical.
    if EMIT_US_OBS2:
        obs2_rng = np.random.default_rng(20240607)
        n_repeat = max(5, int(np.ceil(0.15 * N_PATIENTS)))   # ~15%, >= ICC minimum
        repeat_idx = obs2_rng.choice(N_PATIENTS, size=n_repeat, replace=False)
        for col in ["dstvc_mm", "hmd_neutral_mm", "hmd_extended_mm", "dse_mm"]:
            vals = np.full(N_PATIENTS, np.nan)
            primary = ultrasound[col].to_numpy()
            # second reading = first + small measurement noise (high agreement)
            vals[repeat_idx] = primary[repeat_idx] + obs2_rng.normal(0, 0.5, n_repeat)
            ultrasound[f"{col}_obs2"] = np.round(vals, 1)

    ultrasound.to_csv(config.ULTRASOUND_CSV, index=False)
    print(f"wrote {config.ULTRASOUND_CSV}  ({len(ultrasound)} rows)")

    # --- face_index.csv + real JPEG image files ----------------------------
    views = [
        "frontal_rest", "frontal_open", "left_profile",
        "right_profile", "oblique_left", "oblique_right",
    ]
    rows = []
    for pid, diff in zip(ids, is_difficult):
        for v in views[:IMAGES_PER_PATIENT]:
            rel = f"{pid}_{v}.jpg"
            img = _make_fake_face(rng, bool(diff))
            img.save(config.FACE_IMAGE_DIR / rel, format="JPEG", quality=90)
            rows.append({config.ID_COL: pid, "view_code": v, "file_path": rel})
    face_index = pd.DataFrame(rows)
    face_index.to_csv(config.FACE_INDEX_CSV, index=False)
    print(f"wrote {config.FACE_INDEX_CSV}  ({len(face_index)} rows)")
    print(f"wrote {len(face_index)} real JPEG images to {config.FACE_IMAGE_DIR}")

    print("\nDummy data ready (with real images). Loaders and tests can run.")


if __name__ == "__main__":
    main()
