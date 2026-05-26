"""
Dummy-data generator.

WHY THIS EXISTS
---------------
Week 1's goal is "the pipeline runs end-to-end" — not "the model is good".
You should NOT touch the real patient data while building plumbing. Instead,
this script writes fake-but-correctly-shaped files into data/raw/ so every
loader, the splitter, and the test suite have something to chew on.

When your real data is ready, you simply replace these files with the real
ones (same column names) and nothing else changes.

RUN IT
------
    python -m airway.make_dummy_data
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airway import config

N_PATIENTS = 30          # small on purpose — this is plumbing, not science
IMAGES_PER_PATIENT = 6
SEED = 0


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    (config.FACE_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    ids = [f"P{ i:03d}".replace(" ", "") for i in range(1, N_PATIENTS + 1)]

    # --- labels.csv ---------------------------------------------------------
    # Make difficult airway rare (~20%), like real life.
    cl_grades = rng.choice([1, 2, 3, 4], size=N_PATIENTS, p=[0.45, 0.35, 0.12, 0.08])
    labels = pd.DataFrame({
        config.ID_COL: ids,
        config.CL_GRADE_COL: cl_grades,
        "ids_score": rng.integers(0, 12, size=N_PATIENTS),
    })
    labels.to_csv(config.LABELS_CSV, index=False)
    print(f"wrote {config.LABELS_CSV}  ({len(labels)} rows)")

    # --- ultrasound.csv -----------------------------------------------------
    ultrasound = pd.DataFrame({
        config.ID_COL: ids,
        "dstvc_mm": rng.normal(18, 4, N_PATIENTS).round(1),
        "hmd_neutral_mm": rng.normal(45, 6, N_PATIENTS).round(1),
        "hmd_extended_mm": rng.normal(55, 7, N_PATIENTS).round(1),
        "dse_mm": rng.normal(27, 5, N_PATIENTS).round(1),
    })
    ultrasound.to_csv(config.ULTRASOUND_CSV, index=False)
    print(f"wrote {config.ULTRASOUND_CSV}  ({len(ultrasound)} rows)")

    # --- face_index.csv + placeholder image files --------------------------
    views = [
        "frontal_rest", "frontal_open", "left_profile",
        "right_profile", "oblique_left", "oblique_right",
    ]
    rows = []
    for pid in ids:
        for v in views[:IMAGES_PER_PATIENT]:
            rel = f"{pid}_{v}.jpg"
            # Write a tiny placeholder file so abs_path actually exists.
            (config.FACE_IMAGE_DIR / rel).write_bytes(b"DUMMY_IMAGE")
            rows.append({config.ID_COL: pid, "view_code": v, "file_path": rel})
    face_index = pd.DataFrame(rows)
    face_index.to_csv(config.FACE_INDEX_CSV, index=False)
    print(f"wrote {config.FACE_INDEX_CSV}  ({len(face_index)} rows)")
    print(f"wrote {len(face_index)} placeholder image files to {config.FACE_IMAGE_DIR}")

    print("\nDummy data ready. You can now run the loaders and the tests.")


if __name__ == "__main__":
    main()
