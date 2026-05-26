"""
Central configuration: every path and constant the project uses lives here.

WHY THIS FILE EXISTS
--------------------
Never hard-code a file path inside a script. If your data moves, you change
it in ONE place (here) instead of hunting through ten files.

HOW TO USE
----------
    from airway import config
    df = pd.read_csv(config.LABELS_CSV)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root: this file is at src/airway/config.py, so the root is 3 levels up.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Data directories. These are tracked by DVC, not Git.
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ---------------------------------------------------------------------------
# Expected raw input files.
# >>> EDIT THESE NAMES to match what your collected data is actually called. <<<
# The loaders (loaders.py) read from exactly these paths.
# ---------------------------------------------------------------------------
LABELS_CSV = RAW_DIR / "labels.csv"            # study_id, cl_grade, ids_score, ...
ULTRASOUND_CSV = RAW_DIR / "ultrasound.csv"    # study_id, dstvc_mm, hmd_neutral_mm, ...
FACE_INDEX_CSV = RAW_DIR / "face_index.csv"    # study_id, view_code, file_path
FACE_IMAGE_DIR = RAW_DIR / "face_images"       # folder of the actual image files

# ---------------------------------------------------------------------------
# Output directory for metrics, figures, model files.
# ---------------------------------------------------------------------------
REPORTS_DIR = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------------
# Cross-validation settings. Fixed seed = reproducible results.
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_SPLITS = 5          # number of CV folds
N_REPEATS = 2         # repeated 5x2 CV as described in the plan

# ---------------------------------------------------------------------------
# The column names every part of the code agrees on.
# If your raw CSV uses different names, rename them in the loaders, NOT here.
# ---------------------------------------------------------------------------
ID_COL = "study_id"          # patient identifier — the key everything joins on
CL_GRADE_COL = "cl_grade"    # Cormack-Lehane grade, integer 1-4
LABEL_COL = "label"          # derived binary: 1 = difficult (CL 3-4), 0 = not


def ensure_dirs() -> None:
    """Create the output directories if they don't exist yet."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
