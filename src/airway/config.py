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

# Persisted face crops (Week 4) and per-image embeddings (Week 5) live here.
FACE_CROPS_DIR = PROCESSED_DIR / "face_crops"
FACE_EMBEDDINGS_PARQUET = PROCESSED_DIR / "face_embeddings.parquet"   # per-image 512-d
FACE_FEATURES_PARQUET = PROCESSED_DIR / "face_features.parquet"       # per-patient 1024-d
CLEANED_US_CSV = PROCESSED_DIR / "cleaned_ultrasound_features.csv"    # cleaned ultrasound table

# Column-name prefixes (defined here so torch-free modules can reference them
# without importing the embedding code, which pulls in torch).
FACE_IMG_EMBED_PREFIX = "emb_"     # per-image embedding cols:  emb_000 .. emb_511
FACE_FEATURE_PREFIX = "face_"      # per-patient feature cols:  face_000 .. face_1023

# Optional dlib 68-point landmark model. If present, face_crops uses dlib for
# eye-centred alignment; if absent, it falls back to OpenCV (see face_crops.py).
DLIB_LANDMARK_MODEL = PROJECT_ROOT / "models" / "shape_predictor_68_face_landmarks.dat"

# ---------------------------------------------------------------------------
# Expected raw input files.
# >>> EDIT THESE NAMES to match what your collected data is actually called. <<<
# The loaders (loaders.py) read from exactly these paths.
# ---------------------------------------------------------------------------
LABELS_CSV = RAW_DIR / "labels.csv"            # study_id, cl_grade, cl_grade_obs2, ...
ULTRASOUND_CSV = RAW_DIR / "ultrasound.csv"    # study_id, dstvc_mm, hmd_neutral_mm, ...
FACE_INDEX_CSV = RAW_DIR / "face_index.csv"    # study_id, view_code, file_path
FACE_IMAGE_DIR = RAW_DIR / "face_images"       # folder of the actual image files
PREOP_CSV = RAW_DIR / "preop.csv"              # study_id, demographics + airway exam fields

# ---------------------------------------------------------------------------
# Output directory for metrics, figures, model files.
# ---------------------------------------------------------------------------
REPORTS_DIR = PROJECT_ROOT / "reports"

# Block D explainability artefacts (force plots, notes) live under outputs/.
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPLAIN_DIR = OUTPUTS_DIR / "explainability"
FORCE_PLOTS_DIR = EXPLAIN_DIR / "force_plots"

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

# Second-observer CL grade (optional). If this column is present in the labels
# file, the data audit computes inter-observer Cohen's kappa against CL_GRADE_COL.
CL_GRADE_OBS2_COL = "cl_grade_obs2"

# ---------------------------------------------------------------------------
# Demographic columns (optional). The audit summarises these if present.
# Rename your real columns to these names inside the loaders, NOT here.
# ---------------------------------------------------------------------------
DEMOGRAPHIC_COLS = ["age_years", "sex", "bmi"]

# Surgery type (categorical). Used as a descriptive subgroup in Block D.
SURGERY_TYPE_COL = "surgery_type"

# Continuous variables that Block D splits into tertiles for subgroup analysis.
TERTILE_SUBGROUP_COLS = ["bmi", "age_years"]
# Categorical subgroup variables (reported as-is).
CATEGORICAL_SUBGROUP_COLS = [SURGERY_TYPE_COL]

# ---------------------------------------------------------------------------
# Pre-operative airway-exam columns used to compute Mallampati / LEMON / Wilson.
# These are the names the scoring code (scores.py) reads. If a column is
# missing, the corresponding score component degrades gracefully (see scores.py).
# ---------------------------------------------------------------------------
PREOP_COLS = [
    "mallampati_class",        # clinician Mallampati class, 1-4 (if already scored)
    "mouth_opening_mm",        # inter-incisor distance, mm
    "thyromental_mm",          # thyromental distance, mm
    "neck_movement_deg",       # head/neck range of motion, degrees
    "jaw_subluxation",         # 0 = can protrude lower teeth past upper, else 1
    "buck_teeth",              # 1 = prominent upper incisors ("buck teeth")
    "neck_circumference_cm",   # neck circumference, cm
    "obstructed_airway",       # 1 = OSA / stridor / other obstruction
    "weight_class",            # Wilson weight points: 0 (<90kg), 1 (90-110kg), 2 (>110kg)
    "head_neck_class",         # Wilson head/neck movement points: 0, 1, 2
    "receding_mandible",       # Wilson: 0 normal, 1 moderate, 2 severe
]


def ensure_dirs() -> None:
    """Create the output directories if they don't exist yet."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FACE_CROPS_DIR.mkdir(parents=True, exist_ok=True)
