# Airway — Bi-Modal Difficult Airway Prediction

Predicts difficult airway (Cormack–Lehane Grade 3–4) from facial images and
point-of-care ultrasound measurements.

## Status

- **Week 1 (done):** project skeleton, data loaders, patient-level CV splitter, tests.
- **Week 2 (done):** face alignment, ResNet-18 embeddings, ultrasound feature
  table, baseline logistic-regression model with cross-validated metrics.
- **Week 3 (done):** data audit report, frozen quarantine rules, and computed
  Mallampati / LEMON / Wilson comparator baselines.
- **Weeks 4–5 (done):** dlib eye-centred face crops (persisted, idempotent),
  persisted 512-d per-image embeddings → 1024-d per-patient features, and two
  face classifiers (L2 logistic regression + XGBoost) under patient-level 5×2 CV.
- **Weeks 6–7 (done):** ultrasound feature cleaning/standardisation, safe
  hyomental distance ratio, within-fold mean imputation, L2 logistic regression
  + XGBoost under the same 5×2 CV, and permutation + XGBoost-gain feature
  importance. Manuscript Methods + Results stubs in `docs/manuscript.md`.
- **Block C / Weeks 8–11 (done):** within-fold isotonic calibration of each
  modality (Brier + reliability plots), late fusion (logistic meta-learner) with
  an unweighted-average baseline and a sanity check, clinical comparators
  (Mallampati/LEMON/Wilson) on the same folds, and DeLong tests with Bonferroni
  correction. Manuscript Tables 1–4 in `docs/manuscript.md`.
- **Block D / Weeks 12–14 (coding done):** patient-level bootstrap 95% CIs,
  descriptive subgroup analyses (BMI/age tertiles, surgery type) with
  underpowered flags, SHAP explainability + per-case force plots, automatic
  TP/TN/FP/FN case selection, FN/FP export tables for manual review, and a
  data-freeze memo **template**. Clinical interpretation, the data-freeze
  decision, repo tagging, and MLflow freezing are intentionally left to a human.

## Quick start

```bash
# 1. create and activate the environment (see SETUP guide)
# 2. install the project
pip install -e ".[dev]"
pip install torch torchvision        # installed separately

# 3. generate dummy data (real JPEG images; replace with real data later)
make dummy

# 4. run the tests — all 25 should pass
make test

# 5. run the full pipeline end-to-end
make pilot-report
```

`make pilot-report` writes:
- `reports/baseline_metrics.csv` — AUC, sensitivity, specificity per modality
- `reports/baseline_roc.png` — ROC curves
- `data/processed/face_features.parquet`, `ultrasound_features.parquet`

## Week 3 — data audit

```bash
make quarantine   # freeze the cohort rules  -> reports/quarantine_rules.md
                  #                              data/processed/quarantine_decisions.json
make audit        # one-page audit           -> reports/data_audit_report.md
make scores       # comparator baselines     -> reports/computed_baselines.csv
make week3        # all three, in order
```

- **`quarantine`** is the single source of truth for which patients/images are
  excluded and which ultrasound cells get imputed. The decisions are written to
  `quarantine_decisions.json`; the crop and embedding steps read that file so the
  cohort never silently diverges. Edit the rule constants at the top of
  `src/airway/quarantine.py` to change them, then re-run.
- **`scores`** computes Mallampati, LEMON and Wilson deterministically from the
  pre-op fields (`data/raw/preop.csv`). Clinical cut-points are constants at the
  top of `src/airway/scores.py`.
- **`audit`** reports per-modality usability, missingness, CL-grade distribution,
  demographics, and inter-observer Cohen's κ (only if a second-observer column is
  present).

## Weeks 4–5 — face model

```bash
make crops        # eye-centred 224x224 crops -> data/processed/face_crops/   (idempotent)
make embeddings   # 512-d per image + 1024-d per patient -> data/processed/*.parquet
make face-model   # train + 5x2 CV LogReg & XGBoost
make week45       # all three, in order
```

`make face-model` writes `reports/face_model.pkl`, `reports/face_cv_metrics.csv`,
and `reports/face_roc.png`.

- **Crops are idempotent:** existing crops are skipped. Force a rebuild with
  `python -m airway.face_crops --force`.
- **Embeddings are computed once** (frozen ResNet-18, leakage-free) and cached to
  parquet; supervised model fitting happens *inside* each CV fold.
- **`--force`** on `python -m airway.face_embeddings` recomputes the cached parquet.

### dlib alignment (optional)

`face_crops` uses dlib 68-point landmarks for eye-centred alignment when both
dlib **and** the landmark model are available, and **falls back to OpenCV**
otherwise (a message prints which backend is active). To enable dlib:

```bash
pip install ".[face-dlib]"      # needs cmake + a C++ toolchain
mkdir -p models
curl -L http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
  | bunzip2 > models/shape_predictor_68_face_landmarks.dat
```

The model path is `config.DLIB_LANDMARK_MODEL` (`models/…` by default, git-ignored).

### macOS note (XGBoost)

XGBoost needs the OpenMP runtime. If `import xgboost` fails with a `libomp.dylib`
error, run `brew install libomp`.

## Weeks 6–7 — ultrasound model

```bash
make us-clean     # clean + standardize ultrasound features
                  #   -> data/processed/cleaned_ultrasound_features.csv
make us-model     # train + 5x2 CV LogReg & XGBoost + feature importance
make week67       # us-model (cleans features first)
```

`make us-model` writes:
- `reports/us_cv_metrics.csv` — AUC, sensitivity, specificity, … per classifier
- `reports/us_roc.png` — pooled out-of-fold ROC curves
- `reports/us_feature_importance.csv` / `.png` — permutation + XGBoost gain importance
- `reports/us_model.pkl` — both classifiers refit on all data + metadata
- `data/processed/cleaned_ultrasound_features.csv` — the cleaned feature table

- **Cleaning** coerces measurements to numeric (invalid/blank → missing),
  applies the column aliases in `ultrasound_features.US_COLUMN_ALIASES`, adds
  all-missing placeholders for absent schema columns, and computes the derived
  **hyomental distance ratio** (extended ÷ neutral; zero/negative/missing
  neutral → missing).
- **Imputation is within-fold mean only** — fitted on the training fold inside
  the sklearn pipeline, never on the full dataset before CV. Features that are
  entirely missing across the cohort are dropped with a warning.
- Same outcome (CL 3–4), same patient-level stratified 5×2 CV, and same class
  weighting (`class_weight` / `scale_pos_weight`) as the face model.

## Block C / Weeks 8–11 — fusion & clinical comparison

Run **after** the face features (`make week45`) exist; calibration rebuilds the
cleaned ultrasound table itself.

```bash
make calibration          # isotonic-calibrate face & ultrasound (within-fold)
make fusion               # logistic meta-learner + average baseline
make clinical-comparison  # clinical baselines + DeLong tests
make block-c              # all three, in order
```

Outputs (in `reports/`):
- `calibrated_face_probs.csv`, `calibrated_us_probs.csv` — out-of-fold calibrated
  probabilities, one row per patient per fold (`study_id, repeat, fold_index,
  label, calibrated_prob`)
- `calibration_metrics.csv` — Brier per fold + pooled; `face_calibration.png`,
  `us_calibration.png` — reliability diagrams
- `fused_model.pkl` — meta-learner refit on all OOF probs + metadata
- `fusion_cv_metrics.csv`, `fusion_average_baseline_metrics.csv` — learned vs
  average-baseline metrics; `fusion_roc.png`
- `fusion_fold_predictions.csv` — per patient/fold: `face_prob, us_prob,
  fused_prob, avg_prob, label`
- `per_model_metrics.csv` — AUC + operating-point metrics for every model
- `delong_comparisons.csv` — six DeLong tests (fused vs each comparator),
  Bonferroni α = 0.0083

Notes:
- **Primary model per modality = L2 logistic regression** (calibrated); XGBoost
  remains available in the standalone per-modality reports.
- **No leakage:** one common cohort (face + ultrasound + label); folds generated
  once and reused everywhere via the calibrated-prob fold membership. The
  meta-learner trains only on training-fold calibrated probabilities.
- Each step **fails clearly** if a prerequisite output is missing (e.g. fusion
  requires the calibrated-prob CSVs).
- On the synthetic data the learned fusion does not beat the average baseline
  (both AUC = 1.00); the pipeline prints/persists a sanity-check **warning**
  rather than failing.

## Block D / Weeks 12–14 — validation & explainability (coding only)

Run **after** Block C (these steps read `reports/fusion_fold_predictions.csv`
and `reports/fused_model.pkl`).

```bash
make bootstrap-ci      # patient-level bootstrap 95% CIs for all metrics
make subgroups         # descriptive subgroup metrics + effect sizes
make explainability    # SHAP summary + case selection + force plots
make error-analysis    # FN / FP tables for manual review
make data-freeze       # data-freeze memo TEMPLATE
make block-d           # all five, in order
```

Outputs:
- `reports/bootstrap_metric_cis.csv` — per model × metric: estimate, 95% CI,
  valid-iteration count (1000 patient-level resamples, fixed seed)
- `reports/subgroup_metrics.csv`, `reports/subgroup_effect_sizes.csv` —
  descriptive only; small subgroups flagged `underpowered` (no hypothesis tests)
- `reports/shap_summary_fused.png`, `reports/explainability_feature_summary.csv`
  — SHAP on the fused inputs (falls back to standardised LR coefficients if SHAP
  is unavailable)
- `reports/explanation_case_selection.csv` — up to two each of TP/TN/FP/FN
- `outputs/explainability/force_plots/*.png` — per-case SHAP force plots, or
  `outputs/explainability/force_plot_notes.md` if they cannot be generated
- `reports/false_negatives_for_manual_review.csv`,
  `reports/false_positives_for_manual_review.csv` — full per-patient context
- `reports/data_freeze_memo_TEMPLATE.md` — placeholders only; **not** a freeze

Notes:
- **SHAP is optional.** If `import shap` fails, the summary/force plots are
  skipped, the feature summary falls back to standardised logistic coefficients,
  and `force_plot_notes.md` records why — the pipeline never crashes.
- **Patient-level units throughout:** the bootstrap resamples patients (not
  rows/images); per-patient probabilities are averaged across the CV repeats.
- This block is **coding/automation only** — it produces tables, plots, and a
  template. It does **not** tag the repo, freeze MLflow runs, or make the
  data-freeze decision.

## Assumptions (column names & paths)

The code reads these names; rename your real columns to match **inside the
loaders**, not throughout the codebase:

- **Outcome:** `cl_grade` (1–4); difficult = CL 3–4 → `label`. Optional second
  observer `cl_grade_obs2` enables inter-observer κ.
- **Ultrasound** (`data/raw/ultrasound.csv`): measured `dstvc_mm`,
  `hmd_neutral_mm`, `hmd_extended_mm`, `dse_mm` (+ derived `hmdr`). Differently
  named source columns can be mapped via `ultrasound_features.US_COLUMN_ALIASES`;
  a schema column absent from the export becomes an all-missing placeholder
  (warned), and is dropped from the model if entirely missing. Non-numeric/blank
  values are coerced to missing and imputed within each CV fold (mean).
- **Pre-op / demographics** (`data/raw/preop.csv`): `age_years`, `sex`, `bmi`,
  `surgery_type`, `mallampati_class`, `mouth_opening_mm`, `thyromental_mm`,
  `neck_movement_deg`, `jaw_subluxation`, `buck_teeth`, `obstructed_airway`,
  `weight_class`, `head_neck_class`, `receding_mandible`. Every pre-op column is
  optional — a missing column makes the dependent score component `NaN` (never
  silently 0). Block D subgroups use `bmi`/`age_years` (tertiles) and
  `surgery_type` (categorical); absent columns are skipped with a message.
- **Faces** (`data/raw/face_index.csv`): `study_id`, `view_code`, `file_path`
  (relative to `data/raw/face_images/`). Multiple images per patient.

## Repository layout

```
airway/
├── pyproject.toml              # dependencies + tool config
├── Makefile                    # convenience commands
├── src/airway/
│   ├── config.py               # ALL paths and constants
│   ├── loaders.py              # face / ultrasound / label loaders
│   ├── splits.py               # patient-level CV splitter (most important file)
│   ├── face_align.py           # WEEK 2: crop + resize faces to 224x224
│   ├── face_features.py        # WEEK 2: ResNet-18 embeddings -> per-patient vector
│   ├── ultrasound_features.py  # WEEK 2: ultrasound feature table
│   ├── baseline_model.py       # WEEK 2: leakage-safe CV model + metrics
│   ├── make_dummy_data.py      # generates fake data for plumbing tests
│   └── pilot_report.py         # runs the whole pipeline
├── tests/
│   ├── test_pipeline.py        # Week 1 tests
│   └── test_week2.py           # Week 2 tests
├── data/                       # tracked by DVC, not Git
└── reports/                    # generated metrics and figures
```

## The two rules

1. Never split data with `sklearn.train_test_split` directly. Always use
   `airway.splits.patient_level_folds`.
2. Never scale or impute outside cross-validation. The baseline `Pipeline`
   does it inside each fold — keep it that way.
