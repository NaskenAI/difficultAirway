# Airway — Bi-Modal Difficult Airway Prediction

Predicts difficult airway (Cormack–Lehane Grade 3–4) from facial images and
point-of-care ultrasound measurements.

## Status

- **Week 1 (done):** project skeleton, data loaders, patient-level CV splitter, tests.
- **Week 2 (done):** face alignment, ResNet-18 embeddings, ultrasound feature
  table, baseline logistic-regression model with cross-validated metrics.

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
