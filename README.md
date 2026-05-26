# Airway — Bi-Modal Difficult Airway Prediction

Predicts difficult airway (Cormack–Lehane Grade 3–4) from facial images and
point-of-care ultrasound measurements.

This repository is the **Week 1 foundation**: project skeleton, data loaders,
the patient-level cross-validation splitter, and a test suite. No models yet —
those come in Block B.

## Quick start

```bash
# 1. create and activate the environment (see SETUP guide for details)
# 2. install the project
pip install -e ".[dev]"

# 3. generate dummy data (replace with real data later)
make dummy

# 4. run the tests — all should pass
make test

# 5. run the pipeline end-to-end
make pilot-report
```

## Repository layout

```
airway/
├── pyproject.toml          # dependencies + tool config
├── Makefile                # convenience commands
├── .pre-commit-config.yaml # automatic style checks on commit
├── .gitignore
├── src/airway/
│   ├── config.py           # ALL paths and constants live here
│   ├── loaders.py          # face / ultrasound / label loaders
│   ├── splits.py           # patient-level CV splitter (most important file)
│   ├── make_dummy_data.py  # generates fake data for plumbing tests
│   └── pilot_report.py     # runs the full pipeline
├── tests/
│   └── test_pipeline.py    # 13 tests, incl. the data-leakage test
├── data/                   # tracked by DVC, not Git
│   ├── raw/
│   └── processed/
└── reports/                # generated metrics and figures
```

## The one rule

Never split data with `sklearn.train_test_split` directly. Always use
`airway.splits.patient_level_folds`. A patient's images must never appear in
both train and test — see the long comment at the top of `splits.py`.
