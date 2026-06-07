# Makefile -- convenience commands. Run `make <target>` from the repo root.

.PHONY: help install dummy test lint format check features pilot-report clean \
        audit quarantine scores crops embeddings face-model week3 week45 \
        us-clean us-model week67 calibration fusion clinical-comparison block-c

PYTHON ?= python3
SRC_PATH := src

help:           ## Show this help
	@echo "Available commands:"
	@echo "  make install       - install project dependencies"
	@echo "  make dummy         - generate fake data into data/raw/"
	@echo "  make test          - run the test suite"
	@echo "  make lint          - check code style"
	@echo "  make format        - auto-format code"
	@echo "  make check         - lint + test together"
	@echo "  make features      - build face + ultrasound feature tables"
	@echo "  make pilot-report  - run the full pipeline end-to-end"
	@echo "  --- Week 3 (data audit) ---"
	@echo "  make quarantine    - compute + persist quarantine rules"
	@echo "  make audit         - write reports/data_audit_report.md"
	@echo "  make scores        - compute Mallampati/LEMON/Wilson -> computed_baselines.csv"
	@echo "  make week3         - quarantine + audit + scores"
	@echo "  --- Weeks 4-5 (face model) ---"
	@echo "  make crops         - generate persisted 224x224 face crops (idempotent)"
	@echo "  make embeddings    - 512-d per image + 1024-d per patient features"
	@echo "  make face-model    - train + CV LogReg & XGBoost -> face_model.pkl"
	@echo "  make week45        - crops + embeddings + face-model"
	@echo "  --- Weeks 6-7 (ultrasound model) ---"
	@echo "  make us-clean      - clean + standardize ultrasound features -> cleaned_ultrasound_features.csv"
	@echo "  make us-model      - train + CV LogReg & XGBoost + importance -> us_model.pkl"
	@echo "  make week67        - us-clean + us-model"
	@echo "  --- Block C / Weeks 8-11 (fusion + clinical comparison) ---"
	@echo "  make calibration         - isotonic-calibrate face & US models -> calibrated_*_probs.csv"
	@echo "  make fusion              - late-fusion meta-learner + average baseline -> fused_model.pkl"
	@echo "  make clinical-comparison - clinical baselines + DeLong tests -> delong_comparisons.csv"
	@echo "  make block-c             - calibration + fusion + clinical-comparison"
	@echo "  make clean         - remove caches and generated outputs"

install:        ## Install common dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install numpy pandas scikit-learn torch torchvision matplotlib pytest ruff black opencv-python pyarrow

dummy:          ## Generate dummy data
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m airway.make_dummy_data

test:           ## Run all tests
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m pytest

lint:           ## Check style without changing files
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m ruff check src tests

format:         ## Auto-format the code
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m black src tests
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m ruff check --fix src tests

check: lint test  ## Lint and test in one go

features:       ## Build the face and ultrasound feature tables
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m airway.face_features
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m airway.ultrasound_features

pilot-report: features   ## Run the whole pipeline end-to-end
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m airway.baseline_model
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m airway.pilot_report

quarantine:     ## Compute and persist quarantine rules
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m airway.quarantine

audit:          ## Write the one-page data audit report
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m airway.data_audit

scores:         ## Compute Mallampati/LEMON/Wilson comparator baselines
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m airway.scores

week3: quarantine audit scores   ## Run the whole Week-3 data audit

crops:          ## Generate persisted 224x224 face crops (idempotent)
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m airway.face_crops

embeddings:     ## Build 512-d per-image + 1024-d per-patient face features
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m airway.face_embeddings

face-model:     ## Train + cross-validate LogReg & XGBoost face classifiers
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m airway.face_model

week45: crops embeddings face-model   ## Run the whole Weeks 4-5 face model

us-clean:       ## Clean + standardize the ultrasound features
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m airway.ultrasound_features

us-model:       ## Train + CV LogReg & XGBoost ultrasound classifiers + importance
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m airway.ultrasound_model

week67: us-model   ## Run the whole Weeks 6-7 ultrasound model (cleans features first)

calibration:    ## Isotonic-calibrate the face and ultrasound models
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m airway.calibration

fusion:         ## Late-fusion meta-learner + average baseline
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m airway.fusion

clinical-comparison:  ## Clinical baselines + DeLong tests
	PYTHONPATH=$(SRC_PATH) $(PYTHON) -m airway.clinical_comparison

block-c: calibration fusion clinical-comparison   ## Run all of Block C (Weeks 8-11)

clean:          ## Remove caches and generated outputs
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf reports/*.csv reports/*.png reports/*.md reports/*.pkl
	rm -rf data/processed/*.parquet data/processed/*.json data/processed/*.csv
	rm -rf data/processed/face_crops