# Makefile -- convenience commands. Run `make <target>` from the repo root.

.PHONY: help install dummy test lint format check features pilot-report clean

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

clean:          ## Remove caches and generated outputs
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf reports/*.csv reports/*.png
	rm -rf data/processed/*.parquet	rm -rf data/processed/*.parquet