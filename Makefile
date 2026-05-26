# Makefile — convenience commands. Run `make <target>` from the repo root.
#
# On Windows, `make` is not installed by default. Either:
#   (a) install it via `choco install make` or use Git Bash, or
#   (b) just run the commands after the colon manually.

.PHONY: help dummy test lint format check pilot-report clean

help:           ## Show this help
	@echo "Available commands:"
	@echo "  make dummy         - generate fake data into data/raw/"
	@echo "  make test          - run the test suite (pytest)"
	@echo "  make lint          - check code style (ruff)"
	@echo "  make format        - auto-format code (black)"
	@echo "  make check         - lint + test together"
	@echo "  make pilot-report  - run the full pipeline end-to-end"
	@echo "  make clean         - remove caches and generated reports"

dummy:          ## Generate dummy data
	python -m airway.make_dummy_data

test:           ## Run all tests
	pytest

lint:           ## Check style without changing files
	ruff check src tests

format:         ## Auto-format the code
	black src tests
	ruff check --fix src tests

check: lint test  ## Lint and test in one go

pilot-report:   ## Run the whole pipeline (Week 2 fills this in properly)
	python -m airway.pilot_report

clean:          ## Remove caches and generated outputs
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/**/__pycache__
	rm -rf reports/*.csv reports/*.png
