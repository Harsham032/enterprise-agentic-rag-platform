PYTHON ?= python3.11
VENV   ?= .venv
BIN    := $(VENV)/bin
CONFIG ?= configs/default.yaml

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

.PHONY: install
install: $(BIN)/python ## Install runtime and development dependencies
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pip install -e .

.PHONY: fmt
fmt: ## Apply formatting and import ordering
	$(BIN)/black src tests scripts
	$(BIN)/ruff check --fix src tests scripts

.PHONY: lint
lint: ## Run linters and formatting checks
	$(BIN)/ruff check src tests scripts
	$(BIN)/black --check src tests scripts

.PHONY: typecheck
typecheck: ## Run static type analysis
	$(BIN)/mypy

.PHONY: test
test: ## Run the test suite
	$(BIN)/pytest

.PHONY: coverage
coverage: ## Run tests with a coverage report
	$(BIN)/pytest --cov=rag_platform --cov-report=term-missing

.PHONY: check
check: lint typecheck test ## Run every quality gate

.PHONY: index
index: ## Build the retrieval index from the configured corpus
	$(BIN)/python scripts/build_index.py --config $(CONFIG)

.PHONY: experiment
experiment: ## Run the retrieval and generation benchmark
	$(BIN)/python scripts/run_experiment.py --config $(CONFIG)

.PHONY: serve
serve: ## Start the API locally with reload
	$(BIN)/uvicorn rag_platform.services.api:app --reload --host 0.0.0.0 --port 8000

.PHONY: docker-build
docker-build: ## Build the service container image
	docker build -t rag-platform:local .

.PHONY: up
up: ## Start PostgreSQL, Qdrant, Redis and the API
	docker compose up --build

.PHONY: down
down: ## Stop and remove the local stack
	docker compose down -v

.PHONY: secrets-scan
secrets-scan: ## Look for credential-shaped strings in tracked files
	$(BIN)/python scripts/secrets_scan.py

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
