.DEFAULT_GOAL := help
IMAGE ?= opencv-age-detection:dev
export AD_TEST_NATS_URL ?= nats://127.0.0.1:4222
export AD_TEST_S3_ENDPOINT ?= http://127.0.0.1:8333

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: models
models: ## Download + verify the age classifier weights (not stored in git)
	uv run python scripts/fetch_models.py models

.PHONY: install
install: ## Install dependencies (uv) and git hooks
	uv sync --all-groups
	uv run pre-commit install

.PHONY: lint
lint: ## Ruff lint + format check + mypy
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

.PHONY: fmt
fmt: ## Auto-fix and format
	uv run ruff check . --fix
	uv run ruff format .

.PHONY: test
test: models ## Unit tests (no external services)
	uv run pytest -m "not integration"

.PHONY: test-integration
test-integration: ## Integration tests against `make up` (NATS + S3 on localhost)
	uv run pytest -m integration --cov-fail-under=0

.PHONY: test-all
test-all: ## Unit + integration with the CI coverage gate
	uv run pytest --cov-fail-under=90

.PHONY: run-api
run-api: ## Run the API locally with auto-reload
	uv run uvicorn age_detection.api.app:create_app --factory --reload --port 8000

.PHONY: docker
docker: ## Build the container image
	docker build -t $(IMAGE) --build-arg REVISION=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) .

.PHONY: up
up: ## Start the full local stack (API, workers, NATS, S3, observability)
	docker compose --profile observability up -d --build --wait

.PHONY: down
down: ## Stop the local stack and delete its volumes
	docker compose --profile observability down -v

.PHONY: helm-lint
helm-lint: ## Lint and render the Helm chart
	helm lint deploy/helm/age-detection --strict
	helm template fd deploy/helm/age-detection --set auth.apiKeyHashes=$$(printf x | shasum -a 256 | cut -d' ' -f1) >/dev/null

.PHONY: bench
bench: ## Load-test a running API (see scripts/bench.py)
	uv run python scripts/bench.py --url http://127.0.0.1:8000 --image tests/data/astronaut.jpg

.PHONY: dashboard
dashboard: ## Regenerate the Grafana dashboard JSON
	python3 scripts/gen_dashboard.py > deploy/compose/grafana/dashboards/age-detection.json
	cp deploy/compose/grafana/dashboards/age-detection.json deploy/helm/age-detection/files/age-detection-dashboard.json
