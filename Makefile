# Makefile — standard development workflows
#
# Usage:
#   make test          Run unit + property tests
#   make test-all      Run all tests (requires docker services)
#   make lint          Run ruff linter
#   make format        Auto-format with ruff
#   make typecheck     Run mypy on core modules
#   make ci            Run full CI locally (lint + test + typecheck)
#   make docker-up     Start docker-compose services
#   make docker-test   Start test infrastructure
#   make clean         Remove build artifacts

.PHONY: help test test-all test-unit test-integration test-property \
        lint format typecheck ci \
        docker-up docker-down docker-test docker-test-down \
        clean install dev-install pre-commit-install

PYTHON := python3
PYTEST := $(PYTHON) -m pytest
RUFF   := ruff
MYPY   := mypy

# Default target
help:
	@echo "Available targets:"
	@echo "  make test            Run unit + property tests (fast)"
	@echo "  make test-all        Run all tests (needs docker services)"
	@echo "  make test-unit       Run unit tests only"
	@echo "  make test-integration Run integration tests (needs docker)"
	@echo "  make lint            Check code with ruff"
	@echo "  make format          Auto-format with ruff"
	@echo "  make typecheck       Run mypy type checker"
	@echo "  make ci              Full local CI (lint + test + typecheck)"
	@echo "  make docker-up       Start development services"
	@echo "  make docker-test     Start test infrastructure"
	@echo "  make clean           Remove build artifacts"
	@echo "  make install         Install production dependencies"
	@echo "  make dev-install     Install dev dependencies"

# -------------------------------------------------------------------
# Installation
# -------------------------------------------------------------------

install:
	$(PYTHON) -m pip install -e .

dev-install:
	$(PYTHON) -m pip install -e ".[dev]"

pre-commit-install:
	pre-commit install

# -------------------------------------------------------------------
# Testing
# -------------------------------------------------------------------

test: test-unit test-property

test-all: test-unit test-property test-integration

test-unit:
	$(PYTEST) tests/unit/ -q --tb=short

test-property:
	$(PYTEST) tests/property/ -q --tb=short -m "property"

test-integration:
	$(PYTEST) tests/integration/ -q --tb=short -m "integration"

test-cov:
	$(PYTEST) tests/unit/ tests/property/ \
		-q --tb=short \
		--cov=src/agentic_trading \
		--cov-report=term-missing:skip-covered \
		--cov-report=html:htmlcov

# -------------------------------------------------------------------
# Code quality
# -------------------------------------------------------------------

lint:
	$(RUFF) check src/ tests/

format:
	$(RUFF) format src/ tests/
	$(RUFF) check --fix src/ tests/

typecheck:
	$(MYPY) src/agentic_trading/core/ \
	        src/agentic_trading/governance/ \
	        src/agentic_trading/event_bus/ \
	        --ignore-missing-imports \
	        --no-strict-optional \
	        --allow-untyped-defs

# -------------------------------------------------------------------
# Full CI (local)
# -------------------------------------------------------------------

ci: lint test typecheck
	@echo "✅ All CI checks passed"

# -------------------------------------------------------------------
# Docker
# -------------------------------------------------------------------

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-test:
	docker-compose -f docker-compose.test.yml up -d
	@echo "Waiting for services..."
	@sleep 3
	@echo "Test infrastructure ready (postgres:5433, redis:6380)"

docker-test-down:
	docker-compose -f docker-compose.test.yml down

# -------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage coverage.xml junit.xml
	rm -rf dist build
