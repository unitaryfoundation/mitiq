.PHONY: all
all: dist

.PHONY: build
build: check-all docs test-all

.PHONY: check-all
check-all: check-format check-types

.PHONY: check-format
check-format:
	uv run ruff check
	uv run ruff format --check

.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format

.PHONY: check-types
check-types:
	uv run mypy mitiq --show-error-codes

.PHONY: clean
clean:
	rm -rf dist
	rm -rf mitiq.egg-info
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	rm -rf htmlcov coverage.xml .coverage .coverage.*
	rm -rf .ipynb_checkpoints

.PHONY: dist
dist:
	uv run python setup.py sdist

.PHONY: docs
docs:
	uv run make -C docs html

.PHONY: docs-clean
docs-clean:
	uv run make -C docs clean
	uv run make -C docs html

.PHONY: docs-lite
docs-lite:
	uv run env DOCS_LITE=1 make -C docs html

.PHONY: linkcheck
linkcheck:
	uv run make -C docs linkcheck

.PHONY: install
install:
	uv sync --all-extras --all-groups

.PHONY: install-hooks
install-hooks:
	@git config --local core.hooksPath .git-hooks/
	@chmod +x .git-hooks/*
	@echo "Git hooks installed."

.PHONY: test
test:
	uv run pytest -n auto -v --cov=mitiq --cov-report=term --cov-report=xml --ignore=mitiq/interface/mitiq_pyquil

.PHONY: test-%
test-%:
	uv run pytest -n auto -v --cov=mitiq --cov-report=term --cov-report=xml */$(*)/*

.PHONY: test-pyquil
test-pyquil:
	uv run pytest -v --cov=mitiq --cov-report=term --cov-report=xml mitiq/interface/mitiq_pyquil

.PHONY: test-all
test-all:
	uv run pytest -n auto -v --cov=mitiq --cov-report=term --cov-report=xml
