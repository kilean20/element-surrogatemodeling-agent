.PHONY: help install install-dev test test-cov lint format type-check ext check-card validate-skills conformance clean

DISCOVERY_LOCAL_REF := f88a6097f3cddca84aa57d6f44edd0b5d5039ab1
GENESIS_SKILLS_REF := b00054a6375c18463de751f2eef8c4d4bf8643ee

help:
	@echo "element-surrogate - development tasks"
	@echo ""
	@echo "  make install          Sync the project environment with uv"
	@echo "  make install-dev      Sync dev and test dependencies with uv"
	@echo "  make test             Run tests with uv"
	@echo "  make test-cov         Run tests with an HTML coverage report"
	@echo "  make lint             Run lint and format checks with ruff"
	@echo "  make format           Format code with ruff"
	@echo "  make type-check       Run type checking with mypy"
	@echo "  make ext              Fetch the pinned AI-ModCon checkers into _ext/"
	@echo "  make conformance      Check the agent card and validate skills (needs make ext)"
	@echo "  make clean            Remove build artifacts and cache files"

install:
	uv sync

install-dev:
	uv sync --group dev --group test

test:
	uv run pytest

test-cov:
	uv run pytest --cov=element_surrogate --cov-report=html --cov-report=term-missing

lint:
	uv run ruff check .
	uv run ruff format . --check

format:
	uv run ruff format .

type-check:
	uv run mypy src/element_surrogate

ext:
	git clone https://github.com/AI-ModCon/discovery-local.git _ext/discovery-local
	git -C _ext/discovery-local checkout $(DISCOVERY_LOCAL_REF)
	git clone https://github.com/AI-ModCon/genesis-skills.git _ext/genesis-skills
	git -C _ext/genesis-skills checkout $(GENESIS_SKILLS_REF)

check-card:
	uv run python -I _ext/discovery-local/templates/check_card.py cards/agent_card.md

validate-skills:
	@skills="$$(find skills -name SKILL.md -exec dirname {} \;)"; \
	if [ -z "$$skills" ]; then echo "No skills yet; nothing to validate."; \
	else uv run python -I _ext/genesis-skills/tools/validate_skills.py --profile all --no-color $$skills; fi

conformance: check-card validate-skills

# Preserve .venv and installed package metadata; clean repository outputs only.
clean:
	find src tests -type f -name '*.py[cod]' -delete
	find src tests -depth -type d -name '__pycache__' -empty -delete
	rm -rf build/ dist/ htmlcov/ .coverage .coverage.* .pytest_cache/ .mypy_cache/ .ruff_cache/
