.PHONY: quality test lint typecheck fix

# Run all quality checks — tests, linter, type checker
quality: test lint typecheck
	@echo "All quality checks passed."

# Run the test suite
test:
	uv run pytest -q --tb=short

# Ruff linter
lint:
	uv run ruff check src/neograph/ tests/

# Mypy type checker
typecheck:
	uv run mypy src/neograph/ --ignore-missing-imports

# Auto-fix ruff issues
fix:
	uv run ruff check src/neograph/ tests/ --fix
