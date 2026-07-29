.PHONY: quality test lint typecheck fix live release-gate

# Run all quality checks — tests, linter, type checker.
# NOTE: live external checks (tests/test_observe_trace_live.py) SKIP here when
# credentials are absent. That is intentional for the fast dev loop — but it
# means a green `quality` says NOTHING about the live path. Before tagging a
# release, run `release-gate`, which cannot skip them.
quality: test lint typecheck
	@echo "All quality checks passed."

# Run the test suite (offline only — `live` tests are deselected so this stays
# deterministic and fast even when LANGFUSE_* happen to be exported).
test:
	uv run pytest -q --tb=short -m "not live"

# Ruff linter
lint:
	uv run ruff check src/neograph/ tests/ examples/

# Mypy type checker
typecheck:
	uv run mypy src/neograph/ --ignore-missing-imports

# Auto-fix ruff issues
fix:
	uv run ruff check src/neograph/ tests/ examples/ --fix

# Live external checks — real Langfuse API. Requires LANGFUSE_SECRET_KEY +
# LANGFUSE_PUBLIC_KEY in the environment:
#     set -a && . .env && set +a && make live
# NEOGRAPH_REQUIRE_LIVE=1 makes absent credentials a hard ERROR instead of a
# skip, so this target cannot report success without actually talking to Langfuse.
live:
	NEOGRAPH_REQUIRE_LIVE=1 uv run --extra langfuse pytest -q --tb=short -m live

# THE RELEASE GATE — mandatory on merged `main` AFTER the merge and BEFORE the
# tag. `quality` alone is not sufficient: it lets the live checks skip, which is
# how 0.7.4 was first tagged with a flaky live assertion that a keyless run had
# silently subtracted from the count.
release-gate: quality live
	@echo ""
	@echo "RELEASE GATE PASSED — offline suite + live external checks both green."
	@echo "Safe to tag. Tag the commit you just gated, not a later one."
