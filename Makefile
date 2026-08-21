.PHONY: quality test lint typecheck fix live mcp examples website skipcheck release-gate forward-port-check

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

# Optional shipped surfaces. `neograph[mcp]` is a SECOND top-level package
# (src/neograph_mcp); without these extras its ~74 tests importorskip and the
# default suite says nothing about it.
mcp:
	uv run --extra mcp --extra mcp-examples pytest -q --tb=short -m "not live"

# Examples must run end-to-end — breaking one is a regression. Only the keyless
# ones; 07 and observable_pipeline.py need real OpenRouter credentials.
examples:
	@set -e; for f in examples/01_*.py examples/01c_*.py examples/02_*.py \
	    examples/03_*.py examples/04_*.py examples/05_*.py examples/06_*.py \
	    examples/08_*.py examples/09_*.py examples/10_*.py examples/11_*.py; do \
	  printf '  %s ... ' "$$f"; \
	  uv run python "$$f" >/dev/null 2>&1 && echo ok || { echo FAILED; exit 1; }; \
	done
	uv run --extra mcp-examples pytest -q --tb=short tests/test_mcp_examples_e2e.py

# The website build is part of the product: the api-manifest guard couples page
# content to the public API surface, so a signature change can break the build.
website:
	cd website && npm ci --silent && npm run build

# A skip is invisible in a pass count. This fails on any skip whose reason is not
# in tests/skip_allowlist.txt (which is empty by design).
skipcheck:
	uv run python scripts/check_skips.py -m "not live"

# Forward-port check — run on `develop` AFTER merging a release branch back.
# A green test run cannot tell "the docs merged" from "the docs were discarded";
# this asks the question the suite structurally cannot. Override the refs with
# SOURCE=/TARGET= when porting between other branches.
SOURCE ?= origin/main
TARGET ?= HEAD
forward-port-check:
	uv run python scripts/check_forward_port.py --source $(SOURCE) --target $(TARGET)

# THE RELEASE GATE — mandatory on merged `main` AFTER the merge and BEFORE the
# tag.
#
# `quality` alone is not sufficient, and neither was `quality + live`. Both
# report success while an arbitrary subset of the suite does not run:
#   - 0.7.4 was tagged with a flaky live assertion; the Langfuse tests had
#     silently skipped for want of credentials.
#   - 0.7.7 was tagged green while 74 mcp tests skipped for a missing extra, the
#     examples were never run by the gate, and the website was never built.
# A gate whose success signal is compatible with whole surfaces not executing is
# not a gate. `skipcheck` is the general fix; mcp/examples/website are the
# instances that were missing.
release-gate: quality live mcp examples website skipcheck
	@echo ""
	@echo "RELEASE GATE PASSED"
	@echo "  offline suite + live external checks"
	@echo "  neograph[mcp] surface exercised (not skipped)"
	@echo "  keyless examples run end-to-end"
	@echo "  website builds"
	@echo "  zero unallowlisted skips"
	@echo "Safe to tag. Tag the commit you just gated, not a later one."
