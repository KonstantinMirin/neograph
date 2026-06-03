"""Test fixtures — registry isolation between tests."""

import pytest
import structlog


@pytest.fixture(autouse=True)
def _clean_registries():
    """Clear test-local registries before each test.

    The decoration-time shim registry (inline body-merge / interrupt_when /
    @merge_fn / @tool) lives in the leaf `_runtime_registry` and is reset via
    its own `reset()` — no hand-maintained `.clear()` block reaching into
    `decorators.py` internals (neograph-v3xx HIGH-01). The test-side helpers in
    `tests/fakes.py` (register_scripted etc.) write into test-local dicts that
    `reset_test_registry()` clears.
    """
    from neograph import _runtime_registry
    from neograph._sidecar import _merge_fn_registry
    from neograph.spec_types import _type_registry
    from tests.fakes import reset_test_registry

    reset_test_registry()
    _runtime_registry.reset()
    _merge_fn_registry.clear()
    _type_registry.clear()
    # Reset structlog to defaults so tests that capture warnings via stdout
    # (capsys) are not affected by an earlier test's reconfigure (e.g.
    # tests that route structlog through stdlib logging).
    structlog.reset_defaults()
    yield


