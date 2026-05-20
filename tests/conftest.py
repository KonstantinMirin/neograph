"""Test fixtures — registry isolation between tests."""

import pytest
import structlog


@pytest.fixture(autouse=True)
def _clean_registries():
    """Clear test-local registries before each test.

    Post-§2: src/ no longer holds module-level registries. The test-side
    convenience helpers in `tests/fakes.py` (register_scripted etc.) write
    into test-local dicts; this fixture resets them for test isolation.
    """
    from neograph.decorators import (
        _decorator_conditions,
        _decorator_scripted,
        _decorator_tool_factories,
        _merge_fn_registry,
    )
    from neograph.spec_types import _type_registry
    from tests.fakes import reset_test_registry

    reset_test_registry()
    _merge_fn_registry.clear()
    _type_registry.clear()
    _decorator_scripted.clear()
    _decorator_conditions.clear()
    _decorator_tool_factories.clear()
    # Reset structlog to defaults so tests that capture warnings via stdout
    # (capsys) are not affected by an earlier test's reconfigure (e.g.
    # tests that route structlog through stdlib logging).
    structlog.reset_defaults()
    yield


