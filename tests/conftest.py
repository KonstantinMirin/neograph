"""Test fixtures — registry isolation between tests."""

import pytest


@pytest.fixture(autouse=True)
def _clean_registries():
    """Clear all global registries before each test."""
    from neograph import factory, _llm

    factory._scripted_registry.clear()
    factory._condition_registry.clear()
    factory._tool_factory_registry.clear()
    _llm._llm_factory = None
    _llm._prompt_compiler = None
    yield
