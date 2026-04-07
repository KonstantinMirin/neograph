"""Test fixtures — registry isolation between tests."""

import pytest


@pytest.fixture(autouse=True)
def _clean_registries():
    """Clear all global registries before each test."""
    from neograph import factory, _llm
    from neograph.decorators import _merge_fn_registry
    from neograph.spec_types import _type_registry

    factory._scripted_registry.clear()
    factory._condition_registry.clear()
    factory._tool_factory_registry.clear()
    _llm._llm_factory = None
    _llm._llm_factory_params = set()
    _llm._prompt_compiler = None
    _llm._prompt_compiler_params = set()
    _llm._global_renderer = None
    _merge_fn_registry.clear()
    _type_registry.clear()
    yield
