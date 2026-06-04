"""Runtime regression for examples/vs_langgraph/03_map_reduce.py (neograph-iu05).

The example's NeoGraph path uses ``Node | Oracle(n=3, merge_prompt="pick-best")``
with an inline ``prompt_compiler``. The Oracle merge step calls the
prompt_compiler with ``input_data = {"variants": [<variant models>], ...}`` -- a
dict, not a bare ``list[Pydantic]``. The original example iterated ``data``
directly (``for j in data ...``), which walks the dict's *keys* (strings) and
crashes with ``AttributeError: 'str' object has no attribute 'items'``.

This test runs the actual example module with a fake LLM and asserts the
NeoGraph path completes and returns a joke. It fails (AttributeError) against
the broken lambda and passes once the lambda reads ``data["variants"]``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

EXAMPLE =Path(__file__).resolve().parent.parent / "examples" / "vs_langgraph" / "03_map_reduce.py"


def _load_example():
    os.environ.setdefault("OPENROUTER_API_KEY", "test-dummy-key")
    spec = importlib.util.spec_from_file_location("neograph_example_03_map_reduce", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


class _FakeJokeLLM:
    """Returns a structured ``Jokes`` for both the generators and the merge."""

    def __init__(self, jokes_cls):
        self._jokes_cls = jokes_cls

    def with_structured_output(self, model, **_kw):
        self._model = model
        return self

    def invoke(self, _messages, **_kw):
        return self._jokes_cls(items=["why did the developer go broke? cache flow"])


class TestMapReduceExampleNeoGraphPath:
    """neograph-iu05 — the Oracle merge_prompt compiler must read the variants
    from ``data["variants"]``, not iterate ``data`` as if it were a list."""

    def test_neograph_map_reduce_returns_a_joke(self):
        module = _load_example()
        module.llm = _FakeJokeLLM(module.Jokes)
        result = module.run_neograph()
        assert isinstance(result, str) and result.strip(), (
            "NeoGraph map-reduce path should return a non-empty best-joke string; "
            f"got {result!r}"
        )
