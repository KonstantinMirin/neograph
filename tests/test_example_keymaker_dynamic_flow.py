"""Runtime regression for examples/29_keymaker_dynamic_flow.py (KEYMAKER mode-b).

The example demonstrates the shipped KEYMAKER dynamic-dispatch surface (mode b):
a planner node EMITS a neograph spec at RUNTIME, and the Keymaker
(``route='decide'``) validates -> compiles -> dispatches that emitted flow. The
happy path proves an emitted spec is loaded, compiled, and its typed result lands
on the ``{node}_dispatch`` channel for a downstream consumer. The rejection path
proves a structurally-invalid emitted spec is caught at the ``load_spec`` gate --
the underlying ``ConstructError`` surfaces as a wrapped ``ExecutionError`` naming
the spec, and it fires BEFORE any dispatched sub-node body executes.

This test runs the example's ``main()`` so a future KEYMAKER dispatch API change
that breaks the showcase is caught in CI (the example carries per-demo asserts;
``main()`` raising = a red test). It mirrors ``test_example_keymaker.py``
(example 28).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "29_keymaker_dynamic_flow.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("neograph_example_29_keymaker_dynamic_flow", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so Pydantic forward refs (e.g. under
    # `from __future__ import annotations`) resolve against the module namespace.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_keymaker_dynamic_flow_example_runs_end_to_end(capsys):
    """The mode-(b) dynamic-dispatch showcase compiles + runs keyless: the happy
    demo dispatches a runtime-emitted spec, the rejection demo proves a wrapped
    ConstructError fires before any sub-node body runs. The example's own
    per-demo asserts pin the observable behavior."""
    module = _load_example()
    module.main()  # raises if any demo's assert fails or any surface breaks
    out = capsys.readouterr().out
    assert "DEMO 1" in out, "DEMO 1 (happy dynamic dispatch) did not run"
    assert "DEMO 2" in out, "DEMO 2 (rejection path) did not run"
    assert "KEYMAKER dynamic-flow dispatch verified" in out
