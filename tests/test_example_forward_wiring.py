"""Runtime regression for examples/27_forward_agent_wiring.py (neograph-e9zse showcase).

The example demonstrates every ForwardConstruct imperative surface — branch,
self.loop, self.each fan-out, fan-out-inside-loop cascade, self.ensemble, and
self.interrupt HITL — each traced to IR and run end-to-end with scripted
(keyless) nodes. This test runs the example's ``main()`` so a future
ForwardConstruct API change that breaks the showcase is caught in CI (the
example carries per-demo asserts; ``main()`` raising = a red test). It is the
guard against the exact rot that left the new surfaces unexemplified before.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "27_forward_agent_wiring.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("neograph_example_27_forward_wiring", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so Pydantic forward refs (e.g. list[Claim] under
    # `from __future__ import annotations`) resolve against the module namespace.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_forward_agent_wiring_example_runs_end_to_end(capsys):
    """Every ForwardConstruct surface in the showcase traces + runs; the
    example's own per-demo asserts pin the observable behavior."""
    module = _load_example()
    module.main()  # raises if any demo's assert fails or any surface breaks
    out = capsys.readouterr().out
    for marker in ("DEMO 1", "DEMO 2", "DEMO 3", "DEMO 4", "DEMO 5", "DEMO 6"):
        assert marker in out, f"{marker} did not run"
    assert "All ForwardConstruct surfaces ran" in out
