"""Runtime regression for examples/28_portal_swarm.py (PORTAL T5 showcase).

The example demonstrates the shipped PORTAL peer-routing surface (mode a): a
triage node hands off to declared specialist peers at RUNTIME via ``Command(goto)``,
specialists can route BACK to triage (a genuine cycle), a ``max_hops`` budget
guarantees termination, and ``HANDOFF_END`` leaves the mesh. Every member consumes
the reserved ``handoff`` channel and the routing is decided at runtime yet every
reachable peer is type-checked at compile time.

This test runs the example's ``main()`` so a future PORTAL API change that breaks
the showcase is caught in CI (the example carries per-demo asserts; ``main()``
raising = a red test). It mirrors ``test_example_forward_wiring.py`` (example 27).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "28_portal_swarm.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("neograph_example_28_portal_swarm", EXAMPLE)
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


def test_portal_swarm_example_runs_end_to_end(capsys):
    """Every PORTAL peer-routing surface in the showcase compiles + runs; the
    example's own per-demo asserts pin the observable routing behavior."""
    module = _load_example()
    module.main()  # raises if any demo's assert fails or any surface breaks
    out = capsys.readouterr().out
    for marker in ("DEMO 1", "DEMO 2", "DEMO 3", "DEMO 4"):
        assert marker in out, f"{marker} did not run"
    assert "All PORTAL peer-routing surfaces ran" in out
