"""Runtime regression for examples/33_run_scoped_state.py.

The example is the answer to GH #15 in one runnable file: the three routes a
step has to a value produced earlier in the run, and -- the part that matters --
the difference between a model being SHOWN a value and being unable to
substitute a different one.

It runs rather than imports, because the whole claim is about what happens at
run time when a model composes a tool call. The example carries its own asserts,
so `main()` raising is a red test; this pins the observable output too, so a
change that keeps the asserts true while breaking what the example TEACHES is
still caught.
"""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "33_run_scoped_state.py"


def test_the_example_runs_and_the_model_cannot_substitute_the_bound_argument(capsys):
    """The example's own asserts gate correctness; these gate the lesson."""
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    out = capsys.readouterr().out

    # Route 2: every fanned scripted branch read the run context.
    assert "SKU-114 [warehouse 4822/eu-west]" in out
    assert "SKU-330 [warehouse 4822/eu-west]" in out

    # Route 3: the model asked for warehouse 1, the tool got 4822, and the
    # unbound argument survived untouched. All three lines matter -- the third
    # is what stops the fix being "overwrite everything".
    assert "model emitted   : warehouse_id=1" in out
    assert "tool received   : warehouse_id=4822" in out
    assert "include_reserved=True" in out


def test_the_bound_argument_is_declared_on_the_node():
    """The binding is visible in the IR, not buried in a runtime callback --
    that is what makes it lint-visible and assembly-checked."""
    spec = importlib.util.spec_from_file_location("neograph_example_33", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        summarize = next(n for n in module.pipeline.nodes if n.name == "summarize")
    finally:
        sys.modules.pop(spec.name, None)

    assert summarize.tools[0].bound_args == {"warehouse_id": "audit.warehouse_id"}
    assert summarize.context == ["audit"]
