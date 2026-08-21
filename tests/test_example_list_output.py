"""Runtime regression for examples/32_list_output_type.py.

The example exists because `outputs=list[X]` was the one declaration shape the
framework never exercised: zero occurrences across every other example, while
the parse path could not handle it and the decorator rejected it whenever a
matching return annotation was present (GH #14). A road with potholes and no
signposts.

This test runs the example so the road stays open. It is deliberately a RUN
rather than an import: the reported failure was a pipeline that assembled
cleanly and then produced no decision on any row, so assembling proves nothing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "32_list_output_type.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("neograph_example_32_list_output", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so Pydantic forward refs resolve against the module
    # namespace under `from __future__ import annotations`.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_a_list_output_node_runs_and_a_downstream_node_consumes_it():
    """The whole point: declare `list[Reading]`, get `list[Reading]` back, and
    have a downstream node bind it by name like any other output."""
    from neograph import compile, run

    module = _load_example()
    graph = compile(
        module.pipeline,
        llm_factory=module.llm_factory,
        prompt_compiler=lambda template, data: [{"role": "user", "content": "extract"}],
    )
    result = run(graph, input={"node_id": "REPORT-7"})

    readings = result["extract"]
    assert isinstance(readings, list), f"declared list[Reading], got {type(readings).__name__}"
    assert [r.sensor for r in readings] == ["t1", "t2"]
    assert all(isinstance(r, module.Reading) for r in readings), "elements were not validated"

    assert "hottest t1" in result["summarize"].text, "the downstream consumer did not receive the list"


def test_the_declaration_survives_onto_the_node():
    """`outputs=list[Reading]` with a matching `-> list[Reading]` is the shape
    the decorator used to reject by identity comparison."""
    module = _load_example()

    extract = next(n for n in module.pipeline.nodes if n.name == "extract")
    assert extract.outputs == list[module.Reading]
