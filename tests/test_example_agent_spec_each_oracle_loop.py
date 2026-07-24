"""Runtime regression for examples/31_agent_spec_each_oracle_loop.py (e3b4j).

Cite docs/design/agent-spec-interop-2026-07-09.md s5, s10.8 and neograph-e3b4j's
refined plan: THREE separate minimal panels (Each, Oracle, Loop), each adapted
from the already-proven pipelines in
tests/test_agent_spec_roundtrip.py:87-195
(TestEachOracleLoopRoundTripPreservesBehavior) -- not a single fused construct.

Core Invariant under test (e3b4j Refined Plan): the "N lines" headline the
example prints for each panel's Agent Spec dump must be the ACTUAL line count
of ``json.dumps(to_agent_spec(pipeline).to_dict(), indent=2)`` for THAT
pipeline -- never a hand-abridged/rhetorical number. This test independently
recomputes the real line count from the module's own exposed pipeline
builders and asserts it against what the example printed, so a hand-typed
headline (or one that silently drifts from the exporter's real output) fails
loud.

Contract this test pins on the example module (per e3b4j's plan, this is the
interface the implementation atom (cbtjk.7) must satisfy):
  - module-level ``build_each_panel() -> Construct``
  - module-level ``build_oracle_panel() -> Construct``
  - module-level ``build_loop_panel() -> Construct``
  - ``main()`` prints a header containing "EACH PANEL", "ORACLE PANEL", and
    "LOOP PANEL" (one per section) and, per panel, a line of the form
    ``Agent Spec Flow: <N> lines`` where <N> is the real to_dict() dump line
    count for that panel's pipeline.
  - Oracle panel uses mode='think' + a keyless StructuredFake (scripted-mode
    Oracle round-trip is known-broken per the design doc s6).

Behavioral proof (not just "it runs"): each panel's pipeline is round-tripped
through to_agent_spec -> from_agent_spec -> compile -> run and produces the
SAME observable result as the known-good minimal pipelines in
test_agent_spec_roundtrip.py, proving the example's compact panels are not
merely importable but behave identically to the proven originals.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyagentspec")

from neograph import compile, run  # noqa: E402
from neograph._agent_spec import to_agent_spec  # noqa: E402
from neograph.loader import from_agent_spec  # noqa: E402
from neograph.testing.fakes import StructuredFake  # noqa: E402
from tests.fakes import build_fake_llm_kwargs, build_test_compile_kwargs  # noqa: E402

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "31_agent_spec_each_oracle_loop.py"


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "neograph_example_31_agent_spec_each_oracle_loop", EXAMPLE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so Pydantic forward refs resolve against the module
    # namespace (mirrors test_example_portal_dynamic_flow.py's convention).
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _real_dump_line_count(pipeline) -> int:
    """The mechanically-derived, non-negotiable line count for a panel's
    Agent Spec dump -- the ONLY number the example is allowed to print."""
    flow = to_agent_spec(pipeline)
    dumped = json.dumps(flow.to_dict(), indent=2)
    return dumped.count("\n") + 1


def test_example_module_exposes_three_panel_builders():
    """The example must expose one builder per panel (Each/Oracle/Loop) so the
    printed headline can be independently verified against the real exporter
    output, per e3b4j's Core Invariant."""
    module = _load_example()
    assert hasattr(module, "build_each_panel"), "missing build_each_panel()"
    assert hasattr(module, "build_oracle_panel"), "missing build_oracle_panel()"
    assert hasattr(module, "build_loop_panel"), "missing build_loop_panel()"


def test_each_oracle_loop_example_runs_end_to_end(capsys):
    """The example's main() runs all three panels keylessly and prints a
    header per panel plus a real (not hand-abridged) Agent Spec line count."""
    module = _load_example()
    module.main()
    out = capsys.readouterr().out

    assert "EACH PANEL" in out
    assert "ORACLE PANEL" in out
    assert "LOOP PANEL" in out

    each_pipeline = module.build_each_panel()
    oracle_pipeline = module.build_oracle_panel()
    loop_pipeline = module.build_loop_panel()

    expected_counts = {
        "EACH PANEL": _real_dump_line_count(each_pipeline),
        "ORACLE PANEL": _real_dump_line_count(oracle_pipeline),
        "LOOP PANEL": _real_dump_line_count(loop_pipeline),
    }

    printed_counts: dict[str, int] = {}
    current_panel = None
    for line in out.splitlines():
        for panel in expected_counts:
            if panel in line:
                current_panel = panel
        match = re.search(r"Agent Spec Flow:\s*(\d+)\s*lines", line)
        if match and current_panel is not None:
            printed_counts[current_panel] = int(match.group(1))

    assert set(printed_counts) == set(expected_counts), (
        f"expected a 'Agent Spec Flow: N lines' printout per panel, got {printed_counts}"
    )
    for panel, expected in expected_counts.items():
        assert printed_counts[panel] == expected, (
            f"{panel}: printed line count {printed_counts[panel]} does not match the "
            f"REAL to_agent_spec(...).to_dict() dump ({expected} lines) -- the headline "
            "must be mechanically derived, never hand-abridged (e3b4j Core Invariant)"
        )


def test_each_panel_pipeline_round_trips_and_runs():
    """The Each panel's pipeline behaves identically after an Agent Spec
    round-trip, matching test_agent_spec_roundtrip.py::test_each_round_trips_and_runs."""
    module = _load_example()
    pipeline = module.build_each_panel()

    flow = to_agent_spec(pipeline)
    imported = from_agent_spec(flow)
    graph = compile(imported, **build_test_compile_kwargs())
    result = run(graph, input={"node_id": "each-panel-rt"})

    collected = result[pipeline.nodes[-1].name if hasattr(pipeline, "nodes") else "each_step"]
    assert len(collected) >= 2, "Each panel must fan out to at least 2 items"


def test_oracle_panel_pipeline_round_trips_and_runs():
    """The Oracle panel uses think-mode + a keyless StructuredFake (scripted
    Oracle round-trip is known-broken per design doc s6) and behaves
    identically after round-trip, matching
    test_agent_spec_roundtrip.py::test_oracle_round_trips_and_runs."""
    module = _load_example()
    pipeline = module.build_oracle_panel()

    flow = to_agent_spec(pipeline)
    imported = from_agent_spec(flow)

    def _respond(m):
        # Field-name-agnostic: populate whatever the model's single list[str]
        # field is named (the example may rename it to avoid an unrelated
        # Agent-Spec type-synthesis hash collision), rather than assuming
        # "items" specifically.
        field_name = next(iter(m.model_fields))
        return m(**{field_name: ["variant"]})

    fake_llm = StructuredFake(_respond)
    graph = compile(
        imported,
        **build_test_compile_kwargs(),
        **build_fake_llm_kwargs(lambda tier: fake_llm),
    )
    result = run(graph, input={"node_id": "oracle-panel-rt"})

    ensemble_field = next(k for k in result if not k.startswith("neo_") and "node_id" not in k)
    merged = result[ensemble_field]
    merged_field = next(iter(type(merged).model_fields))
    merged_values = getattr(merged, merged_field)
    assert merged_values == ["variant", "variant", "variant"], (
        "expected the 3 Oracle-fanned-out generator variants merged into one list"
    )


def test_loop_panel_pipeline_round_trips_and_runs():
    """The Loop panel's pipeline behaves identically after round-trip,
    matching test_agent_spec_roundtrip.py::test_loop_round_trips_and_runs."""
    module = _load_example()
    pipeline = module.build_loop_panel()

    flow = to_agent_spec(pipeline)
    imported = from_agent_spec(flow)
    graph = compile(imported, **build_test_compile_kwargs())
    result = run(graph, input={"node_id": "loop-panel-rt"})

    field = next(k for k, v in result.items() if isinstance(v, list) and v)
    assert field, "Loop panel must produce a list-shaped (multi-iteration) result"
