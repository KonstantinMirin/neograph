"""neograph.testing — auto-scaffolded test suites from pipeline definitions.

Introspects a Construct and generates a pytest-compatible test file
structured the way real pipelines are tested:

    1. Topology — structural invariants (auto-verified, zero human input)
    2. Per-node — one test per node, grouped by position in the pipeline
       Scripted: _get_sidecar + direct call + isinstance assertion
       LLM: metadata checks (outputs, mode, prompt)
    3. Sub-constructs — compile + run stubs
    4. E2E — full pipeline with fake LLM
    5. Sync — catches drift when nodes are added/removed

Matches the test patterns proven in production (piarch):
- Tests grouped by pipeline phase, not by abstract category
- Scripted nodes tested via _get_sidecar (direct function call)
- LLM nodes tested via metadata assertions + E2E
- Each node gets a pytest fixture stub for test data

Usage:
    neograph test-scaffold my_pipeline.py -o tests/test_my_pipeline.py

    from neograph.testing import scaffold_tests
    scaffold_tests(my_construct, output_path="tests/test_pipeline.py")
"""

from __future__ import annotations

import os
from typing import Any

from neograph.construct import Construct
from neograph.naming import field_name_for
from neograph.node import Node


def _node_info(node: Node) -> dict[str, Any]:
    """Extract test-relevant metadata from a node."""
    inputs_dict = node.inputs if isinstance(node.inputs, dict) else {}
    outputs_name = (
        node.outputs.__name__
        if hasattr(node.outputs, "__name__")
        else repr(node.outputs)
    )
    input_names = list(inputs_dict.keys()) if inputs_dict else []
    return {
        "name": node.name,
        "field": field_name_for(node.name),
        "mode": node.mode,
        "outputs": node.outputs,
        "outputs_name": outputs_name,
        "inputs_dict": inputs_dict,
        "input_names": input_names,
        "is_scripted": node.mode == "scripted",
        "is_llm": node.mode in ("think", "agent", "act"),
        "has_oracle": node.modifier_set.oracle is not None,
        "has_each": node.modifier_set.each is not None,
        "has_loop": node.modifier_set.loop is not None,
        "prompt": node.prompt,
        "model": node.model,
    }


def _collect_items(construct: Construct) -> list[dict[str, Any]]:
    """Collect all items (nodes + sub-constructs) from a construct."""
    items = []
    for item in construct.nodes:
        if isinstance(item, Node):
            info = _node_info(item)
            info["is_subconstruct"] = False
            items.append(info)
        elif isinstance(item, Construct):
            items.append({
                "name": item.name,
                "field": field_name_for(item.name),
                "is_subconstruct": True,
                "input_name": item.input.__name__ if item.input else "None",
                "output_name": item.output.__name__ if item.output else "None",
                "node_count": len(item.nodes),
            })
    return items


def _collect_edges(construct: Construct) -> list[tuple[str, str]]:
    """Collect declared edges (upstream → downstream)."""
    edges = []
    for item in construct.nodes:
        if not isinstance(item, Node) or not isinstance(item.inputs, dict):
            continue
        for upstream in item.inputs:
            edges.append((upstream, field_name_for(item.name)))
    return edges


def _group_into_phases(
    items: list[dict[str, Any]],
    max_phase_size: int = 6,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group items into phases for test organization.

    Phase boundaries occur at:
    - Sub-construct boundaries
    - Each/Oracle modifier boundaries
    - Every max_phase_size nodes (natural chunking)
    """
    phases: list[tuple[str, list[dict[str, Any]]]] = []
    current: list[dict[str, Any]] = []
    phase_idx = 0

    for item in items:
        # Sub-constructs get their own phase
        if item.get("is_subconstruct"):
            if current:
                phases.append((f"phase_{phase_idx}", current))
                phase_idx += 1
                current = []
            phases.append((f"sub_{item['field']}", [item]))
            continue

        # Modifier boundaries start a new phase
        if current and (item.get("has_each") or item.get("has_oracle")):
            phases.append((f"phase_{phase_idx}", current))
            phase_idx += 1
            current = []

        current.append(item)

        # Size limit
        if len(current) >= max_phase_size:
            phases.append((f"phase_{phase_idx}", current))
            phase_idx += 1
            current = []

    if current:
        phases.append((f"phase_{phase_idx}", current))

    return phases


def scaffold_tests(
    construct: Construct,
    output_path: str,
    *,
    construct_import: str | None = None,
    overwrite: bool = False,
) -> str:
    """Generate a pytest test file from a Construct definition.

    Args:
        construct: The pipeline to scaffold tests for.
        output_path: Where to write the test file.
        construct_import: Import path (e.g., "my_app.pipeline"). Placeholder if None.
        overwrite: Overwrite existing file. Default False.

    Returns:
        The generated test file content.
    """
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Use overwrite=True or delete it first."
        )

    items = _collect_items(construct)
    edges = _collect_edges(construct)
    phases = _group_into_phases(items)
    real_nodes = [i for i in items if not i.get("is_subconstruct")]
    sub_constructs = [i for i in items if i.get("is_subconstruct")]
    has_llm = any(i.get("is_llm") for i in real_nodes)

    construct_var = field_name_for(construct.name)
    mod_import = construct_import or "YOUR_MODULE"
    L = []  # output lines

    # ── Header ───────────────────────────────────────────────────────
    L.append(f'"""Tests for construct \'{construct.name}\'.')
    L.append(f"")
    L.append(f"Auto-scaffolded by neograph.testing.scaffold_tests().")
    L.append(f"Nodes: {len(real_nodes)} | Sub-constructs: {len(sub_constructs)} | Edges: {len(edges)}")
    L.append(f"")
    L.append(f"Structure:")
    L.append(f"  TestTopology          — structural invariants (auto-verified)")
    for phase_name, phase_items in phases:
        node_names = ", ".join(i["name"] for i in phase_items)
        L.append(f"  Test{_class_name(phase_name):20s} — {node_names}")
    L.append(f"  TestEndToEnd          — full pipeline with fake LLM")
    L.append(f"  TestSync              — drift detection")
    L.append(f"")
    L.append(f"Tests with pytest.fail('TODO') need implementation.")
    L.append(f'"""')
    L.append(f"")
    L.append(f"from __future__ import annotations")
    L.append(f"")
    L.append(f"import pytest")
    L.append(f"from neograph._sidecar import _get_sidecar")
    L.append(f"")
    if construct_import:
        L.append(f"from {construct_import} import {construct_var}")
    else:
        L.append(f"# TODO: fill in imports")
        L.append(f"# from {mod_import} import {construct_var}")
        for i in real_nodes:
            L.append(f"# from {mod_import} import {i['field']}")
    L.append(f"")
    L.append(f"")

    # ── Topology ─────────────────────────────────────────────────────
    _section(L, "TOPOLOGY — auto-verified, no human input needed")
    L.append(f"class TestTopology:")
    L.append(f'    """Structural invariants for \'{construct.name}\'."""')
    L.append(f"")
    L.append(f"    def test_node_count(self):")
    L.append(f"        assert len({construct_var}.nodes) == {len(construct.nodes)}")
    L.append(f"")
    L.append(f"    def test_all_nodes_present(self):")
    L.append(f"        names = [getattr(n, 'name', '') for n in {construct_var}.nodes]")
    for i in items:
        L.append(f'        assert "{i["name"]}" in names')
    L.append(f"")

    if edges:
        L.append(f"    def test_topological_ordering(self):")
        L.append(f"        names = [getattr(n, 'name', '') for n in {construct_var}.nodes]")
        for up, down in edges:
            L.append(f'        assert names.index("{up.replace("_", "-")}") < names.index("{down.replace("_", "-")}")')
        L.append(f"")

    L.append(f"    def test_compiles(self):")
    L.append(f"        from neograph import compile")
    if has_llm:
        L.append(f"        # {sum(1 for n in real_nodes if n.get('is_llm'))} LLM node(s) — configure_llm required before compile")
        L.append(f'        pytest.fail("TODO: configure_llm then compile")')
    else:
        L.append(f"        graph = compile({construct_var})")
        L.append(f"        assert graph is not None")
    L.append(f"")
    L.append(f"")

    # ── Per-phase node tests ─────────────────────────────────────────
    for phase_name, phase_items in phases:
        class_name = _class_name(phase_name)
        node_names = ", ".join(i["name"] for i in phase_items)
        _section(L, f"{class_name} — {node_names}")
        L.append(f"class Test{class_name}:")
        L.append(f'    """Tests for: {node_names}"""')
        L.append(f"")

        for item in phase_items:
            if item.get("is_subconstruct"):
                _gen_subconstruct_tests(L, item)
            elif item.get("is_scripted"):
                _gen_scripted_node_test(L, item)
            elif item.get("is_llm"):
                _gen_llm_node_test(L, item)

        L.append(f"")

    # ── Fixtures ─────────────────────────────────────────────────────
    _section(L, "FIXTURES — test data for each node")
    L.append(f"# Fill these in with realistic test data.")
    L.append(f"# Each fixture returns the input type for a specific node.")
    L.append(f"")
    for i in real_nodes:
        if i.get("is_scripted") and i.get("input_names"):
            L.append(f"")
            L.append(f"@pytest.fixture")
            L.append(f"def {i['field']}_input():")
            L.append(f'    """Test input for node \'{i["name"]}\'.')
            L.append(f"    Upstream(s): {', '.join(i['input_names'])}")
            L.append(f'    """')
            L.append(f'    pytest.fail("TODO: return {i["input_names"][0]} fixture")')
    L.append(f"")
    L.append(f"")

    # ── E2E ──────────────────────────────────────────────────────────
    _section(L, "END-TO-END — full pipeline with fake LLM")
    L.append(f"class TestEndToEnd:")
    L.append(f'    """Full pipeline E2E."""')
    L.append(f"")
    if has_llm:
        L.append(f"    # Fake response registry — one entry per LLM node:")
        for n in real_nodes:
            if n.get("is_llm"):
                L.append(f'    #   "{n["name"]}": {n["outputs_name"]}(...)')
        L.append(f"")
    L.append(f"    def test_full_pipeline(self):")
    L.append(f'        """Compile and run \'{construct.name}\' end-to-end."""')
    L.append(f'        pytest.fail("TODO: configure_llm, compile, run, assert outputs")')
    L.append(f"")
    L.append(f"")

    # ── Sync ─────────────────────────────────────────────────────────
    _section(L, "SYNC — catches drift between construct and tests")
    L.append(f"class TestSync:")
    L.append(f'    """Fails if nodes are added/removed without updating tests."""')
    L.append(f"")
    L.append(f"    EXPECTED_NODES = {{")
    for i in items:
        L.append(f'        "{i["name"]}",')
    L.append(f"    }}")
    L.append(f"")
    L.append(f"    def test_no_untested_nodes(self):")
    L.append(f"        actual = {{getattr(n, 'name', '') for n in {construct_var}.nodes}}")
    L.append(f"        missing = actual - self.EXPECTED_NODES")
    L.append(f'        assert not missing, f"New nodes need tests: {{sorted(missing)}}"')
    L.append(f"")
    L.append(f"    def test_no_stale_tests(self):")
    L.append(f"        actual = {{getattr(n, 'name', '') for n in {construct_var}.nodes}}")
    L.append(f"        stale = self.EXPECTED_NODES - actual")
    L.append(f'        assert not stale, f"Removed nodes still in tests: {{sorted(stale)}}"')
    L.append(f"")

    content = "\n".join(L)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)
    return content


# ── Helpers ──────────────────────────────────────────────────────────────

def _section(lines: list[str], title: str) -> None:
    lines.append(f"# {'=' * 70}")
    lines.append(f"# {title}")
    lines.append(f"# {'=' * 70}")
    lines.append(f"")
    lines.append(f"")

def _class_name(phase: str) -> str:
    return "".join(word.capitalize() for word in phase.split("_"))


def _gen_scripted_node_test(L: list[str], n: dict) -> None:
    """Generate test stub for a scripted node using _get_sidecar."""
    fname = n["field"]
    L.append(f"    def test_{fname}(self):")
    L.append(f'        """Node \'{n["name"]}\' (scripted) → {n["outputs_name"]}"""')
    L.append(f"        fn, _ = _get_sidecar({fname})")
    L.append(f"        assert fn is not None, 'no sidecar — was @node used?'")

    if n["input_names"]:
        L.append(f"        # Inputs: {', '.join(n['input_names'])}")
        L.append(f"        # result = fn({', '.join(n['input_names'])})")
        L.append(f"        # assert isinstance(result, {n['outputs_name']})")
    else:
        L.append(f"        # result = fn()")
        L.append(f"        # assert isinstance(result, {n['outputs_name']})")

    L.append(f'        pytest.fail("TODO: provide inputs, call fn, assert result")')
    L.append(f"")


def _gen_llm_node_test(L: list[str], n: dict) -> None:
    """Generate metadata test for an LLM node."""
    fname = n["field"]
    modifiers = []
    if n.get("has_oracle"):
        modifiers.append("Oracle")
    if n.get("has_each"):
        modifiers.append("Each")
    if n.get("has_loop"):
        modifiers.append("Loop")
    mod_str = f" | {' | '.join(modifiers)}" if modifiers else ""

    L.append(f"    def test_{fname}(self):")
    L.append(f'        """Node \'{n["name"]}\' ({n["mode"]}{mod_str}) → {n["outputs_name"]}"""')
    L.append(f"        assert {fname}.outputs is {n['outputs_name']}")
    L.append(f'        assert {fname}.mode == "{n["mode"]}"')
    if n.get("prompt"):
        L.append(f'        assert {fname}.prompt == "{n["prompt"]}"')
    if n.get("model"):
        L.append(f'        assert {fname}.model == "{n["model"]}"')
    L.append(f"")


def _gen_subconstruct_tests(L: list[str], sc: dict) -> None:
    """Generate compile + E2E stubs for a sub-construct."""
    fname = sc["field"]
    L.append(f"    def test_{fname}_compiles(self):")
    L.append(f'        """Sub-construct \'{sc["name"]}\' ({sc["node_count"]} nodes, '
             f'input={sc["input_name"]}, output={sc["output_name"]}) compiles."""')
    L.append(f'        pytest.fail("TODO: compile sub-construct")')
    L.append(f"")
    L.append(f"    def test_{fname}_e2e(self):")
    L.append(f'        """Sub-construct \'{sc["name"]}\' runs end-to-end."""')
    L.append(f'        pytest.fail("TODO: provide {sc["input_name"]} fixture, run, assert {sc["output_name"]}")')
    L.append(f"")
