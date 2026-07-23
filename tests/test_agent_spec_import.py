"""Regression tests for ``from_agent_spec()`` -- Agent Spec ``Flow`` ->
neograph ``Construct``/IR import (neograph-01i0g).

Gated on ``pyagentspec`` via ``pytest.importorskip`` -- the ``[agent-spec]``
optional extra keeps ``src/neograph`` core dependency-light by default. Run
with::

    uv run --extra dev --extra agent-spec pytest tests/test_agent_spec_import.py

TDD red step: ``from_agent_spec`` does not exist yet anywhere in
``src/neograph`` (neither ``neograph.loader`` nor the top-level
``neograph.__all__``), so every test below currently fails with
``ImportError``/``AttributeError`` -- confirmed by running pytest, not by
inspection.

Per the ratified user decision resolving 01i0g's NEEDS_USER_INPUT fork
(2026-07-22): Swarm/handoff import goes onto a Portal mesh (not reject);
per the corrected staleness-detection strategy (f0j1e.38 refine), there is
NO ``Flow.metadata['neograph/source']`` whole-pipeline blob to deserialize
-- fidelity rides only on the PER-GROUP ``neograph/*_spec`` markers that
``to_agent_spec`` (``_agent_spec.py``) emits, and this suite must not assert
a source-blob path exists.

This is the *write-test* atom for 01i0g (neograph-f0j1e.14) -- it establishes
TDD red with a meaningful first test file (plain primitive round-trip +
one per-group marker reconstruction case, Oracle), not an exhaustive
enumeration of every modifier/branch/Swarm/RemoteAgent case. The full matrix
extends during implementation (neograph-f0j1e.15) per the bead's own scope
note.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyagentspec")

from neograph import Construct  # noqa: E402
from neograph.modifiers import ModifierCombo, classify_modifiers  # noqa: E402
from neograph.node import Node  # noqa: E402

from .schemas import Claims, RawText, _consumer, _producer  # noqa: E402


class TestFromAgentSpecExistsAndIsExported:
    """Pins that ``from_agent_spec`` is a real, importable, top-level-exported
    free function -- symmetric with ``to_agent_spec``'s own export pin
    (``test_agent_spec_export.py::test_to_agent_spec_is_exported_from_neograph_top_level``).

    This currently fails with ``ImportError``/``AttributeError`` because
    ``from_agent_spec`` does not exist in ``neograph.loader`` (or anywhere)
    yet.
    """

    def test_from_agent_spec_is_exported_from_neograph_top_level(self):
        import neograph

        assert "from_agent_spec" in neograph.__all__, (
            "from_agent_spec must be a free function re-exported through "
            "neograph/__init__.py's __all__ (layer discipline: not a "
            "Construct/Node method), sibling of load_spec/to_agent_spec"
        )
        assert hasattr(neograph, "from_agent_spec")

    def test_from_agent_spec_is_importable_from_loader_module(self):
        # loader.py is the designated home per the bead's implementation
        # plan point 1 ("sibling of load_spec").
        from neograph.loader import from_agent_spec  # noqa: F401


class TestFromAgentSpecPlainPrimitiveRoundTrip:
    """Pins the minimum bar: a markerless (foreign-or-plain) two-node
    scripted chain exported via ``to_agent_spec`` and re-imported via
    ``from_agent_spec`` yields an equivalent-shape ``Construct`` -- same
    node names, in the same order, with the data-flow-derived ``inputs``
    mapping preserved.

    This is a round trip through the REAL to_agent_spec exporter (i3zsh,
    already shipped), not a hand-built Flow -- so it exercises the exact
    marker-free/plain-primitive shape from_agent_spec's fallback path must
    handle for foreign Agent Specs.
    """

    def test_two_node_scripted_chain_round_trips_through_export_and_import(self):
        from neograph._agent_spec import to_agent_spec
        from neograph.loader import from_agent_spec

        seed = _producer("seed", RawText)
        summarize = _consumer("summarize", RawText, Claims)
        pipeline = Construct("two-node-chain", nodes=[seed, summarize])

        flow = to_agent_spec(pipeline)
        imported = from_agent_spec(flow)

        assert isinstance(imported, Construct)
        imported_names = [n.name for n in imported.nodes]
        assert imported_names == ["seed", "summarize"], (
            "expected import to preserve Construct.nodes order from the "
            "Flow's start_node-reachable topological walk"
        )

        summarize_node = next(n for n in imported.nodes if n.name == "summarize")
        assert isinstance(summarize_node, Node)
        assert summarize_node.mode == "scripted"
        # DataFlowEdge seed -> summarize must resurface as an inputs mapping
        # naming "seed" as the upstream producer key (doc s5 structural table).
        assert summarize_node.inputs is not None
        if isinstance(summarize_node.inputs, dict):
            assert "seed" in summarize_node.inputs
        else:
            assert summarize_node.inputs is RawText

    def test_imported_construct_compiles(self):
        """The round-tripped Construct must be a real, compilable IR -- not
        a special-cased shape the compiler chokes on (doc s7 layer rule:
        from_agent_spec produces Node/Construct instances indistinguishable
        from any of the three existing surfaces)."""
        from neograph import compile as neograph_compile
        from neograph._agent_spec import to_agent_spec
        from neograph.loader import from_agent_spec

        seed = _producer("seed", RawText)
        summarize = _consumer("summarize", RawText, Claims)
        pipeline = Construct("two-node-chain", nodes=[seed, summarize])

        flow = to_agent_spec(pipeline)
        imported = from_agent_spec(flow)

        # _producer/_consumer's scripted_fn="f" is name-only (compile()
        # requires a real registration regardless of provenance -- this is
        # true of the ORIGINAL, non-round-tripped pipeline too, not an
        # import-side defect); supply a no-op shim so compile() only checks
        # the STRUCTURAL shape from_agent_spec reconstructed.
        neograph_compile(imported, scripted={"f": lambda *a, **k: RawText(text="")})


class TestFromAgentSpecReconstructsOracleMarker:
    """Pins the per-group marker reconstruction path (Implementation Plan
    point 4): when a group's ``neograph/oracle_spec`` marker is present AND
    matches the actual flattened variant/merge primitives around it,
    ``from_agent_spec`` must reconstruct the abstract ``Oracle`` modifier
    losslessly -- not leave the import at the expanded (variant nodes +
    merge node) primitive level.

    This is the representative "per-group modifier reconstruction" case the
    write-test atom's scope calls for as the minimum bar alongside the plain
    primitive round-trip above.
    """

    def test_oracle_marker_group_reconstructs_to_oracle_modified_node(self):
        from neograph._agent_spec import to_agent_spec
        from neograph.loader import from_agent_spec
        from neograph.modifiers import Oracle

        node = Node(name="ensemble", mode="think", model="fast", outputs=Claims, prompt="rw/ensemble")
        node = node | Oracle(n=2, merge_fn="combine")
        pipeline = Construct("oracle-pipeline", nodes=[node])

        flow = to_agent_spec(pipeline)
        imported = from_agent_spec(flow)

        assert isinstance(imported, Construct)
        # Reconstruction collapses the exported (2 variants + 1 merge = 3
        # spec nodes) group back down to ONE Oracle-modified Construct item
        # named after the original node, not three separate primitive nodes.
        matching = [item for item in imported.nodes if getattr(item, "name", None) == "ensemble"]
        assert len(matching) == 1, (
            f"expected the Oracle group to collapse back to one 'ensemble' "
            f"item, got {[getattr(i, 'name', i) for i in imported.nodes]!r}"
        )
        reconstructed = matching[0]
        combo, mods = classify_modifiers(reconstructed)
        assert combo == ModifierCombo.ORACLE, (
            f"expected the neograph/oracle_spec marker (group_id shared "
            f"across variant + merge nodes) to reconstruct an Oracle "
            f"modifier, got combo={combo.name}"
        )
        oracle = mods["oracle"]
        assert oracle.n == 2
        assert oracle.merge_fn == "combine"


class TestFromAgentSpecReconstructsAgentNode:
    """INVERTED from the former fail-loud pin (neograph-aa5gq): now that
    neograph-i3zsh.1's ``neograph/agent_spec`` marker is consumed on import,
    an ``AgentNode`` must reconstruct the EXACT agent/act Node losslessly --
    never silently downgrade to a plain ToolNode/scripted stand-in (which
    would drop the ReAct tool-loop semantics), and never fail loud.

    The former ``TestFromAgentSpecFailsLoudOnAgentNode`` asserted the OPPOSITE
    (``ConfigurationError`` on AgentNode); aa5gq's Implementation Plan point 1
    explicitly inverts that pin into this losslessness assertion. The
    exhaustive field-by-field + tool-budget/config/idempotent coverage lives
    in ``tests/test_agent_spec_roundtrip.py::TestAgentNodeRoundTripLosslessness``.
    """

    def test_agent_node_in_flow_reconstructs_to_agent_mode_node(self):
        from neograph._agent_spec import to_agent_spec
        from neograph.loader import from_agent_spec
        from neograph.node import Node
        from neograph.tool import Tool

        node = Node(
            name="explore",
            mode="agent",
            model="research",
            prompt="explore the codebase",
            outputs=RawText,
            tools=[Tool("search_code")],
        )
        pipeline = Construct("agent-pipeline", nodes=[node])

        flow = to_agent_spec(pipeline)  # exports to a real AgentNode (i3zsh.1)

        imported = from_agent_spec(flow)
        reconstructed = next(n for n in imported.nodes if getattr(n, "name", None) == "explore")
        assert reconstructed.mode == "agent"
        assert reconstructed.prompt == "explore the codebase"
        assert reconstructed.model == "research"
        assert reconstructed.tools is not None
        assert [t.name for t in reconstructed.tools] == ["search_code"]
