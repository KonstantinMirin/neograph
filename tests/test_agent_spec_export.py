"""Regression tests for ``to_agent_spec()`` -- neograph ``Construct``/IR ->
Agent Spec ``Flow`` export (neograph-i3zsh).

Gated on ``pyagentspec`` via ``pytest.importorskip`` -- the ``[agent-spec]``
optional extra keeps ``src/neograph`` core dependency-light by default. Run
with::

    uv run --extra agent-spec pytest tests/test_agent_spec_export.py

## Step 1 gate (i3zsh implementation plan, sequenced FIRST)

Per the ratification's residual/unverified item (agent-spec-ratification-
2026-07-13.md s6) and i3zsh's own "Risks & Edge Cases" note: the whole
Layer A/B ``neograph/``-prefixed ``metadata`` marker round-trip strategy
(stamping ``metadata[_MARK_MODIFIER]`` etc. on lowered nodes so an
export stays a lossless neograph round-trip source) depends on
``Component.metadata`` actually surviving a real ``pyagentspec``
``to_dict -> from_dict`` cycle -- including disaggregated-component export,
where a referenced sub-component (e.g. a ``Tool``) is serialized separately
and re-attached via a ``components_registry``. ``TestMetadataMarkerRoundTripSurvivesRealPyagentspec``
proves this directly against the REAL installed package (no neograph
involvement at all) -- this is the smoke-test gate the implementation plan's
step 1 calls for, written and run BEFORE trusting the marker strategy for
anything beyond primitive-level export.

## Primary regression tests (TDD red for i3zsh)

``TestToAgentSpecExportsFlow`` is the pin for ``to_agent_spec()`` itself --
it currently fails because neither ``neograph._agent_spec`` nor
``to_agent_spec`` exist yet. It asserts the Core Invariant's DIRECT
structural mapping: a flat scripted-node ``Construct`` lowers to a
``pyagentspec.flows.flow.Flow`` with one node per neograph ``Node``, an
explicit ``ControlFlowEdge`` per adjacent pair in ``Construct.nodes`` order,
and an explicit ``DataFlowEdge`` per ``Node.inputs`` upstream-name mapping.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyagentspec")

from neograph import Construct  # noqa: E402
from neograph._agent_spec import (  # noqa: E402
    _MARK_AGENT_SPEC,
    _MARK_EACH_SPEC,
    _MARK_GROUP_ID,
    _MARK_LOOP_SPEC,
    _MARK_MODIFIER,
    _MARK_OPERATOR_SPEC,
    _MARK_PORTAL_SPEC,
    _MARK_TOOL_SPEC,
    _MARK_VARIANT,
    Branch,
)

from .agent_spec_flow_walk import holder_flows  # noqa: E402
from .schemas import Claims, ClusterGroup, Clusters, MatchResult, RawText, _consumer, _producer  # noqa: E402


class TestMetadataMarkerRoundTripSurvivesRealPyagentspec:
    """Step-1 gate: does ``Component.metadata`` survive a real pyagentspec
    ``to_dict -> from_dict`` cycle, including disaggregated components?

    Pure pyagentspec -- no neograph import at all. This is the prerequisite
    the Layer A/B ``neograph/``-prefixed marker strategy depends on; it is
    NOT itself a test of ``to_agent_spec()`` (which doesn't exist yet).
    """

    def _build_minimal_flow(self, *, tool_metadata: dict[str, str]):
        from pyagentspec.flows.edges import ControlFlowEdge, DataFlowEdge
        from pyagentspec.flows.flow import Flow
        from pyagentspec.flows.nodes import EndNode, StartNode, ToolNode
        from pyagentspec.property import Property
        from pyagentspec.tools import ServerTool

        x_property = Property(json_schema={"title": "x", "type": "number"})
        y_property = Property(json_schema={"title": "y", "type": "number"})
        tool = ServerTool(
            name="compute",
            description="computes y from x",
            inputs=[x_property],
            outputs=[y_property],
            metadata=tool_metadata,
        )
        start_node = StartNode(name="start", inputs=[x_property])
        end_node = EndNode(name="end", outputs=[y_property])
        tool_node = ToolNode(
            name="compute_node",
            tool=tool,
            metadata={_MARK_MODIFIER: "oracle", _MARK_GROUP_ID: "g1"},
        )
        flow = Flow(
            name="minimal flow",
            start_node=start_node,
            nodes=[start_node, tool_node, end_node],
            metadata={"neograph/source": "FLOW_LEVEL_MARKER"},
            control_flow_connections=[
                ControlFlowEdge(name="start_to_tool", from_node=start_node, to_node=tool_node),
                ControlFlowEdge(name="tool_to_end", from_node=tool_node, to_node=end_node),
            ],
            data_flow_connections=[
                DataFlowEdge(
                    name="x_to_tool",
                    source_node=start_node,
                    source_output="x",
                    destination_node=tool_node,
                    destination_input="x",
                ),
                DataFlowEdge(
                    name="tool_to_end_data",
                    source_node=tool_node,
                    source_output="y",
                    destination_node=end_node,
                    destination_input="y",
                ),
            ],
        )
        return flow, tool, tool_node

    def test_flow_and_node_metadata_survive_plain_to_dict_from_dict_round_trip(self):
        from pyagentspec.flows.flow import Flow

        flow, _tool, _tool_node = self._build_minimal_flow(
            tool_metadata={"neograph/tool_marker": "present"}
        )

        serialized = flow.to_dict()
        rebuilt = Flow.from_dict(serialized)

        assert rebuilt.metadata == {"neograph/source": "FLOW_LEVEL_MARKER"}
        rebuilt_tool_node = next(n for n in rebuilt.nodes if n.name == "compute_node")
        assert rebuilt_tool_node.metadata == {
            _MARK_MODIFIER: "oracle",
            _MARK_GROUP_ID: "g1",
        }

    def test_metadata_survives_disaggregated_component_round_trip(self):
        from pyagentspec.flows.flow import Flow
        from pyagentspec.serialization import AgentSpecDeserializer

        flow, tool, tool_node = self._build_minimal_flow(
            tool_metadata={"neograph/tool_marker": "present"}
        )

        main, disaggregated = flow.to_dict(
            disaggregated_components=[(tool, "tool_ref_1")],
            export_disaggregated_components=True,
        )
        disaggregated_components = AgentSpecDeserializer().from_dict(
            disaggregated, import_only_referenced_components=True
        )
        rebuilt = Flow.from_dict(main, components_registry=disaggregated_components)

        assert rebuilt.metadata == {"neograph/source": "FLOW_LEVEL_MARKER"}
        rebuilt_tool_node = next(n for n in rebuilt.nodes if n.name == "compute_node")
        assert rebuilt_tool_node.metadata == {
            _MARK_MODIFIER: "oracle",
            _MARK_GROUP_ID: "g1",
        }
        assert rebuilt_tool_node.tool.metadata == {"neograph/tool_marker": "present"}


class TestToAgentSpecExportsFlow:
    """Pins ``to_agent_spec()``'s Core Invariant DIRECT structural mapping for
    a flat scripted-node chain: neograph ``Construct`` -> ``pyagentspec``
    ``Flow`` with one node per ``Node``, ``ControlFlowEdge`` per adjacent
    pair, ``DataFlowEdge`` per ``Node.inputs`` upstream-name mapping.

    This currently fails because ``neograph._agent_spec`` (and its
    ``to_agent_spec`` free function) do not exist yet -- confirmed by running
    pytest, not by inspection.
    """

    def test_two_node_scripted_chain_lowers_to_flow_with_control_and_data_edges(self):
        from neograph._agent_spec import to_agent_spec

        seed = _producer("seed", RawText)
        summarize = _consumer("summarize", RawText, Claims)
        pipeline = Construct("two-node-chain", nodes=[seed, summarize])

        flow = to_agent_spec(pipeline)

        from pyagentspec.flows.edges import ControlFlowEdge, DataFlowEdge
        from pyagentspec.flows.flow import Flow

        assert isinstance(flow, Flow)

        node_names = {n.name for n in flow.nodes}
        assert "seed" in node_names
        assert "summarize" in node_names

        control_edges = [e for e in flow.control_flow_connections if isinstance(e, ControlFlowEdge)]
        assert any(
            e.from_node.name == "seed" and e.to_node.name == "summarize" for e in control_edges
        ), "expected an explicit ControlFlowEdge seed -> summarize, one per Construct.nodes order"

        data_edges = [e for e in (flow.data_flow_connections or []) if isinstance(e, DataFlowEdge)]
        assert any(
            e.source_node.name == "seed" and e.destination_node.name == "summarize"
            for e in data_edges
        ), (
            "expected an explicit DataFlowEdge seed -> summarize derived from "
            "summarize.inputs={'seed': RawText}"
        )

    def test_to_agent_spec_is_exported_from_neograph_top_level(self):
        import neograph

        assert "to_agent_spec" in neograph.__all__, (
            "to_agent_spec must be a free function re-exported through "
            "neograph/__init__.py's __all__ (layer discipline: not a "
            "Construct/Node method)"
        )
        assert hasattr(neograph, "to_agent_spec")


class TestToAgentSpecExportsDictFormFanIn:
    """Pins ``to_agent_spec()`` against @node's PRIMARY fan-in shape: a
    downstream node with a typed upstream param (``def f(seed: A)``) compiles
    to dict-form ``inputs={'seed': A}`` (keyed by upstream NODE NAME, not by
    output property title). ``TestToAgentSpecExportsFlow`` above only covers
    the single-type ``inputs=RawText`` shorthand via the ``_consumer`` helper
    -- never this dict-form shape, which is the dominant one @node produces.

    Reproduces neograph-ozxqw: ``_agent_spec.py``'s dict-form fan-in branch
    sets ``DataFlowEdge.source_output=upstream_name`` (the inputs-dict KEY =
    the upstream node's NAME), instead of the source node's real exported
    output Property TITLE -- so pyagentspec's own DataFlowEdge validator
    raises a raw ``pydantic.ValidationError`` ("does not have any property
    with that name") for any two-node dict-form-fan-in chain, even though the
    pipeline compiles and runs fine in neograph.
    """

    def test_two_node_dict_form_fan_in_chain_exports_without_raising(self):
        from neograph._agent_spec import to_agent_spec
        from neograph.node import Node

        seed = _producer("seed", RawText)
        summarize = Node.scripted("summarize", fn="g", inputs={"seed": RawText}, outputs=Claims)
        pipeline = Construct("dict-fanin-chain", nodes=[seed, summarize])

        # Must not raise pydantic.ValidationError -- to_agent_spec's fan-in
        # edge must reference a REAL property of the source node.
        flow = to_agent_spec(pipeline)

        from pyagentspec.flows.edges import DataFlowEdge

        seed_spec_node = next(n for n in flow.nodes if n.name == "seed")
        seed_output_titles = {p.title for p in (seed_spec_node.outputs or [])}
        assert seed_output_titles, "expected 'seed' to export at least one output Property"

        data_edges = [e for e in (flow.data_flow_connections or []) if isinstance(e, DataFlowEdge)]
        fanin_edge = next(
            e for e in data_edges if e.source_node.name == "seed" and e.destination_node.name == "summarize"
        )
        assert fanin_edge.source_output in seed_output_titles, (
            f"DataFlowEdge.source_output={fanin_edge.source_output!r} must be a real output "
            f"Property of node 'seed' ({seed_output_titles}), not the raw inputs-dict key "
            "'seed' (the upstream NODE NAME)"
        )


class TestToAgentSpecRejectsUnrepresentableFields:
    """Pins the Core Invariant's fail-loud contract: a construct that cannot
    be lowered must raise ``ConfigurationError``, never silently downgrade.
    """

    def test_raw_fn_node_is_rejected(self):
        from neograph._agent_spec import to_agent_spec
        from neograph.errors import ConfigurationError
        from neograph.node import Node

        node = Node(name="raw", mode="scripted", outputs=RawText, raw_fn=lambda state, config: state)
        pipeline = Construct("raw-pipeline", nodes=[node])

        with pytest.raises(ConfigurationError, match="raw_fn"):
            to_agent_spec(pipeline)

    def test_skip_when_node_is_rejected(self):
        from neograph._agent_spec import to_agent_spec
        from neograph.errors import ConfigurationError

        node = _producer("seed", RawText)
        node = node.model_copy(update={"skip_when": lambda d: False})
        pipeline = Construct("skip-pipeline", nodes=[node])

        with pytest.raises(ConfigurationError, match="skip_when"):
            to_agent_spec(pipeline)

    def test_portal_handoff_member_is_rejected(self):
        from neograph._agent_spec import to_agent_spec
        from neograph.errors import ConfigurationError

        node = _producer("seed", RawText)
        node = node.model_copy(update={"handoff_param": "handoff", "handoff_channel": "neo_handoff_seed"})
        pipeline = Construct("handoff-pipeline", nodes=[node])

        with pytest.raises(ConfigurationError, match="handoff"):
            to_agent_spec(pipeline)

    def test_callable_gate_tools_when_is_rejected(self):
        from neograph._agent_spec import to_agent_spec
        from neograph.errors import ConfigurationError

        node = _producer("seed", RawText).model_copy(update={"mode": "agent", "gate_tools_when": lambda s: True})
        pipeline = Construct("gate-pipeline", nodes=[node])

        with pytest.raises(ConfigurationError, match="gate_tools_when"):
            to_agent_spec(pipeline)


class TestToAgentSpecLowersAgentActMode:
    """Pins neograph-i3zsh.1's EXPORT-SIDE-ONLY acceptance criteria (re-scoped
    2026-07-22 per architect review, see neograph-f0j1e.36): an agent/act mode
    Node must lower to a real ``pyagentspec`` ``AgentNode``+``Agent``+
    ``ServerTool`` composite -- never the fail-loud placeholder it replaces --
    AND stamp a ``neograph/agent_spec`` marker carrying every field a future
    importer needs to reconstruct the node losslessly (mode, prompt, model,
    tools incl. budget/config/idempotent, gate_tools_when string form,
    context).

    NOTE: this test does NOT exercise an actual export -> import round trip.
    No ``from_agent_spec()`` importer exists yet in this codebase; that is
    EXPLICITLY DEFERRED to neograph-01i0g, which owns the importer and depends
    on this task. This test only proves the marker is lossless-IN-PRINCIPLE
    (contains every field an importer would need) plus JSON-serializability,
    per neograph-i3zsh.1's re-scoped acceptance criteria.
    """

    @pytest.mark.parametrize("mode", ["agent", "act"])
    def test_agent_act_node_lowers_to_agent_node_not_tool_node(self, mode):
        from neograph._agent_spec import to_agent_spec
        from neograph.node import Node
        from neograph.tool import Tool

        node = Node(
            name="explore",
            mode=mode,
            model="research",
            prompt="explore the codebase",
            outputs=RawText,
            tools=[Tool("search_code", budget=5, idempotent=True), Tool("write_file", config={"root": "/tmp"})],
        )
        pipeline = Construct("agent-pipeline", nodes=[node])

        flow = to_agent_spec(pipeline)

        from pyagentspec.flows.nodes import AgentNode, ToolNode

        spec_node = next(n for n in flow.nodes if n.name == "explore")
        assert isinstance(spec_node, AgentNode), (
            f"agent/act mode node must lower to a pyagentspec AgentNode, not {type(spec_node).__name__} "
            "-- the ToolNode placeholder silently dropped prompt/model/tools"
        )
        assert not isinstance(spec_node, ToolNode)

        assert spec_node.agent.system_prompt == "explore the codebase"
        assert spec_node.agent.llm_config.model_id == "research"
        tool_names = {t.name for t in spec_node.agent.tools}
        assert tool_names == {"search_code", "write_file"}

    @pytest.mark.parametrize("mode", ["agent", "act"])
    def test_agent_act_marker_carries_every_reconstruction_field(self, mode):
        """The neograph/agent_spec marker must carry every field the plain
        Agent/ServerTool primitives cannot represent, so a future
        from_agent_spec() can rebuild the exact node -- and it must be
        actually JSON-serializable (no callable/_bound_tool leak)."""
        import json

        from neograph._agent_spec import to_agent_spec
        from neograph.node import Node
        from neograph.tool import Tool

        notes = _producer("explore_notes", RawText)
        node = Node(
            name="explore",
            mode=mode,
            model="research",
            prompt="explore the codebase",
            outputs=RawText,
            tools=[Tool("search_code", budget=5, idempotent=True, config={"depth": 2})],
            gate_tools_when="always",
            context=["explore_notes"],
        )
        pipeline = Construct("agent-pipeline", nodes=[notes, node])

        flow = to_agent_spec(pipeline)

        spec_node = next(n for n in flow.nodes if n.name == "explore")
        marker = spec_node.metadata[_MARK_AGENT_SPEC]

        assert marker["mode"] == mode
        assert marker["prompt"] == "explore the codebase"
        assert marker["model"] == "research"
        assert marker["gate_tools_when"] == "always"
        assert marker["context"] == ["explore_notes"]

        tool_entries = {t["name"]: t for t in marker["tools"]}
        assert tool_entries["search_code"]["budget"] == 5
        assert tool_entries["search_code"]["idempotent"] is True
        assert tool_entries["search_code"]["config"] == {"depth": 2}

        # Round-trip-losslessness IN PRINCIPLE (no importer yet, neograph-01i0g)
        # requires the marker to actually be JSON-serializable end to end --
        # a live _bound_tool or callable leaking through would silently break
        # any future from_agent_spec() reconstruction.
        json.dumps(marker)

    def test_callable_gate_tools_when_still_rejected_for_agent_mode(self):
        """Real lowering must not accidentally swallow the pre-existing
        callable-gate_tools_when rejection -- _reject_unrepresentable_fields
        still runs before the mode dispatch."""
        from neograph._agent_spec import to_agent_spec
        from neograph.errors import ConfigurationError

        node = _producer("seed", RawText).model_copy(
            update={"mode": "agent", "gate_tools_when": lambda s: True}
        )
        pipeline = Construct("gate-pipeline", nodes=[node])

        with pytest.raises(ConfigurationError, match="gate_tools_when"):
            to_agent_spec(pipeline)


class TestToAgentSpecPlaceholderInputGuard:
    """Pins the Option F placeholder translation (neograph-cbpyx, which AMENDS
    m57mn's Option-B fail-loud guard). A think/agent/act node's native
    ``${var}`` prompt is TRANSLATED to pyagentspec's ``{{ flat }}`` syntax --
    the two are the same flat text substitution, so there IS a faithful
    representation and rejection was the wrong call. The remaining fail-loud is
    narrower: a ``${path}`` whose first segment is not a declared input (a
    genuinely dangling reference) still raises a clean ``ConfigurationError``
    (never a leaked pyagentspec ``pydantic.ValidationError``), same convention as
    ``raw_fn``/``skip_when``/callable ``Loop.when``. (See
    tests/test_agent_spec_placeholder_translation.py for the full matrix incl.
    collision and round-trip.)
    """

    def test_bare_think_mode_node_with_dollar_ref_translates(self):
        """A single-type-input think node whose ${var} prompt references its input
        exports cleanly, with the prompt rewritten to {{ flat }} and the LlmNode
        declaring the referenced flat Property (Option F, not rejection)."""
        from pyagentspec.flows.nodes import LlmNode

        from neograph._agent_spec import to_agent_spec
        from neograph.node import Node

        prod = _producer("seed", RawText)  # RawText.text -> single-type title 'text'
        node = Node(name="think1", mode="think", model="fast", prompt="hello ${text}", inputs=RawText, outputs=Claims)
        pipeline = Construct("think-pipeline", nodes=[prod, node])

        flow = to_agent_spec(pipeline)

        spec_node = next(n for n in flow.nodes if n.name == "think1")
        assert isinstance(spec_node, LlmNode)
        assert spec_node.prompt_template == "hello {{ text }}"
        assert {p.title for p in (spec_node.inputs or [])} == {"text"}

    def test_bare_agent_mode_node_with_dollar_ref_translates(self):
        """agent-mode is translated the same way -- the Agent's system_prompt is
        rewritten and the AgentNode declares the referenced flat Property."""
        from pyagentspec.flows.nodes import AgentNode

        from neograph._agent_spec import to_agent_spec
        from neograph.node import Node

        prod = _producer("seed", RawText)
        node = Node(name="agent1", mode="agent", model="fast", prompt="hello ${text}", inputs=RawText, outputs=Claims)
        pipeline = Construct("agent-pipeline", nodes=[prod, node])

        flow = to_agent_spec(pipeline)

        spec_node = next(n for n in flow.nodes if n.name == "agent1")
        assert isinstance(spec_node, AgentNode)
        assert {p.title for p in (spec_node.inputs or [])} == {"text"}
        assert spec_node.agent.system_prompt == "hello {{ text }}"

    def test_dangling_dollar_ref_still_fails_loud(self):
        """The narrower remaining fail-loud: a ${path} whose first segment names no
        declared input has no data path in the exported primitive -> clean
        ConfigurationError, not a leaked pyagentspec ValidationError."""
        from neograph._agent_spec import to_agent_spec
        from neograph.errors import ConfigurationError
        from neograph.node import Node

        prod = _producer("seed", RawText)
        node = Node(
            name="think1", mode="think", model="fast", prompt="hello ${nope}", inputs=RawText, outputs=Claims
        )
        pipeline = Construct("think-pipeline", nodes=[prod, node])

        with pytest.raises(ConfigurationError, match="think1"):
            to_agent_spec(pipeline)

    def test_think_mode_node_with_matching_placeholder_still_exports(self):
        """The guard must not be a blanket rejection -- a prompt that DOES
        spell out a matching ``{{ }}`` placeholder for every real input
        Property still exports cleanly (this is pyagentspec's own supported
        authoring shape, just not neograph's native ``${var}`` syntax)."""
        from pyagentspec.flows.nodes import LlmNode

        from neograph._agent_spec import to_agent_spec
        from neograph.node import Node

        prod = _producer("seed", RawText)
        node = Node(
            name="think1", mode="think", model="fast", prompt="hello {{ text }}", inputs=RawText, outputs=Claims
        )
        pipeline = Construct("think-pipeline", nodes=[prod, node])

        flow = to_agent_spec(pipeline)

        spec_node = next(n for n in flow.nodes if n.name == "think1")
        assert isinstance(spec_node, LlmNode)
        assert spec_node.prompt_template == "hello {{ text }}"
        assert {p.title for p in (spec_node.inputs or [])} == {"text"}


class TestToolToServerToolExportOnlySlice:
    """Pins ``_tool_to_server_tool``'s export-only slice (neograph-l7gvy, refined
    2026-07-22 per architect review, atom neograph-f0j1e.30/.31).

    Scope is deliberately narrow: THIS test only pins the standalone pure
    helper's per-tool metadata shape. It does NOT exercise import or a full
    export->import round trip -- that slice is explicitly deferred, gated on
    neograph-01i0g (from_agent_spec), per the refined Implementation Plan
    step 3.

    Core Invariant under test: a neograph ``Tool``'s budget/config/idempotent
    survive into a per-tool ``neograph/tool_spec`` metadata marker (mirroring
    the ``neograph/oracle_spec`` marker convention), and no concrete
    callable/``_bound_tool`` ever leaks into that serialized marker. Also pins
    that ``ServerTool`` is the uniform Agent Spec primitive for ALL neograph
    Tool exports -- no ``MCPTool`` reference anywhere (pyagentspec 26.1.2 has
    no such class; encoding MCP-ness into the wire format would itself
    violate the name-only Core Invariant).
    """

    def test_tool_to_server_tool_stamps_neograph_tool_spec_marker_with_budget_config_idempotent(self):
        from neograph._agent_spec import _tool_to_server_tool
        from neograph.tool import Tool

        tool = Tool("search_code", budget=5, idempotent=True, config={"depth": 2})

        import pyagentspec.tools as tools_mod
        from pyagentspec.tools import ServerTool

        server_tool = _tool_to_server_tool(tool, tools_mod)

        assert isinstance(server_tool, ServerTool), (
            f"_tool_to_server_tool must return a ServerTool uniformly for ALL neograph Tool "
            f"exports (MCP-bound or not), got {type(server_tool).__name__}"
        )

        marker = server_tool.metadata[_MARK_TOOL_SPEC]
        assert marker == {
            "name": "search_code",
            "budget": 5,
            "config": {"depth": 2},
            "idempotent": True,
        }

    def test_tool_to_server_tool_marker_never_leaks_bound_tool_callable(self):
        """A Tool synthesized from a raw LangChain BaseTool carries a live
        callable on the PrivateAttr ``_bound_tool``. That callable must never
        appear in the serialized ``neograph/tool_spec`` marker -- factory
        binding is exclusively a runtime, post-deserialization concern."""
        import json

        from neograph._agent_spec import _tool_to_server_tool
        from neograph.tool import Tool

        tool = Tool("write_file", budget=3)
        object.__setattr__(tool, "_bound_tool", lambda *a, **kw: "i am a live callable")

        import pyagentspec.tools as tools_mod

        server_tool = _tool_to_server_tool(tool, tools_mod)
        marker = server_tool.metadata[_MARK_TOOL_SPEC]

        assert "_bound_tool" not in marker
        assert not any(callable(v) for v in marker.values()), (
            "no callable may appear anywhere in the serialized neograph/tool_spec marker"
        )

        # Scoped narrowly to the invariant this test guards (per architect
        # review LOW finding): json.dumps just the tool_spec sub-dict, not a
        # blanket dump of arbitrary user-supplied Tool.config.
        json.dumps(marker)

    def test_no_mcp_tool_class_used_anywhere_in_tool_export(self):
        """ServerTool is used uniformly for every neograph Tool export --
        MCP-ness is never encoded as a distinct Agent Spec tool class/type,
        per the ratified Core Invariant (name-only wire format) and the
        architect review's MEDIUM finding foreclosing the MCPTool question."""
        import pyagentspec.tools as tools_mod

        assert not hasattr(tools_mod, "MCPTool"), (
            "pyagentspec has no MCPTool class in the pinned SDK version -- if this "
            "ever changes, neograph's tool export must still NOT adopt it (would "
            "violate the name-only invariant by encoding MCP-ness into the wire format)"
        )

        from neograph._agent_spec import _tool_to_server_tool
        from neograph.tool import Tool

        server_tool = _tool_to_server_tool(Tool("search_code", budget=1), tools_mod)
        assert type(server_tool).__name__ == "ServerTool"


class TestToAgentSpecLowersModifiers:
    """Pins each modifier's LOWER composite per the Core Invariant: every
    modifier flattens to Agent Spec primitives stamped with a
    ``neograph/modifier`` metadata marker (the round-trip contract).
    """

    def test_oracle_lowers_to_variant_nodes_plus_merge_with_group_marker(self):
        from neograph._agent_spec import to_agent_spec
        from neograph.modifiers import Oracle
        from neograph.node import Node

        node = Node(name="ensemble", mode="think", model="fast", outputs=Claims, prompt="rw/ensemble")
        node = node | Oracle(n=2, merge_fn="combine")
        pipeline = Construct("oracle-pipeline", nodes=[node])

        flow = to_agent_spec(pipeline)

        oracle_nodes = [n for n in flow.nodes if n.metadata and n.metadata.get(_MARK_MODIFIER) == "oracle"]
        assert len(oracle_nodes) == 3, "expected 2 variant nodes + 1 merge node, all marker-stamped"
        group_ids = {n.metadata[_MARK_GROUP_ID] for n in oracle_nodes}
        assert len(group_ids) == 1, "all Oracle-group nodes must share one group_id"

    def test_oracle_scripted_mode_variants_lower_to_tool_node_not_llm_node(self):
        """neograph-m57mn Option A: ``_lower_oracle`` must dispatch variant
        construction per ``node.mode`` (mirroring ``_lower_node``), not build
        an unconditional ``LlmNode`` -- a scripted-mode (no prompt=/model=)
        Oracle node's variants have no prompt text at all, so an LlmNode
        variant would ALWAYS fail pyagentspec's placeholder-inference
        validation. ``ToolNode`` has zero placeholder coupling (its inferred
        inputs just echo ``tool.inputs``), so real ``Node.inputs`` pass
        through cleanly."""
        from pyagentspec.flows.nodes import LlmNode, ToolNode

        from neograph._agent_spec import to_agent_spec
        from neograph.modifiers import Oracle
        from neograph.node import Node

        prod = _producer("seed", RawText)
        gen = Node.scripted("gen", fn="gen_fn", inputs=RawText, outputs=Claims)
        gen = gen | Oracle(n=2, merge_fn="combine")
        pipeline = Construct("oracle-scripted-pipeline", nodes=[prod, gen])

        flow = to_agent_spec(pipeline)

        variant_nodes = [
            n
            for n in flow.nodes
            if n.metadata and n.metadata.get(_MARK_MODIFIER) == "oracle" and _MARK_VARIANT in n.metadata
        ]
        assert len(variant_nodes) == 2, "expected 2 scripted-mode Oracle variants"
        for variant in variant_nodes:
            assert isinstance(variant, ToolNode), (
                f"scripted-mode Oracle variant must lower to ToolNode, not {type(variant).__name__} "
                "-- an unconditional LlmNode fails pyagentspec's placeholder-inference validation"
            )
            assert not isinstance(variant, LlmNode)

    def test_oracle_merge_prompt_branch_with_real_gen_outputs_is_rejected(self):
        """neograph-m57mn addendum (post-review 4th site): ``_lower_oracle``'s
        ``oracle.merge_prompt`` branch builds an ``LlmNode`` merge node gated
        on ``oracle.merge_prompt`` truthiness, INDEPENDENT of ``node.mode`` --
        a scripted-mode node can legally carry ``merge_prompt=...``, and
        ``merge_node.inputs=gen_outputs`` (virtually always non-empty) then
        hits the exact same placeholder-coupling wall, with zero prior test
        coverage. Must raise a clean ConfigurationError, not a raw pydantic
        ValidationError."""
        from neograph._agent_spec import to_agent_spec
        from neograph.errors import ConfigurationError
        from neograph.modifiers import Oracle
        from neograph.node import Node

        gen = Node(name="gen", mode="scripted", outputs=Claims)
        gen = gen | Oracle(n=2, merge_prompt="pick best: ${variants}")
        pipeline = Construct("oracle-merge-prompt-pipeline", nodes=[gen])

        with pytest.raises(ConfigurationError, match="gen"):
            to_agent_spec(pipeline)

    def test_each_lowers_to_map_node_with_each_spec_marker(self):
        from pyagentspec.flows.nodes import MapNode

        from neograph._agent_spec import to_agent_spec
        from neograph.modifiers import Each

        node = _consumer("verify", RawText, Claims)
        node = node | Each(over="items", key="label")
        pipeline = Construct("each-pipeline", nodes=[node])

        flow = to_agent_spec(pipeline)

        map_nodes = [n for n in flow.nodes if isinstance(n, MapNode)]
        assert len(map_nodes) == 1
        assert map_nodes[0].metadata[_MARK_MODIFIER] == "each"
        assert map_nodes[0].metadata[_MARK_EACH_SPEC]["over"] == "items"

    def test_map_over_dict_form_fan_out_receiver_exports_without_error(self):
        """neograph-qtfof.1: @node's map_over= sugar (dict-form inputs where
        one key is the Each fan-out RECEIVER, not an upstream node) must
        export cleanly -- the fan-out receiver key is not itself an upstream
        NODE name, so to_agent_spec's dict-form fan-in loop must skip it
        (mirroring _validation_inputs.py's fan_out_param skip) instead of
        raising ConfigurationError."""
        from pyagentspec.flows.nodes import MapNode

        from neograph import node
        from neograph._agent_spec import to_agent_spec
        from neograph.decorators import construct_from_functions

        @node(outputs=Clusters)
        def clusters() -> Clusters:
            return Clusters(groups=[ClusterGroup(label="a", claim_ids=["1"])])

        @node(outputs=MatchResult, map_over="clusters.groups", map_key="label")
        def verify(cluster: ClusterGroup) -> MatchResult:
            return MatchResult(cluster_label=cluster.label, matched=[])

        pipeline = construct_from_functions("fanout-only-export", [clusters, verify])

        flow = to_agent_spec(pipeline)

        map_nodes = [n for n in flow.nodes if isinstance(n, MapNode)]
        assert len(map_nodes) == 1

    def test_programmatic_each_with_dict_form_fan_out_receiver_exports_without_error(self):
        """Three-surface parity for qtfof.1: the programmatic
        ``Node(inputs={...}) | Each(...)`` equivalent must export the same
        as the @node ``map_over=`` sugar -- the normalizer sets
        ``fan_out_param`` for both surfaces, so the same skip applies."""
        from pyagentspec.flows.nodes import MapNode

        from neograph._agent_spec import to_agent_spec
        from neograph.modifiers import Each
        from neograph.node import Node

        make = _producer("make", Clusters)
        canonicalize = Node.scripted(
            "canonicalize",
            fn="f",
            inputs={"group": ClusterGroup},
            outputs=MatchResult,
        ) | Each(over="make.groups", key="label")
        pipeline = Construct("fanout-only-export-programmatic", nodes=[make, canonicalize])

        flow = to_agent_spec(pipeline)

        map_nodes = [n for n in flow.nodes if isinstance(n, MapNode)]
        assert len(map_nodes) == 1

    def test_loop_lowers_to_branching_node_with_back_edge_and_loop_marker(self):
        from pyagentspec.flows.edges import ControlFlowEdge
        from pyagentspec.flows.nodes import BranchingNode

        from neograph._agent_spec import to_agent_spec
        from neograph.modifiers import Loop
        from neograph.node import Node

        node = Node.scripted("refine", fn="refine_fn", inputs=Claims, outputs=Claims)
        node = node | Loop(when="claims_incomplete", max_iterations=3)
        pipeline = Construct("loop-pipeline", nodes=[node])

        flow = to_agent_spec(pipeline)

        branch_nodes = [
            n for n in flow.nodes if isinstance(n, BranchingNode) and n.metadata.get(_MARK_MODIFIER) == "loop"
        ]
        assert len(branch_nodes) == 1
        assert branch_nodes[0].metadata[_MARK_LOOP_SPEC]["when"] == "claims_incomplete"
        back_edges = [
            e
            for e in flow.control_flow_connections
            if isinstance(e, ControlFlowEdge) and e.from_node.name == branch_nodes[0].name and e.from_branch == Branch.CONTINUE
        ]
        assert len(back_edges) == 1, "expected a cyclic ControlFlowEdge back into the loop body"

    def test_loop_dict_form_inputs_self_edge_exports_without_error(self):
        """neograph-qtfof.2: a Loop-modified node with DICT-FORM inputs
        (@node's primary shape, inputs={'refine': Claims} self-referencing)
        must export its self-edge cleanly. ``_lower_loop``'s self-edge
        construction assumed destination_input matches a BARE output
        Property title, but dict-form inputs prefix input Property titles
        as '{upstream}.{field}' (per ``_properties_for``'s dict-form
        convention) -- so the self-edge must target the PREFIXED input
        title for the Loop self-reference key, not the bare title."""
        from neograph._agent_spec import to_agent_spec
        from neograph.modifiers import Loop
        from neograph.node import Node

        node = Node.scripted("refine", fn="refine_fn", inputs={"refine": Claims}, outputs=Claims)
        node = node | Loop(when="claims_incomplete", max_iterations=3)
        pipeline = Construct("loop-dict-form-pipeline", nodes=[node])

        # Must not raise pydantic.ValidationError -- the self-edge must
        # reference a REAL input Property of the body node.
        flow = to_agent_spec(pipeline)

        from pyagentspec.flows.nodes import BranchingNode

        branch_nodes = [
            n for n in flow.nodes if isinstance(n, BranchingNode) and n.metadata.get(_MARK_MODIFIER) == "loop"
        ]
        assert len(branch_nodes) == 1

    def test_callable_loop_when_is_rejected(self):
        from neograph._agent_spec import to_agent_spec
        from neograph.errors import ConfigurationError
        from neograph.modifiers import Loop
        from neograph.node import Node

        node = Node.scripted("refine", fn="refine_fn", inputs=Claims, outputs=Claims)
        node = node | Loop(when=lambda d: d is None, max_iterations=3)
        pipeline = Construct("loop-callable-pipeline", nodes=[node])

        with pytest.raises(ConfigurationError, match="Loop.when"):
            to_agent_spec(pipeline)

    def test_operator_lowers_to_pause_branch_composite(self):
        from pyagentspec.flows.edges import ControlFlowEdge
        from pyagentspec.flows.nodes import BranchingNode, InputMessageNode

        from neograph._agent_spec import to_agent_spec
        from neograph.modifiers import Operator

        node = _producer("gate", Claims)
        node = node | Operator(when="needs_review")
        pipeline = Construct("operator-pipeline", nodes=[node])

        flow = to_agent_spec(pipeline)

        checks = [
            n for n in flow.nodes if isinstance(n, BranchingNode) and n.metadata.get(_MARK_MODIFIER) == "operator"
        ]
        assert len(checks) == 1
        assert checks[0].metadata[_MARK_OPERATOR_SPEC]["when"] == "needs_review"
        assert checks[0].mapping["true"] == "pause"

        pause_nodes = [n for n in flow.nodes if isinstance(n, InputMessageNode)]
        assert len(pause_nodes) == 1

        pause_edges = [
            e
            for e in flow.control_flow_connections
            if isinstance(e, ControlFlowEdge) and e.to_node.name == pause_nodes[0].name
        ]
        assert any(e.from_branch == Branch.PAUSE for e in pause_edges), (
            "expected the PAUSE_BRANCH edge (not DEFAULT_BRANCH) into InputMessageNode"
        )

    def test_forward_construct_if_else_lowers_to_branching_node_not_unconditional_sequence(self):
        """neograph-s7zt3.17: a ForwardConstruct if/else must export to a real
        BranchingNode with divergent true/false ControlFlowEdges into each
        arm, reconverging on the successor -- NOT both arms wired to run
        unconditionally in sequence (the pre-fix behavior, since
        ``iter_with_arms`` drops the ``_BranchNode`` sentinel before
        ``to_agent_spec`` ever sees it)."""
        from pyagentspec.flows.edges import ControlFlowEdge
        from pyagentspec.flows.nodes import BranchingNode
        from pydantic import BaseModel

        from neograph._agent_spec import to_agent_spec
        from neograph.forward import ForwardConstruct
        from neograph.node import Node

        from .fakes import register_scripted

        class Confidence(BaseModel, frozen=True):
            score: float

        # Both arms share ArmResult (neograph-qtfof.9 R2: the Agent Spec EndNode
        # requires the two arms' terminal producers to converge on a compatible
        # output type; distinct types here would be an incidental fixture choice
        # colliding with that unrelated, deliberate invariant -- this test's own
        # claim is about CONTROL FLOW shape, not output types).
        class ArmResult(BaseModel, frozen=True):
            label: str

        register_scripted("s7zt3_17_check", lambda input_data, config: Confidence(score=0.9))
        register_scripted("s7zt3_17_high", lambda input_data, config: ArmResult(label="high"))
        register_scripted("s7zt3_17_low", lambda input_data, config: ArmResult(label="low"))

        class BranchPipeline(ForwardConstruct):
            check = Node.scripted("s7zt3-17-check", fn="s7zt3_17_check", outputs=Confidence)
            high_path = Node.scripted("s7zt3-17-high", fn="s7zt3_17_high", outputs=ArmResult)
            low_path = Node.scripted("s7zt3-17-low", fn="s7zt3_17_low", outputs=ArmResult)

            def forward(self, topic):
                result = self.check(topic)
                if result.score > 0.5:
                    return self.high_path(result)
                else:
                    return self.low_path(result)

        pipeline = BranchPipeline()

        flow = to_agent_spec(pipeline)

        branch_nodes = [n for n in flow.nodes if isinstance(n, BranchingNode)]
        assert len(branch_nodes) == 1, (
            "expected a BranchingNode for the if/else -- got a flattened, "
            "unconditional sequence of both arms instead"
        )

        control_edges = [e for e in flow.control_flow_connections if isinstance(e, ControlFlowEdge)]
        high_node = next(n for n in flow.nodes if n.name == "s7zt3-17-high")
        low_node = next(n for n in flow.nodes if n.name == "s7zt3-17-low")

        # Both arms must be entered ONLY via a labeled branch edge out of the
        # BranchingNode -- never directly from each other in sequence.
        assert not any(e.from_node.name == "s7zt3-17-high" and e.to_node.name == "s7zt3-17-low" for e in control_edges), (
            "high_path must not fall through directly into low_path -- both arms ran "
            "unconditionally in sequence, the exact pre-fix bug"
        )
        assert not any(e.from_node.name == "s7zt3-17-low" and e.to_node.name == "s7zt3-17-high" for e in control_edges), (
            "low_path must not fall through directly into high_path -- both arms ran "
            "unconditionally in sequence, the exact pre-fix bug"
        )

        true_edges = [e for e in control_edges if e.from_node.name == branch_nodes[0].name and e.from_branch == Branch.TRUE]
        false_edges = [e for e in control_edges if e.from_node.name == branch_nodes[0].name and e.from_branch == Branch.FALSE]
        assert any(e.to_node.name == high_node.name for e in true_edges), (
            "expected the branch's 'true' edge to enter high_path"
        )
        assert any(e.to_node.name == low_node.name for e in false_edges), (
            "expected the branch's 'false' edge to enter low_path"
        )

    def test_multi_node_arm_wires_internal_sequence_and_reconverges_on_last_node(self):
        """neograph-s7zt3.17 (architect-review-flagged): an arm with MORE than
        one item must wire its own internal sequential edges and reconverge
        on the successor from its LAST item -- not just its first -- proving
        the fix's per-arm reconvergence chain, not only single-node arms."""
        import operator as op

        from pyagentspec.flows.edges import ControlFlowEdge

        from neograph._agent_spec import to_agent_spec
        from neograph._ir_branch import _BranchMeta, _BranchNode, _ConditionSpec
        from neograph.node import Node

        seed = _producer("mn-seed", Claims)
        step1 = Node.scripted("mn-step1", fn="f", inputs=Claims, outputs=Claims)
        step2 = Node.scripted("mn-step2", fn="f", inputs=Claims, outputs=Claims)
        low_path = Node.scripted("mn-low", fn="f", inputs=Claims, outputs=Claims)

        branch_meta = _BranchMeta(
            condition_spec=_ConditionSpec(source_node=seed, attr_chain=[], op_fn=op.gt, op_str=">", threshold=0),
            true_arm_nodes=[step1, step2],
            false_arm_nodes=[low_path],
        )
        pipeline = Construct("multi-node-arm", nodes=[seed, _BranchNode(branch_meta, 0)])

        flow = to_agent_spec(pipeline)

        control_edges = [e for e in flow.control_flow_connections if isinstance(e, ControlFlowEdge)]
        assert any(e.from_node.name == "mn-step1" and e.to_node.name == "mn-step2" for e in control_edges), (
            "expected mn-step1 -> mn-step2 sequential wiring WITHIN the true arm"
        )
        end_node = next(n for n in flow.nodes if n.name == "multi-node-arm__end")
        assert any(e.from_node.name == "mn-step2" and e.to_node.name == end_node.name for e in control_edges), (
            "expected the arm's LAST item (mn-step2), not just its first, to reconverge on the successor"
        )
        assert not any(e.from_node.name == "mn-step1" and e.to_node.name == end_node.name for e in control_edges), (
            "mn-step1 must NOT reconverge directly -- only the arm's final item's exit does"
        )

    def test_modifier_wrapped_node_inside_arm_gets_its_existing_lowering(self):
        """neograph-s7zt3.17 (architect-review-flagged): an arm item carrying
        its OWN modifier (Loop) must dispatch through the SAME
        _lower_construct_item machinery the top-level loop uses -- proving
        the recursive per-arm-item reuse design generalizes, not just bare
        single-node arms."""
        import operator as op

        from pyagentspec.flows.nodes import BranchingNode

        from neograph._agent_spec import to_agent_spec
        from neograph._ir_branch import _BranchMeta, _BranchNode, _ConditionSpec
        from neograph.modifiers import Loop
        from neograph.node import Node

        seed = _producer("mod-seed", Claims)
        looped = Node.scripted("mod-looped-body", fn="f", inputs=Claims, outputs=Claims) | Loop(
            when="claims_incomplete", max_iterations=3
        )
        low_path = _consumer("mod-low", Claims, Claims)

        branch_meta = _BranchMeta(
            condition_spec=_ConditionSpec(source_node=seed, attr_chain=[], op_fn=op.gt, op_str=">", threshold=0),
            true_arm_nodes=[looped],
            false_arm_nodes=[low_path],
        )
        pipeline = Construct("modifier-in-arm", nodes=[seed, _BranchNode(branch_meta, 0)])

        flow = to_agent_spec(pipeline)

        loop_branch_nodes = [
            n for n in flow.nodes if isinstance(n, BranchingNode) and n.metadata.get(_MARK_MODIFIER) == "loop"
        ]
        assert len(loop_branch_nodes) == 1, (
            "expected the arm's Loop-wrapped node to lower to its own BranchingNode+back-edge, "
            "via the SAME modifier dispatch the top-level loop uses"
        )


# ── neograph-s7zt3.8: Construct-ITEM modifier export (silent-drop bug fix) ────


class TestConstructItemModifierExport:
    """Pins neograph-s7zt3.8 (master architecture doc §5, EACH/ORACLE/LOOP/
    OPERATOR Construct-export rows, all pre-fix ``BROKEN -- silent drop``).

    ``_lower_construct_item``'s ``isinstance(item, Construct)`` branch used to
    wrap a bare ``FlowNode`` and return BEFORE ``classify_modifiers`` ran, so a
    ``Construct(...) | Each()/Oracle()/Loop()/Operator()`` used as ONE ITEM
    inside a parent Construct lost its modifier on export -- no error, no
    marker, no diagnostic. The fix routes Node AND Construct items through the
    SAME modifier dispatch (reusing ``_lower_each``/``_lower_oracle``/
    ``_lower_loop``/``_lower_operator`` over the item's ``_lower_item_body``),
    and FAILS LOUD for the EACH_ORACLE / EACH_ORACLE_OPERATOR fusion combos,
    mirroring ``compiler.py``'s own permanent sub-construct rejection.

    Pre-fix, every ``*_lowers_*`` test below saw a lone ``FlowNode`` with no
    ``neograph/modifier`` marker (the silent drop, reproduced directly by
    building the pipeline and finding zero modifier markers), and the two
    ``*_fails_loud`` tests saw a clean export instead of the mandated
    ConfigurationError.
    """

    @staticmethod
    def _sub(input_type: type, output_type: type) -> Construct:
        from neograph.node import Node

        step = Node.scripted("step", fn="step_fn", inputs=input_type, outputs=output_type)
        return Construct("sub", input=input_type, output=output_type, nodes=[step])

    def test_each_on_construct_item_lowers_to_map_node_with_marker(self):
        from pyagentspec.flows.nodes import MapNode

        from neograph._agent_spec import to_agent_spec
        from neograph.modifiers import Each

        sub = self._sub(RawText, Claims) | Each(over="items", key="label")
        parent = Construct("parent", nodes=[sub])

        flow = to_agent_spec(parent)

        map_nodes = [n for n in flow.nodes if isinstance(n, MapNode)]
        assert len(map_nodes) == 1, "Each on a Construct item must lower to a MapNode, not a bare FlowNode"
        assert map_nodes[0].metadata[_MARK_MODIFIER] == "each"
        assert map_nodes[0].metadata[_MARK_EACH_SPEC]["over"] == "items"

    def test_oracle_on_construct_item_lowers_to_variant_flow_nodes_plus_merge(self):
        from pyagentspec.flows.nodes import FlowNode

        from neograph._agent_spec import to_agent_spec
        from neograph.modifiers import Oracle

        # Same-type boundary: an ensemble re-runs the sub-flow and merges.
        sub = self._sub(Claims, Claims) | Oracle(n=2, merge_fn="combine")
        parent = Construct("parent", nodes=[sub])

        flow = to_agent_spec(parent)

        oracle_nodes = [n for n in flow.nodes if n.metadata and n.metadata.get(_MARK_MODIFIER) == "oracle"]
        assert len(oracle_nodes) == 3, "expected 2 variant nodes + 1 merge node, all marker-stamped"
        variant_nodes = [n for n in oracle_nodes if _MARK_VARIANT in (n.metadata or {})]
        assert len(variant_nodes) == 2
        assert all(isinstance(v, FlowNode) for v in variant_nodes), (
            "a Construct-item Oracle variant is a copy of the sub-flow -- a FlowNode over the exported sub-Flow"
        )
        group_ids = {n.metadata[_MARK_GROUP_ID] for n in oracle_nodes}
        assert len(group_ids) == 1, "all Oracle-group nodes must share one group_id"

    def test_oracle_on_construct_item_gives_each_variant_its_own_subflow_object(self):
        """Characterization pin for neograph-15rpw: N variants, N DISTINCT sub-Flows.

        A Construct-item Oracle variant is a COPY of the sub-flow, so each variant
        FlowNode must wrap its own ``Flow`` component -- distinct Python object AND
        distinct component ``id`` -- not one shared ``Flow`` referenced N times. A
        shared Flow is a different exported spec: one component the variants alias,
        rather than N independent bodies.

        Pins the property against the wrong way to collapse the variant arm onto
        ``_lower_item_body`` -- hoisting the body lowering out of the per-variant
        loop and ``model_copy``-ing it N times shares the ``subflow`` by reference
        (``model_copy`` is shallow), turning a pure refactor into a behaviour
        change. Verified to fail against exactly that mutation.
        """
        from neograph._agent_spec import to_agent_spec
        from neograph.modifiers import Oracle

        sub = self._sub(Claims, Claims) | Oracle(n=3, merge_fn="combine")
        parent = Construct("parent", nodes=[sub])

        flow = to_agent_spec(parent)

        variants = [n for n in flow.nodes if _MARK_VARIANT in (n.metadata or {})]
        assert len(variants) == 3
        # holder_flows reads both spellings, so this stays correct if a variant ever
        # lowers to a plural holder (neograph-498gr).
        subflows = [sub for v in variants for sub in holder_flows(v)]
        assert len(subflows) == len(variants), "each variant holds exactly one sub-Flow"
        assert len({id(s) for s in subflows}) == 3, "each variant must wrap its own sub-Flow object"
        assert len({s.id for s in subflows}) == 3, "each variant's sub-Flow must be its own component"
        assert len({v.id for v in variants}) == 3, "each variant FlowNode must be its own component"
        assert [v.name for v in variants] == ["sub__variant_0", "sub__variant_1", "sub__variant_2"]

    def test_loop_on_construct_item_lowers_to_branching_node_with_marker(self):
        from pyagentspec.flows.edges import ControlFlowEdge
        from pyagentspec.flows.nodes import BranchingNode, FlowNode

        from neograph._agent_spec import to_agent_spec
        from neograph.modifiers import Loop

        sub = self._sub(Claims, Claims) | Loop(when="claims_incomplete", max_iterations=3)
        parent = Construct("parent", nodes=[sub])

        flow = to_agent_spec(parent)

        # The looped body is the Construct's FlowNode; the check is a marker-stamped BranchingNode.
        assert any(isinstance(n, FlowNode) and n.name == "sub" for n in flow.nodes)
        branch_nodes = [
            n for n in flow.nodes if isinstance(n, BranchingNode) and n.metadata.get(_MARK_MODIFIER) == "loop"
        ]
        assert len(branch_nodes) == 1
        assert branch_nodes[0].metadata[_MARK_LOOP_SPEC]["when"] == "claims_incomplete"
        back_edges = [
            e
            for e in flow.control_flow_connections
            if isinstance(e, ControlFlowEdge)
            and e.from_node.name == branch_nodes[0].name
            and e.from_branch == Branch.CONTINUE
        ]
        assert len(back_edges) == 1, "expected a cyclic ControlFlowEdge back into the sub-flow body"

    def test_loop_on_construct_item_keeps_self_data_edge_when_boundaries_match(self):
        """Green control for neograph-rh5fb: the matching-boundary case must
        KEEP emitting the self-feedback DataFlowEdge.

        Pins the boundary condition of the differing-boundary fix below so it
        cannot silently widen into dropping data edges that genuinely have a
        destination property on the body's input port.
        """
        from neograph._agent_spec import to_agent_spec
        from neograph.modifiers import Loop

        sub = self._sub(Claims, Claims) | Loop(when="claims_incomplete", max_iterations=3)
        parent = Construct("parent", nodes=[sub])

        flow = to_agent_spec(parent)

        self_edges = [e for e in (flow.data_flow_connections or []) if e.name.startswith("sub__loop_self_")]
        assert len(self_edges) == 1, "a same-boundary loop body feeds its output back as the next input"
        assert self_edges[0].source_node.name == "sub"
        assert self_edges[0].destination_node.name == "sub"
        assert self_edges[0].source_output == "items"
        assert self_edges[0].destination_input == "items"

    def test_loop_on_construct_item_with_differing_boundaries_is_control_only(self):
        """Red test for neograph-rh5fb.

        A sub-Construct whose ``input`` and ``output`` types differ compiles and
        RUNS (``tests/test_loop.py::TestLoopInputNotEqualOutput``) -- the
        LangGraph loop-back is a pure CONTROL edge over a shared, accumulating
        state dict, with zero field aliasing. Export used to build a literal
        field-aliasing self-edge anyway (``sub__loop_self_items`` pointed at the
        ``sub`` FlowNode's input port, which declares RawText's ``text``), so
        pyagentspec's own ``DataFlowEdge`` validator rejected it with a raw
        pydantic ValidationError.

        There is no destination property for the fed-back output, so the correct
        export carries NO data edge -- only the cyclic ControlFlowEdge.
        """
        from pyagentspec.flows.edges import ControlFlowEdge
        from pyagentspec.flows.nodes import BranchingNode

        from neograph._agent_spec import to_agent_spec
        from neograph.modifiers import Loop

        sub = self._sub(RawText, Claims) | Loop(when="claims_incomplete", max_iterations=3)
        parent = Construct("parent", nodes=[sub])

        flow = to_agent_spec(parent)

        assert not [e for e in (flow.data_flow_connections or []) if e.name.startswith("sub__loop_self_")], (
            "a differing-boundary loop body has no destination property for its fed-back output "
            "-- the loop-back must export as a control-only edge"
        )
        branch = next(
            n for n in flow.nodes if isinstance(n, BranchingNode) and n.metadata.get(_MARK_MODIFIER) == "loop"
        )
        back_edges = [
            e
            for e in flow.control_flow_connections
            if isinstance(e, ControlFlowEdge) and e.from_node.name == branch.name and e.from_branch == Branch.CONTINUE
        ]
        assert len(back_edges) == 1, "the cycle back into the body must survive as a control edge"
        assert back_edges[0].to_node.name == "sub"

    def test_operator_on_construct_item_lowers_to_pause_composite(self):
        from pyagentspec.flows.nodes import BranchingNode, InputMessageNode

        from neograph._agent_spec import to_agent_spec
        from neograph.modifiers import Operator

        sub = self._sub(RawText, Claims) | Operator(when="needs_review")
        parent = Construct("parent", nodes=[sub])

        flow = to_agent_spec(parent)

        checks = [
            n for n in flow.nodes if isinstance(n, BranchingNode) and n.metadata.get(_MARK_MODIFIER) == "operator"
        ]
        assert len(checks) == 1
        assert checks[0].metadata[_MARK_OPERATOR_SPEC]["when"] == "needs_review"
        assert checks[0].mapping["true"] == "pause"
        assert len([n for n in flow.nodes if isinstance(n, InputMessageNode)]) == 1

    def test_bare_construct_item_still_exports_as_plain_flow_node(self):
        """Regression guard: the fix must NOT change the BARE Construct-item
        path -- an unmodified sub-construct still lowers to a single FlowNode."""
        from pyagentspec.flows.nodes import FlowNode

        from neograph._agent_spec import to_agent_spec

        parent = Construct("parent", nodes=[self._sub(RawText, Claims)])

        flow = to_agent_spec(parent)

        flow_nodes = [n for n in flow.nodes if isinstance(n, FlowNode)]
        assert len(flow_nodes) == 1
        assert flow_nodes[0].name == "sub"

    def test_each_oracle_on_construct_item_fails_loud(self):
        from neograph._agent_spec import to_agent_spec
        from neograph.errors import ConfigurationError
        from neograph.modifiers import Each, Oracle

        sub = self._sub(Claims, Claims) | Each(over="items", key="label") | Oracle(n=2, merge_fn="combine")
        parent = Construct("parent", nodes=[sub])

        with pytest.raises(ConfigurationError, match="Each x Oracle fusion is not supported on sub-constructs"):
            to_agent_spec(parent)

    def test_each_oracle_operator_on_construct_item_fails_loud(self):
        from neograph._agent_spec import to_agent_spec
        from neograph.errors import ConfigurationError
        from neograph.modifiers import Each, Operator, Oracle

        sub = (
            self._sub(Claims, Claims)
            | Each(over="items", key="label")
            | Oracle(n=2, merge_fn="combine")
            | Operator(when="needs_review")
        )
        parent = Construct("parent", nodes=[sub])

        with pytest.raises(ConfigurationError, match="Each x Oracle fusion is not supported on sub-constructs"):
            to_agent_spec(parent)


# ── neograph-5x43u: Portal mode (a) peer mesh EXPORT direction ──────────────


class TestPortalMeshExportsToSwarm:
    """Pins 5x43u: ``to_agent_spec``'s export-direction mirror of
    ``loader.py``'s Swarm import (``_reconstruct_swarm_mesh``). A Portal
    mode-(a) peer mesh (``handoff_param``/``handoff_channel`` set by
    ``_ir_normalize.py``) must lower to a top-level pyagentspec ``Swarm`` --
    ``first_agent``/``relationships`` of real ``Agent`` objects (pyagentspec
    ``swarm.py:105-107`` + ``agent.py:23`` type them ``AgenticComponent``, so
    a Flow ``LlmNode`` cannot go there) -- never the current fail-loud
    reject. The entry-only knobs (``max_hops``/``on_exhaust``/``route``) have
    no native ``Swarm`` field, so they ride a ``neograph/portal_spec``
    metadata marker (mirrors the ``oracle_spec``/``each_spec``/``loop_spec``
    per-group marker convention) so the information is not lost even though
    the current Swarm *importer* does not read it back yet.

    FAILS NOW: ``_reject_unrepresentable_fields`` (``_agent_spec.py``) still
    raises ``ConfigurationError`` for ANY node with ``handoff_param``/
    ``handoff_channel`` set -- the reject 5x43u replaces with this lowering.
    """

    def _mesh(self) -> Construct:
        from pydantic import BaseModel

        from neograph import Node, Portal

        class Handoff(BaseModel, frozen=True):
            goto: str
            note: str = ""

        triage = Node.scripted("triage", fn="fn_triage", outputs=Handoff) | Portal(
            to=["billing"], max_hops=6, on_exhaust="exit"
        )
        billing = Node.scripted(
            "billing", fn="fn_billing", inputs={"handoff": Handoff}, outputs=Handoff
        ) | Portal(to=[])
        return Construct("swarm-mesh", nodes=[triage, billing])

    def test_portal_peer_mesh_lowers_to_swarm(self):
        from pyagentspec.agent import Agent
        from pyagentspec.swarm import Swarm

        from neograph._agent_spec import to_agent_spec

        mesh = self._mesh()
        swarm = to_agent_spec(mesh)

        assert isinstance(swarm, Swarm), (
            f"expected to_agent_spec to return a top-level pyagentspec Swarm "
            f"(AgenticComponent) for a Portal peer mesh, got {type(swarm)!r}"
        )

        # first_agent is the entry member (Construct.nodes order) and a real
        # Agent -- NOT an LlmNode -- because Swarm.first_agent/relationships
        # are typed AgenticComponent (pyagentspec/swarm.py:105-107).
        assert isinstance(swarm.first_agent, Agent)
        assert swarm.first_agent.name == "triage-agent"

        relationship_names = {(a.name, b.name) for a, b in swarm.relationships}
        assert relationship_names == {("triage-agent", "billing-agent")}
        assert all(isinstance(a, Agent) and isinstance(b, Agent) for a, b in swarm.relationships)

        # Entry-only knobs (max_hops/on_exhaust) and the routing field name
        # ride a neograph/portal_spec marker -- Swarm has no native field for
        # any of them.
        marker = swarm.metadata[_MARK_PORTAL_SPEC]
        assert marker["max_hops"] == 6
        assert marker["on_exhaust"] == "exit"
        assert marker["route"] == "goto"

    def test_mixed_mesh_and_plain_flow_node_is_rejected(self):
        """A Construct mixing a Portal mesh with an ordinary (non-mesh) Flow
        node has no single top-level export shape (a Swarm is not a Flow
        node) -- the v1 answer is a fail-loud ConfigurationError, never a
        silent partial export of just the mesh or just the plain node.
        """
        from neograph._agent_spec import to_agent_spec
        from neograph.errors import ConfigurationError

        mesh = self._mesh()
        plain = _producer("extra", Claims)
        mixed = Construct("mixed-mesh-and-flow", nodes=[*mesh.nodes, plain])

        with pytest.raises(ConfigurationError):
            to_agent_spec(mixed)


# ── neograph-s7zt3.1: mesh-member ${var} prompts must never ship raw ─────────


class TestPortalMeshMemberPromptNeverShipsRawPlaceholder:
    """Pins s7zt3.1: ``_lower_portal_mesh_to_swarm`` passed ``member.prompt``
    UNTRANSLATED into ``_make_agent`` (``Agent(system_prompt=...)``), so an
    agent-mode mesh member with ``inputs={'handoff': Payload}`` and a
    ``${handoff.note}`` prompt exported a raw neograph ``${...}`` into a
    foreign Swarm runtime that speaks pyagentspec ``{{ }}`` placeholders and
    will NEVER fill it -- and pyagentspec does not flag it (``${...}`` is not
    its grammar), so it shipped SILENTLY. That is exactly the North-Star
    silent-seam class: the export must either Option-F-translate the prompt
    (like every other ``_make_agent`` caller) or fail loud -- never ship raw.

    The in-code assumption these tests falsify: 'mesh Agents carry NO I/O
    Properties, so there is nothing to placeholder-translate'. A mesh member
    CAN declare the reserved ``handoff`` input (``_check_portal_mesh``
    validates it) and reference it from its prompt.
    """

    def _placeholder_mesh(self) -> Construct:
        from pydantic import BaseModel

        from neograph import Node, Portal

        class Handoff(BaseModel, frozen=True):
            goto: str
            note: str = ""

        triage = Node(
            name="triage",
            mode="agent",
            prompt="Handle the case: ${handoff.note}",
            model="gpt-4o",
            inputs={"handoff": Handoff},
            outputs=Handoff,
        ) | Portal(to=["billing"], max_hops=6, on_exhaust="exit")
        billing = Node(
            name="billing",
            mode="agent",
            prompt="Bill it: ${handoff.note}",
            model="gpt-4o",
            inputs={"handoff": Handoff},
            outputs=Handoff,
        ) | Portal(to=[])
        return Construct("swarm-mesh-placeholders", nodes=[triage, billing])

    def _swarm_agents(self, swarm) -> list:
        agents = {id(swarm.first_agent): swarm.first_agent}
        for a, b in swarm.relationships:
            agents[id(a)] = a
            agents[id(b)] = b
        return list(agents.values())

    def test_mesh_member_prompt_placeholders_never_ship_raw(self):
        """North-Star invariant, fix-shape-agnostic: exporting a mesh whose
        member prompts reference ``${handoff.*}`` either fails loud
        (ConfigurationError) or produces Agents whose system_prompt carries
        NO raw ``${`` -- silent raw-`${var}` shipping is the one outcome the
        invariant forbids, and it is what happens today.
        """
        from neograph._agent_spec import to_agent_spec
        from neograph.errors import ConfigurationError

        mesh = self._placeholder_mesh()
        try:
            swarm = to_agent_spec(mesh)
        except ConfigurationError:
            return  # fail-loud is a sanctioned outcome

        for agent in self._swarm_agents(swarm):
            assert "${" not in (agent.system_prompt or ""), (
                f"Swarm member Agent {agent.name!r} shipped a RAW neograph "
                f"${{...}} placeholder in system_prompt "
                f"{agent.system_prompt!r} -- a foreign pyagentspec runtime "
                f"speaks {{{{ }}}}, will never fill it, and does not flag it. "
                f"Translate (Option F) or fail loud; never ship raw."
            )

    def test_mesh_member_prompt_is_option_f_translated_with_declared_props(self):
        """The landed fix shape: each mesh Agent's system_prompt is the
        ``{{ flat }}``-rewritten wire form and the Agent declares exactly the
        referenced flat Properties (so pyagentspec's own placeholder
        inference/validation passes by construction) -- the SAME Option-F
        contract every other ``_make_agent`` caller honors.
        """
        from neograph._agent_spec import to_agent_spec

        swarm = to_agent_spec(self._placeholder_mesh())

        assert swarm.first_agent.system_prompt == "Handle the case: {{ handoff_note }}"
        assert [p.title for p in (swarm.first_agent.inputs or [])] == ["handoff_note"], (
            "the mesh Agent must declare the referenced flat Property so "
            "pyagentspec's ComponentWithIO validation matches the rewritten prompt"
        )

    def test_mesh_member_prompt_round_trips_to_original_grammar(self):
        """Ticket requirement 'must round-trip if translated': the untranslated
        ``${var}`` text rides a per-member ``neograph/prompt_spec`` marker on
        the Agent, and ``from_agent_spec`` prefers it -- so an exported mesh
        imports back with the ORIGINAL neograph prompt grammar, not the
        ``{{ flat }}`` wire form.
        """
        from neograph._agent_spec import to_agent_spec
        from neograph.loader import from_agent_spec

        swarm = to_agent_spec(self._placeholder_mesh())
        with pytest.warns(UserWarning, match="best-effort"):
            mesh = from_agent_spec(swarm)

        # Members import under the exported Agent names ('{node}-agent' -- the
        # pre-existing, documented best-effort rename of the Swarm import, not
        # part of s7zt3.1). What s7zt3.1 pins is the PROMPT grammar: original
        # ${var} text, never the {{ flat }} wire form.
        prompts = {n.name: n.prompt for n in mesh.nodes}
        assert prompts["triage-agent"] == "Handle the case: ${handoff.note}"
        assert prompts["billing-agent"] == "Bill it: ${handoff.note}"


# ── neograph-tjpn4: the generic "no Agent Spec lowering yet" fallthrough ─────
#
# POLARITY: GREEN BEFORE AND AFTER. This class pins TODAY's behavior for the
# COMBO_DECOMPOSITION migration of ``_lower_construct_item`` (neograph-tjpn4),
# which is a zero-behaviour-change refactor. It is NOT a TDD-red artifact --
# do not "fix" it because it passes.

from neograph.modifiers import Each, Loop, ModifierCombo, Operator, Oracle  # noqa: E402
from neograph.node import Node  # noqa: E402

from .test_agent_spec_matrix import UNSUPPORTED_COMBOS  # noqa: E402

# DERIVED, never hand-listed: the matrix's UNSUPPORTED_COMBOS is the loud
# partition over ModifierCombo (SUPPORTED | UNSUPPORTED == set(ModifierCombo),
# asserted in test_agent_spec_matrix.TestAgentSpecMatrixExhaustiveness). The two
# PORTAL combos are removed because they hit a DIFFERENT, dedicated raise (the
# permanent dispatch-mode-Portal rejection), so what remains is exactly the set
# that falls through to the generic "no Agent Spec lowering yet" raise. Deriving
# it this way means a new ModifierCombo cannot silently escape this pin.
_PORTAL_COMBOS: frozenset[ModifierCombo] = frozenset({ModifierCombo.PORTAL, ModifierCombo.PORTAL_OPERATOR})
FALLTHROUGH_COMBOS: tuple[ModifierCombo, ...] = tuple(sorted(UNSUPPORTED_COMBOS - _PORTAL_COMBOS, key=lambda c: c.name))


class TestUnsupportedComboFallthroughRaise:
    """Pins that the generic "no Agent Spec lowering yet" fallthrough is GONE, and
    that the two remaining rejections stayed DISTINCT and specific.

    HISTORY / POLARITY CHANGE -- read before touching. This class was written by
    neograph-tjpn4 to pin the PROVISIONAL ``ConfigurationError`` that
    ``_lower_construct_item`` raised for the five fusion combos, and its own
    docstring named the wording provisional and "owned by neograph-s7zt3.10".
    Phase 7 (neograph-s7zt3.10) is that owner: all five combos now have real
    lowerings AND matching loader.py import recognition, so the raise they pinned
    is unreachable and ``_raise_no_agent_spec_lowering`` was DELETED rather than
    left as dead code behind a green suite.

    What still needs pinning, and is pinned below:
      * FALLTHROUGH_COMBOS is now EMPTY -- derived, so a combo silently LEAVING
        SUPPORTED_COMBOS re-populates it and fails here instead of quietly
        reintroducing a provisional raise.
      * The dispatch-mode-Portal rejection is PERMANENT and must keep its own
        specific message -- the hazard the hoisted, unconditional Operator
        postlude created (a hoisted raise, or a ``case ... if ...`` guard, would
        have collapsed PORTAL's message into a generic one or swapped
        ConfigurationError for assert_never's AssertionError).
      * The SUB_CONSTRUCT_UNSUPPORTED_COMBOS gate still fires for a Construct
        carrying the fusion, and still does NOT capture a bare Node.
    """

    @staticmethod
    def _register() -> None:
        """conftest's autouse fixture resets every registry per test, so the
        scripted fn / merge fn / conditions are registered inside the test."""
        from .fakes import register_condition, register_scripted

        register_scripted("f", lambda *a, **k: None)
        register_scripted("combine", lambda variants, config: variants[0])
        register_condition("needs_review", lambda d: True)
        register_condition("keep_going", lambda d: False)

    @staticmethod
    def _construct_for(combo: ModifierCombo) -> Construct:
        """A minimal two-node Construct whose LAST node carries exactly ``combo``.

        Shapes are the empirically-minimal ones that survive assembly validation:
          * Each needs a real fan-out receiver -- a dict-form input key naming no
            peer producer (``item``), which ``_ir_normalize`` resolves into
            ``fan_out_param`` -- over a peer that produces the collection.
          * Loop needs SAME-type in/out on the looped node, or the loop back-edge
            is rejected before export is ever reached.
        """
        each = Each(over="prod.groups", key="label")
        oracle = Oracle(n=2, merge_fn="combine")
        operator = Operator(when="needs_review")
        loop = Loop(when="keep_going", max_iterations=3)

        if combo is ModifierCombo.ORACLE_OPERATOR:
            prod = Node.scripted("prod", "f", outputs=Clusters)
            target = Node.scripted("target", "f", inputs={"prod": Clusters}, outputs=Claims) | oracle | operator
        elif combo is ModifierCombo.LOOP_OPERATOR:
            prod = Node.scripted("prod", "f", outputs=Claims)
            target = Node.scripted("target", "f", inputs={"prod": Claims}, outputs=Claims) | loop | operator
        elif combo in (
            ModifierCombo.EACH_ORACLE,
            ModifierCombo.EACH_OPERATOR,
            ModifierCombo.EACH_ORACLE_OPERATOR,
        ):
            prod = Node.scripted("prod", "f", outputs=Clusters)
            target = Node.scripted("target", "f", inputs={"item": ClusterGroup}, outputs=Claims) | each
            if combo in (ModifierCombo.EACH_ORACLE, ModifierCombo.EACH_ORACLE_OPERATOR):
                target = target | oracle
            if combo in (ModifierCombo.EACH_OPERATOR, ModifierCombo.EACH_ORACLE_OPERATOR):
                target = target | operator
        else:  # pragma: no cover -- a new fallthrough combo must be shaped here
            raise AssertionError(
                f"{combo.name} is in FALLTHROUGH_COMBOS but this builder has no shape for it -- "
                "add one rather than dropping the combo from the pin"
            )

        return Construct(f"fallthrough-{combo.name.lower()}", nodes=[prod, target])

    def test_no_combo_falls_through_to_a_provisional_raise(self) -> None:
        """Loud partition check: NO ModifierCombo lacks a lowering arm any more.

        Phase 7 emptied this set by giving all five fusion combos real lowerings.
        The set stays DERIVED (UNSUPPORTED_COMBOS - the Portal pair), so a combo
        that later LEAVES SUPPORTED_COMBOS repopulates it and fails here -- the
        ratchet against quietly reintroducing a "not supported yet" raise."""
        assert {c.name for c in FALLTHROUGH_COMBOS} == set(), (
            "a combo fell back out of SUPPORTED_COMBOS -- every non-Portal combo must "
            "have a real Agent Spec lowering since neograph-s7zt3.10; do not re-add a "
            f"provisional raise for {sorted(c.name for c in FALLTHROUGH_COMBOS)}"
        )
        assert _PORTAL_COMBOS <= UNSUPPORTED_COMBOS, (
            "the PORTAL combos must still be UNSUPPORTED -- they raise the SEPARATE, "
            "PERMANENT dispatch-mode-Portal error, not a provisional one"
        )

    def test_the_generic_fallthrough_raise_no_longer_exists(self) -> None:
        """The provisional raiser is DELETED, not merely unreachable.

        Leaving ``_raise_no_agent_spec_lowering`` in the tree with zero call sites
        would let a future arm quietly re-adopt the "…yet" wording for a combo
        that is now genuinely supported. Asserting on the SYMBOL is the only way
        to pin a deletion (there is no behaviour left to observe)."""
        import neograph._agent_spec as agent_spec

        assert not hasattr(agent_spec, "_raise_no_agent_spec_lowering"), (
            "_raise_no_agent_spec_lowering came back -- every non-Portal combo has a real "
            "lowering since neograph-s7zt3.10. If a NEW combo genuinely has no lowering, "
            "give it an explicit, permanent, combo-specific message instead."
        )

    def test_dispatch_mode_portal_keeps_its_own_permanent_message(self) -> None:
        """The hoisted unconditional Operator postlude must NOT have collapsed the
        PORTAL arm's specific rejection into a generic one, and must not have made
        ``case _``/``assert_never`` reachable (which would swap the error TYPE)."""
        from pydantic import BaseModel

        from neograph._agent_spec import to_agent_spec
        from neograph.errors import ConfigurationError
        from neograph.modifiers import Portal

        from .fakes import register_scripted

        class Emitted(BaseModel, frozen=True):
            spec: dict
            dispatch_input: dict

        class Summary(BaseModel, frozen=True):
            text: str

        register_scripted("_p7_planner", lambda i, c: Emitted(spec={}, dispatch_input={}))
        construct = Construct(
            "dispatch-portal",
            nodes=[
                Node.scripted("planner", fn="_p7_planner", outputs=Emitted)
                | Portal(route="decide", spec_field="spec", input_field="dispatch_input", output=Summary, max_depth=3)
            ],
        )

        with pytest.raises(ConfigurationError) as exc_info:
            to_agent_spec(construct)

        message = str(exc_info.value)
        assert "dispatch-mode Portal" in message, message
        assert "no Agent Spec lowering yet" not in message, (
            "PORTAL's PERMANENT rejection was downgraded to the provisional wording"
        )

    @pytest.mark.parametrize("combo", FALLTHROUGH_COMBOS, ids=lambda c: c.name)
    def test_unsupported_combo_raises_configuration_error_with_exact_message(self, combo: ModifierCombo) -> None:
        """The exact user-visible contract: type ``ConfigurationError``, and the
        full four-line structured body (what / expected / found / hint) verbatim.

        Asserting the WHOLE message (not a regex fragment) is deliberate -- the
        migration must not reword, re-order, or drop any structured field, and a
        substring match would not catch a lost ``expected=``/``hint=``."""
        from neograph._agent_spec import to_agent_spec
        from neograph.errors import ConfigurationError

        self._register()
        construct = self._construct_for(combo)

        with pytest.raises(ConfigurationError) as exc_info:
            to_agent_spec(construct)

        assert str(exc_info.value) == (
            f"node 'target' has modifier combination {combo.name} — no Agent Spec lowering yet\n"
            "  expected: BARE, ORACLE, EACH, LOOP, or OPERATOR\n"
            f"  found: {combo.name}\n"
            "  hint: composed modifier lowering (e.g. Each+Oracle) is out of scope for "
            "i3zsh's primitive-level export"
        )

    @pytest.mark.parametrize("combo", FALLTHROUGH_COMBOS, ids=lambda c: c.name)
    def test_fallthrough_is_not_the_dispatch_mode_portal_raise(self, combo: ModifierCombo) -> None:
        """The two raises in ``_lower_construct_item`` must stay DISTINCT sites.

        The dispatch-mode-Portal rejection is PERMANENT ('no Agent Spec
        lowering' -- there is no runtime-flow-synthesis primitive); the fusion
        fallthrough is provisional ('...yet', owned by neograph-s7zt3.10).
        Hoisting the ``has_operator`` check out of the per-shape arms into one
        pre-dispatch test would collapse them and silently downgrade PORTAL's
        permanent message to the provisional one -- so pin that they do not
        cross-fire."""
        from neograph._agent_spec import to_agent_spec
        from neograph.errors import ConfigurationError

        self._register()
        construct = self._construct_for(combo)

        with pytest.raises(ConfigurationError) as exc_info:
            to_agent_spec(construct)

        message = str(exc_info.value)
        assert "dispatch-mode Portal" not in message, (
            f"{combo.name} fired the dispatch-mode-Portal raise instead of the generic fallthrough"
        )
        assert "Each x Oracle fusion is not supported on sub-constructs" not in message, (
            f"{combo.name} fired the SUB_CONSTRUCT_UNSUPPORTED_COMBOS gate -- that gate is "
            "Construct-item-only and must not capture a bare Node"
        )
        assert message.endswith("i3zsh's primitive-level export")
