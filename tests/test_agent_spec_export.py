"""Regression tests for ``to_agent_spec()`` -- neograph ``Construct``/IR ->
Agent Spec ``Flow`` export (neograph-i3zsh).

Gated on ``pyagentspec`` via ``pytest.importorskip`` -- the ``[agent-spec]``
optional extra keeps ``src/neograph`` core dependency-light by default. Run
with::

    uv run --extra dev --extra agent-spec pytest tests/test_agent_spec_export.py

## Step 1 gate (i3zsh implementation plan, sequenced FIRST)

Per the ratification's residual/unverified item (agent-spec-ratification-
2026-07-13.md s6) and i3zsh's own "Risks & Edge Cases" note: the whole
Layer A/B ``neograph/``-prefixed ``metadata`` marker round-trip strategy
(stamping ``metadata["neograph/modifier"]`` etc. on lowered nodes so an
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

from .schemas import Claims, RawText, _consumer, _producer  # noqa: E402


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
            metadata={"neograph/modifier": "oracle", "neograph/group_id": "g1"},
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
            "neograph/modifier": "oracle",
            "neograph/group_id": "g1",
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
            "neograph/modifier": "oracle",
            "neograph/group_id": "g1",
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

        data_edges = [e for e in flow.data_flow_connections if isinstance(e, DataFlowEdge)]
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
        marker = spec_node.metadata["neograph/agent_spec"]

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

        oracle_nodes = [n for n in flow.nodes if n.metadata and n.metadata.get("neograph/modifier") == "oracle"]
        assert len(oracle_nodes) == 3, "expected 2 variant nodes + 1 merge node, all marker-stamped"
        group_ids = {n.metadata["neograph/group_id"] for n in oracle_nodes}
        assert len(group_ids) == 1, "all Oracle-group nodes must share one group_id"

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
        assert map_nodes[0].metadata["neograph/modifier"] == "each"
        assert map_nodes[0].metadata["neograph/each_spec"]["over"] == "items"

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
            n for n in flow.nodes if isinstance(n, BranchingNode) and n.metadata.get("neograph/modifier") == "loop"
        ]
        assert len(branch_nodes) == 1
        assert branch_nodes[0].metadata["neograph/loop_spec"]["when"] == "claims_incomplete"
        back_edges = [
            e
            for e in flow.control_flow_connections
            if isinstance(e, ControlFlowEdge) and e.from_node.name == branch_nodes[0].name and e.from_branch == "continue"
        ]
        assert len(back_edges) == 1, "expected a cyclic ControlFlowEdge back into the loop body"

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
            n for n in flow.nodes if isinstance(n, BranchingNode) and n.metadata.get("neograph/modifier") == "operator"
        ]
        assert len(checks) == 1
        assert checks[0].metadata["neograph/operator_spec"]["when"] == "needs_review"
        assert checks[0].mapping["true"] == "pause"

        pause_nodes = [n for n in flow.nodes if isinstance(n, InputMessageNode)]
        assert len(pause_nodes) == 1

        pause_edges = [
            e
            for e in flow.control_flow_connections
            if isinstance(e, ControlFlowEdge) and e.to_node.name == pause_nodes[0].name
        ]
        assert any(e.from_branch == "pause" for e in pause_edges), (
            "expected the PAUSE_BRANCH edge (not DEFAULT_BRANCH) into InputMessageNode"
        )
